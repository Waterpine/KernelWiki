---
id: blog-simon-nvfp4-gemv
title: NVFP4 GEMV and Improved NVFP4 GEMV
author: Simon Veitner
url: https://veitner.bearblog.dev/nvfp4-gemv/
source_category: community-note
architectures:
- sm100
tags:
- nvfp4
- gemv
- fp4
- block-scale
- cute-dsl
- vectorized-loads
- register-reuse
- batched-gemv
retrieved_at: 2026-08-14
---

# NVFP4 GEMV and Improved NVFP4 GEMV (Simon Veitner)

## Overview

Simon Veitner's two-part blog series on implementing NVFP4 GEMV kernels using CuTe DSL for Blackwell GPUs. Part 1 presents a reference CuTe implementation with hierarchical tensor layouts for FP4 block-scaled GEMV. Part 2 ("Improved") introduces three optimization strategies that parallelize the K-dimension reduction, achieving up to 6.4x speedup over the reference.

## Part 1: Reference CuTe DSL Implementation

### Configuration

- Tile dimensions: (128, 1, 64) for M, N, K
- Data types: FP4 (E2M1) for weight matrix, FP8 (E4M3) for scale factors, FP16 output
- Scale factor block size: 16 elements per FP8 scale value
- Thread block size: 128 threads per CTA

### NVFP4 Format Handling

NVFP4 is composed of two tensors: one in FP4 precision and another in FP8 precision. The scale factor applies to every block of 16 values to minimize quantization error while reducing memory footprint.

### Memory Access Patterns

The kernel uses hierarchical CuTe tensor indexing:
- Matrix A: shape (128, 64, 1, 4, 1) representing one M-tile with four K-tiles
- Scale factors: hierarchical layout ((32,4), (16,4)) enabling broadcast optimization
- Vector B: single N-tile with K-dimension iteration

### Core Computation

The computation loop applies two-level scaling:
```
res += tArA[i] * tArSFA[i] * tBrB[i] * tBrSFB[i]
```
FP4 values are multiplied by their corresponding FP8 scale factors before accumulation. Values convert from FP4/FP8 to FP32 registers for computation, then convert to FP16 for storage.

## Part 2: Improved NVFP4 GEMV

### Three Optimization Strategies

**Strategy 1: Extra Blocks (K-Parallel Grid)**
- Launches grid blocks corresponding to K-tiles, eliminating the K-tile loop
- Uses atomic operations on global memory (F32 accumulation buffer)
- Source benchmark means: ~36,864 (benchmark 0), ~55,399 (benchmark 1); the post's emitted output does not label the unit
- Achieves 6.4x improvement on benchmark 0 versus reference

**Strategy 2: Thread-Level with Atomic Add**
- Distributes work across thread dimensions (threads_per_m=32, threads_per_k=32)
- Uses shared memory atomics for collaborative result calculation
- Avoids separate F32 tensor allocation overhead
- Source benchmark means: ~38,911 (benchmark 0), ~67,258 (benchmark 1); unit not stated

**Strategy 3: Thread-Level with Reduction (No Atomics)**
- Allocates 2D shared memory tensor (K-major stride)
- Each thread pair stores intermediate results, then performs synchronous reduction
- Source benchmark means: ~38,911 (benchmark 0), ~65,599 (benchmark 1); unit not stated

### Key Technical Changes

- Serial K-dimension loop replaced with parallel K-reduction
- Intermediate products converted to FP16 for storage efficiency
- FP32 maintained for scale factor operations
- Synchronization barriers inserted between computation and reduction phases
- K-major memory layouts optimize the reduction step access patterns

### Performance Summary

Reference benchmark-0 mean: ~234,495 in the source's unlabeled output unit. The extra-block result of ~36,864 is about 6.4x lower for that case; other shapes show different gains.

Primary posts: [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/) and [NVFP4 GEMV improved](https://veitner.bearblog.dev/nvfp4-gemv-improved/).

## Key Insights

- CuTe DSL's hierarchical tensor layouts naturally express NVFP4's two-level scaling structure
- K-dimension parallelism is critical for GEMV performance on Blackwell
- Atomic-free reductions in shared memory match or approach atomic-based approaches
- The choice between strategies depends on K-dimension size and available SM resources

## Source-code boundary

The original posts implement these kernels in CuTe DSL Python, including
`@cute.kernel` entry points and CuTe tensor/layout operations. Earlier versions
of this page included reconstructed CUDA C++ fragments that were not present in
the cited posts. Those synthetic fragments have been removed; use the linked
posts for the actual CuTe DSL source and treat the prose above as a structural
summary only.
