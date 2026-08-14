---
id: technique-kernel-fusion
title: "Kernel Fusion"
type: technique
architectures: [sm100, sm90]
tags: [kernel-fusion, fused-kernel, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem]
related: [kernel-fused-moe, kernel-nvfp4-gemm, technique-epilogue-fusion]
sources: [contest-gpumode-p3, contest-flashinfer-track-a, blog-tflops-gap-fp4-moe]
blackwell_relevance: "Blackwell TMEM and tensor-core instructions can support fused GEMM/epilogue designs, subject to the instruction's legal allocation and synchronization rules."
---

# Kernel Fusion

## Overview

Kernel fusion keeps producer results on chip when a consumer can execute in the same launch. It can reduce launch overhead, global-memory traffic, and intervening synchronization. The benefit depends on tensor shapes and on whether fusion lowers occupancy or introduces redundant computation.

## Common Forms

- fuse an activation, scaling operation, bias, or quantization step into a GEMM epilogue;
- compute gate and up projections in one scheduled kernel, then apply the gated activation;
- combine routing, permutation, or reduction stages when their ownership and synchronization can be expressed safely.

```python
def fused_bias_activation(x, weight, bias):
    accumulator = matmul(x, weight)
    accumulator = accumulator + bias
    return activation(accumulator)
```

This mathematical sketch shows the eliminated intermediate, not a CUDA implementation.

On SM100, TMEM accumulators are allocated in documented power-of-two column units by a warp collectively. Their lifetime must include the required MMA completion mechanism, loads, `tcgen05.wait::ld`, and deallocation. A C-like call such as `tmem_alloc(256)` or `tcgen05_mma(...)` is only pseudocode unless it names a real wrapper with those semantics.

## Constraints

- register, shared-memory, and TMEM demand can reduce residency;
- CTA-local synchronization cannot satisfy a dependency between unrelated CTAs;
- a fused epilogue may become the critical path or limit vectorized stores;
- numerical ordering can change, especially for reductions and quantization;
- the best launch boundary varies with batch size, expert distribution, and framework overhead.

Counts such as "seven kernels to one" and fixed traffic savings are implementation-specific measurements, not properties of MoE fusion in general. Preserve them only with a pinned benchmark, exact shapes, hardware, software, and baseline.
