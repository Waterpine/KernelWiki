---
id: blog-tilus-nvidia
title: "Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation"
author: NVIDIA
url: https://github.com/NVIDIA/tilus
source_category: community-note
architectures: [sm100, sm90]
tags: [nvfp4, fp4, fp6, fp8, gemm, swizzling, pipeline-stages, ptx]
retrieved_at: 2026-08-13
---

# Tilus

Tilus is NVIDIA's tile-level kernel language with explicit control of shared-memory and register layouts. Its paper focuses on arbitrary low-precision computation, and the repository provides a Python interface, automatic layout inference, autotuning, and caching.

The repository's project-news timeline records an initial Ampere release
(v0.1.0, April 2025) and v0.2.0 (July 2025) with Hopper and Blackwell support
plus Blackwell matmul tutorials. Those are the dates displayed by the project,
not the GitHub Release-object timestamps: the current release page for v0.2.0
was published 2026-04-17. “Supports arbitrary 1–8-bit computation” describes
the compiler research goal; individual operations, dtypes, and targets still
need to be checked in the current implementation.

TMEM on Blackwell stores MMA accumulators and supported copied data according to tcgen05 rules; it should not be described generically as an operand-memory level. TMA, warp specialization, cluster behavior, and native low-precision instruction selection are target- and kernel-specific.

Primary links:

- [Tilus repository](https://github.com/NVIDIA/tilus)
- [Documentation](https://nvidia.github.io/tilus/)
- [ASPLOS 2026 paper](https://doi.org/10.1145/3760250.3762219)
