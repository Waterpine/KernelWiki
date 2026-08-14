---
id: doc-cutlass-blackwell
title: "NVIDIA CUTLASS Blackwell Support"
url: https://docs.nvidia.com/cutlass/latest/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, 2sm-cooperative, nvfp4, fp8, fp4, fp6, block-scale, cute-dsl]
retrieved_at: 2026-08-13
---

# NVIDIA CUTLASS Blackwell Support

CUTLASS 3.8 introduced its first SM100 building blocks: tcgen05 MMA atoms, TMA
copy extensions, TMEM as a CuTe data locale, TMEM copy atoms, Blackwell
pipelines, CLC APIs, and narrow/block-scaled data types. CUTLASS 4.x adds and
continues to change its Python DSL, examples, builders, and kernels.

## Stable concepts, versioned names

- CuTe represents tcgen05 instructions as MMA atoms and composes them into a
  tiled MMA with legal layouts.
- TMEM allocation, copy, and pipeline objects preserve the collective and
  asynchronous hardware contract.
- TMA atoms/layouts must match the shared-memory layout consumed by the MMA.
- CLC-backed schedulers implement cancellation-based work stealing; raster and
  logical tile mapping remain software.
- block-scaled FP8/MXFP4/NVFP4 use distinct element, scale, granularity, and
  descriptor contracts.

Concrete class and schedule names must be taken from the chosen CUTLASS
release. The former summary listed several unverified schedule/epilogue types
and a fabricated MMA atom spelling, so those code blocks were removed.

## Current documentation state

As of 2026-08-13 the live changelog includes CUTLASS 4.7.0 (2026-08-04),
following 4.6.2 (2026-08-03); maintained release lines also have separate
maintenance entries. The Python documentation now has a dedicated tcgen05
programming guide and API references for `nvgpu.tcgen05`.

No general 98%-of-cuBLAS, DeepGEMM-equivalence, or FlashMLA-equivalence claim is
made by the relevant changelog entries. Such results require a versioned
example, exact shape, build, GPU state, metric definition, and benchmark log.

## Sources

- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/)
- [tcgen05 MMA programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)
- [CUTLASS changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
