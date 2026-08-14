---
id: blog-colfax-cutlass
title: 'Colfax CUTLASS Tutorial: GEMM Kernels Using Tensor Memory for Blackwell'
author: Colfax Research
url: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, cute-dsl, warp-specialization, 2sm-cooperative]
retrieved_at: 2026-08-13
---

## Scope

The Colfax tutorial explains CUTLASS/CuTe abstractions around Blackwell tcgen05 MMA and TMEM. It discusses PTX wrappers/traits, tensor layouts, and sub-byte/block-scaled GEMM construction.

TMEM is organized as 512 columns across 128 lanes with 32-bit cells. Allocation/deallocation is collective for a warp and uses legal power-of-two column counts. MMA issue, commitment to an mbarrier, completion wait, TMEM load, `tcgen05.wait::ld`, and deallocation are distinct protocol steps.

The old summary described the operation as “register-free,” treated a single-thread issue rule as the whole execution/synchronization model, and embedded synthesized PTX/CUTLASS types that omitted completion and deallocation. Those fragments were not the linked tutorial's verbatim code and were removed. CUTLASS wrapper names are version-specific; use the captured artifact and its provenance or the current tutorial source.

The related sub-byte tutorial covers NVFP4/MXFP formats, but their scale kinds and block sizes must not be interchanged.
