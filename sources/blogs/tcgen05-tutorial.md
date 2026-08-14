---
id: blog-tcgen05-tutorial
title: tcgen05 for dummies
author: Gau Nernst
url: https://gau-nernst.github.io/tcgen05/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, swizzling, pipeline-stages, persistent-kernel, warp-specialization, mbarrier, cuda-cpp, ptx]
retrieved_at: 2026-08-13
---

# tcgen05 for dummies

This community tutorial builds one Blackwell GEMM implementation step by step. For its benchmark, it reports about 255 TFLOP/s for the basic version, 695 after the selected swizzle, 940 after pipelining, and 1476 for the persistent version versus 1507 for its cuBLAS comparison.

The progression is source-reported evidence for the tutorial's code, hardware, shapes, and harness. It is not a universal 17%→98% optimization recipe, and its swizzle result does not make 128-byte swizzling legal or optimal for every tensor map.

Technical points that agree with the PTX ISA include direct shared-memory descriptors for supported tcgen05 operands, TMEM's 512-column by 128-lane 32-bit organization, explicit collective allocation/deallocation, and phase-aware mbarrier pipelines.

The former source summary embedded synthesized CUDA/PTX that omitted or misstated allocation, TMA, MMA-completion, and TMEM-load protocols. That code was removed. Use the pinned artifact bundle and its `PROVENANCE.yaml` for captured source; check instruction grammar against the current PTX ISA before reuse.
