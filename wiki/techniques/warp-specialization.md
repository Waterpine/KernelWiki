---
id: technique-warp-specialization
title: Warp Specialization on Blackwell
type: technique
architectures: [sm100, sm90]
tags: [warp-specialization, tcgen05, tmem]
confidence: verified
reproducibility: snippet
prerequisites: [hw-tmem, hw-tcgen05-mma]
related: [technique-persistent-kernels, technique-pipeline-stages, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, pr-cutlass-3106, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-3106
    evidence_type: upstream-code
blackwell_relevance: "tcgen05 hardware issue can be single-threaded while software wrappers and surrounding load/epilogue work remain collective."
artifact_dir: artifacts/kernels/warp-specialization
---

## Overview

Warp specialization assigns persistent producer, MMA/control, softmax, or
epilogue roles to different warps so independent hardware resources can
overlap. The number of warps and role mapping are kernel choices.

```cuda
__global__ void specialized_kernel(Params params) {
    int warp = threadIdx.x / warpSize;
    if (warp == params.producer_warp) producer_loop(params);
    else if (warp == params.mma_warp) mma_loop(params);
    else epilogue_loop(params, warp);
}
```

Although one elected thread issues the tcgen05 hardware MMA, CuTe/CUTLASS
interfaces can require a warp-uniform call and elect internally. TMEM alloc,
load/store, and dealloc are warp-collective. It is therefore wrong to infer that
the other 31 lanes or every other warp are “free.”

The pinned CUTLASS tutorial demonstrates several mappings, pipelines, and CTA
sizes. There is no canonical 16-warp CTA with exactly one producer, one math,
and fourteen epilogue warps, and no rule to use specialization in every
Blackwell kernel. Specialization is worthwhile when overlap outweighs extra
barriers, registers, shared memory, and underfilled roles.

Artifacts are under
[`artifacts/kernels/warp-specialization/`](../../artifacts/kernels/warp-specialization/).
