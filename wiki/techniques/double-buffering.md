---
id: technique-double-buffering
title: "Double/Multi-Buffering Patterns"
type: technique
architectures: [sm100, sm90]
tags: [double-buffering, tmem, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem]
related: [hw-tmem, technique-pipeline-stages, technique-epilogue-fusion]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, doc-nvidia-tuning-guide, pr-flashinfer-2387]
blackwell_relevance: "SMEM stages and TMEM accumulator buffers are separate resources with separate completion protocols."
---

## Overview

Multiple buffers let a producer fill one region while a consumer uses another.
For SM100 kernels, distinguish shared-memory operand stages from TMEM
accumulator buffers. Neither must have exactly two buffers, and splitting all
512 TMEM columns into two 256-column halves is only one legal design.

```cuda
template <int Stages>
__device__ void pipeline_loop(int tiles) {
    for (int tile = 0; tile < tiles; ++tile) {
        int stage = tile % Stages;
        wait_empty(stage);
        produce_async(stage, tile);
        wait_full(stage);
        consume(stage, tile);
    }
}
```

Correct reuse needs a phase-aware full/empty protocol. For TMA, the completion
transaction belongs to the stage's mbarrier. For tcgen05 MMA, completion must be
committed to and observed through the appropriate mbarrier before a dependent
TMEM load or reuse. Fence instructions alone are not completion signals.

Buffer count is constrained by shared memory, TMEM columns/layout, barriers,
registers, and useful work per stage. More stages can reduce latency stalls but
also reduce occupancy or add pipeline overhead. The previous page's fixed
16-warp schedule, 228-KB shared-memory limit, zero-cost claim, and “always use”
rules were not architectural facts and are removed.
