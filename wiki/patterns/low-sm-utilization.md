---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## Symptom

SM utilization below 60% despite sufficient occupancy. Nsight Compute shows idle SMs during portions of kernel execution.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../hardware/clc.md) | sm_100+ (incl. SM120) | Dynamic tile assignment; rebalances work by relocating unlaunched clusters |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM90+ | Amortizes launch overhead and shortens the tail; the last wave is still bounded by the remaining tile count |
| [Tile scheduling](../techniques/tile-scheduling.md) | SM90+ | Better L2 locality, reduce load variance |

## Examples

```
// tcgen05 tutorial progression:
// Non-persistent (2-SM MMA, v5):        86% of cuBLAS
// Persistent, static scheduling (v6):   98% of cuBLAS
// (the tutorial did not use CLC; the author leaves it as an exercise)
```

## Caveats
- CLC requires `sm_100` or higher; the `.multicast::cluster::all` qualifier is additionally supported on `sm_120a` / `sm_120f`, so SM120 parts are not excluded
- Persistent kernels complicate debugging and profiling
- For non-persistent kernels, ensure grid size >> SM count
