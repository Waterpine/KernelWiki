---
id: pattern-tail-effect
title: "Tail Effect — Last Wave Underutilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## Symptom

Performance drops for problem sizes where total_tiles % num_SMs != 0. The last wave of tiles runs with many SMs idle.

## Likely Causes

1. **Wave quantization**: Grid of N tiles on M SMs takes ceil(N/M) waves; last wave may use only N%M SMs
2. **Static assignment**: stride-by-gridDim leaves remainder tiles on few SMs
3. **Non-persistent launch**: each kernel launch has fixed grid, no dynamic rebalancing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC](../hardware/clc.md) | Hardware dynamic scheduling, SMs grab tiles on-demand |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM-count grid, iterate over tiles, no wave boundary |
| [Tile scheduling](../techniques/tile-scheduling.md) | Raster order, swizzle patterns for better distribution |

## Example

```
// B200: 148 SMs
// Problem: 150 tiles
// Without CLC: 2 waves (148 + 2), last wave uses only 2 SMs (1.4%)
// With CLC: the 2 leftover coordinates are taken over by the first 2 workers
//   that finish, and they run without paying a second prologue. Note there is
//   no "wave barrier" to remove even without CLC -- a coordinate that has not
//   started "will be launched as a Worker when there are available
//   resources", so the leftovers start as soon as any SM frees. What CLC adds
//   here is the persistent worker's amortized prologue/epilogue, plus the
//   ability to rebalance when SM availability is uneven or unknown at launch
//   (another kernel or a Green Context holding SMs).
//   The tail is still only 2 tiles wide: CLC relocates an unlaunched
//   cluster, it cannot spread 2 tiles over 148 SMs. Use Stream-K if the
//   tail itself must be parallelized.
//
// Impact in the tcgen05 tutorial: 86% → 98% of cuBLAS, measured for
// non-persistent (v5) vs persistent-with-static-scheduling (v6) — not CLC
```

## Caveats
- Only significant for moderate tile counts (< 4× SM count)
- For very large problems, tail effect is amortized across many waves
- CLC requires `sm_100` or higher (SM120 parts are also supported)
