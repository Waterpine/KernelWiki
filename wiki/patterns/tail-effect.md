---
id: pattern-tail-effect
title: "Tail Effect — Last Wave Underutilization"
type: pattern
architectures: [sm100, sm90]
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
---

## Symptom

A grid's final launch wave contains fewer runnable CTAs or clusters than the
machine can host, leaving execution capacity idle while those last work items
finish. For uniform, one-tile CTAs, a useful first model is
`ceil(tile_count / concurrent_grid_items)` waves. The actual concurrency is a
launch- and resource-dependent quantity, not simply the product's advertised
SM count.

## Likely causes

1. The tile count is not a multiple of the number of simultaneously resident
   grid items.
2. Tile durations vary, so a few long-running items dominate completion.
3. A static mapping strands remaining work on busy workers.

## Candidate techniques

| Technique | What it can change |
|---|---|
| [Persistent kernels](../techniques/persistent-kernels.md) | Let a resident worker process more than one logical tile. |
| [CLC](../hardware/clc.md) | Let a worker claim a not-yet-launched CTA/cluster ID by cancellation. |
| [Tile scheduling](../techniques/tile-scheduling.md) | Change the software mapping or use a software work counter. |

CLC does not merge two waves into one or guarantee full utilization. A worker
can only steal an item that has not launched, and the gain must exceed request
and scheduling overhead. It also does not replace the software mapping from a
returned launch ID to a tile.

The tcgen05 tutorial reports a large improvement in one tuned GEMM progression,
but several changes were introduced across that progression. Treat its result
as a workload-specific observation, not a general `86% -> 98%` CLC guarantee.

## Measurement

Record grid dimensions, cluster shape, occupancy/resource limits, per-CTA
duration distribution, and the number of incomplete items near kernel exit.
Compare a static mapping, a software-persistent mapping, and CLC only when each
is supported by the same workload.
