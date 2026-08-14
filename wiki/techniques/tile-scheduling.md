---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
type: technique
architectures: [sm100, sm90]
tags: [tile-scheduling, clc, persistent-kernel]
confidence: verified
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-persistent-kernels, pattern-low-sm-utilization]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc, doc-cutlass-blackwell, pr-cutlass-2161]
blackwell_relevance: "SM100 CLC can support cancellation-based work stealing; tile-to-problem mapping and raster order remain software decisions."
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
---

## Overview

Tile scheduling maps a logical work index to an output tile and decides how
CTAs acquire additional work. Keep these two questions separate:

1. A software mapping chooses `(tile_m, tile_n, batch_or_group)` from an index.
2. A static, persistent-software, or CLC-assisted policy determines which CTA
   processes that index.

CLC does not provide a configurable Hilbert, Morton, or raster-order queue.
Its PTX operation attempts to cancel a cluster or CTA that has not launched and
returns the canceled launch identifier. Software then maps that identifier to
the corresponding tile.

## Static mappings

```cuda
struct TileCoord { int m, n; };

__device__ TileCoord row_major(unsigned index, unsigned tiles_n) {
    return {int(index / tiles_n), int(index % tiles_n)};
}

// Group nearby N tiles while retaining a one-to-one index mapping.
__device__ TileCoord grouped_n(unsigned index, unsigned tiles_m,
                               unsigned tiles_n, unsigned group_n) {
    unsigned group = index / (tiles_m * group_n);
    unsigned local = index % (tiles_m * group_n);
    unsigned m = local / group_n;
    unsigned n = group * group_n + local % group_n;
    return {int(m), int(n)}; // caller bounds-checks n < tiles_n
}
```

The best mapping is workload-dependent. Grouping can improve reuse of one
operand but worsen balance or reuse of the other, so there is no universally
best swizzle width.

## Persistent software scheduling

A conventional persistent kernel may use a deterministic grid-stride mapping
or a software atomic counter:

```cuda
for (unsigned index = blockIdx.x; index < tile_count; index += gridDim.x) {
    compute(map(index));
}

// Alternative when tile durations vary:
for (;;) {
    unsigned index = atomicAdd(next_tile, 1);
    if (index >= tile_count) break;
    compute(map(index));
}
```

The atomic approach is software scheduling, not CLC. It trades counter traffic
for dynamic balancing.

## CLC-assisted work stealing

On supported Blackwell targets, a running CTA or cluster can issue
`clusterlaunchcontrol.try_cancel.async...mbarrier::complete_tx::bytes.b128`.
After the mbarrier indicates completion, `query_cancel` reports whether a
not-yet-launched item was canceled and, on success, supplies its launch ID.
The requesting CTA can then execute the software tile mapping for that ID.

If cancellation fails because no cancellable launch remains, software follows
the documented termination path. Issuing another request after an observed
failed cancellation has undefined behavior in the PTX ISA. This makes CLC a
specialized work-stealing primitive, not an open-ended task queue.

## Selection guide

- Use a static raster or grouped mapping when tiles have similar cost and the
  launch already exposes enough parallelism.
- Use software persistence when controlling resident CTA count or balancing
  variable-duration work justifies the counter or coordination cost.
- Consider CLC work stealing when the target and runtime support it and the
  launch has not-yet-started CTAs that productive workers can cancel.
- Measure locality and tail behavior for the actual shape. Neither CLC nor a
  swizzle guarantees a speedup.
