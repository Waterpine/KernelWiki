---
id: technique-persistent-kernels
title: Persistent Kernels with CLC
type: technique
architectures: [sm100]
tags: [persistent-kernel, clc, tile-scheduling]
confidence: verified
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc, doc-nvidia-tuning-guide, doc-cutlass-blackwell, pr-cutlass-2161, blog-tcgen05-tutorial]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
artifact_dir: artifacts/kernels/persistent-kernels
---

## Overview

A persistent kernel intentionally uses each resident CTA or cluster for multiple
logical work items. The launch size need not equal the physical SM count: it is
chosen from cluster shape, occupancy, resource limits, and the scheduler design.

Blackwell CLC can assist such a design, but it does not replace software tile
scheduling. A running worker attempts to cancel a grid item that has not yet
launched. On success, the response supplies that item's launch ID, which the
kernel maps to the corresponding tile.

## Static software persistence

```cuda
for (unsigned tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    compute_tile(map_launch_id_to_tile(tile));
}
```

This deterministic grid-stride form is useful when tile costs are similar. A
global atomic counter is another software option for variable-duration work,
at the cost of coordination traffic.

## CLC control flow

```text
compute the tile represented by this worker's original launch ID
issue clusterlaunchcontrol.try_cancel asynchronously
wait for the response transaction on its mbarrier
query_cancel(response)
if successful:
    map the returned launch ID to a tile and compute it
    repeat
else:
    stop; do not issue another request after observing failure
```

The concrete PTX request uses a 16-byte shared-memory response and an mbarrier
completion transaction. Multi-CTA clusters use the documented cluster-level
protocol. CUTLASS may pipeline requests with computation, so use its versioned
implementation instead of treating the pseudocode as inline PTX.

## What CLC does and does not do

CLC can reduce last-wave loss and adapt to uneven worker availability while
not-yet-launched grid work remains. It does not choose a raster order, chain
unrelated problems, eliminate launch overhead, or guarantee balance for
arbitrary variable-cost tasks. A failed cancellation is a terminal result for
that request sequence, not a retry race.

The tcgen05 tutorial's reported persistent-kernel result belongs to one tuned
GEMM progression whose other choices also changed. It is evidence that the
combined implementation worked for that shape, not an isolated or universal
CLC speedup.

## Reproduction path

Verbatim and derived artifacts are under
[`artifacts/kernels/persistent-kernels/`](../../artifacts/kernels/persistent-kernels/).
Use `PROVENANCE.yaml` to distinguish pinned upstream files from teaching
variants.
