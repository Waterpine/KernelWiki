---
id: technique-ping-pong-scheduling
title: Ping-Pong Scheduling
type: technique
architectures: [sm100]
tags: [ping-pong-scheduling, warp-specialization, tmem, pipeline-stages]
confidence: verified
reproducibility: snippet
prerequisites: [hw-tmem, technique-warp-specialization]
related: [kernel-flash-attention-4, technique-double-buffering]
sources: [doc-flash-attention-4, blog-flash-attention-4, pr-flash-attention-2441, blog-tcgen05-tutorial]
evidence_basis:
  - source_id: doc-flash-attention-4
    evidence_type: official-doc
  - source_id: pr-flash-attention-2441
    evidence_type: upstream-code
artifact_dir: artifacts/kernels/ping-pong-scheduling
---

# Ping-Pong Scheduling

FlashAttention-4 computes two query tiles per CTA. While tensor-core operations
run for one tile, a softmax warpgroup works on the other, then the roles
alternate. The key is overlapping independent resource use, not merely toggling
two integer buffer indices.

```python
def ping_pong_schedule(num_kv_tiles):
    for kv_tile in range(num_kv_tiles):
        current = kv_tile & 1
        other = current ^ 1
        launch_mma_for(current, kv_tile)
        run_softmax_for(other, kv_tile - 1)
        synchronize_handoff(current, other)
```

The real kernel stores multiple S/P/O intermediates in carefully partitioned
TMEM and uses separate warpgroups, register budgets, and barriers. A correct
implementation must also handle prologue and tail iterations and cannot have
both branches issue conflicting TMEM operations as the old teaching kernel did.

Ping-pong is not automatically superior. It is useful when the two phases use
different bottleneck resources and there is sufficient state for both tiles
without damaging occupancy.
