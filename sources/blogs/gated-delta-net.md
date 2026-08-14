---
id: blog-gated-delta-net
title: Gated Delta Networks
author: NVlabs
url: https://github.com/NVlabs/GatedDeltaNet
source_category: benchmark-blog
architectures:
- sm90
- sm100
tags:
- gated-delta-net
- linear-attention
- attention
- triton
- chunk-parallelism
retrieved_at: 2026-08-13
artifact_dir: artifacts/blogs/gated-delta-net/code
---

## Summary

The Gated DeltaNet repository accompanies the ICLR 2025 work on combining a delta-rule state update with data-dependent gating. It provides research code and points users to the Flash Linear Attention implementation for optimized kernels.

## Mechanism

For a simplified row-vector convention, a delta-rule update first compares the new value with the value retrieved from the current state:

```python
retrieved = k @ state
error = v - retrieved
state = decay * state + beta * outer(k, error)
```

Exact conventions, normalization, gates, and tensor shapes must be taken from the pinned implementation. Replacing the error term with an unconditional `outer(k, v)` changes the algorithm into an additive linear-attention recurrence.

## Implementation Notes

- recurrent decode carries a fixed-shape state rather than an ever-growing attention KV history;
- chunkwise training/prefill reformulates work inside chunks as matmuls while preserving ordered boundary-state propagation;
- the upstream repository's claims concern the model and its evaluated code, not a universal SM90/SM100 performance guarantee.

This entry does not attribute later Qwen model configurations or throughput claims to the 2025 Gated DeltaNet repository unless they are supported by a separate primary source.
