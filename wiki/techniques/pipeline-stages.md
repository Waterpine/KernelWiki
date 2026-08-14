---
id: technique-pipeline-stages
title: "Software Pipelining and Multi-Stage Buffering"
type: technique
architectures: [sm100, sm90]
tags: [pipeline-stages, double-buffering, tma, mbarrier]
confidence: verified
reproducibility: snippet
prerequisites: [hw-tma, hw-tmem]
related: [technique-warp-specialization, technique-double-buffering, hw-tma]
sources: [doc-ptx-isa-sm100, pr-cutlass-3106, blog-tcgen05-tutorial, blog-modular-blackwell, doc-nvidia-tuning-guide]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-3106
    evidence_type: upstream-code
blackwell_relevance: "TMA completion and tcgen05 completion use distinct barrier events; stage depth is a resource/timing choice."
---

## Overview

A staged pipeline overlaps an asynchronous producer with a consumer by cycling
through separately synchronized buffers. A stage is reusable only after the
consumer has released it; a consumer may use it only after the producer's
transaction has completed.

```cuda
template <int Stages>
__device__ void consume_tiles(int count) {
    for (int tile = 0; tile < count; ++tile) {
        int stage = tile % Stages;
        producer_acquire_and_copy(stage, tile);
        consumer_wait(stage);
        consumer_compute(stage);
        consumer_release(stage);
    }
}
```

Production kernels separate producer and consumer agents and overlap iterations;
the serial-looking snippet names the required state transitions. TMA uses an
expected-byte mbarrier transaction. tcgen05 MMA uses `tcgen05.commit` and a
completion mbarrier before dependent TMEM access. Each reused barrier needs the
correct phase/parity.

Three to five stages are not a default rule. The useful depth depends on copy
latency, K-loop length, tile bytes, shared-memory capacity, barriers, and
occupancy. Tutorial progression numbers combine a particular shape and build
and do not isolate a universal pipeline speedup.
