---
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound, kernel-nvfp4-gemv]
sources: [doc-nvidia-tuning-guide, blog-yue-nvfp4, blog-amandeep-nvfp4]
---

## Diagnosis

Estimate bytes and useful operations for the exact cache/reuse behavior, then
compare achieved bandwidth with a measured roof for the same access pattern.
High DRAM percentage alone can hide uncoalesced over-fetch or unnecessary data.

Candidate changes include coalescing/alignment, reducing bytes, increasing
reuse, vectorizing where it reduces instructions, choosing appropriate cache
hints, or changing tiling. TMA multicast helps only when cluster CTAs genuinely
reuse the same transfer and its layout/protocol costs are justified.

Register limits and wider loads are tradeoffs rather than priorities: both can
increase spills or register footprint. Compute simplification can still matter
when decode/conversion prevents enough memory requests from being in flight.
Nominal product bandwidth is not a speed-of-light result without accounting for
all inputs, scales, outputs, transactions, and achievable efficiency.
