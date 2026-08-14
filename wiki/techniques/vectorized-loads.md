---
id: technique-vectorized-loads
title: "Wide Vectorized Loads and Cache Policies"
type: technique
architectures: [sm100, sm90]
tags: [vectorized-loads, cache-policy, register-budgeting]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-nvfp4-gemv, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, blog-yue-nvfp4, blog-amandeep-nvfp4, contest-gpumode-p1]
blackwell_relevance: "Load width is useful only when alignment, coalescing, instruction support, and downstream unpack/compute all match."
---

## Overview

Vector loads can reduce instruction count and expose aligned contiguous
transactions per thread. They do not change the warp's fundamental coalescing
rules or guarantee more DRAM bandwidth.

```cuda
__device__ uint4 load_aligned_16(const uint4* pointer, int index) {
    uint4 value = pointer[index];
    return value;
}
```

Before widening a load, verify pointer alignment, in-bounds tail handling,
per-lane address contiguity, register cost, and that unpack/conversion can keep
up. A 32-byte per-thread operation increases destination registers and may
reduce occupancy or instruction-level overlap.

Cache eviction hints are a separate choice, not an intrinsic property of a
vector load. The contest-derived “2000 us to 22.4 us” story combined many code
changes and was not an authoritative isolated measurement, so it is not
attributed to load width here.
