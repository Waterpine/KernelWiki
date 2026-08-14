---
id: technique-cache-policy
title: "PTX Cache Policy Differentiation"
type: technique
architectures: [sm100, sm90]
tags: [cache-policy, vectorized-loads]
confidence: verified
reproducibility: snippet
prerequisites: []
related: [technique-vectorized-loads, kernel-nvfp4-gemv, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: blog-simon-nvfp4-gemv
    evidence_type: upstream-code
blackwell_relevance: "PTX eviction-priority hints are advisory and require measurement on the target GPU and workload."
---

# Cache Policy Differentiation

PTX load qualifiers such as `L1::no_allocate`, `L1::evict_first`, and
`L1::evict_last` communicate eviction priority/admission intent. They do not
reserve cache capacity or guarantee residency.

```ptx
ld.global.L1::no_allocate.v4.u32 {a0,a1,a2,a3}, [streaming_ptr];
ld.global.L1::evict_last.v4.u32  {b0,b1,b2,b3}, [reused_ptr];
add.u32 a0, a0, b0;
```

Use a streaming hint only when reuse analysis and profiling support it. A value
that is reused by other CTAs may benefit from L2 even when one CTA uses it once;
conversely, “evict last” can harm other working sets. Alignment, coalescing,
instruction count, and occupancy often dominate the hint itself.

Blog reconstructions from the NVFP4 contest report using different policies by
shape. Their `443 us -> 27 us` progression combined layout, decode, assembly,
and cache changes, so it is not evidence of a 16x cache-policy effect. The
former B200 cache-size and bandwidth thresholds were also removed because they
were neither product-qualified nor necessary to the technique.
