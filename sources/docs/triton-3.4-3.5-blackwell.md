---
id: doc-triton-pre36-blackwell
title: "Triton 3.4 and 3.5 Blackwell Support"
url: https://github.com/triton-lang/triton/releases/tag/v3.4.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, warp-specialization]
retrieved_at: 2026-08-14
---

# Triton 3.4 and 3.5 Blackwell support

Triton 3.6 was not the introduction point for native Blackwell lowering.
Official release records establish this earlier history:

- Triton 3.4.0, published 2025-07-30, reports enhanced Blackwell TMEM
  support, MMAv5 pipelining, and the introduction of automatic warp
  specialization.
- Triton 3.5.0, published 2025-10-21, reports generic lowering for
  `tcgen05.ld`/`tcgen05.st`, exposed `tcgen05_copy`, control-flow support in
  TMEM allocation, TMEM deallocation synchronization, and warp-specialization
  enablement for persistent matmul and FlashAttention paths.

These records establish existence and continued development, not universal
lowering for every `tl.dot` shape. Triton 3.6 remains the repository's
conservative supported-policy baseline for the combined documented surfaces;
it is not described as the first release with Blackwell TMEM, tcgen05, or
automatic warp specialization.

Official records:

- [Triton 3.4.0](https://github.com/triton-lang/triton/releases/tag/v3.4.0)
- [Triton 3.5.0](https://github.com/triton-lang/triton/releases/tag/v3.5.0)
