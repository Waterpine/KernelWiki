---
id: doc-triton-3.6-blackwell
title: "Triton 3.6.0 Release Notes — Blackwell Expansion"
url: https://github.com/triton-lang/triton/releases/tag/v3.6.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, block-scale, nvfp4, warp-specialization]
retrieved_at: 2026-08-14
---

# Triton 3.6.0 Blackwell expansion

Triton 3.6.0 was published 2026-01-21. Its official notes extend Blackwell
support already present in 3.4 and 3.5. Blackwell-relevant 3.6 changes include:

- bitwidth-aware TMEM encoding and more generic TMEM layouts;
- generalized `tcgen05.mma`, `tcgen05.ld`/`st`, and `tcgen05.cp` lowering;
- additional scaled-MMA and warp-specialization work; and
- initial Gluon multi-CTA/two-CTA and cluster support.

Version 3.6 is a defensible conservative policy baseline for the combined
surface above, but it did not introduce Blackwell TMEM, tcgen05 lowering, or
automatic warp specialization. Those existed in earlier official releases.
None of these notes proves that every ordinary `tl.dot` shape uses tcgen05,
and 3.6 is not the current stable release.

The old page also asserted that all pre-3.6 Blackwell code silently fell back to WGMMA. The release notes do not establish that universal runtime behavior, so that statement was removed.

For the earlier boundary see `doc-triton-pre36-blackwell`; for the current
stable snapshot see `doc-triton-3.7-blackwell`.
