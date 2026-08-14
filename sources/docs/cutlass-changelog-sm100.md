---
id: doc-cutlass-changelog-sm100
title: "CUTLASS Changelog: SM100/Blackwell Entries"
url: https://docs.nvidia.com/cutlass/latest/CHANGELOG.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, nvfp4, fp4, fp6, fp8, block-scale, warp-specialization, persistent-kernel, gemm, grouped-gemm, attention, moe, mla, 2sm-cooperative, tile-scheduling, cute-dsl, epilogue-fusion, sparse-attention]
retrieved_at: 2026-08-13
---

# CUTLASS Changelog: SM100/Blackwell

## Verified milestones

- CUTLASS 3.8.0 (2025-01-25) introduced the first SM100 CuTe/CUTLASS building
  blocks: tcgen05 MMA atoms, Blackwell TMA extensions, TMEM data/copy support,
  Blackwell synchronization pipelines, CLC APIs, and supported narrow and
  block-scaled formats.
- CUTLASS 4.0.0 introduced the Python DSL release line and expanded SM100
  examples and kernels.
- CUTLASS 4.5.0 was released 2026-05-01, not 2026-03-27. Its Python DSL changes
  included `block_copy()` and additional block-scaled support.
- CUTLASS 4.6.0 was released 2026-07-01, followed by 4.6.1 on 2026-07-13
  and 4.6.2 on 2026-08-03. CUTLASS 4.7.0 was released 2026-08-04 and is the
  newest live-changelog entry at the 2026-08-13 audit date.

The live changelog has multiple maintained release branches, so numeric version
order alone does not imply that every later-dated maintenance release contains
the same feature set as another branch.

## Audit correction

The former page claimed to be comprehensive but omitted the 4.5.1--4.7.0
releases, used a wrong 4.5.0 date, and assigned several example numbers and
features without direct changelog anchors. This page now records only milestones
confirmed in the live official changelog. For a complete version-by-version
list, consult the linked changelog rather than this summary.
