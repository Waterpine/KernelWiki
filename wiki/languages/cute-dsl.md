---
id: lang-cute-dsl
title: "CUTLASS Python DSL / CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell]
sources: [doc-cutlass-blackwell, doc-cutlass-changelog-sm100, pr-cutlass-3106, pr-cutlass-3021, blog-colfax-cutlass, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-cutlass-blackwell
    evidence_type: official-doc
  - source_id: pr-cutlass-3106
    evidence_type: upstream-code
---

## Overview

CUTLASS Python DSL exposes CuTe layouts, pipelines, TMA copy atoms, tcgen05 MMA
atoms, and TMEM allocators. Names and launch conventions evolve between
CUTLASS releases, so examples must be tied to a pinned version.

## Pinned tutorial API

The captured CUTLASS PR 3106 tutorial constructs its SM100 MMA as follows:

```python
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

op = tcgen05.MmaF16BF16Op(
    io_dtype, acc_dtype, (128, 256, 16),
    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
    tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)
```

Its kernel uses `@cute.kernel`, `utils.TmemAllocator`,
`pipeline.PipelineTmaUmma`, `tcgen05.make_tmem_copy`, and explicit TMEM
allocation/free. These are concrete upstream interfaces. The former page's
`@cute_kernel`, `SM100_MMA_F16BF16_SS`, `SM100_TMA_LOAD_2D`, and
`alloc_tmem()` skeleton were not anchored to the pinned API.

## Tutorial progression

The pinned `fp16_gemm_0.py` through `fp16_gemm_6.py` examples progress through
a simple one-CTA GEMM, 2CTA/TMA multicast, warp specialization, static
persistence, preferred/dynamic clusters, TMA prefetch, and PDL. They are a
versioned learning path, not proof that every operation is automatic or that
one fixed pipeline is optimal.

The local verbatim captures are under
[`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/).
The CLC binding is captured under PR 3021.

Compilation-speed and kernel-throughput numbers cited by downstream projects
remain workload/version-specific. They are not general properties of CuTe DSL
and are omitted here unless attached to a reproducible benchmark.
