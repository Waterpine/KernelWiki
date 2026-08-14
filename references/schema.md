# Schema Reference

Condensed reference for the wiki's controlled vocabulary and page schemas. Full definitions live in `data/schemas.yaml`.

## Page Types and IDs

Every page has a unique `id` with a type-specific prefix:

| Type | ID Prefix | Purpose |
|------|-----------|---------|
| source-pr | `pr-<repo>-<N>` | A merged PR from a tracked repo |
| source-doc | `doc-*` | Official NVIDIA docs, papers |
| source-blog | `blog-*` | Community blog posts, tutorials |
| source-contest | `contest-*` | Competition problems / tracks |
| wiki-hardware | `hw-*` | Blackwell hardware feature pages |
| wiki-technique | `technique-*` | Optimization techniques |
| wiki-kernel | `kernel-*` | Kernel case studies with perf claims |
| wiki-pattern | `pattern-*` | Problem → solution diagnosis |
| wiki-language | `lang-*` | DSL / language guides |
| wiki-migration | `migration-*` | Hopper → Blackwell migration |

## Required Frontmatter by Type

### source-pr

For generated `source-pr` pages, `date` is the UTC calendar date from GitHub's
`created_at` field. It is deliberately distinct from the merge date; a merged
page records merge state and `merge_sha`, but does not repurpose `date`.
The block below is a shape/template example: placeholder fields such as
`username`, `abc12345`, and the abbreviated title are not an authoritative
record of CUTLASS PR 2472.

```yaml
id: pr-cutlass-2472
repo: NVIDIA/cutlass
pr: 2472
title: "Add Blackwell MLA forward"
author: username
date: 2025-07-16
url: https://github.com/NVIDIA/cutlass/pull/2472
source_category: upstream-code
architectures: [sm100]
tags: [mla, attention, prefill]
techniques: [warp-specialization, pipeline-stages]
hardware_features: [tcgen05, tmem, tma]
kernel_types: [mla, attention, prefill]
languages: [cute-dsl]
captured_at: 2026-04-17
status: merged
merge_sha: abc12345
inclusion_reason: "kernel file changes"
changed_paths: [...]
```

### wiki-kernel (must have `performance_claims`)
```yaml
id: kernel-flash-attention-4
title: "FlashAttention-4"
type: kernel
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, flash-attention]
languages: [cute-dsl]
related: [technique-warp-specialization, technique-software-exp, hw-tcgen05-mma]
sources: [doc-flash-attention-4, blog-flash-attention-4, pr-...]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: "maximum over B200 BF16 sweep: seqlen=1K-32K, 32K total tokens"
    metric: TFLOPS
    value: 1613
    utilization: "71%"
    source_id: doc-flash-attention-4
```

### wiki-pattern (diagnostic flow)
```yaml
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound]
sources: [...]
```

## Confidence Levels

- **`verified`**: Requires ≥1 `official-doc` + ≥1 `upstream-code` in sources. Enforced by validator.
- **`source-reported`**: Cited by ≥1 authoritative source (paper, major blog, major repo).
- **`inferred`**: Synthesized from multiple sources, no single authoritative one.
- **`experimental`**: Undocumented, PTX tricks, version-sensitive. Include CUDA version.

## Reproducibility Levels

For `wiki-technique`, `wiki-kernel`, `wiki-language`, must be ≥ `snippet`.

| Level | Meaning |
|-------|---------|
| `concept` | Text only |
| `pseudocode` | Language-agnostic algorithm |
| `snippet` | Language-typed fragment with real syntax; surrounding declarations may be required. The validator performs a structural check, not a compilation. |
| `runnable` | Self-contained buildable example |
| `benchmarked` | Runnable + perf numbers with env metadata |

## Controlled Vocabulary

All values in these fields must appear in `data/tags.yaml`:

- **architectures**: sm100, sm100a, sm103, sm103a, sm90, sm90a, sm120

For generated source-PR pages, `architectures` records only targets explicitly
named by captured PR text or changed paths. An empty list means the exact target
is not established; candidate inclusion by itself is topical relevance, not
architecture evidence.
- **hardware_features**: tcgen05, tmem, tma, clc, 2sm-cooperative, pdl, gdc, nvfp4, mxfp4, fp8, fp6, fp4, block-scale, wgmma, cluster, mbarrier
- **architectures** records exact targets or benchmark devices directly named by the captured source. When both a compiled component target and a different benchmark device are listed, the page body must distinguish those evidence roles.
- **techniques**: warp-specialization, persistent-kernel, swizzling, pipeline-stages, double-buffering, register-reuse, epilogue-fusion, tile-scheduling, tma-multicast, software-exp, fine-grained-quantization, cuda-core-promotion, jit-compilation, vectorized-loads, cache-policy, kernel-fusion, chunk-parallelism, loop-unrolling, register-budgeting, shared-memory-optimization, ping-pong-scheduling, conditional-rescaling, data-reuse, per-k-specialization
- **kernel_types**: gemm, attention, moe, sparse-attention, gemv, grouped-gemm, gated-delta-net, fused-kernel, decode, prefill, quantization, flash-attention, mla, linear-attention, gated-dual-gemm, batched-gemv
- **languages**: cuda-cpp, cute-dsl, triton, tilelang, cutile, ptx, python, jax-pallas
- **source_category**: official-doc, upstream-code, paper, benchmark-blog, contest-report, community-note

## Canonical Aliases (from data/aliases.yaml)

When asking about:
- UMMA → canonical tag is `tcgen05`
- Tensor Memory / TMEM → `tmem`
- Cluster Launch Control / CLC → `clc`
- Blackwell / B200 → architecture `sm100`
- Hopper / H100 → architecture `sm90`
- MoE / Mixture of Experts → `moe`
- MLA / Multi-head Latent Attention → `mla`
- GDN / Gated Delta Net → `gated-delta-net`
- NSA / Native Sparse Attention → `sparse-attention`
- WGMMA / wgmma.mma_async → `wgmma`

## Cross-Reference Fields

- `sources`: list of source IDs whose content backs this wiki page
- `related`: list of wiki page IDs that are topically related
- `prerequisites`: list of wiki page IDs the reader should read first
- `candidate_techniques` (pattern only): list of technique/hw/migration IDs that address the symptoms

## Blackwell-First Scope

Pages including `sm90` in `architectures` without any Blackwell target (`sm100*`, `sm103*`, or `sm120`) must include a `blackwell_relevance:` field explaining why the Hopper content is kept. Enforced by validator.
