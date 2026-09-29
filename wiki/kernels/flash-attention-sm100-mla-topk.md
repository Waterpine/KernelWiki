---
id: kernel-flash-attention-sm100-mla-topk
title: FlashAttention SM100 MLA TopK Sparse Forward
type: kernel
architectures:
- sm100
- sm103
tags:
- attention
- flash-attention
- mla
- sparse-attention
- tma
- tile-scheduling
- top-k-selection
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- flash-attention
- mla
- sparse-attention
- topk
languages:
- cute-dsl
- python
related:
- kernel-flash-attention-4
- kernel-sparse-mla
- technique-tile-scheduling
- technique-external-source-map-research
sources:
- pr-flash-attention-2441
- pr-flash-attention-1236
performance_claims:
- gpu: sm100-class (specific SKU not stated in PR)
  dtype: not stated in PR
  shape: batch=512, seqlen_q=1, seqlen_k=16384, nheads=128, topk=2048
  metric: latency_ms
  value: 0.31
  source_id: pr-flash-attention-2441
  source_locator: https://github.com/Dao-AILab/flash-attention/pull/2441 (PR description, "DSA, no bitmask")
blackwell_relevance: PR-grade CuTe DSL SM100 MLA code is directly relevant to DSA sparse attention and top-k KV-gather routing on SM100-class GPUs.
artifact_dir: artifacts/prs/flash-attention/PR-2441
---

## Shape

FlashAttention PR 2441 adds an SM100 CuTe DSL forward path for MLA shapes with
top-k sparsity. It is useful when an attention candidate has to combine page/KV
layout handling, sparse top-k selection, and tiled forward scheduling.

```python
for i in cutlass.range_constexpr(entries_per_thread):
    topk_idx = rTopk[i]
    if const_expr(not self.disable_bitmask):
        row_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        tPrRowValid[i] = row_valid
    if const_expr(not transpose):
        tPrXPtr[i] = utils.elem_pointer(mX, (topk_idx, 0)).toint()
    else:
        tPrXPtr[i] = utils.elem_pointer(mX, (0, topk_idx)).toint()
```

This is a contiguous excerpt from `topk_gather_kv.py` in the retained PR
snapshot. It is implementation context, not a standalone benchmark. The full
snapshot and PR tests are required to exercise the path.

## Transfer Notes

- Treat top-k gather and tiled attention scheduling as separate evidence paths.
- Profile memory traffic separately from tensor-pipe utilization; sparse top-k
  routing can improve arithmetic work while worsening gather locality.
- Keep full-workload validation because the useful path is shape-specific.

## Local B300 DSA trajectory: cluster transfer and PV work

The separate `dsa_attention_official` CuTe run supplies a measured
top-k sparse-attention A→B→C sequence. Before the performance changes, its
saved [seed](../../data/internal-b300/dsa/before-correctness-fix.jsonl)
produced roughly 40× only on **7/23 passing** workloads. The correctness
repair kept FP32 partials through the merge and split high/low P in PV; the
[next full run](../../data/internal-b300/dsa/after-correctness-fix.jsonl)
passed 23/23 at 27.7786× mean speedup. The failing 40× is not a valid
predecessor score.

| Passing B300 23-case suite | Mean speedup vs reference | Change |
|---|---:|---|
| [A](../../data/internal-b300/dsa/attention-a.jsonl) | 28.1316× | FP32 partials, 128 small DSMEM pushes per row |
| [B](../../data/internal-b300/dsa/attention-b.jsonl) | 29.6105× | One 2 KiB `cp.async.bulk` shared-to-cluster push per consumer |
| [C](../../data/internal-b300/dsa/attention-c.jsonl) | 30.8641× | Relaxed empty synchronization and skip proven zero PV K steps |

B fences shared-memory proxy visibility, then maps the
destination mbarrier to the consumer CTA before the bulk transfer. C
uses relaxed cluster arrival and relaxed empty-flag posts only
where there is no data payload to publish. Its PV loop skips K steps only
when packed valid-row counts prove they contribute zero; it preserves a
straight-line loop for full tiles because per-step guards had slowed them.
On dense T=8 workload `564007ac`, kernel time was
**11.552→10.720→10.656 µs**. All three saved A/B/C runs passed 23/23.

These B300 observations are distinct from PR 2441's benchmark. Each linked
A/B/C snapshot contains the full 23-case result, including per-case latency
and correctness.

## Audited full-suite kernel

The later [selected DSA implementation](../../artifacts/kernels/kda-mlsys26-final/full/kernels/dsa_attention/kernel.py)
passed all 23 B300 workloads at 37.6827× arithmetic-mean speedup. The
[selection and provenance notes](../../data/internal-b300/mlsys26-final.md)
identify the audited snapshot. The A/B/C sequence above records earlier
optimization steps.
