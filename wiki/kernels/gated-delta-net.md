---
id: kernel-gated-delta-net
title: Gated Delta Network kernels
type: kernel
architectures: [sm100, sm103, sm90]
tags: [gated-delta-net, linear-attention, triton, cute-dsl, kernel-fusion]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-delta-net, linear-attention]
languages: [triton, cute-dsl, python]
related: [technique-kernel-fusion, pattern-memory-bound]
sources: [blog-gated-delta-net, contest-flashinfer-track-c, pr-sglang-21019]
performance_claims: []
blackwell_relevance: The FlashInfer contest targeted B200 Gated Delta Net decode and prefill; the retained SGLang projection-fusion example was reported on H200 and is not presented as a Blackwell-specific kernel.
artifact_dir: artifacts/kernels/gated-delta-net
---

# Gated Delta Network kernels

Gated Delta Networks combine a gated recurrent state update with a delta-rule
correction. Implementations commonly separate or fuse projections, local
convolution, recurrence/chunk processing, normalization, and output projection.
The exact recurrence must come from the paper or implementation; a simple gated
outer-product update is not an equivalent substitute.

## Retained implementation excerpt

SGLang PR 21019 fuses the split/reshape/concatenate work around Qwen3.5's GDN
projection. It is not the recurrent update itself. This contiguous excerpt from
the captured Triton file shows its interleaved-input stores:

```python
tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
```

The full upstream file supports both the interleaved Qwen3-Next layout and the
contiguous Qwen3.5 layout. Layout identity is therefore a correctness condition,
not a performance-only choice.

## Evidence and reproduction boundary

- The NVlabs repository provides the research reference and points to FLA for
  a faster variable-length implementation.
- The FlashInfer MLSys 2026 page identifies a B200 Gated Delta Net track and
  winner names, but publishes no latency or throughput table.
- The retained SGLang PR description reports an H200 projection-fusion
  benchmark; it does not validate a `tcgen05.mma` form or a universal GDN
  mainloop.

The former synthetic prefill/decode kernels and invalid tcgen inline PTX were
removed. Reproduction requires an upstream implementation plus its model
layout, recurrence parameters, state initialization, dtype, and tolerance.

## Local B300 trajectory: short-sequence prefill

**Kernel A** used a persistent CuTe path with a 128×128 state in TMEM and
Q/K/V staged through TMA. For `T <= 48`, fixed tensor-map, TMEM
and pipeline setup dominated useful work. **Kernel B** dispatches
one CTA per `(sequence, value head, value slice)` to a single-launch
`mma.sync` path, computes gates inside that launch, and keeps the state in
registers. In the B300 dev suite, T=6, one sequence fell from **11.584 to
8.320 µs** (28.2% less time); T=42, two sequences fell from **12.096 to
9.952 µs**. All 34 cases passed and the suite mean speedup rose from
7.6864× to 8.1563× against its reference.

**Kernel C** narrows the tiny path further: eight warps use a
`(1,8,1)` MMA layout to split the free N dimension, while the M dimension is
walked in 16-row blocks guarded by the actual token count. The T=6 row fell
again from **8.320 to 7.344 µs** (11.7%), with 34/34 passing. The larger-T
persistent path remained in place. The `T <= 48` crossover is specific to
these shapes and this implementation.

A related KDA backward run made a broader pipeline change: it added a cached
host plan and an in-house CuTe K0 for normalization, gates, `Aqk` and
`Akk_inv` preparation, and disabled an expensive K split. Its six-case
geomean moved from **1.1073× to 1.7696×**; all six passed. That is the
combined pipeline result, not an isolated recurrent
kernel speedup. The two full runs used different B300 leases; the improved
run was recorded before its solution/benchmark commit, so the saved results
describe the measured candidate rather than a clean commit. The complete
B300 prefill [A](../../data/internal-b300/gated-delta-net/prefill-a.jsonl),
[B](../../data/internal-b300/gated-delta-net/prefill-b.jsonl) and
[C](../../data/internal-b300/gated-delta-net/prefill-c.jsonl) runs each contain
34 rows. The KDA backward [A](../../data/internal-b300/gated-delta-net/backward-a.jsonl)
and [B](../../data/internal-b300/gated-delta-net/backward-b.jsonl) runs each
contain six rows. These B300 measurements are distinct from the SGLang
projection-fusion H200 example above.

## Audited full-suite kernel

The later [selected GDN prefill implementation](../../artifacts/kernels/kda-mlsys26-final/full/kernels/gdn_prefill/kernel.py)
and its sibling modules passed the 100-workload B300 suite at 15.1598×
arithmetic-mean speedup. The [selection and provenance notes](../../data/internal-b300/mlsys26-final.md)
summarize the full-suite result. The A/B/C sequence above measures a 34-case
development suite and does not identify this final kernel.
