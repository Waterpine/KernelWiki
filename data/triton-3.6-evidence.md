# Triton Blackwell Evidence Memo

## Releases of record

- 3.4.0 — published 2025-07-30; already reported enhanced Blackwell TMEM, MMAv5 pipelining, and automatic warp specialization.
- 3.5.0 — published 2025-10-21; added generic tcgen05 load/store lowering, tcgen05 copy exposure, and further TMEM/warp-specialization support.
- 3.6.0 — published 2026-01-21; generalized those paths and added initial Gluon multi-CTA/two-CTA support.
- 3.7.0 — released 2026-05-07; added and hardened two-CTA, TMA/tcgen05 multicast, TMEM deallocation, and warp-specialization paths.
- 3.7.1 — released 2026-06-18; current stable as verified 2026-08-13, a regression-fix patch with no new API.

Primary official records are `doc-triton-pre36-blackwell`, `doc-triton-3.6-blackwell`, and `doc-triton-3.7-blackwell`.

## Claim supported

For Triton `>=3.6`, native Blackwell lowering paths exist, but 3.6 is a
conservative policy floor rather than their first release. The strongest
documented user-facing paths are supported descriptor/TMA matmul loops with
automatic warp specialization, `tl.dot_scaled` for block-scaled formats, and
the explicit Gluon Blackwell APIs for TMEM/tcgen05/multi-CTA work.

This does not establish that every plain `tl.dot`, dtype, layout, or shape emits tcgen05 or matches a vendor library. Verify generated code when the instruction choice matters.

## Downstream anchors

- `pr-vllm-34597`: captured SM120 Triton MLA decode kernel with actual `tl.dot` operations; useful ecosystem context, but not evidence for SM100 lowering.
- `pr-vllm-29339`: production dispatch around the Triton kernel library; supplementary because it is control-plane code.
- `pr-sglang-22079`: captured Triton attention matmul on an SM100-capable path and the direct downstream SM100 anchor used by the version claim.
- `pr-sglang-21019`: captured SM100 memory-rearrangement kernel without matmul.
- `pr-sglang-5390` and `pr-sglang-21595`: workload-specific evidence that non-Triton backends can remain preferred.
- `pr-pytorch-175826`: ecosystem/toolchain context, not tcgen05 lowering proof.

## Removed claims

- 3.6 is no longer the current release.
- pre-3.6 behavior cannot be summarized as a universal silent WGMMA fallback; official 3.4/3.5 notes already record native Blackwell work.
- downstream `tl.dot` source alone does not prove its final PTX instruction.
- pre-contest AI baseline scores are not final MLSys 2026 winners.

Access date: 2026-08-14.
