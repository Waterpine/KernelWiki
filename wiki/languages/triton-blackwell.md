---
id: lang-triton
title: "Triton on Blackwell"
type: language
tags: [triton, attention, moe, gated-delta-net]
related: [kernel-nsa, kernel-gated-delta-net, kernel-fused-moe, lang-cute-dsl]
sources: [doc-triton-3.7-blackwell, doc-triton-3.6-blackwell, doc-triton-pre36-blackwell, pr-vllm-29339, pr-sglang-22079, pr-sglang-21019, pr-sglang-5390, pr-sglang-21595, pr-pytorch-175826, blog-nsa, blog-gated-delta-net, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm90]
confidence: verified
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-triton-3.7-blackwell
  - evidence_type: upstream-code
    source_id: pr-sglang-22079
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
blackwell_relevance: "Triton 3.4/3.5 already contained native Blackwell TMEM/tcgen05 and warp-specialization work; 3.6 is the conservative policy baseline for the expanded combined surface, and current stable 3.7.1 continues it."
---

# Triton on Blackwell

## Version boundary

Triton 3.4.0 already reported enhanced Blackwell TMEM support and introduced
automatic warp specialization; 3.5.0 added generic `tcgen05.ld`/`st` lowering,
`tcgen05_copy`, and further TMEM and warp-specialization work. Triton 3.6.0
generalized those paths and added initial Gluon multi-CTA/two-CTA support. The
repository retains 3.6 as a conservative supported-policy baseline, not as the
introduction point. As of this audit (2026-08-14), the latest stable release is
3.7.1, released 2026-06-18. Release 3.7.0 added further two-CTA, TMA multicast,
tcgen05 multicast, and warp-specialization fixes; 3.7.1 is a regression-fix
patch with no new APIs.

Do not describe 3.6 as the current release. Also do not reverse the old limitation into the blanket claim that every `tl.dot` shape necessarily lowers to the same tcgen05/TMEM sequence.

## Supported user surfaces

The official current documentation provides:

- `tl.range(..., warp_specialize=True)` for supported simple matmul loops on Blackwell;
- `tl.dot_scaled` for microscaling formats, with hardware or emulated behavior depending on target and format;
- explicit Gluon Blackwell APIs for TMEM, `tcgen05_mma`, scaled MMA, barriers, and multi-CTA construction;
- TMA/tensor-descriptor paths and an official Blackwell fused-attention example.

```python
import triton.language as tl

def tiled_matmul_loop(acc, a_desc, b_desc, k_tiles: tl.constexpr):
    for k in tl.range(0, k_tiles, warp_specialize=True):
        a = a_desc.load([0, k])
        b = b_desc.load([k, 0])
        acc = tl.dot(a, b, acc)
    return acc
```

This fragment illustrates the documented loop surface only; descriptor shapes, launch metadata, and legal dtypes are required for a runnable kernel.

## Evidence and caveats

The captured vLLM and SGLang PRs demonstrate real downstream Triton kernels on SM100. They do not, by themselves, prove the final PTX instruction chosen for every shape. Inspect generated TTGIR/PTX/SASS for that question.

Performance comparisons are workload-specific. A captured SGLang PR reports a CUTLASS MLA backend ahead of its Triton baseline for one setup, and another changes a Blackwell attention default to FA4. Those routing decisions do not establish a universal language ranking.

The earlier AI-agent leaderboard snapshot and fabricated Gated DeltaNet/NSA Triton sketches were removed. Pre-contest baseline scores were not final MLSys 2026 standings, and conceptual sparse or recurrent pseudocode was not a correct implementation.

## Local artifacts

Pinned downstream examples include the vLLM MLA decode kernel under `artifacts/prs/vllm/PR-34597/` and SGLang kernels under `artifacts/prs/sglang/PR-21019/` and `PR-22079/`. Read each bundle's `PROVENANCE.yaml` before treating a file as verbatim.
