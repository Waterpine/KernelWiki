---
id: kernel-flashmla
title: FlashMLA — Multi-head Latent Attention
type: kernel
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention]
confidence: verified
reproducibility: snippet
kernel_types: [mla, attention, decode, prefill, sparse-attention]
languages: [cuda-cpp]
related: [hw-tcgen05-mma, hw-tmem, kernel-nsa]
sources: [blog-flashmla, pr-flashinfer-1117, pr-vllm-39752]
performance_claims: []
evidence_basis:
  - source_id: blog-flashmla
    evidence_type: official-doc
  - source_id: pr-flashinfer-1117
    evidence_type: upstream-code
blackwell_relevance: "Current upstream provides SM100 sparse decode, dense MHA prefill, and sparse MLA prefill paths; each has a distinct data contract."
artifact_dir: artifacts/kernels/flashmla
---

# FlashMLA

FlashMLA provides DeepSeek MLA attention kernels for dense/sparse decoding and
prefill. Its current support matrix distinguishes MQA and MHA modes and does
not expose one implementation that is simultaneously all four variants.

```python
def flashmla_call(q, kvcache, block_table, cache_seqlens, dv, indices):
    """Interface sketch; consult the installed FlashMLA version."""
    is_fp8 = kvcache.element_size() == 1
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk,
    )
    output, lse = flash_mla_with_kvcache(
        q, kvcache, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, is_causal, is_fp8, indices,
    )
    return output, lse
```

This interface sketch follows the current README. Exact names/signatures must
be checked against the installed version.

## FP8 cache contract

The 656-byte record applies to FlashMLA's FP8-with-scale sparse-decode cache:
512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. Computation
dequantizes this cache to BF16 and performs attention in BF16. It should not be
described as native FP8 block-scale tcgen05 attention.

## Reported performance

Upstream reports maxima for several different tests, including 1460 TFLOP/s
forward / 1000 backward for dense MHA prefill on B200 and 1450 for sparse MLA
prefill on B200. Because the README does not attach the former wiki's exact
shapes or a common utilization denominator, these values are retained in the
source summary but removed from structured claims and cross-variant comparison.

Pinned and derived artifacts are under
[`artifacts/kernels/flashmla/`](../../artifacts/kernels/flashmla/).
