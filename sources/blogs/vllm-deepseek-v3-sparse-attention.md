---
id: blog-vllm-deepseek-v3-sparse
title: "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action"
author: vLLM Team
url: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
source_category: community-note
architectures: [sm100, sm100a, sm90]
tags: [sparse-attention, mla, attention, flash-attention, decode, prefill, fp8, quantization, kernel-fusion]
retrieved_at: 2026-08-13
---

# DeepSeek-V3.2-Exp in vLLM

This dated vLLM post describes its initial support for DeepSeek-V3.2-Exp and the model's two-stage DeepSeek Sparse Attention path: a lightning indexer computes relevance and selects up to 2048 positions, then sparse MLA attends to those positions.

## Cache contract reported by the post

For the post's FP8 MLA path, one token's 656-byte cache entry contains:

- 512 `float8_e4m3` NoPE values;
- four FP32 scales (16 bytes), one per 128 values;
- 64 BF16 RoPE values (128 bytes).

The separate indexer key cache has a block-oriented layout. The post says this implementation supports block size 64 and shows the handling of fewer than 2048 available tokens.

## Deployment scope at publication

The article says the initial model could be run on 16×H100, 8×H200, or 8×B200 with tensor parallelism, while expert parallelism still had a bug. It also says accuracy verification against the official results was still in progress; only an earlier weight version had matched expected GSM8K and GPQA-Diamond results.

Those statements describe the September 2025 software snapshot, not current minimum hardware or current vLLM limitations. The previous summary incorrectly said accuracy had been validated and claimed a 50% API-cost reduction not present in the source; both claims were removed.
