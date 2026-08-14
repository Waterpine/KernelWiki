---
id: blog-nsa
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
author: DeepSeek AI
url: https://arxiv.org/abs/2502.11089
source_category: benchmark-blog
architectures: []
tags: [sparse-attention, attention, triton, chunk-parallelism]
retrieved_at: 2026-08-13
---

## Summary

Native Sparse Attention (NSA) is a trainable sparse-attention design with three branches: compressed coarse-grained tokens, selected fine-grained token blocks, and a sliding window. Its blockwise selection and shared selections within a GQA group are intended to align sparsity with GPU memory access.

## Reported Evaluation

The paper reports experiments on eight NVIDIA A100 GPUs, not H100 or B200. In that setup it reports up to 9× forward and 6× backward speedup at 64K sequence length relative to its stated FlashAttention-2 baseline, and up to 11.6× decoding speedup in the decoding experiment.

Those values are paper-specific measurements, not SM90 or SM100 results. Later deployment claims and later FlashMLA kernels require separate sources.
