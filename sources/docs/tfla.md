---
id: doc-tfla
title: "Tiled Flash Linear Attention (TFLA)"
url: https://arxiv.org/abs/2503.14376
source_category: paper
architectures: [sm90]
tags: [linear-attention, gated-delta-net, chunk-parallelism]
retrieved_at: 2026-08-14
blackwell_relevance: "The paper includes theoretical B200 runtime curves, but its measured Triton 3.1 kernels and all experiments use H100; it does not claim a Blackwell implementation."
---

## Summary

Paper on Tiled Flash Linear Attention, which adds a second tiling level to
chunkwise linear attention and supports arbitrarily large chunk sizes.

## Key Techniques
- Two levels of sequence parallelism: standard chunkwise plus tiling within chunks
- Prevents materialization of intermediate memory states
- Implements the measured kernels in Triton 3.1.0 and integrates them through JAX-Triton 0.2.0
- Runs all reported experiments on NVIDIA H100 80 GB GPUs
- Includes theoretical hardware-bound analysis for H100 and B200, but leaves CUDA and next-generation-hardware optimization to future work
- Improves arithmetic intensity for linear-attention variants including Gated Delta Net

The paper does not report inline WGMMA or `tcgen05` implementations. A
theoretical B200 curve is not evidence of a measured Blackwell kernel.
