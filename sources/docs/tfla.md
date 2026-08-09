---
id: doc-tfla
title: "Tiled Flash Linear Attention (TFLA)"
url: https://arxiv.org/abs/2503.14376
source_category: paper
architectures: [sm90]
tags: [linear-attention, chunk-parallelism, triton]
retrieved_at: 2026-04-16
---

## Summary

Paper on Tiled Flash Linear Attention enabling arbitrarily large chunk sizes for linear attention.

## Key Techniques
- Two levels of sequence parallelism: standard chunkwise + tiling within chunks
- Prevents materialization of intermediate memory states
- Kernels are written in Triton 3.1.0; the paper leaves a CUDA implementation to future work
- Improves arithmetic intensity for linear RNNs; applied in the paper to the mLSTM (xLSTM with matrix memory) and to an mLSTM variant with a sigmoid input gate
