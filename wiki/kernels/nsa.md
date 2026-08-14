---
id: kernel-nsa
title: "Native Sparse Attention (NSA)"
type: kernel
architectures: [sm90, sm100]
tags: [sparse-attention, attention, triton]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, attention]
languages: [triton]
related: [kernel-flashmla, technique-pipeline-stages]
sources: [blog-nsa, blog-flashmla, blog-vllm-deepseek-v3-sparse]
performance_claims: []
blackwell_relevance: "NSA's block-sparse access pattern can be implemented on newer GPUs, but the original paper's reported kernel measurements were on A100."
---

# Native Sparse Attention (NSA)

## Overview

NSA is a natively trained sparse-attention architecture with three parallel branches:

1. a compression branch creates coarse-grained key/value representations;
2. a selection branch attends to selected contiguous token blocks;
3. a sliding-window branch retains recent local context.

The branch outputs are combined by the model's learned formulation. Blockwise rather than scattered-token access is central to the hardware-oriented design, and query heads in a GQA group share selections to reduce redundant key/value traffic.

```python
def nsa_structure(query, compressed_kv, selected_kv, local_kv):
    compressed = attend(query, compressed_kv)
    selected = attend(query, selected_kv)
    local = attend(query, local_kv)
    return learned_combine(compressed, selected, local)
```

This structural snippet deliberately omits the learned compression and selection equations; consult the paper for the complete algorithm.

## Implementation Boundary

The selection mechanism, compression weights, block sizes, and branch combination are part of the trained model. A generic sparse-attention loop over externally supplied indices demonstrates only the selection branch; it is not a complete NSA implementation.

The original paper evaluated its system on eight A100 GPUs. It reports up to 9× forward and 6× backward speedup at 64K sequence length and up to 11.6× decoding speedup under the paper's respective setups. Those numbers are not H100 or Blackwell measurements and therefore are not encoded as this page's structured SM90/SM100 performance claims.

## Use and Transfer

NSA is relevant when a model was trained with its compression, selection, and local branches. Porting the block-sparse kernels to SM90 or SM100 requires new benchmark evidence; cache size alone does not predict the speedup.

## Primary Source

- [Native Sparse Attention paper](https://arxiv.org/abs/2502.11089)
