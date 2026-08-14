---
id: kernel-sparse-mla
title: "Sparse MLA (DeepSeek V3.2)"
type: kernel
architectures: [sm100, sm90]
tags: [sparse-attention, mla, fp8, attention, decode, prefill]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, mla, attention, decode, prefill]
languages: [cuda-cpp, cute-dsl]
related: [kernel-flashmla, kernel-nsa, hw-tcgen05-mma]
sources: [blog-flashmla, blog-vllm-deepseek-v3-sparse, blog-nsa]
performance_claims: []
blackwell_relevance: "FlashMLA provides SM100 sparse decode and sparse prefill, while the separate indexer/routing stage determines selected token indices."
---

# Sparse MLA

Sparse MLA consumes an index set and performs attention over selected KV
positions. In a DeepSeek V3.2-style system, an indexer produces candidates and
the attention kernel gathers/uses them; those are separate kernels and evidence
paths.

```cuda
__global__ void validate_indices(const int* indices, int count, int limit,
                                 int* invalid_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && (indices[i] < 0 || indices[i] >= limit))
        atomicAdd(invalid_count, 1);
}
```

The FlashMLA FP8-with-scale decode cache is 656 bytes per token and is
dequantized to BF16 for attention computation. The former page instead showed
an invented native FP8 tcgen05 indexer and treated one block maximum as the
top-k score; that is not the documented algorithm.

Upstream reports performance maxima for its test suites, including up to 1450
TFLOP/s for B200 sparse prefill and up to 350 for B200 sparse decode, which it
calls not yet optimized. These modes are not comparable without their exact
shapes and FLOP definitions, so no structured cross-mode claim is retained.
