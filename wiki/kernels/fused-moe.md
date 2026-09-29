---
id: kernel-fused-moe
title: Fused MoE — Expert GEMM and Adjacent Operations
type: kernel
architectures: [sm100, sm100a, sm103, sm90]
tags: [moe, fused-kernel, fp8, block-scale, kernel-fusion, grouped-gemm, gated-dual-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, fused-kernel, grouped-gemm, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl, triton]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-fine-grained-quantization, technique-tile-scheduling]
sources: [contest-flashinfer-track-a, blog-deepgemm, pr-TensorRT-LLM-11897]
performance_claims: []
blackwell_relevance: Blackwell block-scaled tensor-core operations and TMEM are implementation tools for expert GEMMs; the useful fusion boundary remains workload-specific.
---

# Fused MoE

“Fused MoE” covers kernels that combine an expert GEMM with adjacent data preparation, activation, quantization, or result-combination work. It does not imply that routing, dispatch, both expert projections, and combine are always one device kernel.

## Kernel boundary

A concrete implementation should state which of these operations are inside the launch:

- routed-row preparation or permutation;
- grouped gate/up expert GEMM;
- gated activation;
- intermediate quantization;
- grouped down-projection GEMM;
- weighted output combination.

The profitable boundary depends on batch/routed-row distribution, data type, intermediate traffic, code size, and the available backend. Correctness must cover empty experts, repeated expert IDs, padding, routing weights, and quantization scales.

## Evidence boundary

The FlashInfer contest page defines a fused-MoE track but does not publish the
earlier local performance table. DeepGEMM documents grouped expert GEMM
primitives, not a universal end-to-end fusion speedup. The B300 timings below
come from a separate internal run and apply only to its named workloads.

The previous artifact directory mixed a vLLM test-only PR, an SGLang dispatcher file, a blog extract, and a synthetic routing skeleton. It was removed because it was not a coherent fused-MoE kernel implementation and linked to excluded source PRs.

One retained, narrower example is TensorRT-LLM PR 11897's shared-expert path.
Its BF16-output call site invokes a dense NVFP4 GEMM fused with SwiGLU:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```

This contiguous upstream excerpt demonstrates one expert-projection fusion. It
does not imply that routing, dispatch, both projections, and combine are in the
same device kernel.

## Local B300 trajectory: larger expert GEMM tiles at high token counts

In a saved DeepSeek-V3 FP8 block-scale MoE run, **v5.3** repeatedly read
weights for M sub-tiles. **v6** dispatches both expert GEMMs to
`BM=256` at `T >= 11948`, with eight warps and split=8 for GEMM1 and split=1
for GEMM2. Smaller T keeps the `BM=64` path to avoid large-tile fixed cost.
The change amortizes weight reads across more routed rows.

| B300 routed tokens | v5.3 kernel time | v6 kernel time | Reduction |
|---:|---:|---:|---:|
| 11,948 | 914.7 µs | 728.4 µs | 20.4% |
| 14,107 | 1282.6 µs | 964.8 µs | 24.8% |
| 32,768 | 2383.8 µs | 1739.6 µs | 27.0% |

All 20 saved workloads passed. These are matched-shape **candidate kernel
times** from the saved [v5.3](../../data/internal-b300/fused-moe/full-a.jsonl)
and [v6](../../data/internal-b300/fused-moe/full-b.jsonl) full runs. Their
aggregate geomean scores cannot be compared as an overall improvement: small-case
reference times varied between runs, and the recorded geomean changed from
2.1209× to 1.8368×. The evidence supports the large-T timing improvement;
the exact threshold should be remeasured on a new target GPU.

An earlier routing change replaced the per-expert count loop with
`tl.histogram`. Its T=1 end-to-end fast row improved from **112.35 to
106.08 µs** in the saved [before](../../data/internal-b300/fused-moe/routing-a.jsonl)
and [after](../../data/internal-b300/fused-moe/routing-b.jsonl) runs. Both full
runs contain 19 official rows plus one live large row; the routing fast runs
contain four rows each. The end-to-end result is the relevant measure of the
routing change's effect on the pipeline.

## Audited full-suite kernel

The later [selected FP8 MoE implementation](../../artifacts/kernels/kda-mlsys26-final/full/kernels/moe/kernel.py)
and its sibling modules passed all 19 B300 official workloads at 2.2227×
arithmetic-mean speedup. The [selection and provenance notes](../../data/internal-b300/mlsys26-final.md)
identify the audited snapshot. The earlier v5.3/v6 section compares a
different experiment and metric.
