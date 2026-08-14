---
id: kernel-grouped-gemm
title: "Grouped GEMM for MoE"
type: kernel
architectures: [sm100, sm100a, sm90]
tags: [grouped-gemm, moe, gemm, fp8, nvfp4, tcgen05, persistent-kernel, tile-scheduling]
confidence: source-reported
reproducibility: snippet
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp, cute-dsl]
related: [kernel-fused-moe, kernel-deepgemm, hw-tcgen05-mma, hw-clc, technique-persistent-kernels, technique-tile-scheduling]
sources: [contest-gpumode-p4, blog-deepgemm, doc-cutlass-blackwell]
performance_claims: []
blackwell_relevance: "SM100 provides tensor-core/TMEM paths and CLC cancellation that a grouped-GEMM scheduler may use when their costs fit the workload."
---

# Grouped GEMM for MoE

## Overview

Grouped GEMM evaluates a set of matrix products whose problem sizes or pointers differ. In MoE inference, each expert commonly has a different token count, so its M dimension varies while model dimensions N and K are often shared.

Implementations may pack expert inputs contiguously, use fixed-capacity masked layouts for graph capture, or pass arrays of problem descriptors. Those layouts are library APIs rather than universal grouped-GEMM categories; use the pinned DeepGEMM or CUTLASS version for exact names and contracts.

```python
def grouped_gemm(inputs, expert_weights):
    outputs = []
    for expert_input, weight in zip(inputs, expert_weights):
        outputs.append(expert_input @ weight)
    return outputs
```

This Python fragment defines only the mathematical batch of products; it is not a GPU scheduling implementation.

## Scheduling choices

- a static mapping is cheap when expert sizes are known and balanced;
- a software persistent scheduler can claim logical tiles through a counter or queue;
- SM100 CLC can attempt to cancel a not-yet-started cluster and reuse the returned grid ID;
- CLC does not claim arbitrary application tiles and cannot be retried after a failed cancellation response under the documented protocol.

Small expert M values, last-wave effects, descriptor setup, and padding can dominate. A dynamic scheme also has atomic or cancellation overhead, so it is not uniformly faster.

## Instruction boundary

TMA tensor maps and block-scaled MMA operands must satisfy their exact alignment, stride, box, swizzle, scale, and target requirements. There is no generic `tcgen05_mma(...)` CUDA intrinsic and no universal 128-byte rule for every operand.

## Evaluator incident

The GPU Mode report describes a submission that precomputed later benchmark cases by exploiting evaluator object reuse. Its reported timing was invalid and is not a performance result. The incident is useful only as a benchmark-integrity lesson.

## Sources

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [GPU Mode evaluator postmortem](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/)
