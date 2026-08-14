---
id: blog-deepgemm
title: DeepGEMM — Tensor-Core Kernel Library
author: DeepSeek AI
url: https://github.com/deepseek-ai/DeepGEMM
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale, jit-compilation, tcgen05, wgmma]
retrieved_at: 2026-08-13
artifact_dir: artifacts/blogs/deepgemm/code
---

## Current upstream facts

DeepGEMM is a runtime-compiled tensor-core library for FP8, FP4, BF16, grouped
GEMMs, scoring kernels, and fused MoE work. The current README requires CUDA
12.9 or newer for SM100 and describes a lightweight JIT module.

For its fine-grained FP8 GEMM interface:

- SM90 uses FP32 scale tensors and supports NT layout.
- SM100 uses packed UE8M0 scale tensors (four scales per `torch.int`) and
  supports NT, NN, TN, and TT layouts.
- the LHS scale tensor uses a project-defined TMA-aligned transposed layout.
- M-grouped APIs hold N and K fixed and provide contiguous or masked layouts;
  a separate K-grouped API serves weight gradients.

## Pinned SM100 instruction

PR 304's captured wrapper uses this instruction family for block-scaled FP8:

```ptx
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
    [d], a_desc, b_desc, idesc, [a_sf_tmem], [b_sf_tmem], p;
```

The former summary used the nonexistent spelling
`kind::f8f6f4.block_scale`, placed scale factors in shared memory in an
invented operand list, and asserted full-FP32 TMEM accumulation without tying
it to the instruction descriptor. Those statements are not retained.

## Performance scope

The upstream project announced “up to 1550 TFLOPS on H800” for its April 2025
code and links the contributing changes. It does not make that number a
universal result or attach it to the previously asserted `4096x4096x4096`
configuration on this source page.

## Version note

DeepGEMM's JIT path and supported kernel set have changed over time. Historical
NVRTC statements must be versioned; they do not describe every current release.
