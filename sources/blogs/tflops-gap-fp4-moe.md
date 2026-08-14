---
id: blog-tflops-gap-fp4-moe
title: 'TFLOPS Gap: Why FP4 MoE Kernel Engineering Matters on Blackwell'
author: apsys (HuggingFace)
url: https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, moe, warp-specialization, tma, kernel-fusion, tile-scheduling, persistent-kernel, block-scale, gemm, grouped-gemm, fine-grained-quantization]
retrieved_at: 2026-08-13
---

# Source-reported FP4 MoE comparison

This community benchmark compares vLLM, SGLang, and FlashInfer on a B200 using its linked `advpropsys/fp4-blackwell-bench` harness. Its GPT-OSS-20B setup uses 32 experts, top-4 routing, hidden size 2880, intermediate size 7680, and NVFP4 weights.

## Results in the post

For batch 4096, the post reports 1262, 1225, and 1117 TFLOP/s for SGLang, FlashInfer, and vLLM respectively. For batch 1 it reports 206.9, 481.9, and 369.5 microseconds per layer in the same order. These are source-reported results for the linked harness and software snapshot—not official framework limits or portable B200 expectations.

The post attributes differences to fusion boundaries, CUTLASS schedule selection, TMA staging, and launch/grid heuristics. Its seven-versus-five launch count and 21.9% traffic reduction describe the compared code paths only. They must not be generalized to all vLLM/SGLang versions or MoE kernels.

The original summary also treated 142 enabled SMs as a universal B200 property and described 128-byte TMA alignment as universal. Both are incorrect generalizations: enabled-SM count is device/configuration specific, and each TMA instruction/tensor map has exact documented alignment and layout constraints.

Use the linked benchmark repository and pinned dependency versions before reproducing a number. The page is evidence about one third-party comparison, not an NVIDIA hardware specification.
