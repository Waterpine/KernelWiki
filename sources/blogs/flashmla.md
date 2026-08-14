---
id: blog-flashmla
title: FlashMLA — Multi-head Latent Attention
author: DeepSeek AI
url: https://github.com/deepseek-ai/FlashMLA
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention, tcgen05, tmem]
retrieved_at: 2026-08-13
---

## Current upstream support matrix

- dense decoding: SM90, MQA mode, BF16 KV cache;
- sparse decoding: SM90 and SM100, MQA mode, optional FP8-with-scale cache;
- dense prefill: SM100, MHA mode;
- sparse prefill: SM90 and SM100, MQA mode.

SM100 kernels require CUDA 12.9 or newer in the current README.

For the optional FP8 sparse-decode cache, each token is 656 bytes: 512 E4M3
values, four FP32 scales covering 128 values each (16 bytes), and 64 BF16 RoPE
values (128 bytes). This layout is not the size of every MLA cache or a generic
per-token compression claim.

## Upstream-reported maxima

The README reports up to 3000 GB/s or 660 TFLOP/s for different dense-decode
configurations on H800; 410 TFLOP/s sparse decode on H800 and up to 350 on B200
(explicitly described as not yet optimized); 1460 forward and 1000 backward for
SM100 dense MHA prefill; and 640 on H800 / 1450 on B200 for sparse MLA prefill.

These are maxima from different suites and modes. They must not be placed in a
single like-for-like comparison table or used to infer a fixed utilization
without the corresponding test configuration.

The prior code blocks were teaching reconstructions, not upstream FlashMLA
code. They remain only in artifact paths labeled by provenance and are not
presented as source excerpts here.
