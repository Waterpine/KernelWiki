---
id: blog-qwen3-next-architecture
title: "Qwen3-Next: Hybrid GDN+MoE Architecture on NVIDIA GPUs"
author: NVIDIA / Alibaba
url: https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/
source_category: community-note
architectures: [sm100, sm100a, sm90]
tags: [gated-delta-net, moe, linear-attention, attention, sparse-attention, cluster]
retrieved_at: 2026-08-13
---

# Qwen3-Next architecture announcement

The NVIDIA article announces the Qwen3-Next 80B-A3B Thinking and Instruct preview models. It states:

- 80B total parameters and 3B activated per token;
- more than 260K input-token context;
- 512 routed experts plus one shared expert, with 10 experts activated per token;
- 48 layers, with every fourth layer using GQA and the other three using linear attention based on Gated Delta Networks;
- deployment paths through NVIDIA NIM, SGLang, and vLLM;
- operation on Hopper and Blackwell.

The article also cites 1.8 TB/s for fifth-generation NVLink in its expert-routing discussion. That is an interconnect claim in the deployment context, not evidence that a particular routing kernel achieves that bandwidth.

The active-parameter count is not, by itself, proof that end-to-end inference costs equal a dense 3B model: routing, shared experts, attention, communication, and memory traffic remain. The former summary's “comparable cost” and recommendation to replace full-attention layers with sparse/sliding attention were inferences not stated by the source and were removed.
