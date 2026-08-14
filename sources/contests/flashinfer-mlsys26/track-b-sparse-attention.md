---
id: contest-flashinfer-track-b
title: 'FlashInfer MLSys 2026 - Track B: DeepSeek V3.2 Sparse Attention'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [sparse-attention, mla, fp8, block-scale, decode]
techniques: [kernel-fusion, warp-specialization, pipeline-stages, fine-grained-quantization]
hardware_features: [tcgen05, tmem, fp8, block-scale, tma]
kernel_types: [sparse-attention, mla, attention, decode]
languages: [cuda-cpp, cute-dsl, triton, tilelang]
url: https://mlsys26.flashinfer.ai/
---

# Track B: DeepSeek V3.2 Sparse Attention

## Official scope

Track B covers the two components of DeepSeek V3.2 sparse attention on B200:

- lightning indexer workload `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`;
- sparse MLA workload `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

The identifiers specify the evaluated shapes; the linked FlashInfer-Bench workload definitions are authoritative for tensor layouts, allowed tolerances, and scoring. Optimizations must not assume that a generic NSA selection branch alone implements this contract.

## Official winners

The official results page, accessed 2026-08-13, lists:

| Approach | Rank | Team |
|---|---:|---|
| Agent-assisted | 1 | Dogacel |
| Agent-assisted | 2 | Cong |
| Agent-assisted | 3 | Team Wombat |
| Full-agent | 1 | Dogacel |
| Full-agent | 2 | HAN Lab Kernel Mafia |
| Full-agent | 3 | UW SyFI |

Earlier Gemini/GPT/Claude baseline scores were not the final Track B standings and have been removed from the structured contest record.

## Implementation boundary

Current FlashMLA documentation describes an optional FP8 sparse-decode cache entry as 512 bytes of E4M3 data, four FP32 scales (16 bytes), and 64 BF16 RoPE values (128 bytes), totaling 656 bytes. That layout belongs to the documented FlashMLA path; it is not a universal NSA cache format.

Ideas such as fusing indexer and attention or sorting selected blocks are optimization hypotheses until a pinned, correct submission demonstrates them. This page therefore does not present them as winning techniques.

## Sources

- [Official contest and results](https://mlsys26.flashinfer.ai/)
- [Official Track B indexer workload](https://bench.flashinfer.ai/kernels/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64)
- [Official Track B sparse-attention workload](https://bench.flashinfer.ai/kernels/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64)
- [Organizer writeup archive](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)
