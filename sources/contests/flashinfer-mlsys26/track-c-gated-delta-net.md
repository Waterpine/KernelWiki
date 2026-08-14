---
id: contest-flashinfer-track-c
title: 'FlashInfer MLSys 2026 - Track C: Gated Delta Net'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [gated-delta-net, linear-attention, chunk-parallelism]
techniques: [chunk-parallelism, kernel-fusion, warp-specialization, pipeline-stages]
hardware_features: [tcgen05, tmem, tma]
kernel_types: [gated-delta-net, linear-attention, decode, prefill]
languages: [cuda-cpp, cute-dsl, triton, tilelang]
url: https://mlsys26.flashinfer.ai/
---

# Track C: Gated Delta Net

## Official scope

Track C evaluates Gated Delta Net decode and prefill on B200. The contest names the `qk4_v8_d128_k_last` benchmark family; the FlashInfer-Bench workload definition is authoritative for the exact recurrence, gates, shapes, tolerances, and scoring.

## Official winners

The official results page, accessed 2026-08-13, lists:

| Approach | Rank | Team |
|---|---:|---|
| Agent-assisted | 1 | Kachua |
| Agent-assisted | 2 | UW SyFI |
| Agent-assisted | 3 | LLM-CUDA |
| Full-agent | 1 | UW SyFI |
| Full-agent | 2 | LLM-CUDA |
| Full-agent | 3 | HAN Lab Kernel Mafia |

Earlier Gemini/GPT/Claude baseline scores were not the final Track C standings and have been removed from the structured contest record.

## Algorithm boundary

Gated Delta Net uses a retrieval-error update rather than an unconditional additive outer product. Chunkwise prefill can expose matrix operations inside chunks, but the boundary state remains ordered and requires a valid scan or recurrence. A one-program-per-chunk sketch that races on a shared state is not a correct reference implementation.

This page does not retain preliminary implementation-status claims or cross-model throughput numbers as contest results. Winning techniques should be attributed only from the teams' public submissions and writeups.

## Sources

- [Official contest and results](https://mlsys26.flashinfer.ai/)
- [FlashInfer-Bench](https://bench.flashinfer.ai/)
- [Organizer writeup archive](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)
- [Gated DeltaNet](https://github.com/NVlabs/GatedDeltaNet)
