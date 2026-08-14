---
id: contest-flashinfer-track-a
title: 'FlashInfer MLSys 2026 - Track A: Fused MoE FP8'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [moe, fp8, block-scale, fused-kernel]
techniques: [kernel-fusion, warp-specialization, tile-scheduling, pipeline-stages, fine-grained-quantization]
hardware_features: [tcgen05, tmem, fp8, block-scale, tma]
kernel_types: [moe, fused-kernel, gemm, grouped-gemm]
languages: [cuda-cpp, cute-dsl, triton]
url: https://mlsys26.flashinfer.ai/
---

# Track A: Fused MoE

## Official scope

The FlashInfer AI Kernel Generation Contest at MLSys 2026 describes Track A as
Fused MoE with FP8 support. The official platform evaluates Blackwell B200
kernels for correctness, speed, and win rate against FlashInfer baselines.
Entrants could submit expert-crafted/agent-assisted or fully agent-generated
solutions.

The linked FlashInfer-Bench workload is
`moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`. Its name
encodes top-k 8, 8 groups, top-group 4, 32 local experts per expert-parallel
rank, hidden size 7168, and intermediate size 2048. The benchmark definition
uses 256 global experts with expert parallelism 8 and is the authority for the
full tensor contract.

## Official winners

The results page, accessed 2026-08-13, lists:

| Approach | Rank | Team |
|---|---:|---|
| Agent-assisted | 1 | Team Wombat |
| Agent-assisted | 2 | KernelEvolve |
| Agent-assisted | 3 | LLM-CUDA |
| Full-agent | 1 | HAN Lab Kernel Mafia |
| Full-agent | 2 | GEMM People |
| Full-agent | 3 | Insider |

The official page links each team's writeup and public repository. It does not
rank Gemini, GPT-5, or Claude agent baselines as the final Track A winners; the
former structured submission list did so and has been removed.

## Timeline and reproducibility

The official page gives public launch on 2026-01-22, baseline release on
2026-02-09, submission deadline on 2026-04-24, writeup deadline on 2026-05-01,
winner notification on 2026-05-12, and the award ceremony on 2026-05-22.
Official evaluation ran on bare-metal systems; Modal scores were reference-only
because its clocks could not be locked.

No baseline TFLOPS, launch-count, or launch-latency table is present on the
official contest page. Those values were removed rather than promoted without
a directly reproducible benchmark record.

## Sources

- [Official contest and results](https://mlsys26.flashinfer.ai/)
- [Official Track A benchmark](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)
- [Organizer writeup archive](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)
