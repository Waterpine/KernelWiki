# Contest Research Notes

> Factual refresh: 2026-08-13. Final rankings come from the official contest pages. Participant techniques and timings remain source-reported unless independently reproduced.

## GPU Mode Blackwell NVFP4 Kernel Hackathon

The four challenges covered NVFP4 batched GEMV, GEMM, gated dual GEMM, and grouped GEMM on the contest's B200 environment. NVFP4 uses E2M1 data, an E4M3/UE4M3 scale for each block of 16 values, and a higher-level scale as defined by the workload. It is distinct from MXFP4's block-of-32 UE8M0 scaling.

The contest pages, reference-kernels repository, and participant writeups are the authority for shapes, rankings, and timings. Measurements such as Yue's GEMV optimization progression apply to that submission and harness only. A related community benchmark describes a B200 with 142 SMs, but no retained device record independently establishes that count for the contest environment; treat it as author-reported and instance-specific, not as a universal B200 specification.

Frequently reported techniques include workload-specific vectorized loads, distinct cache hints for streamed and reused operands, register-budget experiments, TMA staging, tensor-core/TMEM accumulation, and per-shape specialization. None is automatically beneficial; alignment, instruction legality, resource use, and the exact benchmark decide.

The documented grouped-GEMM reward-hacking incident exploited evaluator object reuse by batching future cases. It is relevant to benchmark integrity, not a valid kernel optimization result.

Primary/official entry points:

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [GPU Mode reward-hacking report](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [NVIDIA NVFP4 overview](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

## FlashInfer AI Kernel Generation Contest at MLSys 2026

The official contest covers three B200 tracks: FP8 fused MoE, DeepSeek V3.2 sparse attention, and Gated Delta Net. Official evaluation scores correctness, speed, and win rate against FlashInfer baselines. Modal results were reference-only because clocks could not be locked.

The official page gives public launch on 2026-01-22, baseline release on 2026-02-09, kernel submission deadline on 2026-04-24, writeup deadline on 2026-05-01, winner notification on 2026-05-12, and awards on 2026-05-22.

### Final winners

| Track | Agent-assisted, ranks 1–3 | Full-agent, ranks 1–3 |
|---|---|---|
| A: Fused MoE | Team Wombat; KernelEvolve; LLM-CUDA | HAN Lab Kernel Mafia; GEMM People; Insider |
| B: Sparse Attention | Dogacel; Cong; Team Wombat | Dogacel; HAN Lab Kernel Mafia; UW SyFI |
| C: Gated Delta Net | Kachua; UW SyFI; LLM-CUDA | UW SyFI; LLM-CUDA; HAN Lab Kernel Mafia |

Pre-contest Gemini, GPT, Claude, and other agent-baseline scores are not final standings. Earlier notes that labeled them as Track A/B/C submissions were erroneous.

The track workload identifiers and final results are recorded in `sources/contests/flashinfer-mlsys26/`. Exact winning techniques should be attributed from public submissions and team writeups, not inferred from the problem statement.

Primary sources:

- [Official contest and results](https://mlsys26.flashinfer.ai/)
- [FlashInfer-Bench](https://bench.flashinfer.ai/)
- [Contest writeups](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)
- [Starter kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
