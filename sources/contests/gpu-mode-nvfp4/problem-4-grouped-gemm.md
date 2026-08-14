---
id: contest-gpumode-p4
title: 'GPU Mode NVFP4 Hackathon - Problem 4: Grouped GEMM'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, grouped-gemm, fp4, block-scale, tcgen05, tmem, tma, moe]
techniques: [warp-specialization, tile-scheduling, pipeline-stages, kernel-fusion]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma, clc]
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels
---

# Problem 4: NVFP4 Grouped GEMM

The public workload evaluates a set of NVFP4 GEMMs with group-specific problem data. Its tensor lists, shapes, scale layouts, tolerances, and scoring come from the pinned reference-kernel definition.

Grouped scheduling can use a static tile map, a software persistent work structure, or—in a compatible SM100 design—CLC cancellation of not-yet-started grid work. CLC returns a grid ID from a cancellation response; it does not directly pop an arbitrary expert tile. Descriptor setup, small-M padding, group transitions, and load imbalance remain workload-specific costs.

## Evaluator incident

GPU Mode's public postmortem reports that one submission exploited reused benchmark objects: it computed future cases during an earlier timed call and returned precomputed results later. The reported 11.191-microsecond leaderboard result was invalid. It is recorded as an evaluator-integrity incident, not as rank-one kernel performance.

The former page also supplied approximate “legitimate” ranks, 13–15-microsecond values, CLC techniques, and CUTLASS schedules without public submissions or an organizer results table. Those assertions were removed.

Sources:

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [GPU Mode reward-hacking postmortem](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
