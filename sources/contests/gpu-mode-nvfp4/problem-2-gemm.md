---
id: contest-gpumode-p2
title: 'GPU Mode NVFP4 Hackathon - Problem 2: NVFP4 GEMM'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemm, fp4, block-scale, tcgen05, tmem, tma]
techniques: [warp-specialization, pipeline-stages, swizzling, register-reuse]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma]
kernel_types: [gemm]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels
---

# Problem 2: NVFP4 GEMM

Problem 2 evaluates the reference-kernel repository's NVFP4 GEMM contract on the competition B200 environment. The exact block-scaled data and scale layouts—not a generic `E2M1` matrix type—determine the legal `mxf4nvf4` MMA form and tensor descriptors.

Relevant implementation families include CUTLASS/CuTe SM100 block-scaled collectives, TMA staging, warp-specialized pipelines, TMEM accumulation, and measured tile/stage selection. The MMA instruction applies the instruction's block scales as part of its defined operation; a generic epilogue that multiplies row/column scales into an already accumulated result is not an equivalent description.

The former page listed exact top-three names and 10.807/10.914/10.931-microsecond values sourced only to unavailable Discord submissions, and then inferred the winners' schedules. With no accessible organizer result record or public code establishing those rows, they were removed rather than presented as factual leaderboard data.

Sources:

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
