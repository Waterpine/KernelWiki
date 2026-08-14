---
id: contest-gpumode-p3
title: 'GPU Mode NVFP4 Hackathon - Problem 3: Gated Dual GEMM'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemm, fp4, block-scale, tcgen05, tmem, tma]
techniques: [warp-specialization, kernel-fusion, epilogue-fusion, pipeline-stages]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma]
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels
---

# Problem 3: NVFP4 Gated Dual GEMM

The workload evaluates two NVFP4 projections sharing an input, followed by the gated activation defined by the reference contract. A fused implementation may reuse input tiles and avoid global intermediates, subject to its exact output type, scale convention, and numerical tolerance.

On SM100, two accumulator regions must fit legal TMEM allocations and MMA layouts. Completion must be committed to and observed through an mbarrier before TMEM loads, followed by `tcgen05.wait::ld` before reuse or deallocation. No universal “gate columns 0–255, up columns 256–511” partition follows from TMEM capacity; the actual tiled MMA and copy layouts decide.

The former page assigned approximate rankings and techniques from an unpublished Discord thread and presented a synthesized CUTLASS epilogue as a winning implementation. Those claims had insufficient accessible evidence and were removed. The page now records only the public workload and generally valid implementation constraints.

Sources:

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [NVIDIA forum announcement](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
