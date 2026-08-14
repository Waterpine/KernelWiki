---
id: contest-gpumode-p1
title: 'GPU Mode NVFP4 Hackathon - Problem 1: Batched GEMV'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemv, fp4, block-scale]
techniques: [vectorized-loads, cache-policy, register-reuse, per-k-specialization, data-reuse, register-budgeting, loop-unrolling]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels
artifact_dir: artifacts/contests/gpu-mode-nvfp4/problem-1-gemv
---

# Problem 1: NVFP4 Batched GEMV

The reference workload defines a batched matrix-vector product with packed E2M1 data, block-16 NVFP4 scales, and FP16 output. Exact tensor layout, global-scale convention, tolerances, and benchmark shapes must be taken from the pinned reference-kernels revision.

Participant posts describe experiments with vector width, cache hints, unpack/conversion PTX, register limits, per-shape specialization, and reuse of the vector operand. Yue's public post records a final latency near 22.4 microseconds for its stated contest benchmark progression. That is an author-reported submission measurement, not a hardware bound or a complete official leaderboard.

The previous `submissions` block assigned ranks, techniques, and artifact files by reconstructing Discord discussions. In particular, a teaching/source extract was labeled as the winner's submission. Because no accessible organizer result record or verbatim public submission established those mappings, the structured rankings were removed.

The competition's reported device configuration (including enabled-SM count) is specific to that environment. Likewise, a bandwidth-derived “speed of light” that omits conversion, scale, instruction, and transaction costs is not an observed lower bound.

Sources:

- [GPU Mode reference kernels](https://github.com/gpu-mode/reference-kernels)
- [Yue's participant writeup](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Amandeep's participant writeup](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Simon's participant writeup](https://veitner.bearblog.dev/nvfp4-gemv/)
