---
id: blog-modular-blackwell
title: "Modular: Matrix Multiplication on Blackwell"
author: Modular
url: https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction
source_category: community-note
architectures: [sm100, sm100a]
tags: [gemm, tcgen05, tmem, tma, 2sm-cooperative, pipeline-stages, tma-multicast, clc]
techniques: [pipeline-stages, tma-multicast, warp-specialization, double-buffering]
hardware_features: [tcgen05, tmem, tma, 2sm-cooperative, clc]
kernel_types: [gemm]
languages: [cuda-cpp]
retrieved_at: 2026-08-13
---

# Modular Blackwell matmul series

Modular's multi-part series develops a Blackwell GEMM and reports reaching about 85% of the comparison target used by the posts. It discusses shared-memory layout, TMA staging/multicast, a multi-stage circular buffer, warp specialization, and cooperative two-CTA MMA.

The useful transfer lesson is structural: a multicast can reduce duplicate operand traffic only when clustered CTAs actually consume the same tile, and deeper staging can hide latency only if its shared-memory and synchronization costs fit. The two CTAs in `cta_group::2` follow the documented paired-MMA data contract; they are not arbitrary workers sharing an accumulator.

The former summary converted the series into universal rules—“five stages is optimal for B200,” a fixed ordering of optimization impact, and exact percentage steps. The posts' results are specific to their kernel, shapes, baseline, and code snapshot, so those generalizations were removed.

Sources:

- [Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- [Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
