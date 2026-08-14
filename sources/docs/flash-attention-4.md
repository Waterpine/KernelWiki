---
id: doc-flash-attention-4
title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
url: https://arxiv.org/abs/2603.05451
source_category: paper
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling]
retrieved_at: 2026-08-13
---

## Verified paper claims

FlashAttention-4 targets B200/GB200 with BF16 and addresses tensor-core
throughput growing faster than shared-memory, exponential-unit, and scalar-ALU
throughput. Its principal techniques are:

- forward and backward pipelines that overlap MMA, softmax, and memory work;
- two query tiles per CTA in a ping-pong forward schedule;
- partial software emulation of `exp2` using range reduction and a polynomial,
  while retaining hardware exponential for most values to control register use;
- conditional online-softmax rescaling;
- TMEM and two-CTA MMA in backward to reduce shared-memory traffic and dQ
  atomic reductions;
- deterministic backward support and Blackwell-specific scheduling/resource
  allocation.

The implementation is in CuTe DSL. The paper reports 20--30x shorter compile
times than its C++ template comparison.

## Performance context

On B200 with FP16/BF16, batch size chosen for 32k total tokens, sequence lengths
from 1k through 32k, and the paper's head-dimension configurations, the paper
reports up to 1613 TFLOP/s (about 71% of its theoretical maximum), up to 1.3x
over cuDNN 9.13, and up to 2.7x over its Triton baseline.

The previous source page recorded 1605 TFLOP/s and attached it to a single
8192-by-128 shape. The paper's headline is 1613 and is the maximum over the
reported sweep, so the narrower attribution was removed.
