---
id: blog-flash-attention-4
title: FlashAttention-4 Blog
author: Tri Dao
url: https://tridao.me/blog/2026/flash4/
source_category: benchmark-blog
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling, conditional-rescaling, cute-dsl]
retrieved_at: 2026-08-13
---

## Scope

Tri Dao's post explains FlashAttention-4's response to Blackwell's imbalance between increased tensor-core throughput and non-matmul work such as softmax. It should be read alongside the paper rather than as a source of invented CUDA/PTX snippets.

## Reported techniques

- ping-pong scheduling alternates query tiles so matmul and softmax-related work can overlap;
- conditional rescaling avoids some online-softmax correction work;
- a Cody-Waite/polynomial exponential path is used selectively, while the hardware exponential remains preferable for most entries;
- the backward kernel uses a coordinated two-CTA MMA design to partition work and reduce shared-memory traffic;
- the implementation uses CuTe DSL.

The software exponential is not a universal replacement for the hardware path. Likewise, “two CTA” does not mean arbitrary CTAs can share an unconstrained TMEM address.

## Performance

The accompanying paper reports up to 1613 TFLOP/s on its B200 BF16 sweep, about 71% using the paper's theoretical maximum, up to 1.3× over cuDNN 9.13, and up to 2.7× over its Triton baseline. The older 1605-TFLOP/s summary was corrected to the paper value and is not assigned to a single shape.

The artifact bundle contains only files whose provenance mode is stated in its `PROVENANCE.yaml`; conceptual scheduling pseudocode is not represented as verbatim upstream code.
