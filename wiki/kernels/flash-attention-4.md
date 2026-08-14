---
id: kernel-flash-attention-4
title: FlashAttention-4
type: kernel
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp]
confidence: verified
reproducibility: snippet
kernel_types: [attention, flash-attention]
languages: [cute-dsl]
related: [technique-warp-specialization, technique-software-exp, hw-tcgen05-mma, hw-tmem]
sources: [doc-flash-attention-4, blog-flash-attention-4, pr-flash-attention-2441, pr-flashinfer-1850]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: "paper sweep: seqlen 1k-32k, 32k total tokens, documented head dimensions"
    metric: TFLOPS
    value: 1613
    utilization: "~71% paper-reported theoretical maximum"
    source_id: doc-flash-attention-4
evidence_basis:
  - source_id: doc-flash-attention-4
    evidence_type: official-doc
  - source_id: pr-flash-attention-2441
    evidence_type: upstream-code
artifact_dir: artifacts/kernels/flash-attention-4
---

# FlashAttention-4

FlashAttention-4 is an attention algorithm/kernel co-design for Blackwell. It
uses fully asynchronous MMA/TMEM pipelines and restructures non-MMA work that
became comparatively expensive on B200.

## Verified design

```python
def online_attention_step(scores, values, old_max, old_sum, old_out):
    new_max = max(old_max, row_max(scores))
    correction = exp(old_max - new_max)
    probs = exp(scores - new_max)
    new_sum = old_sum * correction + row_sum(probs)
    new_out = old_out * correction + probs @ values
    return new_max, new_sum, new_out
```

The real CuTe DSL kernel overlaps two query tiles: while one tile uses tensor
cores, softmax work proceeds for the other. It uses TMEM for intermediate
matrices and separates correction work into another warpgroup.

The software exponential is not a blanket replacement that makes every
exponential four times faster. The paper applies polynomial emulation to only
about 10--25% of entries and uses the hardware path for the rest, balancing
throughput against register pressure and numerical accuracy.

Backward uses two-CTA MMA to partition operands/TMEM and reduce shared-memory
traffic and dQ atomic reductions. It does not mean two arbitrary CTAs share one
flat TMEM allocation or that a fixed `m256n256k16` instruction implements the
whole backward pass.

## Performance scope

The paper reports a maximum of 1613 TFLOP/s on B200 (about 71%), up to 1.3x over
cuDNN 9.13, and up to 2.7x over its Triton baseline. These are maxima over its
documented sweep, not a guaranteed `seqlen=8192, headdim=128` result. It also
notes that later cuDNN versions incorporated related techniques and achieved
similar performance.

Pinned upstream and derived artifacts are under
[`artifacts/kernels/flash-attention-4/`](../../artifacts/kernels/flash-attention-4/).

`pr-flashinfer-1850` is retained as a separate SM100 FMHA implementation
reference for head-dimension handling; it is not evidence for the FA4 paper's
algorithm or performance number.
