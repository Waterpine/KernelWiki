---
id: kernel-gated-dual-gemm
title: Gated Dual GEMM (Gate-Up + SwiGLU Fusion)
type: kernel
architectures: [sm100, sm90]
tags: [gated-dual-gemm, gemm, fused-kernel, kernel-fusion, nvfp4, tmem]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, cute-dsl]
related: [kernel-nvfp4-gemm, kernel-fused-moe, technique-kernel-fusion, technique-epilogue-fusion]
sources: [contest-gpumode-p3, blog-deepgemm, blog-tflops-gap-fp4-moe, pr-vllm-23696]
performance_claims: []
blackwell_relevance: "TMEM can hold gate and up accumulator regions for a fused kernel when the selected MMA layouts and legal column allocations fit."
artifact_dir: artifacts/kernels/gated-dual-gemm
---

# Gated Dual GEMM

## Operation

A gated MLP commonly computes:

```python
def gated_projection(x, w_gate, w_up):
    gate = matmul(x, w_gate)
    up = matmul(x, w_up)
    return silu(gate) * up

gate = X @ W_gate
up   = X @ W_up
out  = SiLU(gate) * up
```

A fused kernel can reuse an input tile and avoid writing the two projection outputs to global memory before the activation and multiply. The exact saving depends on whether the baseline already fuses the projections or epilogue.

## SM100 implementation constraints

- issue each documented MMA with operands, descriptors, scale layout, and TMEM destination that match its instruction kind;
- allocate TMEM collectively in legal power-of-two column units;
- use the documented commit-to-mbarrier and wait sequence before dependent TMEM loads;
- execute `tcgen05.wait::ld` before reusing or deallocating loaded TMEM;
- budget registers for two epilogue values and verify that fusion does not reduce residency enough to offset saved traffic.

Names such as `tmem_alloc`, `tmem_load`, and `tcgen05_mma` are not CUDA intrinsics. The previous illustrative kernel used those placeholders as if it were compilable and was removed.

The cited contest summary did not provide a complete authoritative leaderboard record, so no numeric performance claim is retained here.

## Artifacts

The bundle under [`artifacts/kernels/gated-dual-gemm/`](../../artifacts/kernels/gated-dual-gemm/) contains its upstream patch/extract provenance. Consult `PROVENANCE.yaml` before treating a file as verbatim upstream code.
