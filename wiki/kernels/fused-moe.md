---
id: kernel-fused-moe
title: Fused MoE — FP8 Block-Scale Routing and Expert Computation
type: kernel
architectures: [sm100, sm100a, sm90]
tags: [moe, fused-kernel, fp8, block-scale, kernel-fusion, grouped-gemm, gated-dual-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, fused-kernel, grouped-gemm, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl, triton]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-fine-grained-quantization, technique-tile-scheduling]
sources: [contest-flashinfer-track-a, blog-deepgemm, pr-vllm-23696]
performance_claims: []
blackwell_relevance: "The MLSys 2026 contest evaluates fused FP8 MoE on B200; implementations may use SM100 tensor-core, TMA, and scheduling facilities."
artifact_dir: artifacts/kernels/fused-moe
---

# Fused MoE

A Mixture-of-Experts forward path selects experts, dispatches token rows,
executes gate/up and down projections, applies the activation, and combines
weighted expert outputs. “Fused MoE” covers several possible boundaries; it
does not imply that routing and both GEMMs execute in one CUDA launch.

```text
tokens -> routing/top-k -> dispatch
       -> per-expert gate/up GEMM -> SwiGLU
       -> per-expert down GEMM -> weighted combine
```

## Correctness contract

A kernel must preserve the reference router's grouping, top-k selection,
weights, expert mapping, quantization scales, output dtype, and numerical
tolerance. FP8 block-scale tensors must follow the selected library/kernel
layout; the number “128” in one workload's scale granularity is not a universal
TMA alignment rule.

```python
def swiglu(gate, up):
    # Schematic operation between the two expert projections.
    activated_gate = gate * sigmoid(gate)
    fused_output = activated_gate * up
    return fused_output
```

## Blackwell implementation notes

An SM100 implementation can stage tensor-map copies through shared memory and
accumulate with a legal block-scaled `tcgen05.mma` form. Before reading an MMA
result from TMEM, it must use the documented `tcgen05.commit`/mbarrier
completion path and the required load ordering. The former page's inline PTX
used the unscaled `kind::f8f6f4` form while passing invented scale operands and
read TMEM without a completion operation; that code was not executable.

Expert work can be mapped statically, through a software counter, or with a
versioned persistent scheduler. CLC only claims a not-yet-launched grid ID; it
does not directly select an arbitrary expert tile.

## Evidence and performance

The official MLSys 2026 FlashInfer contest confirms a B200 Fused MoE track and
publishes separate agent-assisted and full-agent winners with repositories and
writeups. The old framework latency/TFLOPS table and `21.9%` traffic statement
were not present on the official contest page or anchored to a reproducible
artifact, so no numeric performance claim is retained here.

## Reproduction path

Local snippets are under
[`artifacts/kernels/fused-moe/`](../../artifacts/kernels/fused-moe/). Their
provenance labels determine whether they are upstream captures or derived
teaching material.
