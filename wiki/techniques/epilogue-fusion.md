---
id: technique-epilogue-fusion
title: Epilogue Fusion
type: technique
architectures: [sm100, sm90]
tags: [epilogue-fusion, tmem, warp-specialization]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem, technique-warp-specialization]
related: [technique-warp-specialization, hw-tmem, technique-double-buffering]
sources: [doc-cutlass-blackwell, blog-colfax-cutlass, pr-vllm-16032]
blackwell_relevance: "SM100 epilogues read completed TMEM fragments into registers and may overlap with another independent accumulator buffer."
artifact_dir: artifacts/kernels/epilogue-fusion
---

## Overview

An epilogue transforms an MMA accumulator into the output: for example linear
combination, bias, activation, conversion, quantization, or store. Fusing it can
avoid an intermediate global-memory round trip and another launch.

```cuda
__device__ half fused_epilogue(float acc, float alpha,
                               const float* bias, int column) {
    float value = alpha * acc + bias[column];
    value = value > 0.0f ? value : 0.0f;
    return __float2half(value);
}
```

On SM100, the consumer first waits for MMA completion and performs a legal
collective TMEM load followed by its load wait. Elementwise work then occurs in
registers. Overlap with the next MMA requires distinct, non-conflicting TMEM
storage and an explicit handoff/reuse protocol.

CUTLASS epilogue visitor and collective types are release-specific. There is
no architectural default of fourteen epilogue warps, no universally available
half-TMEM double buffer, and no guarantee that fusing a heavy epilogue is faster:
it can increase registers, shared memory, synchronization, or reduce occupancy.

Artifacts under
[`artifacts/kernels/epilogue-fusion/`](../../artifacts/kernels/epilogue-fusion/)
are labeled as upstream or derived in their provenance files.
