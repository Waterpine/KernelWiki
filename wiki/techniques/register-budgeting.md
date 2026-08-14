---
id: technique-register-budgeting
title: "Register Budgeting for Occupancy"
type: technique
architectures: [sm100, sm90]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, kernel-nvfp4-gemv]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
blackwell_relevance: "TMEM removes some accumulator register pressure, but epilogues, softmax, address state, and unrolling still consume registers."
---

# Register Budgeting

Registers can limit resident blocks/warps, but lowering a register budget can
also introduce spills, recomputation, or less unrolling. Occupancy is stepwise,
not simply inverse to registers per thread.

```cuda
__global__ __launch_bounds__(256, 2)
void memory_kernel(const float* input, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] * 2.0f;
}
```

`__launch_bounds__` supplies compiler launch assumptions; `-maxrregcount` sets
a compilation constraint. Neither guarantees a particular occupancy because
shared memory, barrier slots, thread count, cluster shape, and device limits
also apply.

Measure spills, achieved occupancy, eligible warps, memory stalls, and runtime
together. Spills use the memory hierarchy and are not “free” just because the
kernel was already memory-bound. Approximate contest rank/latency comparisons
do not isolate register count, so they are not retained as causal evidence.
