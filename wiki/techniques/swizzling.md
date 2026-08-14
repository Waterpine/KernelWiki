---
id: technique-swizzling
title: "Shared Memory Swizzling"
type: technique
architectures: [sm100, sm90]
tags: [swizzling, shared-memory-optimization, tma]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
reproducibility: snippet
prerequisites: [hw-tma]
related: [hw-tma, technique-pipeline-stages, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-modular-blackwell]
blackwell_relevance: "Blackwell tcgen05 descriptors support multiple legal swizzle/layout combinations; 128-byte swizzling is a frequent performance choice, not a universal correctness requirement."
---

# Shared Memory Swizzling

## Purpose

Shared-memory swizzling permutes addresses so the consumer's access pattern is
distributed across banks. TMA can write a swizzled shared-memory box directly,
and CUTLASS/CuTe composes the matching layout into the MMA descriptor.

The producer and consumer must agree on the layout. A mismatch can produce
incorrect values. That does not mean one swizzle mode is mandatory: PTX lists
legal combinations by element width, major order, operand, and swizzle.

For `tcgen05.mma`, common K-major A and B forms support all swizzling modes.
Some transposed forms narrow the valid choices; for example, certain transposed
32-bit layouts require 128-byte swizzling with 32-byte atomicity.

## Choosing a mode

| Mode | Typical reason to consider it |
|---|---|
| None | Naturally conflict-free or non-matrix shared data |
| 32-byte / 64-byte | Narrower boxes or a layout whose legal descriptor uses that mode |
| 128-byte | Wide tensor-core tiles and transpose patterns where it is legal/required |

Choose by:

1. checking the PTX validity table for the exact type, major order, and operand;
2. constructing the TMA and MMA/CuTe layouts from the same definition;
3. validating numerical results; and
4. measuring shared-bank-conflict and throughput counters.

```cuda
// The mode is selected to match smem_layout and the consumer descriptor.
CUtensorMapSwizzle selected_swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
CUresult result = cuTensorMapEncodeTiled(
    &tensor_map, element_type, rank, global_ptr,
    global_dims, global_strides, box_dims, element_strides,
    CU_TENSOR_MAP_INTERLEAVE_NONE, selected_swizzle,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
if (result != CUDA_SUCCESS) return;
```

## Interpreting tutorial numbers

The source-reported `tcgen05 for dummies` benchmark rose from 255 to 695 TFLOPS
after its 128-byte-swizzled layout change. That result demonstrates a large bank
conflict/layout bottleneck in that kernel and environment. It does not establish
that unswizzled `tcgen05.mma` is invalid or that every kernel receives a 2.7×
gain.

## Sources

- [PTX ISA: valid type/major/swizzle combinations](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-valid-combinations-of-type-size-major-ness-and-swizzling)
- [CUDA Programming Guide: TMA swizzle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
