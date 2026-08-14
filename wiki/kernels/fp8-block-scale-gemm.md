---
id: kernel-fp8-block-scale-gemm
title: "FP8 Block-Scale GEMM"
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: verified
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, ptx]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, blog-deepgemm, pr-deepgemm-304, doc-cutlass-blackwell]
performance_claims: []
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-deepgemm-304
    evidence_type: upstream-code
blackwell_relevance: "SM100 provides the mxf8f6f4 block-scale tcgen05 family; scale encoding and layout must match the selected library contract."
---

# FP8 Block-Scale GEMM

Block-scale FP8 uses multiple scales across a matrix rather than a single
per-tensor scale. “FP8 block-scale” is not one universal layout: DeepGEMM,
CUTLASS, and model formats may arrange their scale tensors differently.

For the captured DeepGEMM SM100 path, the project accepts packed UE8M0 scale
tensors and the PTX wrapper uses:

```ptx
.reg .b32 d;
.reg .b64 a_desc, b_desc;
.reg .pred p;
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
    [d], a_desc, b_desc, idesc, [a_sf_tmem], [b_sf_tmem], p;
```

Scale-factor operands are TMEM addresses in this form, not invented shared
memory descriptors appended to unscaled `kind::f8f6f4`. Completion follows the
same asynchronous commit/mbarrier contract as other tcgen05 MMA operations.

DeepGEMM's current README distinguishes FP32 scale tensors on SM90 from packed
UE8M0 tensors on SM100. Its project-specific Hopper promotion technique should
not be converted into a generic claim that every WGMMA accumulator has exactly
“FP22” precision or that one interval is universally required.

The upstream “up to 1550 TFLOPS on H800” headline is not tied to the former
wiki's hard-coded `4096^3` shape, so that structured claim was removed.
