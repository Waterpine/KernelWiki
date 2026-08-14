---
id: hw-nvfp4
title: "NVFP4 and Block-Scaled Narrow Precision"
type: hardware
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale, fp8]
confidence: verified
related: [technique-fine-grained-quantization, kernel-nvfp4-gemm, kernel-nvfp4-gemv, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, contest-gpumode-p1, contest-gpumode-p2, blog-yue-nvfp4]
aliases: [NVFP4, E2M1, "FP4 E2M1", "nv_float4"]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

## Overview

NVFP4 represents matrix elements in 4-bit E2M1 and uses a scale for each block
of 16 values plus a higher-level FP32 tensor scale. NVIDIA's Blackwell Tensor
Cores have a dedicated NVFP4 block-scaled MMA mode.

The scale terminology is easy to misread. User-facing material commonly calls
the per-block scale FP8 E4M3. In the PTX ISA, the nonnegative scale encoding for
the `mxf4nvf4` mode is named `ue4m3`. This is distinct from the unsigned E8M0
power-of-two scale used by MX-format modes.

## Format relationship

```text
NVFP4 element:       E2M1, 4 bits
NVFP4 block:         16 elements per E4M3/UE4M3 scale
Tensor-level scale:  FP32

MXFP4 element:       E2M1, 4 bits
MXFP4 block:         32 elements per UE8M0 scale
```

The two formats therefore do not have interchangeable scale tensors. A native
NVFP4 path does not first convert E4M3 block scales to UE8M0.

## PTX instruction family

The PTX family for the NVFP4 operation—E2M1 inputs with UE4M3 block scales—is
canonically written
`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`. Current PTX also
accepts `.block16` as an alias for `.scale_vec::4X`; the alias was added in PTX
8.8, so `.scale_vec::4X` is the less version-ambiguous spelling. Either form is
one scale-vector-size qualifier and is not followed by a second qualifier. The
broader `mxf4nvf4` kind also has UE8M0 combinations, so legal MMA shapes,
operand and scale types, layouts, CTA groups, and scale-factor TMEM forms must
be selected together from the relevant PTX ISA validity table.

`tcgen05.mma.kind::mxf4.block_scale` is the related MXFP4 mode. It is not an
alias for NVFP4.

## Performance interpretation

Native support removes the need for a software dequantization loop before the
MMA, but it does not establish a universal speedup. End-to-end performance also
depends on shapes, scale-layout preparation, memory traffic, scheduling, and
the comparison baseline. Competition latency numbers in the related source
pages are source-reported workload results, not architectural guarantees.

## Related

- [fine-grained-quantization](../techniques/fine-grained-quantization.md)
- [nvfp4-gemm](../kernels/nvfp4-gemm.md)
- [nvfp4-gemv](../kernels/nvfp4-gemv.md)
