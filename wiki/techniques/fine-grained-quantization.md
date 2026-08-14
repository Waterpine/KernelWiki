---
id: technique-fine-grained-quantization
title: "Fine-Grained FP8/FP4 Quantization"
type: technique
architectures: [sm100, sm90]
tags: [fine-grained-quantization, fp8, fp4, nvfp4, block-scale]
confidence: verified
reproducibility: snippet
prerequisites: [hw-nvfp4]
related: [hw-nvfp4, kernel-deepgemm, kernel-nvfp4-gemm]
sources: [doc-ptx-isa-sm100, blog-deepgemm, pr-deepgemm-304, pr-cutlass-2139, doc-nvidia-tuning-guide, pr-vllm-23696]
blackwell_relevance: "SM100 has distinct native MX block-scale and NVFP4 block-16 MMA modes; their scale encodings and layouts are not interchangeable."
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-deepgemm-304
    evidence_type: upstream-code
---

## Overview

Fine-grained quantization assigns scales to smaller regions than a whole tensor.
It can reduce the influence of outliers, but stores and moves more scale values.
Granularity, scale encoding, and the MMA instruction must be designed together.

## Two different schemes

DeepGEMM's FP8 scheme uses project-defined fine-grained scales: commonly one
scale per 1x128 activation region and one per 128x128 weight region. Its Hopper
and Blackwell kernels implement those tensors differently. The pinned SM100
code uses the `tcgen05.mma.kind::mxf8f6f4.block_scale` family and PTX-defined
scale-factor TMEM layouts.

NVFP4 is a different data contract:

```text
matrix element       E2M1 (4 bits)
block scale          E4M3 / PTX nonnegative UE4M3 encoding
block granularity    16 elements
tensor scale         FP32
native PTX family    mxf4nvf4.block_scale.scale_vec::4X
```

MXFP4 instead uses UE8M0 power-of-two scales over blocks of 32 values and the
`mxf4.block_scale` family. A native NVFP4 kernel does not convert its E4M3
scales to UE8M0 before the MMA.

Current PTX accepts `.block16` as an alias for `.scale_vec::4X`, but that alias
was introduced in PTX 8.8. The explicit scale-vector spelling above avoids
implying that the alias exists in every earlier PTX version supporting the
operation.

## Schematic selection

```cpp
enum class ScaleContract { DeepGemmFp8, Nvfp4Block16, Mxfp4Block32 };

// Illustrative dispatch fragment: the surrounding program supplies versioned
// wrappers for each complete, matching PTX/CUTLASS contract.
void dispatch(ScaleContract contract) {
  if (contract == ScaleContract::DeepGemmFp8)
    launch_mxf8f6f4_with_pinned_deepgemm_layout();
  else if (contract == ScaleContract::Nvfp4Block16)
    launch_mxf4nvf4_block16_with_ue4m3_scales();
  else
    launch_mxf4_block32_with_ue8m0_scales();
}
```

This is an interface fragment, not inline PTX or a standalone program. The
instruction descriptor, TMEM scale-factor layout, accumulator type, and
completion sequence must follow the selected PTX form. In particular,
`kind::f8f6f4` is the unscaled family; adding an invented `.block_scale` suffix
to it is not equivalent to `kind::mxf8f6f4.block_scale`.

## Accuracy and performance

Smaller blocks can track local ranges more closely, but consume more scale
bandwidth and metadata. The best scheme depends on model calibration, error
tolerance, shapes, and kernel support. Project-specific promotion intervals or
accumulator techniques should not be generalized as architectural guarantees.
