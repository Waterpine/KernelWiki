---
id: doc-blackwell-compatibility-guide
title: "NVIDIA Blackwell Compatibility Guide"
url: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, cuda-cpp]
retrieved_at: 2026-08-13
---

# NVIDIA Blackwell Compatibility Guide

The live guide is published with CUDA 13.3 documentation. It distinguishes
native cubins from forward-compatible PTX:

- a cubin is compatible within the same compute-capability major revision when
  the target GPU's minor revision is at least the cubin's minor revision;
- ordinary PTX is supported on GPUs with a compute capability at least as high
  as the one used to generate that PTX;
- architecture-conditional `compute_100a`/`sm_100a` code is neither forward nor
  backward compatible, and Hopper `compute_90a` PTX is not supported on
  Blackwell.

CUDA 12.8 introduced native `sm_100` cubin generation. The guide's recommended
Blackwell pair is:

```bash
nvcc -gencode=arch=compute_100,code=sm_100 \
     -gencode=arch=compute_100,code=compute_100 \
     -c mykernel.cu
```

Including the native cubin avoids JIT compilation on the matching target, while
including PTX preserves forward compatibility. Existing applications can test
whether PTX is present by running with `CUDA_FORCE_PTX_JIT=1`, then unsetting
the variable after the test.

The former page added an unsupported product mapping and asserted that ordinary
`compute_100` PTX cannot JIT on a higher compute capability. That contradicts
the guide's general PTX rule; only accelerated architecture-conditional targets
have the stated non-forward-compatible restriction.
