---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [doc-ptx-isa-sm100, doc-cuda-13, doc-nvidia-tuning-guide, pr-cutlass-2139, blog-tcgen05-tutorial]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

## Overview

CUDA C++ implementations can reach Blackwell features through CUDA libraries,
CUTLASS/CuTe wrappers, or carefully versioned inline PTX. The compilation
target must match the instruction's PTX target requirements: use family or
architecture-accelerated targets only as required by the selected form.

## Inline PTX obligations

tcgen05 allocation, loads, stores, and deallocation are warp-collective where
the PTX ISA says so. They cannot be safely wrapped in `if (threadIdx.x == 0)`.
Likewise, `tcgen05.ld` has shape-specific register counts and is asynchronous;
the matching `tcgen05.wait::ld` is required before using its results.

```ptx
// Schematic unscaled f16 form. The real idesc encodes the selected shape/types.
.reg .b32 d;
.reg .b64 a_desc, b_desc;
.reg .pred p;
tcgen05.mma.cta_group::1.kind::f16
    [d], a_desc, b_desc, idesc, p;
```

`tcgen05.mma` is also asynchronous. A dependent TMEM read normally follows a
`tcgen05.commit...mbarrier::arrive`, an mbarrier wait, the required
after-thread fence, and a legal collective load/wait. The ordering fences alone
do not signal MMA completion.

## Prefer versioned wrappers

```cuda
template <class Sm100Collective, class Params>
__global__ void kernel(Params params) {
    Sm100Collective collective(params);
    collective.run();
    collective.finish();
}
```

This snippet illustrates the recommended ownership boundary: keep descriptor
construction, collective participation, and completion inside a wrapper tested
against the selected toolkit. The previous page's hand-written wrappers had an
invalid single-thread allocation, wrong MMA operands, and a load/register-count
mismatch and have been removed.

Reported tutorial performance belongs to its exact shape, build, and combined
optimization sequence; CUDA C++ or inline PTX alone does not guarantee a
percentage of a library baseline.
