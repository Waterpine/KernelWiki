---
id: kernel-nvfp4-gemm
title: NVFP4 GEMM — 4-bit Floating Point Matrix Multiply
type: kernel
architectures: [sm100, sm100a]
tags: [gemm, nvfp4, fp4, block-scale, tcgen05, tmem, warp-specialization]
confidence: verified
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl, ptx]
related: [hw-nvfp4, hw-tcgen05-mma, hw-tmem, kernel-nvfp4-gemv, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell, pr-cutlass-2139, contest-gpumode-p2]
performance_claims: []
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
artifact_dir: artifacts/kernels/nvfp4-gemm
---

# NVFP4 GEMM

## Verified data contract

An NVFP4 GEMM uses E2M1 matrix elements, one E4M3/UE4M3 scale per 16 values,
and a tensor-level FP32 scale. It must not reuse an MXFP4 scale layout, which
instead uses UE8M0 scales over blocks of 32 values.

At the PTX level, the native block-scaled operation belongs to the
`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` family. Current PTX
accepts `.block16` as a PTX 8.8-and-later alias for `.scale_vec::4X`; both are
single scale-vector-size qualifiers. The selected form also requires an
instruction descriptor, operand descriptors, and legal scale-factor TMEM
layouts. Those operands are omitted below because a shortened inline-assembly
string would look executable while being incomplete.

```text
global NVFP4 A/B and scale tensors
        |
        | TMA or ordinary copies, using legal tensor-map/layout constraints
        v
shared-memory A/B tiles + TMEM scale-factor tiles
        |
        | tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X
        v
TMEM accumulator
        |
        | tcgen05.commit -> mbarrier wait -> tcgen05.ld/wait
        v
register epilogue and global output
```

```ptx
.reg .b32 d;
.reg .b64 a_desc, b_desc;
.reg .pred p;
// Schematic family only; the selected form supplies the remaining operands.
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [d], a_desc, b_desc, idesc, [a_sf_tmem], [b_sf_tmem], p;
```

## Synchronization obligations

`tcgen05.mma` is asynchronous. An implementation that consumes its result must
use the documented completion path, normally `tcgen05.commit` associated with
an mbarrier, wait for that barrier phase, then perform the required
`tcgen05.fence::after_thread_sync` and `tcgen05.ld`/`tcgen05.wait::ld`
sequence. The before/after fences alone do not wait for MMA completion.

TMEM allocation and deallocation are warp-collective operations. The allocated
column count and address mapping must follow the PTX rules, and all allocations
must be explicitly released before kernel exit.

## CUTLASS implementation notes

CUTLASS encodes the PTX descriptor and scale layouts in its SM100 collective
MMA types. Prefer a schedule selected by the installed CUTLASS version instead
of copying a class name from an older example. A correct schedule still needs
shape-specific choices for tile size, cluster shape, pipeline depth, and
epilogue.

The contest source reports that participants targeted B200 and compared against
vendor-library implementations. Its public summary does not provide enough
authoritative evidence to retain the former exact `10.807 us` leaderboard value
as a verified performance claim, so this page makes no numeric claim.

## Reproduction path

Pinned upstream and derived teaching artifacts are under
[`artifacts/kernels/nvfp4-gemm/`](../../artifacts/kernels/nvfp4-gemm/). Check
their provenance before treating a snippet as verbatim upstream code.
