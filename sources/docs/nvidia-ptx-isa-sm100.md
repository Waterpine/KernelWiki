---
id: doc-ptx-isa-sm100
title: "PTX ISA SM100 Instructions Reference"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, tcgen05, tmem, clc, tma, nvfp4, fp4, fp8, fp6, block-scale, mbarrier]
retrieved_at: 2026-08-13
---

# PTX ISA SM100 Instructions Reference

## Scope and version

PTX ISA 8.6 introduced the principal fifth-generation Tensor Core instructions
used by SM100 accelerated targets: `tcgen05.mma`, TMEM allocation and data
movement, `tcgen05.commit`, and the tcgen05 ordering fences. Later PTX releases
extend the supported targets and forms. Always use the exact syntax and target
requirements from the PTX version shipped with the selected CUDA toolkit.

## `tcgen05.mma`

`tcgen05.mma` is asynchronous. A single elected thread issues the hardware MMA,
although compiler or CuTe interfaces may require the call site to be
warp-uniform and perform election internally. The destination is TMEM. Depending
on the MMA kind and mode, operand A may be described in shared memory or reside
in TMEM; operand B is described in shared memory.

The dense instruction family includes unscaled floating-point kinds
`f16`, `tf32`, and `f8f6f4`, integer kind `i8`, and block-scaled kinds
`mxf8f6f4`, `mxf4`, and `mxf4nvf4`. Legal M, N, K, layout, type, scale-vector,
and CTA-group combinations are kind-specific. The often-used
`m128n256k16` BF16 tile is a maximum/example configuration, not the only legal
shape.

```ptx
// Schematic only: idesc and predicate operands are required by the selected form.
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;
```

## Completion and ordering

`tcgen05.fence::before_thread_sync` and
`tcgen05.fence::after_thread_sync` are ordering/code-motion fences. They do not
by themselves wait for an asynchronous MMA to finish.

For an MMA result that will be consumed through `tcgen05.ld`, the documented
completion path is:

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];

wait:
mbarrier.try_wait.parity.b64 p, [mma_done], phase;
@!p bra wait;

tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.16x128b.x16.b32 {r0, r1, r2, r3}, [d_tmem];
tcgen05.wait::ld.sync.aligned;
```

The register list and load shape above are abbreviated; consult the PTX table
for the required number of destination registers. Cross-thread producer and
consumer patterns additionally need the documented execution synchronization
between the before/after fences.

## Tensor Memory (TMEM)

On SM100-family targets covered by the PTX reference, a CTA's TMEM view has 512
columns and 128 lanes, with 32-bit cells. Allocation is collective across one
warp, in a power-of-two number of columns with a 32-column allocation unit.
All allocated TMEM must be explicitly deallocated before kernel exit.

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], 32;
ld.shared.b32 taddr, [smem_slot];
// ... use taddr ...
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 32;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

`tcgen05.ld` and `tcgen05.st` are warp-collective operations with documented
lane-access restrictions. `tcgen05.st` is asynchronous and is completed with
`tcgen05.wait::st`; `tcgen05.ld` is completed with `tcgen05.wait::ld`.

## Shared-memory layout and swizzling

128-byte swizzling is not universally mandatory for `tcgen05.mma`. The PTX ISA
lists legal combinations by element width, major order, matrix, and swizzle.
For example, K-major A and B layouts for common element widths support all
swizzling modes, while some transposed 32-bit forms require 128-byte swizzling
with 32-byte atomicity. The producer layout and the MMA descriptor must agree.

## Cluster Launch Control

The SM100 CLC instruction is `clusterlaunchcontrol.try_cancel`, not
`clc.arrive` or `clc.wait`. It asynchronously attempts to cancel a cluster that
has not started, writes a 16-byte opaque result to shared memory, and reports
completion through an mbarrier. Query instructions determine whether the
request succeeded and recover the first canceled CTA ID.

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], 16;
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.b128 [response], [bar];
// wait on bar, load the 16-byte response, then query it
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response_reg;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, _}, response_reg;
```

After a failed cancellation has been observed, issuing another cancellation
request is undefined. CLC steals a not-yet-launched grid work item; it does not
accept an arbitrary tile coordinate to cancel.

## TMA and mbarrier

TMA tensor copies use `cp.async.bulk.tensor.*` and a tensor-map descriptor.
Shared-memory completion is tracked through an mbarrier transaction count.
Swizzling is selected by the tensor map when needed; it is a layout choice with
mode-specific constraints, not an unconditional requirement.

## Authoritative sources

- [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Archived PTX ISA 8.8 (CUDA 12.9.1)](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/)
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
