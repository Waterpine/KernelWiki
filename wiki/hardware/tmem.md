---
id: hw-tmem
title: "Tensor Memory (TMEM)"
type: hardware
architectures: [sm100, sm100a]
tags: [tmem, tcgen05]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tcgen05-mma, technique-double-buffering, pattern-register-pressure]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [TMEM, "tensor memory", "Tensor Memory"]
---

# Tensor Memory (TMEM)

## Organization

TMEM is dedicated on-chip storage used by fifth-generation Tensor Core
operations. For SM100-family targets described by PTX, a CTA sees 512 columns
and 128 lanes with a 32-bit cell at each lane/column location: 256 KiB of
addressable cells in that view.

TMEM addresses contain a lane index and a column index. Warp-collective
`tcgen05.ld`/`tcgen05.st` operations impose lane-access restrictions: warp 0 of
a warpgroup accesses lanes 0–31, warp 1 lanes 32–63, and so on. This is an
access rule, not a claim that ordinary CUDA threads independently own TMEM rows.

## Allocation lifecycle

Allocation and deallocation are warp-collective. The allocation unit is 32
columns, and the requested number of columns must be a power of two. The
allocated address is written to a shared-memory slot.

```ptx
// Every live lane of the issuing warp executes these collective operations.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], 256;
ld.shared.b32 taddr, [smem_slot];

// ... MMA and epilogue using taddr ...

tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 256;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

All allocations must be explicitly deallocated before kernel exit. For
`.cta_group::2`, peer CTAs participate collectively and must synchronize TMEM
accesses before deallocation.

## Data movement and completion

`tcgen05.ld` and `tcgen05.st` are warp-collective, asynchronous data-movement
operations. Use the legal shape/repeat forms from PTX and wait with
`tcgen05.wait::ld` or `tcgen05.wait::st`. `tcgen05.cp` copies a shaped matrix
from a shared-memory descriptor to TMEM.

For an MMA result, first establish MMA completion with `tcgen05.commit` and its
mbarrier wait. A `tcgen05.fence::before_thread_sync` alone does not make an
in-flight MMA complete.

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [taddr], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];
// wait for mma_done parity ...
tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.16x128b.x16.b32 { /* registers */ }, [taddr];
tcgen05.wait::ld.sync.aligned;
```

## Capacity planning

Column allocations reserve all 128 lanes. Two 256-column allocations consume
the 512-column view, but whether that corresponds to two complete accumulator
tiles depends on the selected MMA's data-path layout. Use CuTe/CUTLASS layout
objects to size allocations rather than equating every logical matrix column
with one TMEM column.

TMEM latency and throughput vary by access shape and instruction. The earlier
blanket “420-cycle cache-miss latency versus 30-cycle SMEM” table mixed a
microbenchmark result with an unsupported cache model and has been removed.

## Sources

- [PTX ISA: Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
- [PTX ISA: allocation and management](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-dealloc-tcgen05-relinquish-alloc-permit)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
