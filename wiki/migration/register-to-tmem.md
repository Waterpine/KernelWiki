---
id: migration-register-to-tmem
title: "Register Accumulators to TMEM"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tmem, tcgen05]
related: [hw-tmem, hw-tcgen05-mma, pattern-register-pressure]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, pr-cutlass-2139, pr-vllm-22738]
blackwell_relevance: "SM100 tcgen05 uses TMEM for destination/accumulator storage, changing allocation, completion, and epilogue design."
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
reproducibility: pseudocode
---

# Register Accumulators to TMEM

## What changes

For a Hopper `wgmma` tile, each participating thread owns a fragment of the
register accumulator. For example, an `m64n256` FP32 destination contains
16,384 FP32 values, or 128 accumulator registers per thread when distributed
across 128 threads. That figure describes the accumulator fragment; total
register use and occupancy still depend on compiler allocation and the rest of
the kernel.

SM100 `tcgen05.mma` writes the destination to TMEM. This removes that
register-resident fragment, but it does not imply a fixed register count or a
particular CTA occupancy. TMEM capacity, SMEM, barriers, warp count, and the
epilogue may become the limiting resources.

## Correct lifecycle

```text
collective warp allocation (32-column unit, power-of-two column count)
    -> shared-memory slot receives TMEM address
TMA/SMEM producer synchronization
    -> asynchronous tcgen05.mma operations
tcgen05.commit to completion mbarrier
    -> wait for mbarrier phase
tcgen05.fence::after_thread_sync when required by the consumption pattern
    -> collective tcgen05.ld
tcgen05.wait::ld
    -> register epilogue and global store
collective tcgen05.dealloc before exit
```

`tcgen05.fence::before_thread_sync` plus `__syncthreads()` is not an MMA
completion mechanism. The documented non-pipelined completion path uses
`tcgen05.commit` and an mbarrier wait.

## Allocation example

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], 256;
ld.shared.b32 taddr, [smem_slot];
// ... issue MMA, commit, wait, load, and complete the epilogue ...
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 256;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

All live lanes of the issuing warp execute the collective allocation and
deallocation instructions. For `.cta_group::2`, the peer-CTA warp participates.

## Capacity and double buffering

A CTA's SM100 TMEM view has 512 columns and 128 lanes of 32-bit cells. Two
256-column allocations consume that view, but logical accumulator dimensions do
not always map one-for-one to TMEM columns; the selected MMA data-path layout is
authoritative. Confirm the allocation with CUTLASS/CuTe layout objects before
assuming that two logical tiles fit.

Double buffering can overlap an epilogue with later MMA work only when the
pipeline also preserves all data and synchronization dependencies. Two TMEM
allocations alone do not create overlap.

## Sources

- [PTX ISA: Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
- [PTX ISA: tcgen05 completion](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
