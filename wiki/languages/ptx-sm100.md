---
id: lang-ptx
title: "PTX Instructions for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

# PTX Instructions for SM100

## Target and version first

Most fifth-generation Tensor Core instructions were introduced in PTX ISA 8.6
and require accelerated/family-specific SM100 targets as listed by the selected
PTX release. Compile against the CUDA toolkit's own PTX manual; later PTX 9.x
documents additional targets and instruction forms.

## TMEM allocation and MMA completion

```ptx
// Warp-collective allocation; result is written to shared memory.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], 256;
ld.shared.b32 taddr, [smem_slot];

// Single elected hardware issue; call-site rules depend on the wrapper used.
tcgen05.mma.cta_group::1.kind::f16
    [taddr], a_desc, b_desc, idesc, enable_input_d;

// A fence alone does not wait. Commit completion to an mbarrier.
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];
wait_mma:
mbarrier.try_wait.parity.b64 p, [mma_done], phase;
@!p bra wait_mma;

tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.16x128b.x16.b32 { /* full register list */ }, [taddr];
tcgen05.wait::ld.sync.aligned;

tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 256;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

The load destination list is abbreviated. PTX requires the exact register count
for the selected `.shape` and `.num` qualifiers.

## Cluster Launch Control

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], 16;
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.b128 [response], [bar];
// wait, load the b128 response, then use query_cancel.is_canceled and
// query_cancel.get_first_ctaid.
```

The request steals a not-yet-launched CTA/cluster ID. It neither returns a
free-form “next tile” nor accepts a caller-selected tile to cancel.

## TMA and mbarrier

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], bytes;
cp.async.bulk.tensor.2d.shared::cta.global
    .mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {x, y}], [bar];
```

State-space, multicast, and async-group qualifiers vary by direction. Use an
exact example from the relevant PTX section rather than composing qualifiers by
analogy.

## Cache hints and conversions

Cache eviction hints (`L1::no_allocate`, `L1::evict_last`) are performance hints,
not guarantees. Wide vector loads require natural alignment. FP4/FP6/FP8
conversion forms have strict packed source/destination types and rounding-mode
rules; verify the exact spelling in the toolkit's PTX version.

## Sources

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
