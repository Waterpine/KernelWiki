---
id: lang-ptx
title: "PTX Instructions for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## Overview

SM100 PTX instructions for Blackwell-specific hardware features.

## tcgen05 Instructions

```ptx
// Allocate TMEM columns
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [smem_tmem_addr], num_cols;

// MMA: inputs from SMEM, accumulator in TMEM.
// {m0..} is the optional disable-output-lane vector -- 4 elements for
// cta_group::1, 8 for cta_group::2. Omitting it leaves every output lane
// enabled. enable_input_d is a predicate.
tcgen05.mma.cta_group::1.kind::f16  [tmem_addr], desc_a, desc_b, idesc, {m0, m1, m2, m3}, enable_input_d;

// 2-SM cooperative MMA
tcgen05.mma.cta_group::2.kind::f16  [tmem_addr], desc_a, desc_b, idesc, {m0, m1, m2, m3, m4, m5, m6, m7}, enable_input_d;

// Load TMEM to registers
tcgen05.ld.sync.aligned.32x32b.x1.b32  {regs}, [tmem_addr];

// Store registers to TMEM
tcgen05.st.sync.aligned.32x32b.x1.b32  [tmem_addr], {regs};

// Copy a shaped SMEM matrix descriptor into TMEM
tcgen05.cp.cta_group::1.128x256b  [tmem_addr], sdesc;

// Orders subsequent tcgen05 operations after a prior thread-synchronization
// (e.g. before issuing tcgen05.mma once a flag from the producer is observed)
tcgen05.fence::after_thread_sync;

// Deallocate TMEM (MUST before kernel exit)
tcgen05.dealloc.cta_group::1.sync.aligned.b32  tmem_addr, num_cols;
```

## NVFP4 Conversion Instructions

```ptx
// Convert two packed FP4 values to two FP16 values
cvt.rn.f16x2.e2m1x2  result_f16x2, packed_fp4;

// Byte unpacking (faster than bitwise extraction)
mov.b32  {byte0, byte1, byte2, byte3}, packed_word;
```

## Cache Control for Memory-Bound Kernels

```ptx
// Streaming data (use once): bypass L1
ld.global.L1::no_allocate.v2.u64  {r0, r1}, [addr];

// Reused data: keep in L1
ld.global.L1::evict_last.v2.u64  {r0, r1}, [addr];

// Wide vectorized loads
ld.global.v4.u64  {r0, r1, r2, r3}, [addr];  // 256-bit
```

## Cluster Launch Control

```ptx
// Request a not-yet-launched cluster (persistent kernel loop). Asynchronous:
// the 16-byte opaque response lands at [addr]; completion signals [mbar].
mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 state, [mbar], 16;
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [mbar];
// After waiting on [mbar], read and interpret the response:
ld.shared.b128 handle, [addr];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x, y, z, _}, handle;
```

## TMA (Tensor Memory Accelerator)

```ptx
// Bulk tensor copy: global → shared
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_ptr], [tensorMap, {x, y}], [mbarrier];

// Multicast to cluster SMs
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_ptr], [tensorMap, {x, y}], [mbarrier], multicast_mask;
```

## Related
- [cuda-cpp](cuda-cpp.md) — Inline PTX in CUDA C++
- [tcgen05-mma](../hardware/tcgen05-mma.md) — MMA instruction details
- [nvfp4](../hardware/nvfp4.md) — FP4 format details
