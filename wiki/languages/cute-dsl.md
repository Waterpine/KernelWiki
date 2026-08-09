---
id: lang-cute-dsl
title: "CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell]
sources: [doc-cutlass-blackwell, blog-colfax-cutlass, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## Overview

CuTe (CUDA Templates) DSL is the primary abstraction layer in CUTLASS 4.5.0 for Blackwell kernels. FlashAttention-4 was implemented entirely in CuTe-DSL (Python variant), achieving 20-30× faster compilation than C++ templates.

## SM100 MMA Atoms

```python
# CuTe-DSL: SM100 MMA operation for FP16/BF16
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

# 1-SM MMA (cta_group::1); the accumulator lives in TMEM
op = tcgen05.MmaF16BF16Op(
    io_dtype,                       # e.g. cutlass.BFloat16
    acc_dtype,                      # e.g. cutlass.Float32
    mma_inst_shape_mnk,             # e.g. (128, 256, 16)
    tcgen05.CtaGroup.ONE,           # tcgen05.CtaGroup.TWO for the 2-CTA instruction
    tcgen05.OperandSource.SMEM,     # A and B read from shared memory
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)
```

## TMEM as CuTe Locale

```python
# The accumulator tensor comes from the tiled MMA; TMEM is allocated by the
# kernel via tcgen05.alloc (see utils.blackwell_helpers) rather than by hand.
tmem_tensor = tiled_mma.make_fragment_C(acc_shape)

# TMEM → registers for the epilogue: build a tmem copy atom, then copy
copy_atom = tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom, tmem_tensor)
cute.copy(tiled_copy_t2r, tmem_frag, reg_frag)  # lowers to tcgen05.ld
```

## TMA Copy Atoms

```python
# TMA bulk-tensor copy operation: global → shared.
# Use CopyBulkTensorTileG2SOp() when the tile is not multicast, or the
# multicast variant when one tile is broadcast to the CTAs of a cluster.
op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)

# Build the TMA atom for operand A against the tiled MMA partitioning
tma_atom_a, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op,
    a,                       # global tensor
    a_smem_layout_slice,     # shared-memory layout of one stage
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
)
```

## Warp-Specialized Kernel Skeleton

```python
@cute_kernel
def blackwell_gemm(A, B, C):
    # Warp 0: TMA producer
    if warp_id == 0:
        for stage in pipeline:
            tma_copy(A_tile, smem_A[stage])
            tma_copy(B_tile, smem_B[stage])
            arrive(mbarrier[stage])

    # Warp 1: MMA consumer
    elif warp_id == 1:
        tmem = alloc_tmem(256)  # columns
        for stage in pipeline:
            wait(mbarrier[stage])
            fence_after_thread_sync()
            mma(smem_A[stage], smem_B[stage], tmem)
        signal_epilogue()

    # Warps 2+: Epilogue
    else:
        wait_epilogue()
        regs = load_tmem(tmem)
        store_global(C, regs)
        dealloc_tmem(tmem)
```

## Why CuTe-DSL for Blackwell

1. **20-30× faster compilation** than C++ CUTLASS templates
2. Python-native: easier to iterate and debug
3. Same performance as hand-written C++ (FlashAttention-4: 1605 TFLOPS)
4. First-class TMEM and tcgen05 support in CUTLASS 4.5.0
5. Automatic layout computation and swizzle handling

## Full Examples (verbatim upstream code shipped locally)

The following CuTe DSL files ship **verbatim** in this repository under `artifacts/prs/cutlass/` (pinned at each PR's merge SHA). Open them with `python3 scripts/get_page.py <pr-id> --include-code` or read them directly.

| File | Purpose | Size |
|---|---|---|
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | Step 0 — FP16 GEMM baseline | 447 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | Step 1 — 2CTA MMA + TMA multicast | 535 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | Step 2 — Warp specialization (TMA / MMA / epilogue warps) | 679 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | Step 3 — Static persistent tile scheduler | 769 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | Step 4 — Preferred + dynamic clusters | 1065 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | Step 5 — TMA prefetch | 919 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | Step 6 — Programmatic Dependent Launch (PDL) | 1002 lines |
| [`artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py`](../../artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py) | Persistent GEMM with prefetch | full |
| [`artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py`](../../artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py) | CLC (Cluster Launch Control) Python binding | full |

The `fp16_gemm_{0..6}.py` series from `examples/python/CuTeDSL/blackwell/tutorial_gemm/` in NVIDIA/cutlass PR-3106 is the authoritative CuTe DSL learning path: it walks from a naive FP16 GEMM baseline through 2CTA MMA with TMA multicast, warp specialization, static persistent scheduling, preferred / dynamic clusters, TMA prefetch, and ends with Programmatic Dependent Launch (PDL). Reading them in order is the recommended on-ramp.

## Related
- [tcgen05-mma](../hardware/tcgen05-mma.md) — Underlying MMA instruction
- [flash-attention-4](../kernels/flash-attention-4.md) — CuTe-DSL implementation
- [CUTLASS Blackwell docs](../../sources/docs/nvidia-cutlass-blackwell.md) — Official reference
