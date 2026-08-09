---
id: blog-colfax-cutlass
title: 'Colfax CUTLASS Tutorial: GEMM Kernels Using Tensor Memory for Blackwell'
author: Colfax Research
url: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tmem-for-nvidia-blackwell-gpus/
source_category: community-note
architectures:
- sm100
tags:
- tcgen05
- tmem
- cute-dsl
- warp-specialization
- 2sm-cooperative
retrieved_at: 2026-04-16
artifact_dir: artifacts/blogs/colfax-cutlass-blackwell/code
---

## Summary

Detailed tutorial on CUTLASS abstraction for Blackwell UMMA (tcgen05.mma) with sub-byte GEMM support.

## Key Content
- UMMA replaces WGMMA: register-free operation, single-thread launch, built-in block scaling
- TMEM: 512 columns × 128 rows of 32-bit cells (256KB/SM)
- 32-bit addressing: bits 31-16 = lane ID, bits 15-0 = column
- CUTLASS two-level abstraction: MMA_Atom (PTX wrapper) + MMA_Traits (CuTe layouts)
- Architectural progression: Volta → Hopper TMA → Blackwell TMEM+UMMA
- Sub-byte GEMM tutorial covering NVFP4, MXFP4, block scaling

## Key Code

### TMEM allocation + tcgen05.mma (single-thread launch)

```cuda
// UMMA on Blackwell: one thread drives the MMA for the whole CTA.
// Accumulator lives in TMEM, not registers.
__shared__ uint32_t tmem_addr;

// tcgen05.alloc is .sync.aligned: with cta_group::1 one whole warp of the CTA
// must issue it, all 32 threads with identical operands. A single-thread guard
// would be undefined behaviour, so this is a warp guard, not a lane guard.
if (threadIdx.x < 32) {
    // Allocate 256 TMEM columns (all 128 lanes of each column) for the accumulator
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
                 :: "r"(smem_addr));
}
__syncthreads();

// Issue UMMA: A and B live in SMEM, C accumulates into TMEM.
// disable-output-lane is an optional 4-element vector for cta_group::1 -- an
// all-zero mask, like omitting the operand, keeps every output lane enabled.
// enable-input-d is a predicate, hence the setp.
uint32_t mask[4] = {0, 0, 0, 0};
if (threadIdx.x == 0) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n"
        "}\n"
        :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}
```

### TMEM load into registers for epilogue

```cuda
// Epilogue warps drain TMEM → registers using tcgen05.ld
// Each warp loads 32 columns (=128 bytes) at a time.
float reg[4];
asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x4.b32 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=f"(reg[0]), "=f"(reg[1]), "=f"(reg[2]), "=f"(reg[3])
    : "r"(tmem_addr + warp_col_offset));
```

### CUTLASS MMA_Atom wrapping

```cpp
// The CUTLASS two-level abstraction: MMA_Atom wraps the PTX intrinsic,
// MMA_Traits maps logical MxNxK shapes to TMEM addressing.
using Atom = cute::MMA_Atom<cute::SM100_MMA_F16BF16_SS<
    cute::half_t, cute::half_t, float,     // A, B, C types
    128, 256,                               // MxN tile
    cute::UMMA::Major::K, cute::UMMA::Major::K
>>;
```
