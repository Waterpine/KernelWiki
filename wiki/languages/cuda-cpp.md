---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [blog-tcgen05-tutorial, doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## Overview

Plain CUDA C++ with inline PTX is used for hand-optimized Blackwell kernels. The tcgen05 tutorial achieved 98% of cuBLAS performance using this approach.

## tcgen05 via Inline PTX

```cuda
// Allocate TMEM. tcgen05.alloc writes the allocated address into SMEM.
__device__ uint32_t tmem_alloc_cta(uint32_t* smem_tmem_addr,
                                   uint32_t num_cols) {
    if (threadIdx.x < 32) {  // one whole warp: tcgen05.alloc is .sync.aligned
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(smem_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_addr), "r"(num_cols)
        );
    }
    __syncthreads();
    return *smem_tmem_addr;
}

// Issue MMA (single thread, typically warp 1 lane 0)
// idesc: the 32-bit instruction descriptor (shapes, types, sparsity)
// enable_input_d: predicate selecting D = A*B + D over D = A*B
__device__ void tcgen05_mma(uint32_t tmem_addr,
                             uint64_t desc_a, uint64_t desc_b,
                             uint32_t idesc, bool enable_input_d) {
    // disable-output-lane: optional 4-element vector for cta_group::1;
    // an all-zero mask keeps every output lane enabled
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16"
        " [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
        :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b),
           "r"(idesc), "r"((uint32_t)enable_input_d),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
    );
}

// Load TMEM to registers
__device__ void tmem_load(float* dst, uint32_t tmem_addr, int cols) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
        : "=f"(*dst) : "r"(tmem_addr)
    );
}

// Deallocate TMEM. Every tcgen05.alloc must be matched by a tcgen05.dealloc
// before the kernel exits. Like the allocation, this instruction is
// .sync.aligned: for .cta_group::1 one whole warp must execute it, so call
// this from all 32 lanes of the allocating warp, never from a single thread.
__device__ void tmem_dealloc(uint32_t addr, uint32_t num_cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(num_cols)
    );
}
```

## mbarrier Synchronization

```cuda
// TMA-MMA synchronization via mbarrier
// expected_bytes: total bytes the TMA will deliver to this stage
__device__ void mbarrier_arrive(uint64_t* mbar, uint32_t expected_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(expected_bytes)
    );
}

__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(phase)
    );
}
```

## Warp Role Dispatch

```cuda
__global__ void blackwell_gemm_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id == 0 && lane_id == 0) {
        // TMA producer: issue cp.async.bulk.tensor
        tma_producer_loop(...);
    } else if (warp_id == 1 && lane_id == 0) {
        // MMA consumer: issue tcgen05.mma
        mma_consumer_loop(...);
    } else if (warp_id >= 2) {
        // Epilogue: read TMEM, write to global
        epilogue_loop(...);
    }
}
```

## Related
- [ptx-sm100](ptx-sm100.md) — PTX instruction reference
- [tcgen05 tutorial](../../sources/blogs/tcgen05-tutorial.md) — Step-by-step guide
