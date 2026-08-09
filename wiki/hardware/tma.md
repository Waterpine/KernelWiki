---
id: hw-tma
title: "Tensor Memory Accelerator (TMA)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [tma, mbarrier]
confidence: source-reported
related: [hw-tcgen05-mma, technique-pipeline-stages, technique-swizzling]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-flashinfer-2387]
aliases: [TMA, "tensor memory accelerator", "cp.async.bulk"]
blackwell_relevance: "TMA is shared with Hopper but enhanced on Blackwell. The TMA swizzle mode must match the swizzle mode encoded in the tcgen05 shared memory descriptor; several modes are valid, not only 128-byte."
---

# Tensor Memory Accelerator (TMA)

## Overview

The Tensor Memory Accelerator (TMA) is a hardware unit that performs **asynchronous bulk data transfers** between global memory and shared memory. First introduced on Hopper (SM90), TMA carries forward to Blackwell (SM100) with stricter requirements for tcgen05 compatibility.

TMA offloads data movement from CUDA cores entirely -- a single thread issues the transfer, and the TMA hardware engine handles the multi-dimensional copy, address calculation, out-of-bounds clamping, and format conversion.

## Key Properties

| Property | Detail |
|---|---|
| Transfer direction | GMEM <-> SMEM (bidirectional) |
| Dimensionality | 1D to 5D tensor copies |
| Max box (tile) size | Up to 256 elements per dimension (`boxDim[i] <= 256`), with `boxDim[0] * sizeof(element)` a multiple of 16 bytes |
| Swizzle modes | None, 32B, 64B, 128B (must match the swizzle field of the tcgen05 shared memory descriptor) |
| Format conversion | FP32<->BF16, FP32<->FP16 during transfer |
| Multicast | Single GMEM tile -> multiple CTAs in a cluster |
| Synchronization | mbarrier-based (arrive/wait) |
| Thread requirement | Single thread issues the operation |

## TMA Descriptor

TMA operations are driven by a **descriptor** that encodes the tensor layout, addressing, and transfer parameters. The descriptor is created on the host and passed to the kernel.

### Host-Side Descriptor Creation

```cuda
#include <cuda.h>

// Create a 2D TMA descriptor for a row-major FP16 matrix
CUtensorMap create_tma_descriptor_2d(
    const half* global_ptr,
    int M, int N,           // Global tensor dimensions
    int tile_m, int tile_n, // Tile dimensions for each transfer
    int swizzle_bytes       // Swizzle mode: 0, 32, 64, 128
) {
    CUtensorMap tensor_map;

    // Tensor dimensions (outermost to innermost)
    uint64_t global_dims[2] = {(uint64_t)N, (uint64_t)M};
    uint64_t global_strides[1] = {(uint64_t)(N * sizeof(half))};

    // Tile box dimensions
    uint32_t box_dims[2] = {(uint32_t)tile_n, (uint32_t)tile_m};

    // Element strides (1 = contiguous)
    uint32_t elem_strides[2] = {1, 1};

    CUtensorMapSwizzle swizzle;
    switch (swizzle_bytes) {
        case 0:   swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case 32:  swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case 64:  swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case 128: swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  // Element type
        2,                                  // Dimensionality
        (void*)global_ptr,                 // Global pointer
        global_dims,                       // Tensor dimensions
        global_strides,                    // Byte strides (exclude innermost)
        box_dims,                          // Tile/box dimensions
        elem_strides,                      // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    return tensor_map;
}
```

### Blackwell Requirement: TMA Swizzle Must Match the SMEM Descriptor

`tcgen05.mma` accepts several swizzling modes: all modes are valid for K-major
operands, all except 128-byte-with-32-byte-atomicity for MN-major 8/16-bit
operands, and only 128-byte-with-32-byte-atomicity for MN-major 32-bit operands.
What matters is that the mode encoded in bits 61-63 of the shared memory
descriptor matches the layout the TMA actually wrote.

```cuda
// The TMA swizzle mode and the descriptor swizzle field must agree:
CUtensorMap desc = create_tma_descriptor_2d(ptr, M, N, 128, 64, 128);
// -> shared memory descriptor swizzle field must be 2 (128-Byte swizzling)

CUtensorMap desc = create_tma_descriptor_2d(ptr, M, N, 128, 64, 64);
// -> shared memory descriptor swizzle field must be 4 (64-Byte swizzling)
```

## Asynchronous Copy Operations

### GMEM to SMEM (Load)

```cuda
// TMA load: single thread issues, hardware executes asynchronously
__device__ void tma_load_tile(
    const CUtensorMap* desc,
    void* smem_ptr,
    uint64_t* mbar_ptr,  // mbarrier for synchronization
    int coord_x, int coord_y
) {
    if (threadIdx.x == 0) {
        // Set expected bytes on the mbarrier
        uint32_t expected_bytes = TILE_M * TILE_N * sizeof(half);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "r"(expected_bytes)
        );

        // Issue TMA copy
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
              "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr))
        );
    }
}
```

### SMEM to GMEM (Store)

```cuda
// TMA store: write a tile from shared memory back to global memory
__device__ void tma_store_tile(
    const CUtensorMap* desc,
    const void* smem_ptr,
    int coord_x, int coord_y
) {
    if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0, {%1, %2}], [%3];"
            :
            : "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(smem_ptr))
        );

        // Commit the store
        asm volatile("cp.async.bulk.commit_group;");
    }
}
```

## mbarrier Synchronization

TMA uses mbarriers (memory barriers) for producer-consumer synchronization. The pattern is:

1. **Producer** (TMA): arrives at the barrier when the transfer completes, decrementing the expected transaction count.
2. **Consumer** (compute warps): waits on the barrier before reading the loaded data.

### Pipeline Stage Pattern

```cuda
// Multi-stage pipeline with TMA + mbarrier
__device__ void pipelined_mainloop(
    const CUtensorMap* desc_a,
    const CUtensorMap* desc_b,
    void* smem_a_stages[NUM_STAGES],
    void* smem_b_stages[NUM_STAGES],
    uint64_t* mbar[NUM_STAGES],
    int num_k_tiles
) {
    // Prologue: fill the first NUM_STAGES-1 stages
    for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; ++s) {
        tma_load_tile(desc_a, smem_a_stages[s], mbar[s], 0, s);
        tma_load_tile(desc_b, smem_b_stages[s], mbar[s], s, 0);
    }

    // Mainloop
    for (int k = 0; k < num_k_tiles; ++k) {
        int stage = k % NUM_STAGES;

        // Wait for data to arrive in this stage
        mbarrier_wait(mbar[stage]);

        // Issue MMA using this stage's SMEM buffers.
        // disable-output-lane is an optional 4-element vector for
        // cta_group::1; all-zero keeps every lane enabled, and the operand
        // may be dropped entirely. enable-input-d is a predicate.
        uint32_t mask[4] = {0, 0, 0, 0};
        if (threadIdx.x == 0) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                "}\n"
                :
                : "r"(tmem_acc),
                  "l"(make_desc(smem_a_stages[stage])),
                  "l"(make_desc(smem_b_stages[stage])),
                  "r"(0), "r"(1),
                  "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
            );
        }

        // Prefetch next stage
        int next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            tma_load_tile(desc_a, smem_a_stages[next_stage],
                         mbar[next_stage], 0, next_k);
            tma_load_tile(desc_b, smem_b_stages[next_stage],
                         mbar[next_stage], next_k, 0);
        }
    }
}
```

### mbarrier Operations

```cuda
// Initialize an mbarrier
__device__ void mbarrier_init(uint64_t* mbar, int arrive_count) {
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar)),
              "r"(arrive_count)
        );
    }
}

// Wait for an mbarrier to complete (phase-based)
__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP;\n"
        "}\n"
        :
        : "r"(mbar_addr), "r"(phase)
    );
}
```

## TMA Multicast

TMA multicast sends a single GMEM tile to **multiple CTAs within a cluster** simultaneously. This is critical for GEMM where the B operand is shared across M-axis tiles.

```cuda
// Multicast TMA: load B tile to all CTAs in the cluster
__device__ void tma_multicast_load(
    const CUtensorMap* desc,
    void* smem_ptr,
    uint64_t* mbar_ptr,
    int coord_x, int coord_y,
    uint16_t multicast_mask  // bitmask: which CTAs in cluster receive the data
) {
    if (threadIdx.x == 0) {
        uint32_t expected_bytes = TILE_K * TILE_N * sizeof(half);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "r"(expected_bytes)
        );

        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
            "[%0], [%1, {%2, %3}], [%4], %5;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
              "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "h"(multicast_mask)
        );
    }
}
```

### Multicast in GEMM

```
Cluster: 2 CTAs (CTA0 and CTA1) each computing different M-tiles of the same N column

CTA0: computes C[0:128, 0:256]   -- needs A[0:128, :] and B[:, 0:256]
CTA1: computes C[128:256, 0:256] -- needs A[128:256, :] and B[:, 0:256]

B[:, 0:256] is SHARED -- multicast it once from GMEM to both CTAs
A tiles are UNIQUE -- each CTA loads its own A tile

Result: B bandwidth is halved (1 GMEM read serves 2 CTAs)
```

## Blackwell-Specific Enhancements

### Swizzling for tcgen05

TMA loads feeding `tcgen05.mma` may use any swizzling mode the operand's type-size and major-ness admit; 128-byte swizzling is the common choice for K-major 16-bit operands. The swizzle pattern rearranges bytes within each 128-byte line to match the tensor core's internal data layout:

```
Without swizzle (linear):
  Row 0: bytes [0, 1, 2, ..., 127]
  Row 1: bytes [128, 129, ..., 255]

With 128B swizzle:
  Row 0: bytes [0, 1, ..., 127]     (unchanged)
  Row 1: bytes [128, 129, ..., 255] XOR pattern applied
  Row 2: bytes [256, ...] XOR pattern applied differently
  ...
```

The swizzle eliminates bank conflicts when the tensor core reads operand tiles from SMEM.

### TMA + TMEM Integration

On Blackwell, data flows through a characteristic pipeline:

```
GMEM --[TMA]--> SMEM --[tcgen05.mma]--> TMEM --[tcgen05.ld]--> Registers --[st.global]--> GMEM
                  ^                        |
                  |                        v
                  +---- (epilogue) --------+
                        (bias, activation, etc.)
```

## Performance Considerations

| Tip | Detail |
|---|---|
| Maximize TMA utilization | Keep the TMA unit busy with back-to-back loads across pipeline stages |
| Use multicast for shared operands | Reduces GMEM bandwidth by cluster_size x for shared tiles |
| Match TMA swizzle to the SMEM descriptor | tcgen05 reads the swizzle mode from the descriptor; a mismatch between it and the layout TMA wrote produces incorrect results |
| Prefer 2D TMA over manual addressing | TMA handles out-of-bounds clamping, padding, and strided access |
| Pipeline depth | 3-5 stages typically optimal; more stages increase SMEM usage |

## CuTe-DSL Example

```python
# CuTe-DSL TMA atom setup for a Blackwell GEMM
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

# Operand A: plain bulk-tensor copy, global -> shared
op_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
tma_atom_a, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op_a,
    tensor_a,                # global tensor
    a_smem_layout_slice,     # shared-memory layout of one stage
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
)

# Operand B: multicast the same tile to every CTA of the cluster
op_b = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
tma_atom_b, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
    op_b,
    tensor_b,
    b_smem_layout_slice,
    mma_tiler_mnk,
    tiled_mma,
    cluster_layout_vmnk.shape,
)
```
