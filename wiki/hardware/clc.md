---
id: hw-clc
title: "Cluster Launch Control (CLC)"
type: hardware
architectures: [sm100, sm100a]
tags: [clc, persistent-kernel, tile-scheduling]
confidence: source-reported
related: [technique-persistent-kernels, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-nvidia-tuning-guide, doc-cutlass-blackwell, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2161]
aliases: [CLC, "cluster launch control"]
---

# Cluster Launch Control (CLC)

## Overview

Cluster Launch Control (CLC) is a Blackwell hardware mechanism for **dynamic tile scheduling** in persistent kernels. It does not replace the ordinary grid launch: the kernel still launches a grid with as many thread blocks as there are output tiles -- exactly as a non-persistent kernel would -- and each worker's own `{blockIdx.x, blockIdx.y, blockIdx.z}` is its **first** output tile. CLC adds a *second* path by which a running worker can take over the work of a cluster that has not started yet.

CUTLASS states the two rules directly: a `ClcID` (a coordinate of the launched grid) "will be launched as a Worker when there are available resources" **or** "can be queried by an existing Worker via `clusterlaunchcontrol.try_cancel`", and "every `ClcID` is guaranteed to be processed by either (1) or (2)".

After finishing a tile, a worker asks the hardware for another one instead of exiting, enabling:

- **Better load balancing**: the number of SMs a kernel can actually use is not known at launch (other kernels or a Green Context may hold some), so a fixed CTA-to-tile assignment can idle. CLC lets whichever worker is free absorb the remaining coordinates.
- **Tail-effect mitigation**: a worker that finishes early takes over a not-yet-launched cluster and processes its tile without paying a fresh prologue.
- **Hardware work stealing**: A running cluster issues `clusterlaunchcontrol.try_cancel` to atomically cancel the launch of a cluster that has not started yet; on success the response carries the `ctaid` of the first CTA of the canceled cluster, which the requesting cluster then processes.

## Static Scheduling vs CLC

### Static Scheduling (Hopper and earlier)

```
Launch grid: 256 CTAs for 256 tiles
CTA 0 -> tile (0,0)     [fixed at launch]
CTA 1 -> tile (0,1)     [fixed at launch]
CTA 2 -> tile (0,2)     [fixed at launch]
...
CTA 255 -> tile (15,15)  [fixed at launch]

Problem: If SM count = 132, first wave = 132 CTAs.
         Second wave = 124 CTAs -> 8 SMs idle = 6% waste.
         For small GEMMs, tail effect dominates.
```

### CLC Dynamic Scheduling (Blackwell)

```
Launch grid: 256 CTAs for 256 tiles -- the SAME grid a non-persistent kernel
would launch. The hardware runs as many of them as fit (here 132 at a time).

CTA 0   (launched): compute OWN blockIdx tile (0,0) -> try_cancel -> got (2,4)
                    -> compute -> try_cancel -> declined -> exit
CTA 1   (launched): compute OWN blockIdx tile (0,1) -> try_cancel -> ...
...
CTA 131 (launched): compute OWN blockIdx tile (0,131) -> try_cancel -> ...

CTA 132..255 (not started): each is either launched later, when an SM frees
                    up, or cancelled by a running worker that takes over its
                    tile. Exactly one of the two happens.

Every ClcID is processed exactly once. A worker never skips its own blockIdx
tile -- that tile is its first unit of work, not something CLC hands back.
```

## How CLC Works

### Hardware Queue

CLC operates by letting a running CTA or cluster issue `clusterlaunchcontrol.try_cancel` to cancel a not-yet-launched ClcID and take over that work. There is no `clusterlaunchcontrol.try_acquire` PTX instruction.

### CLC Programming Model

```cuda
__global__ void persistent_gemm_clc(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    int num_tiles_m, int num_tiles_n
) {
    // Allocate persistent resources (TMEM, pipeline state)
    uint32_t tmem_acc = tmem_alloc(256);

    // Shared storage for CLC results (visible to all threads in CTA).
    // clc_response is the naturally aligned 16-byte slot try_cancel writes to;
    // clc_mbar is the mbarrier it signals. clc_*_addr are their shared-memory
    // addresses (__cvta_generic_to_shared).
    __shared__ alignas(16) uint32_t clc_response[4];
    __shared__ uint64_t clc_mbar;
    __shared__ uint2 clc_tile_coord;
    __shared__ int clc_has_tile;
    uint32_t clc_phase = 0;

    const uint32_t clc_response_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(clc_response));
    const uint32_t clc_mbar_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&clc_mbar));

    // The mbarrier must be initialized before any arrive/wait on it.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(clc_mbar_addr) : "memory");
    }
    __syncthreads();

    // The FIRST tile is this CTA's own grid coordinate. The grid was launched
    // with one CTA per output tile, so skipping blockIdx would drop that tile:
    // CLC never hands a worker its own coordinate back.
    uint2 tile = make_uint2(blockIdx.x, blockIdx.y);

    // CLC tile loop: process the current tile, ask for another, repeat.
    while (true) {
        int tile_m = tile.x;
        int tile_n = tile.y;

        // Zero accumulator
        tmem_zero(tmem_acc, 256);

        // Mainloop: iterate over K dimension
        for (int k = 0; k < K / TILE_K; ++k) {
            // TMA load A and B tiles to SMEM
            tma_load_a(smem_a, A, tile_m, k);
            tma_load_b(smem_b, B, k, tile_n);
            wait_barrier();

            // Issue MMA. disable-output-lane is an optional 4-element
            // vector for cta_group::1; enable-input-d is a predicate.
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
                    : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
                      "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
                );
            }
        }

        // Epilogue
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        store_output(tmem_acc, C, tile_m, tile_n);

        // Only now ask for more work. (A real kernel issues the request
        // *before* the mainloop so the response latency overlaps compute --
        // that is what the CUTLASS CLC pipeline does; it is serialized here
        // to keep the control flow readable.)
        //
        // One thread issues the asynchronous request: try_cancel writes a
        // 16-byte opaque response into `clc_response` and signals `clc_mbar`
        // via complete-tx.
        if (threadIdx.x == 0) {
            asm volatile(
                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%0], 16;"
                :: "r"(clc_mbar_addr) : "memory");
            asm volatile(
                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                ".mbarrier::complete_tx::bytes.b128 [%0], [%1];"
                :: "r"(clc_response_addr), "r"(clc_mbar_addr) : "memory");
        }

        // Every thread waits for the response, then decodes the handle.
        mbarrier_wait(clc_mbar, clc_phase);
        clc_phase ^= 1;

        uint32_t ctaid_x, ctaid_y, ctaid_z, acquired;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .b128 clc_result;\n\t"
            "ld.shared.b128 clc_result, [%4];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, clc_result;\n\t"
            "selp.u32 %3, 1, 0, p;\n\t"
            "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
            "{%0, %1, %2, _}, clc_result;\n\t"
            "}\n"
            : "=r"(ctaid_x), "=r"(ctaid_y), "=r"(ctaid_z), "=r"(acquired)
            : "r"(clc_response_addr) : "memory");

        if (threadIdx.x == 0) {
            clc_tile_coord = make_uint2(ctaid_x, ctaid_y);
            clc_has_tile = (int)acquired;
        }
        __syncthreads();  // All threads see the shared result

        // A failed request means no cluster could be canceled: there is no
        // work left. Once a CTA has observed a failed try_cancel, issuing
        // another one is undefined behaviour, so exit rather than retry.
        if (!clc_has_tile) break;

        // Otherwise the response carries the first ctaid of the cluster we
        // cancelled; that becomes the next tile.
        tile = clc_tile_coord;
    }

    // Cleanup
    tmem_dealloc(tmem_acc, 256);
}
```

## try_cancel / query_cancel API

`clusterlaunchcontrol.try_cancel` takes no tile argument. It asks the hardware to
atomically cancel *some* cluster of the same grid that has not started running yet,
and writes a 16-byte opaque response into shared memory. The requesting cluster then
queries that response to learn whether it won a cluster and, if so, which one.

```cuda
// Request cancellation of a not-yet-launched cluster and take over its work.
// `clc_response` is a 16-byte-aligned __shared__ slot; `clc_mbar` is an mbarrier.
__device__ void clc_request(uint32_t clc_response_addr, uint32_t clc_mbar_addr,
                            uint64_t& clc_state) {
    asm volatile(
        "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 %0, [%1], 16;"
        : "=l"(clc_state) : "r"(clc_mbar_addr));
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta"
        ".mbarrier::complete_tx::bytes.b128 [%0], [%1];"
        :: "r"(clc_response_addr), "r"(clc_mbar_addr) : "memory");
}
```

After the mbarrier signals completion, load the response and query it:

```ptx
ld.shared.b128 handle, [clc_response];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x, y, z, _}, handle;
```

On success the response carries the `ctaid` of the first CTA of the canceled
cluster, and no other successful `try_cancel` in the same grid returns that id --
this is what makes CLC a race-free hardware work queue. If a CTA has already
observed a `try_cancel` complete as *failed*, issuing another one is undefined
behaviour, so the tile loop must exit on the first failure.

## CUTLASS Integration

CUTLASS 4.5.0 for SM100 provides CLC support through the `PersistentScheduler` class:

```cuda
// CUTLASS SM100 persistent GEMM with CLC scheduling
// SM100 uses the CUTLASS 3.x API: build the two collectives with the
// CollectiveBuilders, compose them into kernel::GemmUniversal, then wrap
// the result in device::GemmUniversalAdapter.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,    // ProblemShape <M,N,K,L>
    CollectiveMainloop,                 // built for MmaTileShape 128x256x64
    CollectiveEpilogue,                 //   and ClusterShape 2x1x1
    cutlass::gemm::PersistentScheduler  // CLC scheduler (void picks the same default)
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Launch: CUTLASS handles CLC internally
Gemm gemm_op;
gemm_op(args, workspace, stream);
```

### CUTLASS CLC Tile Scheduler

```cpp
// Simplified CUTLASS CLC scheduler logic
struct ClcTileScheduler {
    CUTLASS_DEVICE
    WorkTileInfo get_next_work() {
        WorkTileInfo work;
        // Fetch next tile by cancelling a not-yet-launched ClcID.
        bool valid = clc_try_cancel(&work.tile_coord);
        work.is_valid = valid;

        if (valid) {
            // Convert linear tile index to 2D coordinates
            work.tile_m = work.tile_coord.x;
            work.tile_n = work.tile_coord.y;

            // Apply swizzle for L2 locality
            apply_l2_swizzle(work.tile_m, work.tile_n);
        }
        return work;
    }
};
```

## Performance Impact

CLC delivers significant performance gains, especially for small-to-medium GEMMs where tail effects dominate:

### Tail Effect Mitigation

| GEMM Size | Static Scheduler | CLC Scheduler | Improvement |
|---|---|---|---|
| 2048x2048 (small) | 86% SM utilization | 98% SM utilization | +14% |
| 4096x4096 (medium) | 92% SM utilization | 98% SM utilization | +6.5% |
| 8192x8192 (large) | 97% SM utilization | 99% SM utilization | +2% |

The "tcgen05 for dummies" tutorial reaches 1476 TFLOPS -- 98% of its cuBLAS reference of 1507 TFLOPS -- with a persistent kernel using *static* scheduling, after warp specialization and 2-SM MMA; the author notes that Cluster Launch Control was not used and leaves it as an exercise. That result therefore bounds what static persistent scheduling alone achieves on this shape, not what CLC adds.

### Why CLC Matters for Inference

Production LLM inference typically hits shapes where tail effects are severe:

```python
# Typical LLM GEMM shapes during decode (batch_size=1-64)
# M is small (batch * seq_len for decode), N and K are large (model dim)
# Example: Llama-70B decode, batch=32
M = 32     # small!
N = 8192   # hidden dim
K = 8192   # hidden dim

# Tile = 128x256 -> tiles_m = 1, tiles_n = 32 -> only 32 tiles total
# On B200 (148 SMs): 32 tiles cannot fill 148 SMs, so 116 SMs are idle.
#
# CLC does NOT fix this. There are only 32 ClcIDs in the grid and all 32
# launch immediately, so no worker has an unlaunched cluster to cancel and
# every try_cancel is declined. Widening the parallelism needs a smaller
# tile, a split-K/Stream-K decomposition, or batching -- not CLC.
# CLC helps when there are MORE tiles than can run at once, or when SM
# availability is uneven and unknown at launch.
```

## CLC with 2-SM Cooperative Mode

When using 2-SM cooperative MMA (`cta_group::2`), CLC distributes work in **cluster-sized units**:

```cuda
// 2-SM cooperative CLC: each successful cancel gets a cluster-sized tile.
// As in the 1-SM loop, the cluster's OWN launched coordinate is its first
// tile; try_cancel only supplies the tiles after that.
__device__ void cooperative_clc_loop() {
    ClusterTile tile = cluster_tile_from_blockidx();
    while (true) {

        // Both CTAs in the cluster share the tile
        // CTA 0 handles rows 0-127, CTA 1 handles rows 128-255
        int my_row_start = (blockIdx.x % 2) * 128;

        // Issue cooperative MMA. cta_group::2 takes an 8-element
        // disable-output-lane vector.
        uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (threadIdx.x == 0) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::f16 "
                "[%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
                "}\n"
                :
                : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
                  "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
                  "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7])
            );
        }
        // ... epilogue for this tile ...

        // Only after the current tile is finished, try to take over a
        // not-yet-launched cluster. A declined response is terminal.
        if (!clc_try_cancel_cluster(&tile)) break;
    }
}
```

## L2 Cache Swizzling with CLC

CLC tile ordering can be customized with swizzle patterns to improve L2 cache hit rates:

```cuda
// Swizzle tile coordinates for better L2 locality
// Tiles are visited in a Z-order (Morton) curve pattern
__device__ void apply_l2_swizzle(int& tile_m, int& tile_n, int swizzle_bits) {
    // Convert linear tile index to swizzled 2D coordinates
    // This groups spatially adjacent tiles together, improving
    // L2 reuse for the B matrix (shared across M tiles)
    int linear = tile_m * num_tiles_n + tile_n;
    int swizzle_mask = (1 << swizzle_bits) - 1;
    int group = linear >> swizzle_bits;
    int within = linear & swizzle_mask;

    tile_m = group / num_tiles_n;
    tile_n = (group % num_tiles_n) ^ (tile_m & swizzle_mask);
}
```

## Comparison: CLC vs Software Persistent Scheduling

| Feature | CLC (Hardware) | Software Atomics |
|---|---|---|
| Scheduling overhead | Near zero (hardware) | atomicAdd contention |
| Tail-effect handling | Optimal | Good with careful design |
| Cancellation | try_cancel API | Complex (flags + barriers) |
| L2 swizzle | Configurable at launch | Manual implementation |
| Portability | SM100+ only | SM70+ |
| CUTLASS support | Built-in | Manual scheduler |
