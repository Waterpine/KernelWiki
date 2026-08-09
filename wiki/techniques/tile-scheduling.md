---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
type: technique
architectures: [sm100, sm90]
tags: [tile-scheduling, clc, persistent-kernel]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-persistent-kernels, pattern-low-sm-utilization]
sources: [doc-nvidia-tuning-guide, doc-cutlass-blackwell, pr-cutlass-2161]
blackwell_relevance: "CLC (sm_100 and higher, including SM120) replaces static scheduling; Hopper patterns provide baseline comparison."
---

## Overview

Tile scheduling determines the order in which output tiles of a GEMM (or attention) kernel are assigned to CTAs. The scheduling order affects L2 cache hit rates, tail-effect severity, and overall GPU utilization. On Blackwell, the CLC hardware unit supports dynamic scheduling policies including swizzled raster, while Hopper relies on software-based static stride or swizzled patterns computed at launch time.

## Scheduling Strategies

### Linear Raster (Naive)

Tiles are assigned in row-major order. Simple but poor L2 locality: consecutive tiles share no B-matrix data until the entire M-dimension is traversed.

```cuda
// Linear raster: tile_idx maps directly to (tile_m, tile_n)
__device__ void linear_raster(int tile_idx, int tiles_n,
                               int& tile_m, int& tile_n) {
    tile_m = tile_idx / tiles_n;
    tile_n = tile_idx % tiles_n;
}

// Access pattern for a 4x4 tile grid:
//  0  1  2  3
//  4  5  6  7
//  8  9 10 11
// 12 13 14 15
//
// Problem: tiles 0,1,2,3 all load different B columns.
// By tile 4, B column 0 has been evicted from L2.
```

### Swizzled Raster

Tiles are assigned in a blocked pattern that groups nearby M and N tiles together, maximizing reuse of both A rows and B columns in L2 cache:

```cuda
// Swizzled raster: group tiles into blocks that share A and B data
// swizzle_size controls the block width (typically 4-8)
__device__ void swizzled_raster(int tile_idx, int tiles_m, int tiles_n,
                                 int swizzle_size, int& tile_m, int& tile_n)
{
    // Number of tile columns per swizzle group
    int group_cols = min(swizzle_size, tiles_n);
    int tiles_per_group = tiles_m * group_cols;

    // Which swizzle group
    int group_idx = tile_idx / tiles_per_group;
    int within_group = tile_idx % tiles_per_group;

    // Within the group, iterate in column-major order
    tile_m = within_group / group_cols;
    tile_n = group_idx * group_cols + within_group % group_cols;
}

// Access pattern with swizzle_size=2 on a 4x4 grid:
//  0  1 |  8  9
//  2  3 | 10 11
//  4  5 | 12 13
//  6  7 | 14 15
//
// Tiles 0,1,2,3 share the same B columns (0,1).
// Tiles 0,2,4,6 share the same A rows.
// Much better L2 reuse.
```

### Static Stride (Hopper Persistent)

Each CTA processes tiles at fixed intervals equal to the grid size:

```cuda
// Static stride: CTA i processes tiles i, i+gridDim.x, i+2*gridDim.x, ...
__device__ void static_stride(int cta_id, int total_ctas,
                               int iteration, int tiles_n,
                               int& tile_m, int& tile_n) {
    int tile_idx = cta_id + iteration * total_ctas;
    tile_m = tile_idx / tiles_n;
    tile_n = tile_idx % tiles_n;
}
```

### CLC Dynamic Scheduling (Blackwell)

The CLC hardware scheduler assigns tiles at runtime, combining the benefits of dynamic load balancing with configurable scheduling policies:

```cuda
// CLC-based scheduling on Blackwell
// The scheduling policy is set once during CLC initialization
enum class ClcSchedulePolicy {
    LinearRaster,       // Simple row-major order
    SwizzledRaster,     // Blocked pattern for L2 locality
    ColumnFirst,        // Column-major for specific workloads
    Hilbert             // Space-filling curve (experimental)
};

__device__ void clc_init_scheduler(
    void* clc_buffer,
    int tiles_m, int tiles_n,
    ClcSchedulePolicy policy)
{
    if (threadIdx.x == 0) {
        // Configure the software scheduler metadata used around CLC queries.
        // The CLC PTX surface is try_cancel/query_cancel, not clusterctl.init.
        uint32_t config = encode_clc_config(tiles_m, tiles_n, policy);
        init_clc_scheduler_metadata(clc_buffer, tiles_m, tiles_n, config);
    }
    __syncwarp();
}
```

## CUTLASS Tile Schedulers

CUTLASS provides several tile schedulers that abstract these strategies:

```cuda
// CUTLASS tile scheduler selection for SM100.
// Users name a tag type in `cutlass::gemm`; `TileSchedulerSelector` maps it
// to one of three sibling implementations in `cutlass::gemm::kernel::detail`.

// 1. Default CLC scheduler -> detail::PersistentTileSchedulerSm100
using Scheduler_Default = cutlass::gemm::PersistentScheduler;

// 2. Stream-K scheduler   -> detail::PersistentTileSchedulerSm100StreamK
//    Splits K-dimension across CTAs for the last wave
using Scheduler_StreamK = cutlass::gemm::StreamKScheduler;

// 3. Grouped GEMM scheduler -> detail::PersistentTileSchedulerSm100Group
//    Each group has different M, shared N and K
using Scheduler_Grouped = cutlass::gemm::GroupScheduler;

// Usage in CUTLASS kernel definition. GemmUniversal takes the problem
// shape, the two collectives and the scheduler tag; element types,
// layouts, tile shape and TiledMma are baked into the collectives by
// their builders, not passed here.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,             // ProblemShape <M,N,K,L>
    CollectiveMainloop,
    CollectiveEpilogue,
    Scheduler_Default                            // Tile scheduler tag
>;
```

## L2 Cache Locality Analysis

The choice of scheduling strategy directly impacts L2 cache hit rates. On B200 with 126 MB L2:

```python
# L2 cache reuse analysis for different schedulers
# Problem: M=8192, N=8192, K=4096, BF16
# Tile: 128x256, giving 64x32 = 2048 tiles
# B200: 148 SMs, 126 MB L2

tile_bytes_A = 128 * 4096 * 2  # 1 MB per tile row of A
tile_bytes_B = 4096 * 256 * 2  # 2 MB per tile column of B

# Linear raster (row-major, tile_m = idx / 32): the first wave of 148 tiles
# spans ceil(148/32) = 5 A row-blocks and ALL 32 B column-blocks.
#   A footprint: 5 * 1 MB  =  5 MB
#   B footprint: 32 * 2 MB = 64 MB
#   total       = 69 MB
# Each A row-block is reused by 32 consecutive tiles, but a B column-block is
# only reused once per full row-pass, so all 64 MB of B must stay resident to
# get any B reuse at all.

# Swizzled raster (swizzle=4): the first wave snakes down 4-wide column groups,
# covering 4 B columns x ~37 A rows.
#   A footprint: 37 * 1 MB =  37 MB
#   B footprint: 4 * 2 MB  =   8 MB
#   total       = 45 MB (fits in B200's 126 MB L2 with room to spare)
# Now the reused operand is the small one: 8 MB of B is hit by all 37 rows,
# and each A row-block is reused across 4 tiles.

# Conclusion: swizzled raster shrinks the first-wave footprint (69 MB -> 45 MB)
# and, more importantly, makes the heavily-reused operand small enough to stay
# resident -- the reason it beats linear raster on large problems.
```

## Tail Effect Mitigation

The "tail effect" occurs when the last wave of tiles does not fully occupy all SMs. Different schedulers handle this differently:

| Scheduler | Tail Handling | SM Utilization (Last Wave) |
|-----------|---------------|---------------------------|
| Linear raster | None | `(total_tiles % num_SMs) / num_SMs` |
| Static stride | None | Same as linear |
| CLC dynamic | Automatic | Workers that finish early take over not-yet-launched clusters |
| Stream-K | K-splitting | Near 100% (splits partial tiles across SMs) |

For a problem with 150 tiles on 148 SMs:
- Static: last wave has 2 tiles on 2 SMs, 146 SMs idle (1.4% utilization)
- CLC: the 2 extra coordinates are cancelled and absorbed by whichever workers finish first, without a second prologue. `try_cancel` can only cancel a cluster that has **not started running**; it cannot take work away from a slow CTA that is already executing.
- Stream-K: the 2 remaining tiles are split across all 148 SMs

## When to Use

- **Swizzled raster**: Default choice for large GEMMs. Always better than linear for L2 locality.
- **CLC dynamic**: Recommended on Blackwell for all persistent kernels. Combines dynamic load balancing with swizzled ordering.
- **Stream-K**: Best for small-to-medium problems where the tail effect dominates. Adds complexity for K-dimension synchronization.
- **Grouped scheduler**: Essential for MoE and batched GEMM where problem sizes vary across groups.

## Caveats

- Swizzle size must be tuned per problem shape. Too large a swizzle group exceeds L2 capacity; too small loses the locality benefit.
- CLC scheduling adds a small latency per tile acquisition (~10s of cycles). For extremely small tiles, this overhead is proportionally larger.
- Stream-K requires atomic accumulation where K-splits meet, adding synchronization overhead. Only worthwhile when tail utilization is a proven bottleneck.
