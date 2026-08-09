---
id: technique-persistent-kernels
title: Persistent Kernels with CLC
type: technique
architectures:
- sm100
tags:
- persistent-kernel
- clc
- tile-scheduling
confidence: source-reported
reproducibility: snippet
prerequisites:
- hw-clc
related:
- hw-clc
- technique-tile-scheduling
- pattern-tail-effect
sources:
- doc-nvidia-tuning-guide
- blog-tcgen05-tutorial
- doc-cutlass-blackwell
artifact_dir: artifacts/kernels/persistent-kernels
---

## Overview

A *software* persistent kernel launches exactly as many CTAs as SMs, and each CTA processes multiple output tiles in a loop rather than exiting after one tile. On Blackwell, the CLC (Cluster Launch Control) hardware unit provides the same effect without an SM-sized grid: per the CUTLASS CLC documentation, a CLC kernel "launches a grid containing as many threadblocks as there are output tiles to compute in the kernel -- just like one would in a non-persistent kernel", and "each worker uses the `{blockIdx.x, blockIdx.y, blockIdx.z}` coordinate as the first output tile to process and uses the CLC query for subsequent processing of output tiles". Every grid coordinate is processed exactly once, either by being launched normally when resources free up or by being cancelled by a running worker that takes over its tile.

## CLC Loop Pattern

The core persistent kernel loop on Blackwell uses CLC to dynamically assign tiles:

```cuda
// Persistent kernel with CLC tile scheduling (Blackwell SM100)
__global__ void __launch_bounds__(512)
persistent_gemm_clc(const __grid_constant__ GemmParams params)
{
    // The grid has one CTA per output tile, so this CTA's own coordinate is
    // its FIRST tile. Starting the loop with a CLC query instead would drop
    // every tile that was directly launched.
    TileCoord tile = tile_from_blockidx();

    // CLC-managed persistent loop: each CTA processes multiple tiles
    while (true) {
        // Standard GEMM tile computation
        int tile_m = tile.m;
        int tile_n = tile.n;

        // TMA producer loads A[tile_m, :] and B[:, tile_n] tiles
        // MMA consumer accumulates K-dimension
        // Epilogue writes C[tile_m, tile_n]
        compute_gemm_tile(params, tile_m, tile_n);

        // Only then ask the CLC unit to cancel the launch of a cluster that
        // has not started yet. On success the response carries the ctaid of
        // that cluster's first CTA, which becomes this CTA's next tile.
        // (CUTLASS issues this query early and pipelines it so the latency
        // overlaps the mainloop; it is shown serialized here for clarity.)
        if (!clc_try_cancel_and_get_tile(&tile)) {
            // The request failed: no unlaunched cluster remains, so there is
            // no work left. Do not retry -- once a CTA has observed a failed
            // try_cancel, issuing another one is undefined behaviour.
            return;
        }
    }
}
```

At the PTX level, the CLC interaction is a cancel/query sequence. The exact
inline PTX is usually hidden behind CUTLASS/CuTe wrappers, but the control flow
looks like this:

This mirrors the structure of the worked example in PTX ISA §9.7.14.18, which
issues the cancellation request for the *next* cluster and then falls through to
`processCurrentCluster` while that request completes asynchronously:

```text
    tile = blockIdx            // the directly launched coordinate

TILE_LOOP:
    // Request cancellation of a not-yet-launched cluster for the NEXT
    // iteration. Issued first so its latency overlaps the compute below.
    clusterlaunchcontrol.try_cancel(response_smem, mbarrier)

    // ... compute the CURRENT tile while the request is in flight ...

    // Only now wait for and query the 16-byte response.
    wait(mbarrier)
    has_work, tile_m, tile_n = clusterlaunchcontrol.query_cancel(response_smem)
    if (!has_work) return       // terminal: never re-issue after a failure

    tile = (tile_m, tile_n)
    goto TILE_LOOP
```

## Comparison: CLC vs Static Stride (Hopper)

On Hopper (SM90), persistent kernels use a static stride pattern where each CTA computes tiles at fixed intervals:

```cuda
// Hopper-style static stride persistent kernel
__global__ void hopper_persistent_gemm(GemmParams params)
{
    int cta_id = blockIdx.x;
    int total_ctas = gridDim.x;
    int total_tiles = params.num_tiles_m * params.num_tiles_n;

    // Static stride: CTA i handles tiles i, i+total_ctas, i+2*total_ctas, ...
    for (int tile_idx = cta_id; tile_idx < total_tiles; tile_idx += total_ctas) {
        int tile_m = tile_idx / params.num_tiles_n;
        int tile_n = tile_idx % params.num_tiles_n;
        compute_gemm_tile(params, tile_m, tile_n);
    }
}
```

| Aspect | Hopper Static Stride | Blackwell CLC |
|--------|---------------------|---------------|
| Scheduling | Software loop with fixed stride | Hardware CLC unit assigns tiles |
| Load balancing | Fixed; uneven if tile costs vary | Dynamic; CLC rebalances automatically |
| Tail effect | Last wave may have partial occupancy | CLC minimizes by giving fast CTAs more tiles |
| Launch overhead | Grid launch for each new problem | CLC can chain multiple problems |
| Termination | Implicit when loop ends | Loop exits when a `try_cancel` request fails |
| L2 locality | Depends on stride pattern | CLC can apply swizzled raster |

## CUTLASS PersistentTileSchedulerSm100

CUTLASS 4.5.0 provides `PersistentTileSchedulerSm100` that wraps the CLC hardware:

```cuda
// CUTLASS SM100 persistent tile scheduler (simplified)
template <class TileShape>
struct PersistentTileSchedulerSm100 {

    // Initialize the CLC with the problem geometry
    CUTLASS_DEVICE static void init(
        dim3 problem_tiles,
        void* clc_smem_buffer)
    {
        if (threadIdx.x == 0) {
            // Program CLC with total tile count and scheduling policy
            clc_init(clc_smem_buffer,
                     problem_tiles.x,  // tiles along M
                     problem_tiles.y,  // tiles along N
                     ClcPolicy::SwizzledRaster);
        }
        __syncthreads();
    }

    // Shared storage for CTA-wide CLC result broadcast
    // __shfl_sync is warp-local and cannot reach warps 1-15.
    struct SharedClcState {
        int tile_m, tile_n;
        int valid;       // 1 = got tile, 0 = no more work
        int cancelled;
    };

    // Get next tile assignment from CLC
    CUTLASS_DEVICE static bool get_next_tile(
        void* clc_smem_buffer,
        SharedClcState& shared_clc,
        int& tile_m,
        int& tile_n)
    {
        if (threadIdx.x == 0) {
            int m, n;
            bool v = clc_query_tile(clc_smem_buffer, m, n);
            shared_clc.tile_m = m;
            shared_clc.tile_n = n;
            shared_clc.valid  = v ? 1 : 0;
        }
        __syncthreads();  // All warps see the result
        tile_m = shared_clc.tile_m;
        tile_n = shared_clc.tile_n;
        return shared_clc.valid != 0;
    }

    // Try to cancel the CTA when no more work
    CUTLASS_DEVICE static bool try_cancel(
        void* clc_smem_buffer,
        SharedClcState& shared_clc)
    {
        if (threadIdx.x == 0) {
            shared_clc.cancelled = clc_try_cancel(clc_smem_buffer) ? 1 : 0;
        }
        __syncthreads();
        return shared_clc.cancelled != 0;
    }
};
```

## Performance Impact

The tcgen05-tutorial progression demonstrates the impact of persistent kernels:

```
Pipelined, non-persistent (v3):           940 TFLOPS  (62% of cuBLAS)
+ warp specialization (v4):              1209 TFLOPS  (80% of cuBLAS)
+ 2-SM MMA (v5):                         1302 TFLOPS  (86% of cuBLAS)
+ persistent, static scheduling (v6):    1476 TFLOPS  (98% of cuBLAS)
```

The last step alone is worth about 13% (1302 -> 1476), and the tutorial's author notes that Cluster Launch Control was *not* used -- so this progression measures static persistent scheduling, not CLC. The mechanisms a persistent kernel exploits are:
1. **Reduced launch and setup overhead**: A single kernel launch covers all tiles; per-CTA prologue work happens once.
2. **Overlapped epilogue**: Profiling of the non-persistent kernel showed epilogue and new-threadblock setup leaving the tensor cores idle.
3. **Better L2 cache utilization**: A swizzled raster over tiles improves spatial locality across neighbouring tiles.
With CLC on top, the scheduler also **shortens the tail** by handing the next unlaunched cluster to whichever CTA asks first. It does not remove the tail: `try_cancel` relocates a whole cluster and cannot subdivide a tile, so a last wave of N tiles still occupies N CTAs.

## When to Use

- **Large GEMM problems**: Persistent kernels are most beneficial when the number of output tiles exceeds the SM count by at least 2-3x.
- **Grouped GEMMs / MoE**: CLC can chain multiple problem instances, eliminating inter-kernel launch gaps.
- **Workloads with uneven tile cost**: CLC's dynamic scheduling naturally handles variable-cost tiles (e.g., triangular attention masks).

## Caveats

- CLC requires `sm_100` or higher (SM120 parts included); Hopper kernels must use software-based scheduling.
- A failed `try_cancel` is terminal for that CTA: the PTX ISA makes it undefined behaviour to issue another `try_cancel` after one has been observed to fail, so the loop must exit rather than retry.
- For very small problems (fewer tiles than SMs), CLC overhead may not justify the complexity. A simple single-wave grid launch suffices.

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/persistent-kernels/full/`](../../artifacts/kernels/persistent-kernels/full/); labeled derived variants (each with the required `// provenance: derived from ...; not upstream code` header) live in [`artifacts/kernels/persistent-kernels/variants/`](../../artifacts/kernels/persistent-kernels/variants/). Every file's SHA-256 and upstream-pinning metadata is in `PROVENANCE.yaml` inside each bundle.

Query via:

```bash
python3 scripts/get_page.py technique-persistent-kernels --include-code
```
