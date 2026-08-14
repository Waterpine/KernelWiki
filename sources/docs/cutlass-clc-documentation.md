---
id: doc-cutlass-clc
title: "CUTLASS Cluster Launch Control (CLC) Documentation"
url: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [clc, cluster, tile-scheduling, persistent-kernel, mbarrier, 2sm-cooperative, pipeline-stages, gemm]
retrieved_at: 2026-08-13
---

# CUTLASS Cluster Launch Control (CLC) Documentation

## Mechanism

CUTLASS uses Blackwell CLC to implement cancellation-backed work stealing in
persistent kernels. The logical grid still contains work-item CTA/cluster IDs.
After a worker processes its directly launched ID, it may issue
`clusterlaunchcontrol.try_cancel`; success transfers responsibility for one
not-yet-launched ID to that worker.

CLC is not a free-form hardware queue that assigns arbitrary tile coordinates.
It cannot cancel a caller-selected tile, and the code must stop issuing requests
after it observes a failed cancellation, as required by PTX.

## Asynchronous response pipeline

The request writes a 16-byte opaque response into shared memory and completes a
transaction on an mbarrier. CUTLASS's scheduler pipeline overlaps future
cancellation requests with compute on the current tile and broadcasts/decodes
the resulting ClcID for the participating warps.

Pipeline depth, scheduler-warp identity, arrival counts, and class names are
CUTLASS-version and kernel-schedule details. They must be read from the pinned
CUTLASS implementation rather than treated as universal constants (the former
page asserted depth 3 and a fixed warp number without a versioned code anchor).

## Cluster granularity

For a multi-CTA cluster, cancellation and the returned ID are cluster-granular.
The multicast request form and each CTA's completion barrier participate in the
protocol. Preferred and fallback cluster shapes are launch/scheduler concerns;
CLC itself does not make any particular 2-SM shape optimal.

## What CLC can improve

- last-wave utilization when not-yet-launched grid work remains;
- imbalance caused by uneven worker availability, including partial-SM contexts;
- persistent scheduling without a contended global atomic counter.

It does not guarantee ideal load balance, remove request latency, pack small MoE
experts, or eliminate all tail effects. Those outcomes depend on the surrounding
tile scheduler and workload.

## Sources

- [CUTLASS CLC documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [PTX ISA: cluster launch control](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
