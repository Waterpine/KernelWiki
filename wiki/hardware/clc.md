---
id: hw-clc
title: "Cluster Launch Control (CLC)"
type: hardware
architectures: [sm100, sm100a]
tags: [clc, persistent-kernel, tile-scheduling]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
related: [technique-persistent-kernels, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc, doc-nvidia-tuning-guide, pr-cutlass-2161]
aliases: [CLC, "cluster launch control"]
---

# Cluster Launch Control (CLC)

## Correct model: cancellation-backed work stealing

CLC is a Blackwell work-stealing mechanism. A running CTA or CTA cluster asks
the launch controller to cancel a grid item that has not begun. If the request
succeeds, the response contains the first CTA ID of the canceled cluster and
the requester processes that logical work itself.

This differs from a generic hardware tile queue:

- the host still launches the logical grid;
- the request does not name an arbitrary tile to cancel;
- the returned ID comes from a not-yet-launched CTA/cluster;
- a failed request means no usable cancellation was returned (or another
  documented failure condition occurred);
- after code observes a failed request, submitting another request is undefined.

CLC can improve last-wave utilization and adapt to uneven SM availability, but
the benefit depends on grid size, cluster shape, per-tile duration, and request
overhead.

## PTX flow

The request writes a 16-byte opaque response in shared memory and completes an
mbarrier transaction. The response must be loaded and decoded with
`clusterlaunchcontrol.query_cancel`.

```ptx
// Abbreviated single-CTA/1x1-cluster pattern.
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], 16;
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.b128 [response], [bar];

wait:
mbarrier.try_wait.parity.b64 ready, [bar], phase;
@!ready bra wait;

.reg .b128 handle;
ld.shared.b128 handle, [response];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
    canceled, handle;
@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, _}, handle;
```

For multi-CTA clusters, use the multicast request form and the cluster-scope
barrier/proxy-fence protocol from the PTX ISA or the libcu++ wrapper. The full
protocol also protects the shared response between loop iterations.

## CUTLASS integration

CUTLASS pipelines CLC requests so the scheduler can request a future work item
while compute warps process the current one. CUTLASS's `PersistentTileScheduler`
and pipeline types are version-sensitive implementation details; use the names
present in the pinned CUTLASS release rather than the schematic class names
formerly shown here.

CLC is useful for persistent GEMM and grouped workloads, but it does not by
itself pack multiple small experts into one MMA tile or guarantee ideal load
balance. Those remain scheduler and kernel-design decisions.

## Sources

- [CUDA Programming Guide: Work Stealing with Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [PTX ISA: `clusterlaunchcontrol.try_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [CUTLASS CLC documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)
- [CUTLASS PR 2161](https://github.com/NVIDIA/cutlass/pull/2161)
