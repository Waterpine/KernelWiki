---
id: pattern-pipeline-stalls
title: "Pipeline Stalls"
type: pattern
tags: [pipeline-stages, warp-specialization, tma, tcgen05, mbarrier]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-double-buffering, technique-ping-pong-scheduling]
related: [pattern-compute-bound, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, blog-flash-attention-4, doc-nvidia-tuning-guide]
---

# Pipeline Stalls

## Symptom

Nsight Compute shows TMA or tcgen05 units idle despite nominally compute-bound workload. Tensor core utilization drops during specific phases of the kernel. Warp-level profiling reveals threads blocked on `mbarrier.try_wait` more than expected.

## Likely Causes

1. **Insufficient pipeline depth**: the selected stages do not cover the measured producer-to-consumer latency
2. **Incorrect mbarrier phase tracking**: Consumer observes stale arrivals, waits for next
3. **Missing completion wait**: consumer uses SMEM before the TMA mbarrier phase completes, or uses TMEM before `tcgen05.commit` completion
4. **Single-tile scheduling**: All warps serialized on one tile's softmax/epilogue
5. **Producer over-arrives**: An extra arrival or incorrect expected-byte count advances the phase early

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Pipeline stages](../techniques/pipeline-stages.md) | Tune `NUM_STAGES` against measured latency and SMEM occupancy |
| [Warp specialization](../techniques/warp-specialization.md) | Dedicated warps for TMA/MMA/epilogue eliminate role-switching stalls |
| [Double-buffering](../techniques/double-buffering.md) | TMEM buffer A while MMA runs on buffer B |
| [Ping-pong scheduling](../techniques/ping-pong-scheduling.md) | Two query tiles alternate softmax/MMA (FA4 pattern) |

## Diagnosis Checklist

```
1. Profile with Nsight Compute, check tensor core active cycles
2. Inspect mbarrier wait stalls in warp state breakdown
3. Track parity independently for each reused stage barrier
4. Check TMA expected bytes and the copy's complete-tx target
5. Check that MMA completion uses tcgen05.commit + an mbarrier wait
6. Use tcgen05 before/after fences only for their documented cross-thread ordering role
7. Measure whether another stage hides latency without reducing useful occupancy
```

## Interpreting the tutorial progression

The cited tutorial reports workload-specific improvements from swizzling,
pipelining, and persistent scheduling. It does not report the former synthetic
1-stage/3-stage/warp-specialized/2-SM percentages shown here. Treat its numbers
as measurements of that kernel and environment, not a general Blackwell ladder.

## Caveats

- Too many stages consume SMEM and can reduce occupancy; query the device and kernel limits instead of assuming one product-wide budget
- Phase tracking bugs are notoriously hard to debug — add assertions in development
- Profile first — pipeline is a waste of effort on memory-bound kernels
