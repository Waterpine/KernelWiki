---
id: pattern-compute-bound
title: "Not Reaching Peak FLOPS"
type: pattern
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion, technique-software-exp]
related: [pattern-low-sm-utilization, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, doc-flash-attention-4, blog-tcgen05-tutorial]
---

## Diagnose first

Low percentage of peak does not establish a compute-bound kernel. Use achieved
memory throughput, instruction mix, eligible warps, tensor-pipe utilization,
stall reasons, and a roofline based on the actual operation count.

Common limits include dependency bubbles, inadequate input staging, descriptor
or bank-conflict problems, non-MMA work, insufficient grid parallelism, and
resource limits that reduce occupancy.

| Candidate | Test |
|---|---|
| Pipeline stages | Compare stalls and occupancy as stage depth changes. |
| Warp specialization | Confirm producer/consumer work overlaps and roles are occupied. |
| Two-CTA MMA | Compare legal group-1/group-2 schedules for the same problem. |
| Epilogue fusion | Include register/resource cost and removed memory traffic. |
| Software exponential | Applicable to the measured exponential bottleneck and validated accuracy only. |

FlashAttention-4 reports up to 1613 TFLOP/s on its B200 sweep. That case shows
that non-MMA resources can be the limiting roof; it does not define 70% as a
universal diagnosis threshold.
