---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm]
---

# MoE Expert Load Imbalance

## Symptom

MoE grouped GEMM shows uneven per-expert compute time. Some SMs finish their expert quickly and sit idle while others are still processing. Overall latency is dominated by the slowest expert.

## Likely Causes

1. **Skewed token distribution**: Router assignments create unequal expert row counts
2. **Static tile assignment**: Precomputed tile→SM mapping cannot rebalance at runtime
3. **Masked layout waste**: Fixed M_max per expert wastes compute on padding rows
4. **Small-M per expert**: When M < BLOCK_M, thin-GEMM underutilizes tensor cores

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC (Cluster Launch Control)](../hardware/clc.md) | Workers may claim not-yet-launched grid IDs and map them to expert tiles |
| [Persistent kernels](../techniques/persistent-kernels.md) | Let resident workers process multiple logical tiles |
| [Contiguous layout](../kernels/grouped-gemm.md) | Pack variable-M experts sequentially; offsets array indexes expert boundaries |
| [Masked layout](../kernels/grouped-gemm.md) | Good for CUDA graph capture; wastes compute on padding |
| [K-grouped layout](../kernels/grouped-gemm.md) | For weight gradient computation with variable K per expert |
| Expert replication/load balancing | System-level option; measure communication, memory, and routing effects separately |

## Example: Reward Hack in GPU Mode Problem 4

The 1st-place submission exploited the evaluation harness rather than truly balancing load:
- Correctness phase: real kernel ran on cloned data
- Timing phase: detected reused objects, fired 120-group super-batch in call 1, returned cached results for calls 2-15

This is recorded only as an unverified report about the earlier harness. No
authoritative source in this page establishes that it prompted a particular
MLSys 2026 evaluator change.

## Caveats

- Use the PTX target table for CLC support; do not infer it from product branding alone
- Dynamic scheduling has request/coordination overhead that must be measured
- Small experts may not benefit — minimum viable tile size is a floor
- EPLB works at cluster scale, not single-device

## When NOT An Issue

- Sufficiently uniform routing for the measured batch
- Very large batch sizes (statistics average out)
- Training with auxiliary load balancing loss
