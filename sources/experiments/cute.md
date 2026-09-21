---
id: experiment-family-cute
title: 'Optimization trace family: CuTe MoE'
source_category: optimization-trace
task_family: cute
architectures:
- sm100
tags:
- moe
- grouped-gemm
- cute-dsl
- tma-transfers
- warp-specialization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- tcgen05
- tmem
- tma
- fp8
techniques:
- tma-transfers
- warp-specialization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
hardware_features:
- tcgen05
- tmem
- tma
- fp8
kernel_types:
- moe
- grouped-gemm
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/cute
experiment_count: 1
experiments:
- experiment_id: experiment-cute-d5bd6c063016d26e
  logical_run_id: run-a6fdfde21b497cd1
  selected_code_path: kernel.py
  evidence_sha256: e818cae5c0abaf4ac69b81ba9119db14e87574f3bf3717bc021be6f1c06be7d9
  manifest_row_sha256: b00074b4bde543c574fd142bfce0bc574f1618772151c6f0609d554fc4aaa98c
  artifact_dir: artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 56 unreplayed mutation(s) outside the selected complete source pair.
performance_claims:
- gpu: B200
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.133696
  before_value: 0.138464
  after_value: 0.133696
  unit: ms
  source_id: experiment-family-cute
  source_locator: artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/performance.md#comparison-1
  experiment_id: experiment-cute-d5bd6c063016d26e
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: CuTe MoE

This page groups 1 independently receipted `cute` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `sm100`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/TASK.md); [before](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/before.py); [after](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/after.py); [diff](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/changes.diff); [measurements](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/performance.md); [receipt](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/performance.md#comparison-1): latency improved from 0.138464 ms to 0.133696 ms.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
