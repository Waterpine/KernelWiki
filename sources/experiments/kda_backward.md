---
id: experiment-family-kda-backward
title: 'Optimization trace family: KDA backward'
source_category: optimization-trace
task_family: kda_backward
architectures:
- blackwell
tags:
- linear-attention
- cute-dsl
- tma-transfers
- kernel-fusion
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- tensor-core-optimization
- warp-specialization
- tcgen05
- tmem
- tma
- pdl
techniques:
- tma-transfers
- kernel-fusion
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- tensor-core-optimization
- warp-specialization
hardware_features:
- tcgen05
- tmem
- tma
- pdl
kernel_types:
- linear-attention
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/kda_backward
experiment_count: 2
experiments:
- experiment_id: experiment-kda-backward-166c8416d6cd0edb
  logical_run_id: run-2bbed16f9effb922
  selected_code_path: kernel.py
  evidence_sha256: 1a0d4da500027fb327dd0b5738b48d85447c5519aa2d8533096e126bf0819440
  manifest_row_sha256: 8c6e001ffeb9bfe0d12a65536ddb5e8a01b012dd19d5cd1ad9a92d57d50435c9
  artifact_dir: artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 153 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-kda-backward-7f9fffe3fe7517f6
  logical_run_id: run-903ce4581b531c63
  selected_code_path: kernel.py
  evidence_sha256: 2ce4d66b006909771afe1d91178003540768cdc222094d373f2de9f3a601ea1a
  manifest_row_sha256: 9528acb0a92a099c31da682a2acffca4d9cfa672ae4b22c1b1abfa507f2e330d
  artifact_dir: artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 112 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
performance_claims:
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: score
  value: 2.3154
  before_value: 2.3146
  after_value: 2.3154
  unit: ratio
  source_id: experiment-family-kda-backward
  source_locator: artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/performance.md#comparison-1
  experiment_id: experiment-kda-backward-166c8416d6cd0edb
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.32
  before_value: 1.632
  after_value: 0.32
  unit: us
  source_id: experiment-family-kda-backward
  source_locator: artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-1
  experiment_id: experiment-kda-backward-7f9fffe3fe7517f6
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.288
  before_value: 1.664
  after_value: 0.288
  unit: us
  source_id: experiment-family-kda-backward
  source_locator: artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-2
  experiment_id: experiment-kda-backward-7f9fffe3fe7517f6
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: KDA backward

This page groups 2 independently receipted `kda_backward` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `blackwell`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — FULL benchmark workload set, with every workload passing correctness.
- [Task](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/TASK.md); [before](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/before.py); [after](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/after.py); [diff](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/changes.diff); [measurements](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/performance.md); [receipt](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/performance.md#comparison-1): score improved from 2.3146 ratio to 2.3154 ratio.

### Variant 2

- Architectures: `blackwell`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — FULL benchmark workload set, with every workload passing correctness.
- [Task](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/TASK.md); [before](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/before.py); [after](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/after.py); [diff](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/changes.diff); [measurements](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md); [receipt](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-1): latency improved from 1.632 us to 0.32 us.
- [Comparison 2](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-2): latency improved from 1.664 us to 0.288 us.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
