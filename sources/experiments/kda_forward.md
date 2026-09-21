---
id: experiment-family-kda-forward
title: 'Optimization trace family: KDA forward'
source_category: optimization-trace
task_family: kda_forward
architectures:
- blackwell
- sm103
tags:
- linear-attention
- prefill
- cute-dsl
- tma-transfers
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- precision-specialization
- tcgen05
- tmem
- tma
- ldmatrix
techniques:
- tma-transfers
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- precision-specialization
hardware_features:
- tcgen05
- tmem
- tma
- ldmatrix
kernel_types:
- linear-attention
- prefill
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/kda_forward
experiment_count: 2
experiments:
- experiment_id: experiment-kda-forward-6370e3dd96982da1
  logical_run_id: run-a60ae0f276b9cd89
  selected_code_path: kernel.py
  evidence_sha256: 244b7ce724df367039f91cac4524e7acf417e1221a2be200706e60ac2c1e8b66
  manifest_row_sha256: d75df835a21f52a97099a55c0b95694b398f6cfbe1eef546bfddb9de79e4b5db
  artifact_dir: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 124 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
- experiment_id: experiment-kda-forward-f601f83a1b54d82a
  logical_run_id: run-59897e975fd9ac5a
  selected_code_path: kernel.py
  evidence_sha256: c7177c1b72f02c625070f1f048ef5e64f9c431bf7b40d018dd7f90956ee43cb3
  manifest_row_sha256: 578a768e7d524957fbe34f552f3cfde9cda654e50472432fbf991efdeab6c552
  artifact_dir: artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 18 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
performance_claims:
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 4.224
  before_value: 4.436
  after_value: 4.224
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-1
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 5.343
  before_value: 5.589
  after_value: 5.343
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-2
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: score
  value: 0.2435
  before_value: 0.2381
  after_value: 0.2435
  unit: ratio
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-3
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.124
  before_value: 1.167
  after_value: 1.124
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-4
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.749
  before_value: 0.786
  after_value: 0.749
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-5
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.67
  before_value: 2.251
  after_value: 1.67
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-6
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 6.536
  before_value: 7.226
  after_value: 6.536
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-7
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 4.6
  before_value: 5.1
  after_value: 4.6
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-8
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 5.962
  before_value: 6.536
  after_value: 5.962
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-9
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 5.13
  before_value: 5.962
  after_value: 5.13
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-10
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.641
  before_value: 3.207
  after_value: 2.641
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-11
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.661
  before_value: 2.692
  after_value: 2.661
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-12
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 4.371
  before_value: 5.13
  after_value: 4.371
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-13
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.513
  before_value: 1.538
  after_value: 1.513
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-14
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.441
  before_value: 1.513
  after_value: 1.441
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-15
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 3.441
  before_value: 4.275
  after_value: 3.441
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-16
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.843
  before_value: 1.024
  after_value: 0.843
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-17
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.975398
  before_value: 1.176613
  after_value: 0.975398
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-18
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: Blackwell GPU (exact model not stated)
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.739955
  before_value: 0.924707
  after_value: 0.739955
  unit: ms
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-19
  experiment_id: experiment-kda-forward-6370e3dd96982da1
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 263.712
  before_value: 267.744
  after_value: 263.712
  unit: us
  source_id: experiment-family-kda-forward
  source_locator: artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/performance.md#comparison-1
  experiment_id: experiment-kda-forward-f601f83a1b54d82a
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: KDA forward

This page groups 2 independently receipted `kda_forward` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `blackwell`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — benchmark workload set, with every workload passing correctness.
- [Task](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/TASK.md); [before](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/before.py); [after](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/after.py); [diff](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/changes.diff); [measurements](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md); [receipt](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-1): latency improved from 4.436 ms to 4.224 ms.
- [Comparison 2](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-2): latency improved from 5.589 ms to 5.343 ms.
- [Comparison 3](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-3): score improved from 0.2381 ratio to 0.2435 ratio.
- [Comparison 4](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-4): latency improved from 1.167 ms to 1.124 ms.
- [Comparison 5](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-5): latency improved from 0.786 ms to 0.749 ms.
- [Comparison 6](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-6): latency improved from 2.251 us to 1.67 us.
- [Comparison 7](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-7): latency improved from 7.226 us to 6.536 us.
- [Comparison 8](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-8): latency improved from 5.1 ms to 4.6 ms.
- [Comparison 9](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-9): latency improved from 6.536 us to 5.962 us.
- [Comparison 10](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-10): latency improved from 5.962 us to 5.13 us.
- [Comparison 11](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-11): latency improved from 3.207 us to 2.641 us.
- [Comparison 12](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-12): latency improved from 2.692 ms to 2.661 ms.
- [Comparison 13](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-13): latency improved from 5.13 us to 4.371 us.
- [Comparison 14](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-14): latency improved from 1.538 us to 1.513 us.
- [Comparison 15](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-15): latency improved from 1.513 us to 1.441 us.
- [Comparison 16](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-16): latency improved from 4.275 us to 3.441 us.
- [Comparison 17](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-17): latency improved from 1.024 us to 0.843 us.
- [Comparison 18](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-18): latency improved from 1.176613 ms to 0.975398 ms.
- [Comparison 19](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-19): latency improved from 0.924707 ms to 0.739955 ms.

### Variant 2

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — benchmark workload set, with every workload passing correctness.
- [Task](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/TASK.md); [before](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/before.py); [after](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/after.py); [diff](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/changes.diff); [measurements](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/performance.md); [receipt](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/performance.md#comparison-1): latency improved from 267.744 us to 263.712 us.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
