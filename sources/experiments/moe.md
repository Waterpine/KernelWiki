---
id: experiment-family-moe
title: 'Optimization trace family: Mixture-of-Experts'
source_category: optimization-trace
task_family: moe
architectures:
- sm100
- sm103
tags:
- moe
- grouped-gemm
- cute-dsl
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- cache-policy
- tma-transfers
- kernel-fusion
- tcgen05
- tmem
- tma
- fp8
- pdl
- fp4
techniques:
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- cache-policy
- tma-transfers
- kernel-fusion
hardware_features:
- tcgen05
- tmem
- tma
- fp8
- pdl
- fp4
kernel_types:
- moe
- grouped-gemm
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/moe
experiment_count: 4
experiments:
- experiment_id: experiment-moe-13030440ee02446b
  logical_run_id: run-7021e96eda438890
  selected_code_path: kernel.py
  evidence_sha256: ba8bd25a9ed9829f4defd7138f5453dca6ba133f2a2fedfebe97017315b1e604
  manifest_row_sha256: 2f2be1342cbafe798b67bdd9f1ad20cae73b5cf7e0167aa3370b6bfa4b974846
  artifact_dir: artifacts/experiments/moe/experiment-moe-13030440ee02446b
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 376 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-moe-55b35550b3d4d426
  logical_run_id: run-235139a13d0371ad
  selected_code_path: kernel.py
  evidence_sha256: d2c655cc4b2055a0eb59048f5188e1533c9d53d83ce189450d2277f3d105a95a
  manifest_row_sha256: a4785fdc2a85032b46a5d90152630fd67672a9c1c7086e9931a136cf0aa21242
  artifact_dir: artifacts/experiments/moe/experiment-moe-55b35550b3d4d426
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 76 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-moe-704e7b99864597ef
  logical_run_id: run-c8648f5bf5e7c8b4
  selected_code_path: kernel.py
  evidence_sha256: 8bb41a951e3f6148fc29ff1808205b29a33ed4c76742e5ed7cc197c7a1bec896
  manifest_row_sha256: 0f68675c7576d901333da3c0ac1c015d2f029bd6fc33ace58c92f800f1294367
  artifact_dir: artifacts/experiments/moe/experiment-moe-704e7b99864597ef
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 123 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-moe-712f5040611492e9
  logical_run_id: run-ceccb41ea67c322f
  selected_code_path: kernel.py
  evidence_sha256: 1a927a41f4b20db66f16729c505c3dcb38e075d8c84c94ee4c59d523675a1824
  manifest_row_sha256: 46c14c3217a85d8b435a702d04f8e2b6e40561760a009549c0046700e94162f8
  artifact_dir: artifacts/experiments/moe/experiment-moe-712f5040611492e9
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 62 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
performance_claims:
- gpu: B200
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: score
  value: 1.5755
  before_value: 1.5116
  after_value: 1.5755
  unit: ratio
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-13030440ee02446b/performance.md#comparison-1
  experiment_id: experiment-moe-13030440ee02446b
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 125.872
  before_value: 126.144
  after_value: 125.872
  unit: us
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-1
  experiment_id: experiment-moe-55b35550b3d4d426
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 150.944
  before_value: 151.264
  after_value: 150.944
  unit: us
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-2
  experiment_id: experiment-moe-55b35550b3d4d426
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 230.592
  before_value: 230.816
  after_value: 230.592
  unit: us
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-3
  experiment_id: experiment-moe-55b35550b3d4d426
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 322.224
  before_value: 322.496
  after_value: 322.224
  unit: us
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-4
  experiment_id: experiment-moe-55b35550b3d4d426
- gpu: B300
  dtype: fp8/fp4
  shape: not stated in the retained comparison
  metric: latency
  value: 178.304
  before_value: 180.096
  after_value: 178.304
  unit: us
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-704e7b99864597ef/performance.md#comparison-1
  experiment_id: experiment-moe-704e7b99864597ef
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.1535
  before_value: 0.2351
  after_value: 0.1535
  unit: ms
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-1
  experiment_id: experiment-moe-712f5040611492e9
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: score
  value: 1.738
  before_value: 1.6938
  after_value: 1.738
  unit: ratio
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-2
  experiment_id: experiment-moe-712f5040611492e9
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: score
  value: 1.7668
  before_value: 1.738
  after_value: 1.7668
  unit: ratio
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-3
  experiment_id: experiment-moe-712f5040611492e9
- gpu: B300
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: score
  value: 1.7668
  before_value: 1.738
  after_value: 1.7668
  unit: ratio
  source_id: experiment-family-moe
  source_locator: artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-4
  experiment_id: experiment-moe-712f5040611492e9
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: Mixture-of-Experts

This page groups 4 independently receipted `moe` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `sm100`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/TASK.md); [before](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/before.py); [after](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/after.py); [diff](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/changes.diff); [measurements](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/performance.md); [receipt](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/performance.md#comparison-1): score improved from 1.5116 ratio to 1.5755 ratio.

### Variant 2

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/TASK.md); [before](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/before.py); [after](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/after.py); [diff](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/changes.diff); [measurements](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md); [receipt](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-1): latency improved from 126.144 us to 125.872 us.
- [Comparison 2](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-2): latency improved from 151.264 us to 150.944 us.
- [Comparison 3](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-3): latency improved from 230.816 us to 230.592 us.
- [Comparison 4](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-4): latency improved from 322.496 us to 322.224 us.

### Variant 3

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — failing workload zeroes the official score, so correctness on every workload
- [Task](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/TASK.md); [before](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/before.py); [after](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/after.py); [diff](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/changes.diff); [measurements](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/performance.md); [receipt](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/performance.md#comparison-1): latency improved from 180.096 us to 178.304 us.

### Variant 4

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/TASK.md); [before](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/before.py); [after](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/after.py); [diff](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/changes.diff); [measurements](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md); [receipt](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-1): latency improved from 0.2351 ms to 0.1535 ms.
- [Comparison 2](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-2): score improved from 1.6938 ratio to 1.738 ratio.
- [Comparison 3](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-3): score improved from 1.738 ratio to 1.7668 ratio.
- [Comparison 4](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-4): score improved from 1.738 ratio to 1.7668 ratio.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
