---
id: experiment-family-dsa-attention
title: 'Optimization trace family: DeepSeek sparse attention'
source_category: optimization-trace
task_family: dsa_attention
architectures:
- sm103
tags:
- attention
- sparse-attention
- mla
- decode
- cute-dsl
- shared-memory-optimization
- precision-specialization
- parallel-reduction
- tma-transfers
- tensor-core-optimization
- pipeline-stages
- kernel-fusion
- tile-scheduling
- cache-policy
- tcgen05
- tmem
- tma
- mbarrier
techniques:
- shared-memory-optimization
- precision-specialization
- parallel-reduction
- tma-transfers
- tensor-core-optimization
- pipeline-stages
- kernel-fusion
- tile-scheduling
- cache-policy
hardware_features:
- tcgen05
- tmem
- tma
- mbarrier
kernel_types:
- attention
- sparse-attention
- mla
- decode
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/dsa_attention
experiment_count: 4
experiments:
- experiment_id: experiment-dsa-attention-044fa7d7d954d384
  logical_run_id: run-01f38b4bd25c90a5
  selected_code_path: kernel.py
  evidence_sha256: 076e2541d734af738c76dc5974bc22e27f0abeb6098a44ff5c73d6bad8013e2c
  manifest_row_sha256: 2628b329e76d3efed19161274ef1c285e54348582705ffe31d13a55978f384e3
  artifact_dir: artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 646 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
- experiment_id: experiment-dsa-attention-09a9ec45abf67f99
  logical_run_id: run-691555ef1a53ab8b
  selected_code_path: kernel.py
  evidence_sha256: fefb4ab77ad7f2d65baca33b9fa41c7d46a563afe5a2ad0ef87b5cb09aa026b4
  manifest_row_sha256: 6b7e1f10f9825a1660d706bb066f1bf0a238c95a25c7d9a6337e34682dc5726a
  artifact_dir: artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 13 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-dsa-attention-1a2df21eeb64e7ff
  logical_run_id: run-a7d6e1cbf63bb545
  selected_code_path: kernel.py
  evidence_sha256: 57b349e1ac77fc83c60e9c1e970ae31e9a8167aa4c76dfd51dbb5362ff02aaff
  manifest_row_sha256: ba31da5917bdf05b23fe8a28eadc168d2a7a97d27d9455d6fa91afe0669c6671
  artifact_dir: artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 1270 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
- experiment_id: experiment-dsa-attention-6351d0b3ec4f7f9c
  logical_run_id: run-e9d463fc70bf4c71
  selected_code_path: kernel.py
  evidence_sha256: 7ec645c6563378efb502e05096b216494b1dd0b3aebfe5c2de8bfa70c2204d32
  manifest_row_sha256: 9582d1f9fc61dece346c12b3059989460596bb9e918b5121523cce59a2138ddf
  artifact_dir: artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 102 unreplayed mutation(s) outside the selected complete source pair.
performance_claims:
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 43.231
  before_value: 54.88
  after_value: 43.231
  unit: us
  source_id: experiment-family-dsa-attention
  source_locator: artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/performance.md#comparison-1
  experiment_id: experiment-dsa-attention-044fa7d7d954d384
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 32.97625
  before_value: 33.40025
  after_value: 32.97625
  unit: us
  source_id: experiment-family-dsa-attention
  source_locator: artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/performance.md#comparison-1
  experiment_id: experiment-dsa-attention-09a9ec45abf67f99
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 10.91
  before_value: 11.33
  after_value: 10.91
  unit: us
  source_id: experiment-family-dsa-attention
  source_locator: artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/performance.md#comparison-1
  experiment_id: experiment-dsa-attention-1a2df21eeb64e7ff
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 55.52
  before_value: 56.064
  after_value: 55.52
  unit: us
  source_id: experiment-family-dsa-attention
  source_locator: artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-1
  experiment_id: experiment-dsa-attention-6351d0b3ec4f7f9c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 4.96
  before_value: 5.664
  after_value: 4.96
  unit: us
  source_id: experiment-family-dsa-attention
  source_locator: artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-2
  experiment_id: experiment-dsa-attention-6351d0b3ec4f7f9c
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: DeepSeek sparse attention

This page groups 4 independently receipted `dsa_attention` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — passing correctness.
- [Task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/TASK.md); [before](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/before.py); [after](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/after.py); [diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/changes.diff); [measurements](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/performance.md); [receipt](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/performance.md#comparison-1): latency improved from 54.88 us to 43.231 us.

### Variant 2

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — passing correctness.
- [Task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/TASK.md); [before](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/before.py); [after](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/after.py); [diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/changes.diff); [measurements](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/performance.md); [receipt](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/performance.md#comparison-1): latency improved from 33.40025 us to 32.97625 us.

### Variant 3

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — Any failing workload zeroes the official score, so correctness on
- [Task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/TASK.md); [before](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/before.py); [after](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/after.py); [diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/changes.diff); [measurements](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/performance.md); [receipt](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/performance.md#comparison-1): latency improved from 11.33 us to 10.91 us.

### Variant 4

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — passing correctness.
- [Task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/TASK.md); [before](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/before.py); [after](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/after.py); [diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/changes.diff); [measurements](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md); [receipt](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-1): latency improved from 56.064 us to 55.52 us.
- [Comparison 2](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-2): latency improved from 5.664 us to 4.96 us.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
