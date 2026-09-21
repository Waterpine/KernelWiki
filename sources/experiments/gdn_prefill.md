---
id: experiment-family-gdn-prefill
title: 'Optimization trace family: Gated Delta Net prefill'
source_category: optimization-trace
task_family: gdn_prefill
architectures:
- sm100
- sm103
tags:
- gated-delta-net
- linear-attention
- prefill
- cute-dsl
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- tile-scheduling
- cache-policy
- precision-specialization
- parallel-reduction
- tma-transfers
- kernel-fusion
- tcgen05
- tmem
- tma
- mbarrier
- ldmatrix
techniques:
- tensor-core-optimization
- pipeline-stages
- shared-memory-optimization
- tile-scheduling
- cache-policy
- precision-specialization
- parallel-reduction
- tma-transfers
- kernel-fusion
hardware_features:
- tcgen05
- tmem
- tma
- mbarrier
- ldmatrix
kernel_types:
- gated-delta-net
- linear-attention
- prefill
languages:
- cute-dsl
captured_at: unknown
artifact_dir: artifacts/experiments/gdn_prefill
experiment_count: 5
experiments:
- experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
  logical_run_id: run-bba423a4485e1ff2
  selected_code_path: gdn_small.py
  evidence_sha256: 7ac413b490d16e0b2011f3cd21e1afc7b6cd2c9388eff913a7595d1f40312eb8
  manifest_row_sha256: 8e23ddc0b3c6bb5e7962d07f9540afb99d796ba55e4f70ec7f4562825073af20
  artifact_dir: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 135 unreplayed mutation(s) outside the selected complete source pair.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
- experiment_id: experiment-gdn-prefill-8d2c808b80cf75d0
  logical_run_id: run-4fc2d0ca2daa8e85
  selected_code_path: kernel.py
  evidence_sha256: 067c9260ddf94e2451279c98f328f0f1c6d218e13f7a84f0cced703e52df9507
  manifest_row_sha256: 2ce32de189724f88513364b31716b6d40403fce12522eb3dbd93d27d085ecae2
  artifact_dir: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 3 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-gdn-prefill-9e7d19b93417849f
  logical_run_id: run-682e856ee53cc01e
  selected_code_path: kernel.py
  evidence_sha256: 68f048f37c665acead6895a67e32e8072439baa1cf55eb4a0262e4c83c5bf497
  manifest_row_sha256: 6b1be42670bdadd2959caa7ac9361876338a13a9717c2c2afea508107c0ef152
  artifact_dir: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 153 unreplayed mutation(s) outside the selected complete source pair.
- experiment_id: experiment-gdn-prefill-c5d8efebe630ec24
  logical_run_id: run-d6fa1f200c9d96c6
  selected_code_path: gate_beta.py
  evidence_sha256: 8fbe6bcb08c83863727805940bd4fe00a5254bbcc5aae10c95c9c9ea34c3324f
  manifest_row_sha256: f72fe6d80ae1371bfea2a0c12740f24b55629040193721ae4e847dfd6b9b968e
  artifact_dir: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - Performance snippets that did not prove a direct positive old-to-new comparison were omitted.
- experiment_id: experiment-gdn-prefill-f68f418db298e09c
  logical_run_id: run-29c12181a623012f
  selected_code_path: kernel.py
  evidence_sha256: 3f9a74a11b7998dac6438d3e6bb77bc547adf069d086d46a32a3c14c8a5cfbc7
  manifest_row_sha256: b931fdbab3fd27e47ee7a0fd3bdae4cbe036fde5280c93a5ecb8360bf98e3e46
  artifact_dir: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c
  correctness_status: gate-described-no-explicit-result
  evidence_limitations:
  - The local corpus preserves the selected comparison, not the complete session trace.
  - Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
  - The retained comparison has no separate correctness result; the applicable task gate is documented.
  - The manifest reports 82 unreplayed mutation(s) outside the selected complete source pair.
performance_claims:
- gpu: B200
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 165.759
  before_value: 168.944
  after_value: 165.759
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-1
  experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
- gpu: B200
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 13.76
  before_value: 14.18
  after_value: 13.76
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-2
  experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
- gpu: B200
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 36.701
  before_value: 36.976
  after_value: 36.701
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-3
  experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
- gpu: B200
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 87.681
  before_value: 87.906
  after_value: 87.681
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-4
  experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
- gpu: B200
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 121.137
  before_value: 121.314
  after_value: 121.137
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-5
  experiment_id: experiment-gdn-prefill-1b8c8c37fa32577a
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 60.608
  before_value: 61.216
  after_value: 60.608
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-1
  experiment_id: experiment-gdn-prefill-8d2c808b80cf75d0
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 60.24
  before_value: 61.088
  after_value: 60.24
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-2
  experiment_id: experiment-gdn-prefill-8d2c808b80cf75d0
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 101.92
  before_value: 109.952
  after_value: 101.92
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-1
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 16.939
  before_value: 18.47
  after_value: 16.939
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-2
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 84.992
  before_value: 91.616
  after_value: 84.992
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-3
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 56.768
  before_value: 58.528
  after_value: 56.768
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-4
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 15.863
  before_value: 17.355
  after_value: 15.863
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-5
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 25.312
  before_value: 27.936
  after_value: 25.312
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-6
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.771
  before_value: 1.1365
  after_value: 0.771
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-7
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.119
  before_value: 2.1769
  after_value: 2.119
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-8
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 3.4707
  before_value: 3.5465
  after_value: 3.4707
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-9
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.944
  before_value: 3.008
  after_value: 2.944
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-10
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 109.184
  before_value: 111.616
  after_value: 109.184
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-11
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 3.4707
  before_value: 3.5465
  after_value: 3.4707
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-12
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.4718
  before_value: 2.5045
  after_value: 2.4718
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-13
  experiment_id: experiment-gdn-prefill-9e7d19b93417849f
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 270.464
  before_value: 296.352
  after_value: 270.464
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-1
  experiment_id: experiment-gdn-prefill-c5d8efebe630ec24
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 261.728
  before_value: 270.464
  after_value: 261.728
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-2
  experiment_id: experiment-gdn-prefill-c5d8efebe630ec24
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 287.572
  before_value: 306.437
  after_value: 287.572
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-3
  experiment_id: experiment-gdn-prefill-c5d8efebe630ec24
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.309591
  before_value: 1.335834
  after_value: 1.309591
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-1
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.54375
  before_value: 1.553052
  after_value: 1.54375
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-2
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 16.657
  before_value: 16.737
  after_value: 16.657
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-3
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.737542
  before_value: 0.739102
  after_value: 0.737542
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-4
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 2.168
  before_value: 2.219
  after_value: 2.168
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-5
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.76
  before_value: 1.77
  after_value: 1.76
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-6
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 16.83
  before_value: 17.05
  after_value: 16.83
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-7
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.230107
  before_value: 1.276406
  after_value: 1.230107
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-8
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.737408
  before_value: 0.739102
  after_value: 0.737408
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-9
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.322
  before_value: 1.336
  after_value: 1.322
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-10
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.889
  before_value: 1.979
  after_value: 1.889
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-11
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.598
  before_value: 1.615
  after_value: 1.598
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-12
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.321514
  before_value: 1.32159
  after_value: 1.321514
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-13
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.957253
  before_value: 1.963148
  after_value: 1.957253
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-14
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.958225
  before_value: 1.963148
  after_value: 1.958225
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-15
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 1.267183
  before_value: 1.276406
  after_value: 1.267183
  unit: ms
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-16
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
- gpu: B300
  dtype: bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 16.319
  before_value: 16.737
  after_value: 16.319
  unit: us
  source_id: experiment-family-gdn-prefill
  source_locator: artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-17
  experiment_id: experiment-gdn-prefill-f68f418db298e09c
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization trace family: Gated Delta Net prefill

This page groups 5 independently receipted `gdn_prefill` optimization experiment(s).
Each bundle remains self-contained and independently hash-verified.

## Experiment catalog

### Variant 1

- Architectures: `sm100`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — suite and the large geomean, with every workload passing correctness.
- [Task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/TASK.md); [before](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/before.py); [after](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/after.py); [diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/changes.diff); [measurements](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md); [receipt](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-1): latency improved from 168.944 us to 165.759 us.
- [Comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-2): latency improved from 14.18 us to 13.76 us.
- [Comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-3): latency improved from 36.976 us to 36.701 us.
- [Comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-4): latency improved from 87.906 us to 87.681 us.
- [Comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-5): latency improved from 121.314 us to 121.137 us.

### Variant 2

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/TASK.md); [before](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/before.py); [after](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/after.py); [diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/changes.diff); [measurements](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md); [receipt](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-1): latency improved from 61.216 us to 60.608 us.
- [Comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-2): latency improved from 61.088 us to 60.24 us.

### Variant 3

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/TASK.md); [before](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/before.py); [after](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/after.py); [diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/changes.diff); [measurements](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md); [receipt](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-1): latency improved from 109.952 us to 101.92 us.
- [Comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-2): latency improved from 18.47 us to 16.939 us.
- [Comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-3): latency improved from 91.616 us to 84.992 us.
- [Comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-4): latency improved from 58.528 us to 56.768 us.
- [Comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-5): latency improved from 17.355 us to 15.863 us.
- [Comparison 6](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-6): latency improved from 27.936 us to 25.312 us.
- [Comparison 7](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-7): latency improved from 1.1365 us to 0.771 us.
- [Comparison 8](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-8): latency improved from 2.1769 us to 2.119 us.
- [Comparison 9](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-9): latency improved from 3.5465 us to 3.4707 us.
- [Comparison 10](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-10): latency improved from 3.008 us to 2.944 us.
- [Comparison 11](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-11): latency improved from 111.616 us to 109.184 us.
- [Comparison 12](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-12): latency improved from 3.5465 us to 3.4707 us.
- [Comparison 13](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-13): latency improved from 2.5045 us to 2.4718 us.

### Variant 4

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — with every workload passing correctness.
- [Task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/TASK.md); [before](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/before.py); [after](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/after.py); [diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/changes.diff); [measurements](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md); [receipt](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-1): latency improved from 296.352 us to 270.464 us.
- [Comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-2): latency improved from 270.464 us to 261.728 us.
- [Comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-3): latency improved from 306.437 us to 287.572 us.

### Variant 5

- Architectures: `sm103`
- Languages: `cute-dsl`
- Correctness: `gate-described-no-explicit-result` — language or profiler clauses elsewhere; every correctness, benchmark,
- [Task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/TASK.md); [before](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/before.py); [after](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/after.py); [diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/changes.diff); [measurements](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md); [receipt](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/PROVENANCE.yaml)

- [Comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-1): latency improved from 1.335834 ms to 1.309591 ms.
- [Comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-2): latency improved from 1.553052 ms to 1.54375 ms.
- [Comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-3): latency improved from 16.737 us to 16.657 us.
- [Comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-4): latency improved from 0.739102 ms to 0.737542 ms.
- [Comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-5): latency improved from 2.219 ms to 2.168 ms.
- [Comparison 6](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-6): latency improved from 1.77 ms to 1.76 ms.
- [Comparison 7](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-7): latency improved from 17.05 us to 16.83 us.
- [Comparison 8](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-8): latency improved from 1.276406 ms to 1.230107 ms.
- [Comparison 9](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-9): latency improved from 0.739102 ms to 0.737408 ms.
- [Comparison 10](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-10): latency improved from 1.336 ms to 1.322 ms.
- [Comparison 11](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-11): latency improved from 1.979 ms to 1.889 ms.
- [Comparison 12](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-12): latency improved from 1.615 ms to 1.598 ms.
- [Comparison 13](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-13): latency improved from 1.32159 ms to 1.321514 ms.
- [Comparison 14](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-14): latency improved from 1.963148 ms to 1.957253 ms.
- [Comparison 15](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-15): latency improved from 1.963148 ms to 1.958225 ms.
- [Comparison 16](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-16): latency improved from 1.276406 ms to 1.267183 ms.
- [Comparison 17](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-17): latency improved from 16.737 us to 16.319 us.

## Family-level limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
