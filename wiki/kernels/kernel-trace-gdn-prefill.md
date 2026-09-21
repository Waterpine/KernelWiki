---
id: kernel-trace-gdn-prefill
title: 'Optimization case studies: Gated Delta Net prefill'
type: kernel
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
confidence: experimental
reproducibility: snippet
kernel_types:
- gated-delta-net
- linear-attention
- prefill
languages:
- cute-dsl
related:
- kernel-gated-delta-net
- technique-pipeline-stages
- pattern-memory-bound
- technique-tile-scheduling
- technique-cache-policy
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
- hw-mbarrier
- technique-kernel-fusion
sources:
- experiment-family-gdn-prefill
artifact_dir: artifacts/experiments/gdn_prefill
experiment_count: 5
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

# Optimization case studies: Gated Delta Net prefill

## Family overview

This page summarizes 5 distinct `gdn_prefill` optimization experiment(s). The [family source record](../../sources/experiments/gdn_prefill.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `sm100`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-1): 168.944 us before and 165.759 us after.
- [Latency comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-2): 14.18 us before and 13.76 us after.
- [Latency comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-3): 36.976 us before and 36.701 us after.
- [Latency comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-4): 87.906 us before and 87.681 us after.
- [Latency comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/performance.md#comparison-5): 121.314 us before and 121.137 us after.

Complete local evidence: [task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/TASK.md), [before code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/before.py), [after code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/after.py), [unified diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/changes.diff), and [provenance](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-1b8c8c37fa32577a/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 219 lines and removes 164 lines.
- Added or rewritten symbols: `run_small`.
- Removed or replaced symbols: `_compile_small`, `run_small`.
- Retuned configuration or scheduling values: `acc`, `base`, `cache`, `chunk_base`, `coord`, `d`, `gSin`, `gSout`, `g_cum`, `kb`, `kern`, `neg_exp_alog`, `row`, `sp`, `stride`, `tGgK`, `tGgQ`, `tGgV`, `tGsK`, `tGsQ`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: tensor-core-optimization, pipeline-stages, shared-memory-optimization, tile-scheduling, cache-policy, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

### Variant 2

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-1): 61.216 us before and 60.608 us after.
- [Latency comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/performance.md#comparison-2): 61.088 us before and 60.24 us after.

Complete local evidence: [task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/TASK.md), [before code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/before.py), [after code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/after.py), [unified diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/changes.diff), and [provenance](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-8d2c808b80cf75d0/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 170 lines and removes 58 lines.
- Retuned configuration or scheduling values: `_K_PER_LANE`, `_V_PER_WARP`, `_WARPS`, `block`, `delta`, `gate`, `k_idx`, `linear`, `old_value`, `out_value`, `softplus`, `state_value`, `token`, `update_gate`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: pipeline-stages, shared-memory-optimization, tile-scheduling, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

### Variant 3

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-1): 109.952 us before and 101.92 us after.
- [Latency comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-2): 18.47 us before and 16.939 us after.
- [Latency comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-3): 91.616 us before and 84.992 us after.
- [Latency comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-4): 58.528 us before and 56.768 us after.
- [Latency comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-5): 17.355 us before and 15.863 us after.
- [Latency comparison 6](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-6): 27.936 us before and 25.312 us after.
- [Latency comparison 7](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-7): 1.1365 us before and 0.771 us after.
- [Latency comparison 8](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-8): 2.1769 us before and 2.119 us after.
- [Latency comparison 9](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-9): 3.5465 us before and 3.4707 us after.
- [Latency comparison 10](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-10): 3.008 us before and 2.944 us after.
- [Latency comparison 11](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-11): 111.616 us before and 109.184 us after.
- [Latency comparison 12](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-12): 3.5465 us before and 3.4707 us after.
- [Latency comparison 13](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/performance.md#comparison-13): 2.5045 us before and 2.4718 us after.

Complete local evidence: [task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/TASK.md), [before code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/before.py), [after code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/after.py), [unified diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/changes.diff), and [provenance](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-9e7d19b93417849f/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 560 lines and removes 647 lines.
- Added or rewritten symbols: `ew_view_k`, `ew_view_mn`, `exec_mma`, `ptr`.
- Removed or replaced symbols: `_get_compiled`, `exec_mma`, `maker`, `map_slot`, `ptr`, `slot_base`.
- Retuned configuration or scheduling values: `BAR_CORES`, `BAR_MD`, `BAR_RT1`, `BAR_RT2`, `BAR_S_READY`, `BAR_T64`, `BAR_TAILS`, `BAR_UWK`, `BAR_Z_FREE`, `C`, `D`, `HQ`, `HV`, `Lc`, `Qt`, `THREADS`, `acc_v`, `base`, `base_j`, `blk`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **operator fusion**: can remove intermediate writes, reads, and launch overhead.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: tma-transfers, tensor-core-optimization, pipeline-stages, shared-memory-optimization, kernel-fusion, tile-scheduling, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

### Variant 4

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-1): 296.352 us before and 270.464 us after.
- [Latency comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-2): 270.464 us before and 261.728 us after.
- [Latency comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/performance.md#comparison-3): 306.437 us before and 287.572 us after.

Complete local evidence: [task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/TASK.md), [before code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/before.py), [after code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/after.py), [unified diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/changes.diff), and [provenance](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-c5d8efebe630ec24/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 1 lines and removes 0 lines.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

### Variant 5

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-1): 1.335834 ms before and 1.309591 ms after.
- [Latency comparison 2](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-2): 1.553052 ms before and 1.54375 ms after.
- [Latency comparison 3](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-3): 16.737 us before and 16.657 us after.
- [Latency comparison 4](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-4): 0.739102 ms before and 0.737542 ms after.
- [Latency comparison 5](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-5): 2.219 ms before and 2.168 ms after.
- [Latency comparison 6](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-6): 1.77 ms before and 1.76 ms after.
- [Latency comparison 7](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-7): 17.05 us before and 16.83 us after.
- [Latency comparison 8](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-8): 1.276406 ms before and 1.230107 ms after.
- [Latency comparison 9](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-9): 0.739102 ms before and 0.737408 ms after.
- [Latency comparison 10](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-10): 1.336 ms before and 1.322 ms after.
- [Latency comparison 11](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-11): 1.979 ms before and 1.889 ms after.
- [Latency comparison 12](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-12): 1.615 ms before and 1.598 ms after.
- [Latency comparison 13](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-13): 1.32159 ms before and 1.321514 ms after.
- [Latency comparison 14](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-14): 1.963148 ms before and 1.957253 ms after.
- [Latency comparison 15](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-15): 1.963148 ms before and 1.958225 ms after.
- [Latency comparison 16](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-16): 1.276406 ms before and 1.267183 ms after.
- [Latency comparison 17](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/performance.md#comparison-17): 16.737 us before and 16.319 us after.

Complete local evidence: [task](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/TASK.md), [before code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/before.py), [after code](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/after.py), [unified diff](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/changes.diff), and [provenance](../../artifacts/experiments/gdn_prefill/experiment-gdn-prefill-f68f418db298e09c/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 2016 lines and removes 40 lines.
- Added or rewritten symbols: `SharedStorage`, `_Chunk64GDN`, `_TmaRecurrentGDN`, `__call__`, `__init__`, `_chunk_compile_cache`, `_gemm_sync`, `_make_tiled_mma`, `_recurrent_paired_kernel`, `_run_chunk_impl`, `_run_impl`, `_store_mma_operand`, `_store_mma_operand_k64`, `kernel`, `run`, `run_iket`.
- Removed or replaced symbols: `run`.
- Retuned configuration or scheduling values: `WARPS_PER_BLOCK`, `cache`, `delta`, `k_col`, `out_value`, `r_state`, `value`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **operator fusion**: can remove intermediate writes, reads, and launch overhead.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: tma-transfers, tensor-core-optimization, pipeline-stages, shared-memory-optimization, kernel-fusion, tile-scheduling, cache-policy, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
