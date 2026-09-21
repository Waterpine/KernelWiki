---
id: kernel-trace-dsa-attention
title: 'Optimization case studies: DeepSeek sparse attention'
type: kernel
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
confidence: experimental
reproducibility: snippet
kernel_types:
- attention
- sparse-attention
- mla
- decode
languages:
- cute-dsl
related:
- kernel-sparse-mla
- kernel-flashmla
- pattern-memory-bound
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
- technique-pipeline-stages
- technique-kernel-fusion
- technique-tile-scheduling
- technique-cache-policy
- hw-mbarrier
sources:
- experiment-family-dsa-attention
artifact_dir: artifacts/experiments/dsa_attention
experiment_count: 4
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

# Optimization case studies: DeepSeek sparse attention

## Family overview

This page summarizes 4 distinct `dsa_attention` optimization experiment(s). The [family source record](../../sources/experiments/dsa_attention.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/performance.md#comparison-1): 54.88 us before and 43.231 us after.

Complete local evidence: [task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/TASK.md), [before code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/before.py), [after code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/after.py), [unified diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/changes.diff), and [provenance](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-044fa7d7d954d384/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 34 lines and removes 10 lines.
- Retuned configuration or scheduling values: `block_max`, `local_max`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: shared-memory-optimization, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

### Variant 2

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/performance.md#comparison-1): 33.40025 us before and 32.97625 us after.

Complete local evidence: [task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/TASK.md), [before code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/before.py), [after code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/after.py), [unified diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/changes.diff), and [provenance](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-09a9ec45abf67f99/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 705 lines and removes 67 lines.
- Added or rewritten symbols: `_consume_sparse_entry`, `_consume_sparse_group4`, `_sparse_score`.
- Retuned configuration or scheduling values: `_HEADS_PER_CTA`, `_WARPS_PER_CTA`, `ckv`, `head`, `inv_sum`, `kpe`, `min_blocks_per_mp`, `new_max`, `old_scale`, `probability`, `r_kv`, `r_out`, `r_q`, `r_qpe`, `row_max`, `row_sum`, `score`, `token`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **operator fusion**: can remove intermediate writes, reads, and launch overhead.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: tma-transfers, tensor-core-optimization, pipeline-stages, kernel-fusion, tile-scheduling, cache-policy, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

### Variant 3

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/performance.md#comparison-1): 11.33 us before and 10.91 us after.

Complete local evidence: [task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/TASK.md), [before code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/before.py), [after code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/after.py), [unified diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/changes.diff), and [provenance](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-1a2df21eeb64e7ff/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 48 lines and removes 78 lines.
- Added or rewritten symbols: `_load_sibling_run`, `run`.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.

Possible mechanism: cache-policy. This is an evidence-bounded hypothesis, not measured causality.

### Variant 4

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-1): 56.064 us before and 55.52 us after.
- [Latency comparison 2](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/performance.md#comparison-2): 5.664 us before and 4.96 us after.

Complete local evidence: [task](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/TASK.md), [before code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/before.py), [after code](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/after.py), [unified diff](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/changes.diff), and [provenance](../../artifacts/experiments/dsa_attention/experiment-dsa-attention-6351d0b3ec4f7f9c/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 593 lines and removes 33 lines.
- Added or rewritten symbols: `_compile_dense_shared_kernel`, `_kernel`, `_launch`, `_make_dense_shared_launcher`, `_make_split_launcher`, `_merge_kernel`, `_split_kernel`.
- Retuned configuration or scheduling values: `compiled`, `score`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: shared-memory-optimization, tile-scheduling, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
