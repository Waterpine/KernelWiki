---
id: kernel-trace-kda-backward
title: 'Optimization case studies: KDA backward'
type: kernel
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
confidence: experimental
reproducibility: snippet
kernel_types:
- linear-attention
languages:
- cute-dsl
related:
- technique-kernel-fusion
- technique-tile-scheduling
- technique-persistent-kernels
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
- technique-warp-specialization
- hw-pdl-gdc
sources:
- experiment-family-kda-backward
artifact_dir: artifacts/experiments/kda_backward
experiment_count: 2
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

# Optimization case studies: KDA backward

## Family overview

This page summarizes 2 distinct `kda_backward` optimization experiment(s). The [family source record](../../sources/experiments/kda_backward.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `blackwell`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Score comparison 1](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/performance.md#comparison-1): 2.3146 ratio before and 2.3154 ratio after.

Complete local evidence: [task](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/TASK.md), [before code](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/before.py), [after code](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/after.py), [unified diff](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/changes.diff), and [provenance](../../artifacts/experiments/kda_backward/experiment-kda-backward-166c8416d6cd0edb/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 398 lines and removes 88 lines.
- Added or rewritten symbols: `_get_prep_stream`, `_get_scan_stream`, `_get_workspace`, `_vh`, `big`.
- Retuned configuration or scheduling values: `TA`, `argsA`, `argsB`, `argsC`, `argsD`, `assumed_align`, `beta0`, `bf`, `cu_t`, `dA`, `db`, `dbias`, `device`, `dg`, `dht`, `dk`, `dk_`, `dq`, `dv`, `dva`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **operator fusion**: can remove intermediate writes, reads, and launch overhead.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **persistent-kernel scheduling**: can amortize launch and scheduling overhead while improving work distribution.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: tma-transfers, kernel-fusion, tile-scheduling, persistent-kernel, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

### Variant 2

Architectures: `blackwell`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-1): 1.632 us before and 0.32 us after.
- [Latency comparison 2](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/performance.md#comparison-2): 1.664 us before and 0.288 us after.

Complete local evidence: [task](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/TASK.md), [before code](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/before.py), [after code](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/after.py), [unified diff](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/changes.diff), and [provenance](../../artifacts/experiments/kda_backward/experiment-kda-backward-7f9fffe3fe7517f6/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 51 lines and removes 27 lines.
- Retuned configuration or scheduling values: `bufs`, `dg_pre`, `dk_pre`, `dq_pre`, `fin`, `prioritize_fwd`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **warp specialization**: lets producer and consumer warps overlap data movement with compute.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: tma-transfers, tensor-core-optimization, warp-specialization, tile-scheduling, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
