---
id: kernel-trace-kda-forward
title: 'Optimization case studies: KDA forward'
type: kernel
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
confidence: experimental
reproducibility: snippet
kernel_types:
- linear-attention
- prefill
languages:
- cute-dsl
related:
- technique-pipeline-stages
- pattern-memory-bound
- technique-vectorized-loads
- technique-tile-scheduling
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
sources:
- experiment-family-kda-forward
artifact_dir: artifacts/experiments/kda_forward
experiment_count: 2
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

# Optimization case studies: KDA forward

## Family overview

This page summarizes 2 distinct `kda_forward` optimization experiment(s). The [family source record](../../sources/experiments/kda_forward.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `blackwell`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-1): 4.436 ms before and 4.224 ms after.
- [Latency comparison 2](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-2): 5.589 ms before and 5.343 ms after.
- [Score comparison 3](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-3): 0.2381 ratio before and 0.2435 ratio after.
- [Latency comparison 4](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-4): 1.167 ms before and 1.124 ms after.
- [Latency comparison 5](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-5): 0.786 ms before and 0.749 ms after.
- [Latency comparison 6](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-6): 2.251 us before and 1.67 us after.
- [Latency comparison 7](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-7): 7.226 us before and 6.536 us after.
- [Latency comparison 8](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-8): 5.1 ms before and 4.6 ms after.
- [Latency comparison 9](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-9): 6.536 us before and 5.962 us after.
- [Latency comparison 10](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-10): 5.962 us before and 5.13 us after.
- [Latency comparison 11](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-11): 3.207 us before and 2.641 us after.
- [Latency comparison 12](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-12): 2.692 ms before and 2.661 ms after.
- [Latency comparison 13](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-13): 5.13 us before and 4.371 us after.
- [Latency comparison 14](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-14): 1.538 us before and 1.513 us after.
- [Latency comparison 15](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-15): 1.513 us before and 1.441 us after.
- [Latency comparison 16](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-16): 4.275 us before and 3.441 us after.
- [Latency comparison 17](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-17): 1.024 us before and 0.843 us after.
- [Latency comparison 18](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-18): 1.176613 ms before and 0.975398 ms after.
- [Latency comparison 19](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/performance.md#comparison-19): 0.924707 ms before and 0.739955 ms after.

Complete local evidence: [task](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/TASK.md), [before code](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/before.py), [after code](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/after.py), [unified diff](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/changes.diff), and [provenance](../../artifacts/experiments/kda_forward/experiment-kda-forward-6370e3dd96982da1/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 1320 lines and removes 203 lines.
- Added or rewritten symbols: `PrepStorage`, `RecurrenceStorage`, `_ChunkedKDA`, `__init__`, `_compile`, `_dual_gemm_to_bf16`, `_gemm_state_update`, `_gemm_to_bf16`, `_gemm_to_fp16`, `_get_aux_stream`, `_half2_add`, `_movmatrix_transpose`, `_neumann_inverse_fp16`, `_sigmoid_approx`, `_tanh_approx`, `prepare_kernel`, `recurrence_kernel`.
- Removed or replaced symbols: `_RecurrentKDA`, `__init__`, `_compiled_kernel`, `kernel`.
- Retuned configuration or scheduling values: `a_scale`, `block`, `bos`, `compiled`, `eos`, `gate_sigmoid`, `gate_x`, `k_inv`, `k_sq`, `key`, `lane`, `op`, `output`, `q_inv`, `q_sq`, `s_k`, `s_q`, `self.heads`, `self.sequences`, `self.total_tokens`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **vectorized memory access**: can reduce the number of memory instructions and improve coalescing.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: tma-transfers, tensor-core-optimization, pipeline-stages, shared-memory-optimization, vectorized-loads, tile-scheduling, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

### Variant 2

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/performance.md#comparison-1): 267.744 us before and 263.712 us after.

Complete local evidence: [task](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/TASK.md), [before code](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/before.py), [after code](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/after.py), [unified diff](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/changes.diff), and [provenance](../../artifacts/experiments/kda_forward/experiment-kda-forward-f601f83a1b54d82a/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 35 lines and removes 16 lines.
- Retuned configuration or scheduling values: `cu_seqlens`, `initial_state`, `out`. These changes can alter tile shape, work distribution, or launch configuration.

Possible mechanism: the observed changes. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
