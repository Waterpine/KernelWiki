---
id: kernel-trace-moe
title: 'Optimization case studies: Mixture-of-Experts'
type: kernel
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
confidence: experimental
reproducibility: snippet
kernel_types:
- moe
- grouped-gemm
languages:
- cute-dsl
related:
- kernel-fused-moe
- kernel-grouped-gemm
- technique-pipeline-stages
- pattern-memory-bound
- technique-vectorized-loads
- technique-tile-scheduling
- technique-persistent-kernels
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
- hw-pdl-gdc
- technique-cache-policy
- technique-kernel-fusion
sources:
- experiment-family-moe
artifact_dir: artifacts/experiments/moe
experiment_count: 4
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

# Optimization case studies: Mixture-of-Experts

## Family overview

This page summarizes 4 distinct `moe` optimization experiment(s). The [family source record](../../sources/experiments/moe.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `sm100`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Score comparison 1](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/performance.md#comparison-1): 1.5116 ratio before and 1.5755 ratio after.

Complete local evidence: [task](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/TASK.md), [before code](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/before.py), [after code](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/after.py), [unified diff](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/changes.diff), and [provenance](../../artifacts/experiments/moe/experiment-moe-13030440ee02446b/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 2459 lines and removes 256 lines.
- Added or rewritten symbols: `_NativeWeightWorkspace`, `_finish_routing_prefix`, `_gather_exact_kernel`, `_gather_real_routes_kernel`, `_get_native_weights`, `_hybrid_native_remap_kernel`, `_invoke_compiled`, `_invoke_requant`, `_launch_device_pipeline`, `_launch_gather_exact`, `_launch_gather_real_routes`, `_launch_gemm1_split`, `_launch_gemm2`, `_launch_hybrid_native_remap`, `_launch_native_device_pipeline`, `_launch_native_gemm1`, `_launch_native_gemm1_hybrid`, `_launch_native_gemm2`, `_launch_native_gemm2_hybrid`, `_launch_route_prefix`.
- Removed or replaced symbols: `_launch_gemm2`, `_launch_swiglu_quant`, `_max_active_clusters`.
- Retuned configuration or scheduling values: `acc`, `activated`, `begin`, `block`, `cluster_shape_mn`, `col`, `compute`, `count`, `end`, `expert`, `grid`, `group_index`, `group_value`, `local_expert`, `m`, `mma_tiler_mn`, `offsets`, `padded`, `permuted_to_expanded`, `raw_score`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **vectorized memory access**: can reduce the number of memory instructions and improve coalescing.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **persistent-kernel scheduling**: can amortize launch and scheduling overhead while improving work distribution.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: tensor-core-optimization, pipeline-stages, shared-memory-optimization, vectorized-loads, tile-scheduling, persistent-kernel, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

### Variant 2

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-1): 126.144 us before and 125.872 us after.
- [Latency comparison 2](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-2): 151.264 us before and 150.944 us after.
- [Latency comparison 3](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-3): 230.816 us before and 230.592 us after.
- [Latency comparison 4](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/performance.md#comparison-4): 322.496 us before and 322.224 us after.

Complete local evidence: [task](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/TASK.md), [before code](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/before.py), [after code](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/after.py), [unified diff](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/changes.diff), and [provenance](../../artifacts/experiments/moe/experiment-moe-55b35550b3d4d426/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 474 lines and removes 31 lines.
- Added or rewritten symbols: `_MappedMState`, `_OutputClearState`, `__del__`, `_check_cuda`, `_compute_workspace_key`, `allocate`, `begin`, `begin_pointer_use`, `close`, `join`, `record`, `synchronize_pointer_use`, `wait`.
- Retuned configuration or scheduling values: `key`, `packed_rows`, `seq_len`, `workspace`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMEM / Tensor Core paths**: can move work onto high-throughput matrix hardware and reduce register pressure.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.

Possible mechanism: tensor-core-optimization, pipeline-stages, tile-scheduling, cache-policy. This is an evidence-bounded hypothesis, not measured causality.

### Variant 3

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/performance.md#comparison-1): 180.096 us before and 178.304 us after.

Complete local evidence: [task](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/TASK.md), [before code](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/before.py), [after code](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/after.py), [unified diff](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/changes.diff), and [provenance](../../artifacts/experiments/moe/experiment-moe-704e7b99864597ef/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 200 lines and removes 15 lines.
- Retuned configuration or scheduling values: `G12_FUSE_LO`, `RK_TINY_HI`, `out`, `ws`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **operator fusion**: can remove intermediate writes, reads, and launch overhead.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **persistent-kernel scheduling**: can amortize launch and scheduling overhead while improving work distribution.
- Diff signal — **caching and prefetching**: can hide memory latency and avoid redundant loads.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: tma-transfers, pipeline-stages, shared-memory-optimization, kernel-fusion, tile-scheduling, persistent-kernel, cache-policy, precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

### Variant 4

Architectures: `sm103`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-1): 0.2351 ms before and 0.1535 ms after.
- [Score comparison 2](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-2): 1.6938 ratio before and 1.738 ratio after.
- [Score comparison 3](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-3): 1.738 ratio before and 1.7668 ratio after.
- [Score comparison 4](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/performance.md#comparison-4): 1.738 ratio before and 1.7668 ratio after.

Complete local evidence: [task](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/TASK.md), [before code](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/before.py), [after code](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/after.py), [unified diff](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/changes.diff), and [provenance](../../artifacts/experiments/moe/experiment-moe-712f5040611492e9/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 40 lines and removes 33 lines.
- Retuned configuration or scheduling values: `expert`, `global_expert`, `raw`, `score`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.

Possible mechanism: precision-specialization. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
