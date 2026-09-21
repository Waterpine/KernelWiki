---
id: kernel-trace-cute
title: 'Optimization case studies: CuTe MoE'
type: kernel
task_family: cute
architectures:
- sm100
tags:
- moe
- grouped-gemm
- cute-dsl
- tma-transfers
- warp-specialization
- pipeline-stages
- shared-memory-optimization
- vectorized-loads
- tile-scheduling
- persistent-kernel
- precision-specialization
- parallel-reduction
- tcgen05
- tmem
- tma
- fp8
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
- technique-warp-specialization
- technique-pipeline-stages
- pattern-memory-bound
- technique-vectorized-loads
- technique-tile-scheduling
- technique-persistent-kernels
- lang-cute-dsl
- hw-tcgen05-mma
- hw-tmem
- hw-tma
sources:
- experiment-family-cute
artifact_dir: artifacts/experiments/cute
experiment_count: 1
performance_claims:
- gpu: B200
  dtype: fp8/bf16
  shape: not stated in the retained comparison
  metric: latency
  value: 0.133696
  before_value: 0.138464
  after_value: 0.133696
  unit: ms
  source_id: experiment-family-cute
  source_locator: artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/performance.md#comparison-1
  experiment_id: experiment-cute-d5bd6c063016d26e
correctness_status: gate-described-no-explicit-result
evidence_limitations:
- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
---

# Optimization case studies: CuTe MoE

## Family overview

This page summarizes 1 distinct `cute` optimization experiment(s). The [family source record](../../sources/experiments/cute.md) carries the per-experiment evidence hashes.

Every retained comparison is directionally positive for its metric. Variants remain separate below because they may target different architectures, workloads, or implementations.

## Variants and measured improvements

### Variant 1

Architectures: `sm100`. Languages: `cute-dsl`. Correctness: `gate-described-no-explicit-result`.

- [Latency comparison 1](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/performance.md#comparison-1): 0.138464 ms before and 0.133696 ms after.

Complete local evidence: [task](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/TASK.md), [before code](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/before.py), [after code](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/after.py), [unified diff](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/changes.diff), and [provenance](../../artifacts/experiments/cute/experiment-cute-d5bd6c063016d26e/PROVENANCE.yaml).

Observed changes:

- The optimized version adds 1605 lines and removes 153 lines.
- Added or rewritten symbols: `_bitonic32_desc`, `_bitonic8_desc`, `_bitonic_merge8_desc`, `_build_mtile_map_jit`, `_build_mtile_map_kernel`, `_combine_token_jit`, `_combine_token_kernel`, `_get_gemm`, `_get_mtile_map`, `_get_route`, `_merge4x8_top8_desc`, `_mma_m`, `_mtile_desc_rows`, `_quantize_mtile_jit`, `_quantize_mtile_kernel`, `_swiglu_quant_mtile64_jit`, `_swiglu_quant_mtile64_kernel`, `_swiglu_quant_mtile_jit`, `_swiglu_quant_mtile_kernel`, `_swiglu_quant_mtile_recip_jit`.
- Removed or replaced symbols: `_get_gemm`, `_get_route`.
- Retuned configuration or scheduling values: `block`, `cluster_shape_mn`, `compiled`, `dtype`, `expert`, `gemm1`, `gemm2`, `grid`, `hidden_idx`, `key`, `local_expert`, `max_active_clusters`, `mma_tiler_mn`, `options`, `position`, `scope`, `scores`, `sem`, `t`, `token`. These changes can alter tile shape, work distribution, or launch configuration.
- Diff signal — **TMA asynchronous transfers**: can overlap global-memory movement with computation and reduce load stalls.
- Diff signal — **warp specialization**: lets producer and consumer warps overlap data movement with compute.
- Diff signal — **software pipelining and asynchronous execution**: can hide memory and instruction latency behind useful work.
- Diff signal — **shared-memory reuse**: can reduce repeated global-memory traffic.
- Diff signal — **vectorized memory access**: can reduce the number of memory instructions and improve coalescing.
- Diff signal — **tiling / blocking**: can improve locality, reuse, and hardware occupancy.
- Diff signal — **persistent-kernel scheduling**: can amortize launch and scheduling overhead while improving work distribution.
- Diff signal — **precision or data-type specialization**: can increase arithmetic throughput and reduce memory bandwidth demand.
- Diff signal — **parallel reduction**: can shorten serial dependency chains and expose more parallel work.

Possible mechanism: tma-transfers, warp-specialization, pipeline-stages, shared-memory-optimization, vectorized-loads, tile-scheduling, persistent-kernel, precision-specialization, parallel-reduction. This is an evidence-bounded hypothesis, not measured causality.

## Applicability and limitations

- This family page aggregates distinct experiments; results must not be treated as one benchmark run.
- Complete code, diffs, measurements, and receipts remain isolated in per-experiment artifact bundles.
- Causal mechanisms are hypotheses unless explicitly attributed by the retained evidence.
- At least one retained experiment documents its correctness gate without a separate explicit result.
- These bundles support evidence review; they do not include the original benchmark runtime for an independent rerun.
