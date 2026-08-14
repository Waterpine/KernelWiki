---
id: kernel-gated-delta-net
title: Gated Delta Net — Linear Attention
type: kernel
architectures:
- sm100
- sm90
tags:
- gated-delta-net
- linear-attention
- attention
confidence: source-reported
reproducibility: snippet
kernel_types:
- gated-delta-net
- linear-attention
- decode
- prefill
- attention
languages:
- triton
- cuda-cpp
related:
- technique-pipeline-stages
- technique-chunk-parallelism
sources:
- blog-gated-delta-net
- doc-tfla
- pr-vllm-37303
performance_claims: []
blackwell_relevance: "Tiled linear-attention matmuls may target SM100 tensor-core instructions; the model recurrence and its synchronization requirements are architecture-independent."
artifact_dir: artifacts/kernels/gated-delta-net
---

# Gated Delta Net — Linear Attention

## Algorithm

Gated Delta Networks combine a recurrent linear-attention state with a delta-rule correction and data-dependent decay. In one simplified orientation:

```python
def gated_delta_step(state, q, k, v, decay, beta):
    retrieved = k @ state
    error = v - retrieved
    next_state = decay * state + beta * outer(k, error)
    output = q @ next_state
    return next_state, output
```

The exact upstream equations determine whether output is read before or after the update, as well as key normalization and gate placement. The important distinction is that the update uses a retrieval error; `state += outer(k, v)` is not the delta rule.

## Parallelization

- decode advances the recurrent state in token order;
- training and prefill can express work inside a chunk as dense matrix operations;
- boundary states between chunks still require an ordered recurrence, scan, or equivalent two-stage construction.

The TFLA work tiles the matrix operations used by linear-attention algorithms. Its architecture-specific assembly must be taken from the pinned source; a fabricated `tcgen05.mma.kind::f16f16f32` spelling is not a valid substitute for the documented PTX grammar.

## Evidence Boundary

This page makes no cross-model throughput claim. Model architecture, state dimensions, chunk size, and benchmark speedups need their own pinned primary sources and environments.

## Artifacts

Verbatim upstream code is in [`artifacts/kernels/gated-delta-net/full/`](../../artifacts/kernels/gated-delta-net/full/), with provenance metadata in the bundle's `PROVENANCE.yaml`.

```bash
python3 scripts/get_page.py kernel-gated-delta-net --include-code
```
