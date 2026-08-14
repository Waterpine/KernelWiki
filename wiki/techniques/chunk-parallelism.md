---
id: technique-chunk-parallelism
title: "Chunk-Based Parallelism for Linear Attention"
type: technique
architectures: [sm100, sm90]
tags: [chunk-parallelism, linear-attention, gated-delta-net, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-gated-delta-net, kernel-nsa]
sources: [blog-gated-delta-net, doc-tfla]
blackwell_relevance: "Chunkwise linear-attention matmuls can use Blackwell tensor-core paths, but useful chunk sizes are algorithm- and shape-dependent rather than determined by TMEM capacity alone."
---

# Chunk-Based Parallelism

## Overview

Linear-attention recurrences have linear sequence complexity but retain a state dependency from one token to the next. Chunkwise algorithms reformulate work inside a chunk as matrix operations while carrying a boundary state between chunks. That boundary dependency means chunks are not simply independent grid programs.

## Dependency Structure

For a recurrence such as `S_t = update(S_{t-1}, k_t, v_t)` and `o_t = query(q_t, S_t)`, a chunked implementation normally has three logical phases:

1. compute chunk-local quantities in parallel;
2. obtain or scan the boundary states in sequence order;
3. combine each boundary state with the chunk-local result.

Implementations differ in whether those phases are separate kernels, a hierarchical scan, or a tiled fused kernel. A kernel that launches one program per chunk and lets every program read and overwrite a single state buffer without an ordering protocol is incorrect.

```python
def chunked_recurrence(chunks, initial_state):
    state = initial_state
    outputs = []
    for chunk in chunks:  # boundary states remain ordered
        local, state = evaluate_chunk(chunk, state)
        outputs.append(local)
    return outputs, state
```

This is a dependency sketch, not a parallel GPU implementation.

## Size Tradeoff

Chunk size controls a workload-specific balance among tensor-core efficiency, temporary storage, launch count, and the cost of boundary-state propagation. Values such as 64 or 128 appear in particular implementations, but there is no universal decode/prefill default. Benchmark the exact model dimensions, sequence lengths, and software version.

TFLA adds a second tiling level to chunkwise linear attention so that matrix operations inside larger chunks can be mapped efficiently; this does not remove the recurrence's ordered state semantics.

## When To Use

- linear-attention training or prefill where within-chunk matrix work is substantial;
- recurrent sequence layers with an associative or otherwise parallelizable boundary-state formulation;
- implementations whose numerical formulation matches the model's gates and normalization.

Chunking is an implementation transformation, not a claim that different chunks have no dependency.
