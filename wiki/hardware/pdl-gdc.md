---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90]
tags: [pdl, gdc]
confidence: verified
related: [technique-persistent-kernels, hw-clc]
sources: [doc-cuda-13, pr-cutlass-2161, pr-cutlass-3106, doc-cutlass-changelog-sm100]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL is an explicit launch/synchronization protocol on supported devices; Blackwell libraries can opt into it for suitable dependent launches."
evidence_basis:
  - source_id: doc-cuda-13
    evidence_type: official-doc
  - source_id: pr-cutlass-3106
    evidence_type: upstream-code
---

## Overview

Programmatic Dependent Launch lets a secondary kernel in the same stream become
eligible to launch after every block of the primary has reached a launch
completion point, while the primary may still have independent work remaining.
The overlap is opportunistic and is never a correctness assumption.

```cuda
__global__ void primary() {
    produce_independent_prefix();
    cudaTriggerProgrammaticLaunchCompletion();
    finish_work_that_may_overlap();
}

__global__ void secondary() {
    do_independent_prefix();
    cudaGridDependencySynchronize();
    consume_primary_results();
}
```

The host launches `secondary` with the extensible launch API, sets the
attribute ID to `cudaLaunchAttributeProgrammaticStreamSerialization`, and sets
its value field
`attr.val.programmaticStreamSerializationAllowed = 1`. The latter is a field,
not an attribute-enum constant. If the primary does not call the trigger,
launch completion is implicitly triggered only after all its blocks exit.

## Correctness and limitations

The secondary can start before the primary's writes are visible. It must call
`cudaGridDependencySynchronize()` (or use another documented mechanism) before
dependent accesses. NVIDIA explicitly warns that concurrent execution is not
guaranteed and that relying on it can deadlock.

PDL is opt-in; it is not enabled by default on SM100. Back-to-back launches do
not automatically overlap, and the feature does not reduce latency without
code changes. CUTLASS's `use_pdl` launch paths and tutorial are versioned
examples, not a device-wide default.
