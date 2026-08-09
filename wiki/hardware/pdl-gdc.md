---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90]
tags: [pdl, gdc]
confidence: source-reported
related: [technique-persistent-kernels, hw-clc]
sources: [doc-nvidia-tuning-guide, pr-cutlass-2161, doc-cutlass-changelog-sm100]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL is available from compute capability 9.0 (Hopper) onward, including Blackwell SM100. It is opt-in per launch on every architecture: the secondary kernel must be launched with cudaLaunchAttributeProgrammaticStreamSerialization via cudaLaunchKernelEx."
---

## Overview

PDL/GDC allows overlapping execution of dependent kernel launches. The primary kernel signals it is finishing; the secondary kernel begins before the primary fully completes.

## How It Works

```cuda
// Primary kernel signals near completion
cudaGridDependencySynchronize();  // or PTX equivalent

// Secondary kernel can start overlapping with primary's tail
// Opt-in on every architecture: launch the secondary kernel with
// cudaLaunchAttributeProgrammaticStreamSerialization (cudaLaunchKernelEx)
```

## Behavior on Blackwell

SM100 supports PDL, but it is still opt-in per launch. This means:
- Back-to-back kernel launches overlap only when the secondary kernel is
  launched with `cudaLaunchAttributeProgrammaticStreamSerialization`
- The secondary kernel must call `cudaGridDependencySynchronize()` (or use
  another mechanism) before reading the primary kernel's results
- Overlap is opportunistic: the CUDA Programming Guide states the behaviour
  "is opportunistic and not guaranteed to lead to concurrent kernel execution"

## When It Matters
- Chains of small kernels (e.g., MoE dispatch → compute → combine)
- Pipeline-parallel training with many sequential kernel launches
- Reduces overall wall-clock time, but requires code changes: the trigger call in the primary kernel, `cudaGridDependencySynchronize()` in the secondary, and the launch attribute on the secondary launch

## Related
- [persistent-kernels](../techniques/persistent-kernels.md) — Alternative approach to reducing launch overhead
- [clc](clc.md) — Dynamic scheduling within persistent kernels
