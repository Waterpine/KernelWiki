---
id: doc-cuda-13
title: "NVIDIA CUDA Toolkit 13.x for Blackwell (through 13.3 Update 1)"
url: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, clc, tma, pdl, gdc, nvfp4, fp8, fp4, fp6, block-scale]
retrieved_at: 2026-08-13
---

# NVIDIA CUDA Toolkit 13.x for Blackwell

## Historical context

CUDA 12.8 was the first toolkit with native Blackwell cubin generation. CUDA
13.x continues and extends Blackwell support; it did not introduce SM100 support
from scratch. Use `sm_100` for family-compatible SM100 code and an accelerated
target such as `sm_100a` only when the selected instruction requires that
target. `sm_100a` is not a blanket requirement for all code running on B200.

## PTX-facing Blackwell features

The low-level interfaces are the PTX `tcgen05.*` and
`clusterlaunchcontrol.*` families, or compiler/library wrappers in libcu++ and
CUTLASS/CuTe. TMEM allocation counts columns, not rows, and the allocation and
deallocation instructions are warp-collective:

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], 32;
ld.shared.b32 taddr, [smem_slot];
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 32;
```

Cluster Launch Control uses an asynchronous cancellation request plus an
mbarrier and query instructions. There are no PTX instructions named
`clc.arrive` or `clc.wait`.

```cpp
#include <cuda/ptx>

// Preferred high-level shape, abbreviated: one selected thread submits a
// cancellation request; completion is observed with the shared mbarrier.
// cuda::ptx::clusterlaunchcontrol_try_cancel(&response, &barrier);
```

## Programmatic Dependent Launch

PDL is opt-in. The host launch uses
`cudaLaunchAttributeProgrammaticStreamSerialization`; the primary kernel calls
`cudaTriggerProgrammaticLaunchCompletion`, and the dependent kernel calls
`cudaGridDependencySynchronize` before consuming dependent results. It is not
enabled by default merely because the toolkit or GPU is Blackwell.

## Toolkit-specific features

CUDA 13.0 added and changed many compiler, library, and platform features; CUDA
13.1 introduced CUDA Tile as a higher-level tile programming model. These
toolkit release claims are version-sensitive and should be cited to the
corresponding release notes rather than generalized to earlier toolkits.
As of 2026-08-13, NVIDIA's toolkit archive lists CUDA 13.3 Update 1 as the
latest stable release and CUDA 13.4.0 only as a developer preview.

## Authoritative sources

- [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
- [CUDA 13.1 release notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/)
- [CUDA 13.3 Update 1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [CUDA Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
