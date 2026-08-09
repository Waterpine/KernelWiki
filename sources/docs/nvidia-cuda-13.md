---
id: doc-cuda-13
title: "NVIDIA CUDA Toolkit 13.x for Blackwell"
url: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, clc, tma, pdl, gdc, nvfp4, fp8, fp4, fp6, block-scale]
retrieved_at: 2026-04-16
---

# NVIDIA CUDA Toolkit 13.x for Blackwell

## Overview

CUDA 13.0 and 13.1 introduce full Blackwell (SM100) support, including new compiler intrinsics, PTX instructions, and runtime features for tcgen05, TMEM, CLC, and sub-byte data types.

## CUDA 13.0 Key Features

### SM100 Compiler Support

- New target architectures: `sm_100`, `sm_100a`
- NVCC supports `-arch=sm_100` and `-arch=sm_100a`
- `sm_100a` includes architecture-accelerated features (required for B200-specific instructions)
- PTX ISA 8.7+ for SM100 instructions

### tcgen05 Intrinsics

```cpp
// tcgen05.mma via PTX inline assembly
uint32_t mask[4] = {0, 0, 0, 0};   // disable-output-lane: mask nothing
asm volatile(
    "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 "
    "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
    : : "r"(tmem_addr), "l"(smem_desc_a), "l"(smem_desc_b), "r"(idesc), "r"(1),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
);

// CUDA C++ wrappers via CUTLASS/CuTe
// (No direct CUDA runtime intrinsics -- PTX or CuTe abstraction required)
```

### TMEM Access

```cpp
// TMEM allocation and access (via PTX).
// Both alloc and dealloc are .sync.aligned: with .cta_group::1 one whole warp
// of the CTA must perform the allocation and the deallocation.
asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    : : "r"(smem_dst), "r"(num_cols));

// TMEM to register readback. .16x256b.x1 loads a 4-register vector (Table 52).
asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
    : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3) : "r"(tmem_addr));

// TMEM release (mandatory: all allocated TMEM must be deallocated before the
// kernel exits)
asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
    : : "r"(tmem_addr), "r"(num_cols));
```

### CLC APIs

```cpp
// CLC dynamic tile scheduling
asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta"
             ".mbarrier::complete_tx::bytes.b128 [%0], [%1];"
    :: "r"(clc_response_addr), "r"(clc_mbar_addr) : "memory");
// then, after waiting on the mbarrier:
//   ld.shared.b128 handle, [clc_response];
//   clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;

// CLC replaces manual tile queue management
// Hardware schedules tiles to available SMs
```

### Sub-Byte Type Support

```cpp
// FP4 type (E2M1)
#include <cuda_fp4.h>
__nv_fp4_e2m1 val;

// FP6 type
#include <cuda_fp6.h>

// Block-scale types
#include <cuda_fp8.h>
__nv_fp8_e4m3 scale;

// Conversion intrinsics
__half2_raw result = __nv_cvt_fp4x2_to_halfraw2(packed_fp4, __NV_E2M1);
```

### PDL (Opt-In per Launch)

Programmatic Dependent Launch is available from compute capability 9.0 and is
opt-in per launch:
- The secondary kernel must be launched with
  `cudaLaunchAttributeProgrammaticStreamSerialization` via `cudaLaunchKernelEx`
- The primary kernel calls `cudaTriggerProgrammaticLaunchCompletion()`; the
  secondary calls `cudaGridDependencySynchronize()` before reading its results
- Overlap is opportunistic and not guaranteed to produce concurrent execution

## CUDA 13.1 Additions

### cuTile (NVIDIA)

New tile-level programming model:
- Higher-level abstraction than raw PTX
- Automatic TMA and MMA scheduling
- Targets SM100, SM103, SM110, SM120, SM121

### Performance Improvements

- Improved NVRTC (Runtime Compilation) for JIT kernels
- Better NVCC code generation for SM100 instruction scheduling
- Enhanced profiling support in Nsight Compute for TMEM and CLC

## PTX ISA SM100 Highlights

Key new PTX instructions for SM100:

| Instruction | Purpose |
|---|---|
| `tcgen05.mma.*` | Tensor core MMA (7 variants) |
| `tcgen05.alloc` | Allocate TMEM columns |
| `tcgen05.dealloc` | Release TMEM columns |
| `tcgen05.ld` | Load from TMEM to registers |
| `tcgen05.st` | Store from registers to TMEM |
| `clusterlaunchcontrol.try_cancel` | Request cancellation of a not-yet-launched cluster |
| `clusterlaunchcontrol.query_cancel` | Interpret the try_cancel response |
| `cvt.rn.f16x2.e2m1x2` | FP4 to FP16 conversion |
| `cvt.rn.satfinite.e2m1x2.f16x2` | FP16 to FP4 conversion |

## Compiler Flags for SM100

```bash
# Basic SM100 compilation
nvcc -arch=sm_100 kernel.cu

# SM100a (architecture-accelerated, required for B200)
nvcc -arch=sm_100a kernel.cu

# Register budget control (critical for memory-bound kernels)
nvcc -arch=sm_100a -maxrregcount=32 kernel.cu

# Generate PTX + SASS
nvcc -arch=sm_100a --ptx kernel.cu
nvcc -arch=sm_100a --cubin kernel.cu
```

## Sources

- [CUDA 13.0 Blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [CUDA 13.1 Blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [PTX ISA 8.7 Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
