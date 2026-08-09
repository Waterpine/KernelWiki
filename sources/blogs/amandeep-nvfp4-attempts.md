---
id: blog-amandeep-nvfp4
title: "Twelve Attempts at an FP4 Kernel"
author: Amandeep Singh
url: https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/
source_category: community-note
architectures: [sm100, sm100a]
tags: [nvfp4, gemv, fp4, block-scale, batched-gemv]
techniques: [vectorized-loads, cache-policy, register-budgeting, per-k-specialization, data-reuse]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx]
retrieved_at: 2026-04-16
---

# Twelve Attempts at an FP4 Kernel (Amandeep Singh)

## Overview

Amandeep Singh's detailed blog documenting twelve attempts at Problem 1 (NVFP4 Batched GEMV) in the GPU Mode hackathon — "twelve attempts, of which only one really worked well". The working kernel (attempt 7) landed roughly 3x off speed-of-light; attempts 8-12 all regressed or had no effect. The blog is valuable for its honest documentation of failed approaches, for the debugging methodology using Nsight Compute, and for its post-hackathon breakdown of what the top three solutions did differently.

## Final Performance

Attempt 7 (raw CUDA C++) measured against the organizers' speed-of-light figures:

| M | K | L | Kernel (us) | Speed of light (us) | Ratio |
|---|---|---|---|---|---|
| 7168 | 16384 | 1 | 26.7 | 8.6 | 3.1x |
| 4096 | 7168 | 8 | 45.1 | 17.3 | 2.6x |
| 7168 | 2048 | 4 | 16.4 | 4.3 | 3.8x |

Roughly 3x off speed of light, and none of the five later attempts improved on it.

## The 12 Attempts

### Attempts 1-4: CuTe Python DSL

- One thread per output row element, walking the whole K dimension:
  decode FP4 -> FP16, multiply by the scale factors, accumulate
- Correct, but no parallelism along K, so not competitive
- Later attempts varied K-dimension tiling and thread configuration
  within the same basic structure

### Attempt 5: Split-K with atomics in CuTe

- K split across threads, partial sums accumulated with atomicAdd
- Needed a custom @dsl_user_op atomic_add_fp32 (nvvm.atomicrmw FADD),
  since CuTe does not expose it
- The atomics were too expensive

### Attempt 6: Warp-shuffle reduction in CuTe

- 128 threads as 4 warps, one M row per warp, 32-lane K-tile splitting
- Better than atomics, but blocked on the compute side: the DSL exposes
  fma_packed_f32x2 and has no fma_packed_f16x2, and whether scalar FP16
  arith fuses into fma.rn.f16x2 is up to LLVM, not the programmer
- Noted in hindsight as a skill gap: another competitor reached the top 10
  (21.6us) with a pure CuTe kernel that emits the whole decode-FMA-reduce
  pipeline as one llvm.inline_asm block using cvt.rn.f16x2.e2m1x2 and
  fma.rn.f16x2

### Attempt 7: Rewrite in raw CUDA C++ — the one that worked

- 32 threads (one warp) per output row, each thread taking a strided K
  slice; 128 threads per block, 4 rows per block
- FP4 decode with __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1)
- The whole decode-scale-multiply pipeline in packed half2 ops
  (__hmul2 / __hfma2) over 4-byte (8 FP4 element) chunks, which is where
  the real speedup came from
- #pragma unroll 4 over scale-factor groups, __ldg for the scale loads
- Warp reduction with __shfl_down_sync, lane 0 stores the FP16 result
- Result: 26.7 / 45.1 / 16.4 us -> the baseline for everything after

### Attempts 8-12: five experiments that did not work

- Attempt 8: split-K with atomics again, this time in C++
  - Atomic contention and extra memory traffic outweighed the parallelism

- Attempt 9: one uint2 (64-bit) load instead of two uchar4 loads
  - 16-25% slower: extracting bytes from a uint2 costs bitwise ops, and the
    compiler already merges two consecutive uchar4 loads

- Attempt 10: four independent accumulator chains for ILP
  - Worst regression at +32-55%: the kernel is memory-bound, and the chains
    raised register pressure enough to spill while hurting coalescing

- Attempt 11: register-count and block-size tuning
  - -maxrregcount 80 -> 64 had zero effect (the kernel already uses fewer
    than 64), BLOCK_SIZE=256 with ROWS_PER_BLOCK=8 changed nothing, and
    #pragma unroll 8 instead of 4 dropped performance by 5-87%

- Attempt 12: software pipelining with an explicit prefetch prologue
  - Redundant: hardware prefetch plus __ldg was already doing it

### What the top solutions did differently (post-hackathon analysis)

- All three top solutions (clustered around 18.5us geometric mean, about
  2x speed of light) wrote load and decode in raw PTX rather than C
  intrinsics: cvt.rn.f16x2.e2m1x2 instead of __nv_cvt_fp4x2_to_halfraw2,
  ld.global with explicit qualifiers instead of __ldg
- Cache policy control was the biggest gap: L1::no_allocate for the
  streamed matrix A, L1::evict_last for the reused vector B
- Much wider loads: ld.global.v2.u64 (128-bit) and ld.global.v4.u64
  (256-bit), fetching 32 or 64 FP4 values per transaction, decoded with
  PTX byte unpacking (mov.b32 {tmp0, tmp1, tmp2, tmp3}, %reg)
- Templating on the exact K dimension with launch-time dispatch; the rank 1
  solution also tuned cache hints per K (K=3584, K=8192, K=1024)
- Tighter register budgets: rank 1 used -maxrregcount=32, rank 3 used 45,
  against 80 here
- The rank 2 solution processed BLOCK_M rows per block so that threads
  reading B are shared across rows
- A pure PyTorch solution using torch._scaled_mm with multi-stream
  parallelism over L scored 22.4us, within 20% of the top three

## Key Debugging Methodology

### Nsight Compute Analysis

The blog emphasizes using Nsight Compute to confirm the kernel is memory-bound:

```
// Nsight Compute key metrics to check:
// 1. Memory throughput: how close to 8 TB/s?
// 2. Compute throughput: should be low for memory-bound
// 3. Achieved occupancy: higher is better for memory-bound
// 4. L1 hit rate: should be high for B vector (evict_last)
// 5. L2 hit rate: confirms data reuse patterns

// Key insight from Amandeep:
// "The single most important thing I could have done after attempt 7 was
//  run Nsight Compute and confirm the kernel was memory-bound."
// Many optimizations are counterproductive if you misidentify
// the bottleneck (e.g., compute optimizations on memory-bound kernel)
```

### Performance Model

```
// Speed-of-light calculation:
// Total data to read:
//   A: M * K * 0.5 bytes (FP4)
//   B: 1 * K * 0.5 bytes (FP4)
//   sfa: M * (K/16) bytes (FP8)
//   sfb: 1 * (K/16) bytes (FP8)
//   Total ≈ M * K * 0.5625 bytes
//
// For M=7168, K=16384, L=1:
//   Total = 7168 * 16384 * 0.5625 = 66 MB
//   At 8 TB/s: 66 MB / 8 TB/s = 8.25us
//
// Actual: 26.7us = 3.2x off SOL
// The gap comes from: FP4 decode overhead, scale factor application,
// partial sum reduction, and less-than-perfect vectorization
```

## Failed Approaches (Instructive)

1. **Split-K with atomics** (attempts 5 and 8, in CuTe and then in C++): atomic contention plus the extra memory traffic outweighed the parallelism. The kernel is memory-bound, so more blocks only add scheduling overhead and atomic serialization on the same addresses.

2. **A wider single load** (attempt 9): replacing two `uchar4` loads with one `uint2` was 16-25% slower, because extracting the bytes again costs bitwise operations while the compiler was already merging the two `uchar4` loads.

3. **Four accumulator chains for ILP** (attempt 10): the worst regression at +32-55%. FMA latency hiding is irrelevant when every cycle waits on memory, and the extra chains raised register pressure enough to spill while the strided K pattern hurt coalescing.

4. **Register and block-size tuning** (attempt 11): `-maxrregcount` 80 -> 64 had zero effect because the kernel already used fewer than 64 registers, and `#pragma unroll 8` instead of 4 cost 5-87%.

5. **Manual software pipelining** (attempt 12): hardware prefetch combined with `__ldg` was already covering it, so the explicit prologue just doubled register pressure.

## Key Lessons

1. **Profile first, optimize second**: Nsight Compute should be the first tool, not the last. Knowing whether a kernel is memory-bound or compute-bound determines the entire optimization strategy.

2. **FP4 decode is the hidden bottleneck**: The sub-byte format introduces decode overhead that doesn't exist for standard FP16/FP32 kernels. Hardware intrinsics and PTX byte unpacking are essential.

3. **Intuition about the wrong bottleneck fails predictably**: split-K does not help memory-bound kernels; wider loads only help when the data can be used directly without unpacking; ILP is irrelevant when the bottleneck is memory; register tuning does nothing if you are already under the limit; and software pipelining is redundant when hardware prefetch is sufficient.

4. **3x off SOL is realistic for FP4 GEMV**: The FP4 decode overhead, scale factor application, and reduction operations add unavoidable computation that a pure bandwidth model doesn't account for.

5. **Systematic exploration beats intuition**: Documenting 12 attempts with measurements at each step is more productive than guessing at the optimal configuration.

## Sources

- [Twelve Attempts at an FP4 Kernel](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels)
