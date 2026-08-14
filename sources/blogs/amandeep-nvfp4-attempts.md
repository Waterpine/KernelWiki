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
languages: [cuda-cpp, ptx, cute-dsl]
retrieved_at: 2026-08-14
---

# Twelve Attempts at an FP4 Kernel (Amandeep Singh)

## Overview

Amandeep Singh's worklog covers twelve attempts at GPU Mode's NVFP4 batched
GEMV problem. It is not a monotonic twelve-step optimization sequence: attempts
1--6 explore CuTe DSL implementations, attempt 7 is the successful raw-CUDA
kernel, and attempts 8--12 are reported regressions or no-ops.

## Attempt 7 results

The post reports three separate benchmark cases, not a 26.7-microsecond
geometric mean:

| M | K | L | Attempt 7 | bandwidth speed-of-light estimate | ratio |
|---:|---:|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 26.7 us | 8.6 us | 3.1x |
| 4096 | 7168 | 8 | 45.1 us | 17.3 us | 2.6x |
| 7168 | 2048 | 4 | 16.4 us | 4.3 us | 3.8x |

These are author-reported measurements and speed-of-light estimates for the
post's B200 contest workload, not general hardware limits.

## What the attempts actually did

- **Attempts 1--4:** CuTe Python DSL variants kept the basic structure of one
  thread walking K for an output row while experimenting with K tiling and
  thread configuration. They were correct but not competitive.
- **Attempt 5:** split K across threads and used global FP32 `atomicAdd` for
  partial sums. The author reports that the atomics were too expensive.
- **Attempt 6:** replaced atomics with warp-shuffle reduction. Four warps in a
  128-thread block each handled one M row; this improved on attempt 5 but did
  not break the compute-side wall the author encountered in CuTe.
- **Attempt 7:** switched to raw CUDA. A warp handled a row, each lane walked a
  strided portion of K, and the final sum used warp shuffles. The decode path
  used `__nv_cvt_fp4x2_to_halfraw2`; paired `half2` operations handled two FP4
  values at a time. This is the attempt that produced the table above.
- **Attempt 8:** retried split-K with C++ atomics and an FP32 intermediate;
  contention and extra traffic made it worse.
- **Attempt 9:** replaced two `uchar4` loads with one `uint2` load. Byte
  extraction overhead made it 16--25% slower.
- **Attempt 10:** used four accumulator chains for instruction-level
  parallelism. Higher register pressure and worse coalescing caused a reported
  32--55% regression.
- **Attempt 11:** varied register limits, block size, and unrolling. Lowering
  the limit from 80 to 64 registers and changing block size had no effect;
  heavier unrolling regressed.
- **Attempt 12:** added explicit software pipelining. It raised register
  pressure without improving the already-prefetched memory path.

Warp shuffles therefore belong to the successful attempt-7 kernel; they are
not one of the post's failed experiments.

## Post-hoc comparison with top solutions

After the contest, the author studied the top three solutions, which the post
says clustered near an 18.5-microsecond geometric mean. The techniques below
describe those competitors and must not be presented as the author's own
twelve-attempt progression:

- inline PTX for conversion and loads;
- different cache policies for streamed A and reused B;
- 128- or 256-bit PTX vector loads coupled with byte-unpack instructions;
- compile-time K specialization and per-K cache tuning;
- more aggressive register budgets; and
- reuse of B loads across multiple output rows in one solution.

The author also notes that a pure-PyTorch multi-stream `torch._scaled_mm`
solution scored 22.4 microseconds in the contest context.

## Source boundary

All performance values and causal explanations on this page are attributed to
the participant post. The page does not turn them into architectural
guarantees, and it does not infer unsupported tensor-core shape restrictions.

## Sources

- [Twelve Attempts at an FP4 Kernel](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels)
