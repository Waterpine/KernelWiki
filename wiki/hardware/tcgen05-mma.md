---
id: hw-tcgen05-mma
title: "tcgen05.mma — Blackwell MMA Instruction"
type: hardware
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tmem, hw-2sm-cooperative, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05.mma — Blackwell MMA Instruction

## What it is

`tcgen05.mma` is the fifth-generation Tensor Core PTX family used by Blackwell
accelerated targets. Its destination/accumulator is Tensor Memory (TMEM), which
removes the large register-resident accumulator fragments used by Hopper
`wgmma.mma_async`.

An elected thread issues the hardware operation. CuTe and compiler interfaces
may require a warp-uniform call and perform election internally, so do not add
manual `threadIdx.x == 0` guards around a framework API unless that API's
contract asks for one.

## Operand and shape constraints

- The destination is TMEM.
- Operand B is described in shared memory.
- Operand A may use a shared-memory descriptor or, for supported forms, TMEM.
- `.cta_group::1` and `.cta_group::2` forms are available; group 2 coordinates
  a CTA pair.
- Legal M, N, K, type, layout, scale-vector, and CTA-group combinations depend
  on the MMA kind. `m128n256k16` BF16 and `m256n256k16` CTA-pair examples are
  common maximum tiles, not the only legal shapes.
- Dense kinds include `f16`, `tf32`, `f8f6f4`, `i8`, `mxf8f6f4`, `mxf4`, and
  `mxf4nvf4`, with block-scale qualifiers on the MX/NV forms as specified by
  PTX.

Use the PTX tables or a CUTLASS/CuTe atom rather than hand-constructing an
instruction descriptor from a prose summary.

## Completion is not a fence

The MMA is asynchronous. A `tcgen05.fence` orders tcgen05 operations across an
execution synchronization point; it does not wait for MMA completion.

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];

wait:
mbarrier.try_wait.parity.b64 p, [mma_done], phase;
@!p bra wait;

// Required before a dependent asynchronous tcgen05 load in this pattern.
tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.16x128b.x16.b32 { /* registers */ }, [d_tmem];
tcgen05.wait::ld.sync.aligned;
```

When producer and consumer are different threads, pair
`tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` with
the documented execution-order synchronization. `__syncthreads()` plus a
before-fence is not a substitute for `tcgen05.commit` and an mbarrier completion
wait.

## Shared-memory layouts

128-byte swizzling is a frequent high-performance choice, but it is not a
universal correctness requirement. PTX permits all swizzle modes for common
K-major A and B layouts and imposes narrower combinations for some transposed
forms. The shared-memory producer layout, matrix descriptor, and MMA form must
agree.

## Performance evidence

The 255 → 695 → 940 → 1476 TFLOPS progression recorded in the `tcgen05 for
dummies` source is a source-reported benchmark for that tutorial's particular
GEMM and environment. It demonstrates that swizzling, pipelining, and persistent
scheduling mattered there; it is not a general percentage-of-peak guarantee.

## Sources

- [PTX ISA: fifth-generation Tensor Core operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-family-instructions)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
- [CUTLASS PR 2139](https://github.com/NVIDIA/cutlass/pull/2139)
