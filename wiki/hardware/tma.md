---
id: hw-tma
title: "Tensor Memory Accelerator (TMA)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [tma, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-flashinfer-2387
    evidence_type: upstream-code
related: [hw-tcgen05-mma, technique-pipeline-stages, technique-swizzling]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, pr-flashinfer-2387, blog-tcgen05-tutorial]
aliases: [TMA, "tensor memory accelerator", "cp.async.bulk"]
blackwell_relevance: "Blackwell retains Hopper's TMA model; layouts feeding tcgen05 must satisfy the selected MMA descriptor, but 128-byte swizzling is not universally mandatory."
---

# Tensor Memory Accelerator (TMA)

## What TMA does

TMA provides asynchronous bulk copies, including multidimensional tensor copies
described by a tensor map. It can move data between global and shared memory and
supports cluster-shared and multicast forms where the selected PTX instruction
allows them. A single selected thread commonly issues a copy while other warps
compute.

For global-to-shared tensor copies, completion is reported through an mbarrier
transaction count. Shared-to-global bulk copies use async groups and the
corresponding commit/wait operations.

```ptx
// Abbreviated global -> CTA shared tensor copy.
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], bytes;
cp.async.bulk.tensor.2d.shared::cta.global
    .mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {x, y}], [bar];

wait:
mbarrier.try_wait.parity.b64 ready, [bar], phase;
@!ready bra wait;
```

Exact state-space ordering and operands vary by copy direction and multicast
form; copy the syntax from the PTX version used to compile the kernel.

## Tensor maps and swizzling

Tensor maps encode global dimensions and strides, the shared-memory box, element
strides, interleave, swizzle, L2 promotion, and out-of-bounds behavior. Swizzle
choices include none and several byte-group modes. The mode constrains the box
layout and must match how the consumer interprets shared memory.

128-byte swizzling is common for wide GEMM tiles because it can reduce bank
conflicts. It is not required for every TMA copy and is not universally required
for `tcgen05.mma`: the PTX ISA permits all swizzle modes for common K-major A and
B layouts and restricts some transposed/type combinations separately.

```cuda
CUtensorMap map;
CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B; // workload choice
CUresult result = cuTensorMapEncodeTiled(
    &map, type, rank, global_ptr, global_dims, global_strides,
    box_dims, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

Do not describe TMA as performing arbitrary FP32↔FP16/BF16 conversion during a
normal tensor copy. The tensor map's element type describes the copied tensor;
separate conversion/reduction instruction forms have their own contracts.

## Multicast

A multicast tensor load can deliver one global tile to selected CTAs in a
cluster, reducing redundant global reads for shared operands. Each destination
CTA participates in the cluster-scope mbarrier protocol and must use a valid
multicast mask. Multicast does not mean that CTAs share one physical SMEM
allocation.

## Pipeline guidance

- Keep one phase bit per reused barrier/stage.
- Match `expect_tx` bytes to all async transactions completing the phase.
- Do not add an extra arrival that advances the barrier too early.
- Release a shared-memory stage only after all consumers are finished.
- Tune stage count against latency and SMEM occupancy; “3–5 stages” is a common
  starting range, not a Blackwell rule.

## Sources

- [CUDA Programming Guide: TMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
- [PTX ISA: bulk asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [PTX ISA: tcgen05 shared-memory layouts](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-layout-and-swizzling)
