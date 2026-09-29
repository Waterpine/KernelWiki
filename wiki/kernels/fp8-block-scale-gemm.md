---
id: kernel-fp8-block-scale-gemm
title: FP8 block-scale GEMM
type: kernel
architectures: [sm100, sm103, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl, python]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [blog-deepgemm, pr-cutlass-2139, doc-cutlass-changelog-sm100]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: best reported benchmark; shape not specified in README news entry
    metric: TFLOPS
    value: 1550
    source_id: blog-deepgemm
    source_locator: https://github.com/deepseek-ai/DeepGEMM#news (2025-04-18 entry)
blackwell_relevance: CUTLASS and DeepGEMM provide SM100 block-scaled paths, but their scale formats and layouts are part of the API contract.
---

# FP8 block-scale GEMM

An FP8 block-scale GEMM multiplies low-precision operands while applying scale
metadata at a granularity finer than the full tensor. The exact granularity,
scale type, layout, promotion policy, and accumulator behavior belong to the
selected implementation; they must not be combined from different libraries.

CUTLASS PR 2139's Blackwell example wires scale layouts into its collective
builder. This contiguous excerpt is a reproducible configuration fragment:

```cpp
using ScaleConfig = decltype(
    cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;
```

The full retained PR artifact is the executable source. The former raw-PTX block
was removed because it omitted the required instruction descriptor and did not
match the official block-scale operand form.

DeepGEMM's README reports up to 1550 TFLOPS on H800 without naming the shape in
that news entry. It remains an attributed maximum, not a portable expectation
for this kernel class.

## Local B300 trajectory: fuse activation scale packing into a large GEMM

In the saved B300 block-scaled dense GEMM run,
**kernel A** launched a Triton scale-pack kernel to convert live FP32 activation
scales into packed E8M0 words, then launched the two-CTA CuTe GEMM. **Kernel B**
moves that pack into the resident GEMM grid for `M >= 256`. Each CTA packs a
disjoint set of scale words; an `async.global` proxy fence and a GPU-scope
count/epoch barrier
publish all words before the existing TMA mainloop reads them. The launch uses
148 CTAs, or 74 two-CTA clusters, matching the measured B300 residency cap.
The M64 path retains the separate pack because the fused path was neutral
there. The implementation also reserves eight bytes beside the packed buffer
for the reusable count and epoch.

| B300 M=4096, paired CUPTI trials | Separate pack A | Fused pack B | Result |
|---|---:|---:|---|
| First lease, median | 48.461 µs | 47.018 µs | B won 48/48 trials |
| Second lease, median | 47.688 µs | 46.989 µs | B won 48/48 trials |

The accepted large/all score rose from **1.6261× to 1.6780×** against the
task baseline; correctness passed 1/1. This A→B gain comes from removing a
launch and reusing the resident GEMM grid while preserving the same packed
scale input to TMA. The [saved paired A/B report](../../data/internal-b300/fp8-block-scale-gemm/paired-ab-report.md)
records the trial medians, confidence intervals, M64 control and correctness
checks. These B300 results are separate from the H800 DeepGEMM headline in
this page's frontmatter.
