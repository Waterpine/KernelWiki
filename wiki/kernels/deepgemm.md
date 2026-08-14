---
id: kernel-deepgemm
title: DeepGEMM — FP8 GEMM with Fine-Grained Scaling
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, fine-grained-quantization, block-scale]
confidence: verified
reproducibility: snippet
kernel_types: [gemm, grouped-gemm]
languages: [cuda-cpp, ptx]
related: [technique-fine-grained-quantization, hw-tcgen05-mma, hw-nvfp4]
sources: [blog-deepgemm, pr-deepgemm-304, pr-cutlass-2139]
performance_claims: []
blackwell_relevance: "DeepGEMM's SM100 FP8 kernels use tcgen05/TMEM and packed UE8M0 scale tensors; this is separate from its NVFP4/FP4 paths."
evidence_basis:
  - source_id: blog-deepgemm
    evidence_type: official-doc
  - source_id: pr-deepgemm-304
    evidence_type: upstream-code
artifact_dir: artifacts/kernels/deepgemm
---

# DeepGEMM

DeepGEMM is DeepSeek's runtime-compiled tensor-core kernel library. Its current
scope includes FP8, FP4, BF16, grouped GEMMs, and fused MoE-related kernels.
The upstream README requires CUDA 12.9 or newer for its SM100 implementation as
of the audit date.

## Fine-grained FP8 interface

The project uses fine-grained scale tensors rather than one scale for an entire
matrix. Its current README describes different scale storage contracts:

- SM90 accepts FP32 scaling factors.
- SM100 accepts packed UE8M0 scaling factors, four values per `torch.int`.
- the LHS scale tensor must use the project's TMA-aligned transposed layout.

The SM100 block-scaled wrapper in pinned PR 304 emits the documented family:

```ptx
// Verbatim instruction family from the pinned wrapper; C++ operands omitted.
.reg .b32 d;
.reg .b64 a_desc, b_desc;
.reg .pred p;
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
    [d], a_desc, b_desc, idesc, [a_sf_tmem], [b_sf_tmem], p;
```

`kind::f8f6f4` without the `mx` prefix is the unscaled family. TMEM does not by
itself imply “full precision”; the accumulator type and behavior come from the
selected instruction descriptor.

## Layouts and grouped GEMM

The current upstream interface supports NT, NN, TN, and TT for SM100, while its
SM90 implementation is NT-only. For M-grouped MoE GEMMs, N and K remain fixed
and expert segments are packed or masked according to the selected API. The
project also exposes a K-grouped weight-gradient API.

These are library contracts, not generic tcgen05 restrictions. In particular,
the M-segment alignment must come from the matching DeepGEMM query/configuration
function.

## JIT and performance claims

Current DeepGEMM uses a lightweight runtime JIT module. Older descriptions that
characterize all versions as NVRTC-based are stale: the upstream history says
NVRTC was disabled during the 2025 SM100 refactor and later work changed JIT
behavior again.

The project reports “up to 1550 TFLOPS on H800” for its April 2025 code. That is
an upstream, version- and shape-dependent headline, not a verified result for
the former hard-coded `4096^3` shape; the structured performance claim was
therefore removed.

## Reproduction path

Pinned upstream code and derived variants are under
[`artifacts/kernels/deepgemm/`](../../artifacts/kernels/deepgemm/). Consult each
`PROVENANCE.yaml` before quoting a file as upstream.
