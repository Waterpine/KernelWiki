---
id: kernel-nvfp4-gemv
title: NVFP4 Batched GEMV
type: kernel
architectures: [sm100, sm100a]
tags: [gemv, nvfp4, fp4, block-scale, cache-policy, register-budgeting, vectorized-loads]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemv, batched-gemv]
languages: [cuda-cpp, ptx]
related: [hw-nvfp4, kernel-nvfp4-gemm, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, contest-gpumode-p1, blog-yue-nvfp4, blog-amandeep-nvfp4]
performance_claims: []
artifact_dir: artifacts/kernels/nvfp4-gemv
---

# NVFP4 Batched GEMV

The competition workload multiplies packed E2M1 matrices/vectors with one
E4M3 block scale per 16 values and tensor-level scales. Its low reuse makes
memory traffic, decode/scale work, reduction, and launch geometry important.

```cpp
template <int K>
__global__ void nvfp4_gemv(const uint8_t* packed_a,
                           const uint8_t* packed_b,
                           const uint8_t* scale_a,
                           const uint8_t* scale_b,
                           half* output, int rows) {
    int row = blockIdx.x;
    if (row < rows) output[row] = compute_row<K>(packed_a, packed_b, scale_a, scale_b, row);
}
```

This is interface pseudocode, not a competition submission. The public source
page records reconstructed blog-derived strategies such as specialization,
vector loads, cache hints, and register tuning. Their benefit is conditional:
lower register counts can reduce spills or increase occupancy, but can also
reduce instruction-level parallelism; cache hints depend on the actual reuse
and working set.

The former page presented reconstructed rank code and approximate `22.4 us`
geometric mean as a verified exact-shape result, and compared it with an
`8.6 us` bound derived from a nominal bandwidth. Because the public contest
repository does not provide an authoritative leaderboard/log for those values,
the numeric structured claim is removed and the estimates are tracked as
unverified source-reported claims.

Pinned upstream and derived artifacts are under
[`artifacts/kernels/nvfp4-gemv/`](../../artifacts/kernels/nvfp4-gemv/); their
provenance files identify reconstructions.
