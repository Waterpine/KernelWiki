---
id: blog-jax-pallas-blackwell-matmul
title: "Writing High-Performance Matrix Multiplication Kernels for Blackwell with JAX Pallas"
author: JAX Team (Google)
url: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
source_category: community-note
architectures: [sm100]
tags: [gemm, warp-specialization, 2sm-cooperative, persistent-kernel, tma, tmem, pipeline-stages, swizzling, jax-pallas, tcgen05, double-buffering, epilogue-fusion]
retrieved_at: 2026-08-13
---

# JAX Pallas / Mosaic GPU Blackwell matmul

The official JAX tutorial develops an FP16 `4096 × 4096 × 8192` Blackwell matmul through a basic kernel, warp specialization, tiled epilogue, collective two-CTA MMA, persistence, a dedicated epilogue warpgroup, and grid tiling.

For the tutorial's IID-normal inputs, its table reports tensor-core utilization progressing from 37.62% to 69.44%. The final value is 109.6% of its `jax.dot`/cuBLAS utilization and close to the page's CUTLASS-profiler result (69.30%). These are reproducible tutorial measurements with a stated input distribution and shape, not general Pallas-versus-cuBLAS results.

## Actual programming surface

The tutorial uses JAX Pallas Mosaic GPU APIs such as `plgpu.kernel`, `plgpu.emit_pipeline`, `plgpu.tcgen05_mma`, `plgpu.TMEM`, `plgpu.SMEM`, barriers, and `pl.core_map`/`plgpu.WarpMesh`. It does not use cuTile's `ct.bid`, `ct.load`, or `ct.store`; the former summary mixed the two DSLs.

The page explains that a Pallas Mosaic GPU thread of execution corresponds to a warpgroup, and later specializes individual warps. Two-CTA MMA, pipeline depths, and grid order are tied to its implementation. The former “collective MMA achieves ~5× speedup” and universal “first guide” claims were not supported by the table and were removed.

Consult the linked test file and current JAX version before reproducing the code.
