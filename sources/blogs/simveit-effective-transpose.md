---
id: blog-simveit-effective-transpose
title: simveit effective_transpose
author: Simon Veitner
url: https://github.com/simveit/effective_transpose
source_category: community-note
architectures:
- sm90
- sm100
tags:
- tma
- swizzling
- vectorized-loads
- shared-memory-optimization
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for TMA transpose, swizzle, and memory-layout examples.
---

The effective_transpose repository provides compact CUDA C++ transpose and
layout examples built on TMA (`cuda::device::experimental` bulk-tensor copies)
and shared-memory swizzling. Use it when a kernel profile indicates poor sector utilization,
shared-memory bank conflicts, or a need to reason about tiled load/store
layouts.
