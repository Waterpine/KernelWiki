---
id: kernel-kda-mlsys26-final
title: MLSys26 final B300 kernels — GDN prefill, DSA, FP8 MoE
type: kernel
architectures: [sm103]
tags: [gated-delta-net, sparse-attention, moe, fp8, cute-dsl]
confidence: source-reported
reproducibility: runnable
kernel_types: [gated-delta-net, prefill, sparse-attention, moe]
languages: [cute-dsl, python]
related: [kernel-gated-delta-net, kernel-flash-attention-sm100-mla-topk, kernel-fused-moe]
sources: []
performance_claims: []
blackwell_relevance: Complete selected CuTe DSL implementations for three B300 MLSys26 workloads.
artifact_dir: artifacts/kernels/kda-mlsys26-final/full
---

# Audited MLSys26 final kernels

This bundle contains the selected GDN prefill, DSA sparse attention, and FP8
MoE kernels from the archived 2026-08-18 `main` snapshot. The three entry
points are [GDN](../../artifacts/kernels/kda-mlsys26-final/full/kernels/gdn_prefill/kernel.py),
[DSA](../../artifacts/kernels/kda-mlsys26-final/full/kernels/dsa_attention/kernel.py),
and [MoE](../../artifacts/kernels/kda-mlsys26-final/full/kernels/moe/kernel.py).
Their sibling modules are in the same artifact bundle.

The archived B300 acceptance report records GDN at 15.1598× (100/100), DSA at
37.6827× (23/23), and MoE at 2.2227× (19/19) arithmetic-mean speedup. See the
[selection and provenance notes](../../data/internal-b300/mlsys26-final.md)
for the benchmark method, integrity audit, and limits of the retained evidence.
The bundle has not been rebenchmarked in this checkout.

Retrieve the complete local source bundle with:

```bash
python3 scripts/query.py --type kernel --has-code 'MLSys26 final'
python3 scripts/get_page.py kernel-kda-mlsys26-final --include-code
```

The [GDN](gated-delta-net.md),
[DSA](flash-attention-sm100-mla-topk.md), and [MoE](fused-moe.md) case studies
describe earlier optimization trajectories and link to these final snapshots.
