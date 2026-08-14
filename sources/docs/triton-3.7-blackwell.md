---
id: doc-triton-3.7-blackwell
title: "Triton 3.7.1 and Blackwell"
url: https://github.com/triton-lang/triton/releases/tag/v3.7.1
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, block-scale, warp-specialization, tma]
retrieved_at: 2026-08-13
---

# Triton 3.7.1 and Blackwell

Triton 3.7.0 was released 2026-05-07 and 3.7.1 on 2026-06-18. The 3.7.1 notes describe it as a regression-fix patch over 3.7.0 with no new features or API changes.

Blackwell-relevant 3.7.0 changes include end-to-end Gluon two-CTA work, TMA multicast support, tcgen05 MMA multicast, TMEM deallocation timing fixes, and additional warp-specialization correctness and scheduling work. These extend Blackwell paths already present before 3.6 and expanded in 3.6.

Current official tutorials document two distinct surfaces:

- classic Triton `tl.range(..., warp_specialize=True)` for supported Blackwell matmul loops and `tl.dot_scaled` for microscaling formats;
- explicit Gluon APIs for Blackwell TMEM, tcgen05 MMA/scaled MMA, barriers, warp specialization, and multi-CTA kernels.

The API docs warn that automatic loop warp specialization currently applies to supported simple matmul loops. `tl.dot_scaled` may use software emulation on hardware without native support. Therefore “supported by Triton” is not the same claim as “uses one fixed instruction sequence on every target.”

## Official links

- [3.7.1 release](https://github.com/triton-lang/triton/releases/tag/v3.7.1)
- [3.7.0 release](https://github.com/triton-lang/triton/releases/tag/v3.7.0)
- [Triton release history](https://github.com/triton-lang/triton/blob/main/RELEASE.md)
- [Blackwell block-scaled matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [Gluon tcgen05 scaled-MMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html)
