---
id: blog-blackwell-microbenchmarking
title: "Microbenchmarking NVIDIA's Blackwell Architecture"
author: Aaron Jarmusch et al.
url: https://arxiv.org/abs/2512.02189
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, fp4, fp8, fp6, gemm, wgmma, cluster, 2sm-cooperative]
retrieved_at: 2026-08-13
---

# Blackwell microbenchmarking paper

The paper reports measurements from its tested B200 system, including memory hierarchy, tcgen05 throughput/latency experiments, TMEM reads, decompression, and application comparisons with H200. A companion paper at arXiv:2507.10789 studies an RTX 5080 (SM120); its results must not be merged into SM100 specifications.

## Source-reported B200 results

For STREAM Triad, the paper reports 4.134--4.141 TB/s on its 4--16 GB
working sets and 7.42--7.48 TB/s (92.8--93.5% of peak) on its 64--128 GB
working sets. It also reports roughly 16 TB/s for its TMEM-read experiment and
more than 96% of the paper's theoretical peak for several isolated tensor-core
microbenchmarks. It reports its device as 148 enabled SMs. These describe the
tested sample and benchmark methodology, not every product configuration.

The paper also reports instruction-latency, cache-hit, decompression, GEMM, application, and energy measurements. Each comparison has its own operand shapes, clocks, software, and baseline; consult the tables in the paper rather than transferring one ratio to a different kernel.

## Interpretation boundary

`tcgen05.mma` can be issued by a designated thread, but its operation, memory visibility, and completion remain CTA/cluster scoped according to the PTX instruction form and mbarrier protocol. “One issuer” is not equivalent to ordinary warp-synchronous execution.

TMEM capacity and measured bandwidth do not make it automatically preferable to shared memory. TMEM is required for tcgen05 accumulator paths and useful where supported copies/lifetimes fit; the best staging strategy remains workload-specific.

The previous summary elevated microbenchmark observations into universal recommendations (“TMEM is essential,” “all bottlenecks are data movement”) and presented its tested SM count as a generic specification. Those inferences were removed.
