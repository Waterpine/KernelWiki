# DeepSeek and Qwen Kernel Research Notes

> Factual refresh: 2026-08-13. These notes summarize the named primary sources. Canonical records live under `sources/` and `wiki/`.

## DeepGEMM

Primary source: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

The current project builds GPU kernels at runtime with a JIT system. Its documented Hopper FP8 path uses fine-grained activation and weight scaling and periodically promotes limited-precision tensor-core partial sums into FP32 CUDA-core accumulators. Its SM100 support requires the project's documented CUDA version and uses Blackwell tensor-core/TMEM paths.

The local pinned SM100 wrapper for PR 304 spells the block-scaled instruction family as `kind::mxf8f6f4.block_scale` and passes two scale descriptors. UE8M0 is relevant to MX-family scale representations; it must not be described as the NVFP4 E4M3/UE4M3 block scale.

The upstream headline of up to 1550 TFLOP/s is an H800 source-reported maximum. It is not evidence for an exact `4096³` shape or for B200.

## FlashMLA

Primary source: [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA).

The current README's support matrix is operation-specific:

- dense decode: SM90, MQA, BF16 KV cache;
- sparse decode: SM90/SM100, MQA, optional FP8 KV cache;
- dense prefill: SM100, MHA;
- sparse prefill: SM90/SM100, MQA.

For the optional FP8 sparse-decode cache, one documented token entry is 656 bytes: 512 E4M3 bytes, four FP32 scales (16 bytes), and 64 BF16 RoPE values (128 bytes). This is not the format of every FlashMLA mode.

The README reports different maxima for different configurations: dense decode on H800 reaches either bandwidth- or compute-oriented headline values; sparse decode reports 410 TFLOP/s on H800 and up to 350 TFLOP/s on B200 while noting the B200 path was not optimized; dense MHA prefill reports up to 1460 forward and 1000 backward TFLOP/s on B200; sparse prefill reports up to 640 on H800 and 1450 on B200. These are upstream measurements, not cross-mode comparisons.

## Native Sparse Attention

Primary source: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089).

NSA combines compressed tokens, selected token blocks, and a local sliding window. Its GQA groups share selections to make sparse access more hardware-aligned. The paper's system experiments use eight A100 GPUs. It reports up to 9× forward and 6× backward speedup at 64K sequence length, plus up to 11.6× decoding speedup in its respective setup. Those results are not H100 or B200 benchmarks, and later DeepSeek deployment claims require separate sources.

## Gated DeltaNet and chunking

Primary source: [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet).

The delta rule updates state from the error between a new value and the current value retrieved by its key. An unconditional `outer(k, v)` update is a different additive recurrence. Chunkwise implementations turn work inside chunks into matrix operations but retain an ordered boundary-state dependency. Chunk sizes and throughput claims are implementation-specific.

This source does not, by itself, establish later Qwen3.5 layer counts, context length, or throughput. Those facts must be taken from a separately dated primary model source.

## FlashAttention-4

Primary source: [arXiv:2603.05451](https://arxiv.org/abs/2603.05451).

The paper reports up to 1613 TFLOP/s (about 71% under its theoretical-peak convention) on a B200 BF16 sweep covering sequence lengths 1K–32K with 32K total tokens and documented head dimensions. It reports up to 1.3× over cuDNN 9.13 and up to 2.7× over its Triton baseline. These maxima should not be attached to one invented shape.

Its exponential strategy uses software approximation only for a minority of entries (about 10–25% in the paper's discussion), with the hardware exponential path for the rest. Its two-CTA backward is a documented coordinated MMA design, not permission for arbitrary CTAs to share a flat TMEM allocation.

## Blackwell programming facts

Primary sources: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/), and [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

- `tcgen05.fence::before_thread_sync` and `::after_thread_sync` provide ordering around asynchronous tensor operations; the documented completion sequence uses an mbarrier commitment/wait before TMEM load and `tcgen05.wait::ld` before reuse or deallocation.
- TMEM has 512 columns across 128 lanes, with 32-bit cells. Allocation is warp-collective, uses documented power-of-two column units, and must be explicitly deallocated.
- `clusterlaunchcontrol.try_cancel` attempts to cancel a not-yet-started cluster and returns a response through an mbarrier. It is not an arbitrary tile queue.
- ordinary `compute_100` PTX has the compatibility behavior documented for PTX; accelerated `compute_100a`/`sm_100a` features are architecture-specific and not forward- or backward-compatible.
- CUDA 12.8 was the first toolkit with native Blackwell target support; later toolkit versions do not change that historical first-support fact.

## Version snapshot

As accessed 2026-08-13, CUTLASS documentation lists 4.7.0 (2026-08-04), while cuTile documentation lists 1.5.0 (2026-07-08). Consult the repository's version-claims and refresh-cutoff records before repeating any “current” statement.
