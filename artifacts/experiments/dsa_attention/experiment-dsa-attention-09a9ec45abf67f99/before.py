"""CuTe-DSL sparse MLA decode attention for DeepSeek-V3.2.

The initial path is a fused SIMT online-softmax kernel.  A warp owns one
query/head pair and keeps its 512-dimensional output fragment in registers.
Eight warps share a CTA so heads for the same token execute together and reuse
the gathered cache lines through L1.  The sparse index list is a valid prefix
followed by ``-1`` padding, so no separate length/preparation kernel is needed.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


_HEADS = 16
_HEADS_PER_CTA = 8
_WARPS_PER_CTA = _HEADS_PER_CTA
_THREADS = _WARPS_PER_CTA * 32
_TOPK = 2048
_NOPE_DIM = 512
_PE_DIM = 64
_VALUES_PER_LANE = _NOPE_DIM // 32
_PE_VALUES_PER_LANE = _PE_DIM // 32


@cute.kernel
def _dsa_simt_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    sm_scale: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.warp_idx()
    warp = cute.arch.make_warp_uniform(warp)
    block, _, _ = cute.arch.block_idx()

    linear_head = block * _HEADS_PER_CTA + warp
    token = linear_head // _HEADS
    head = linear_head % _HEADS

    # Query and online-softmax output fragments remain resident for the whole
    # sparse row.  Each lane owns dimensions lane + 32*j.
    r_q = cute.make_rmem_tensor((_VALUES_PER_LANE,), cutlass.Float32)
    r_out = cute.make_rmem_tensor((_VALUES_PER_LANE,), cutlass.Float32)
    r_kv = cute.make_rmem_tensor((_VALUES_PER_LANE,), cutlass.Float32)
    r_qpe = cute.make_rmem_tensor((_PE_VALUES_PER_LANE,), cutlass.Float32)

    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
        dim = lane + j * 32
        r_q[j] = cutlass.Float32(q_nope[token, head, dim])
        r_out[j] = cutlass.Float32(0.0)

    for j in cutlass.range_constexpr(_PE_VALUES_PER_LANE):
        dim = lane + j * 32
        r_qpe[j] = cutlass.Float32(q_pe[token, head, dim])

    row_max = cutlass.Float32(-3.402823466e38)
    row_sum = cutlass.Float32(0.0)
    pos = cutlass.Int32(0)

    # All lanes observe the same index and therefore take uniform control flow.
    # Setting pos to TOPK on the first sentinel gives an early exit without a
    # separate length scan or a second launch.
    while pos < _TOPK:
        token_idx = sparse_indices[token, pos]
        if token_idx < 0:
            pos = cutlass.Int32(_TOPK)
        else:
            page = token_idx // 64
            page_offset = token_idx - page * 64

            score = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = lane + j * 32
                kv = cutlass.Float32(ckv_cache[page, page_offset, dim])
                r_kv[j] = kv
                score += r_q[j] * kv

            for j in cutlass.range_constexpr(_PE_VALUES_PER_LANE):
                dim = lane + j * 32
                score += r_qpe[j] * cutlass.Float32(
                    kpe_cache[page, page_offset, dim]
                )

            # Full-warp dot-product reduction; every lane receives the score so
            # the online softmax state stays replicated and synchronization-free.
            for offset in [16, 8, 4, 2, 1]:
                score += cute.arch.shuffle_sync_bfly(
                    score, offset=offset, mask=-1, mask_and_clamp=31
                )
            score *= sm_scale

            new_max = score if score > row_max else row_max
            old_scale = cute.exp(row_max - new_max)
            probability = cute.exp(score - new_max)
            row_sum = row_sum * old_scale + probability

            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                r_out[j] = r_out[j] * old_scale + probability * r_kv[j]

            row_max = new_max
            pos += 1

    inv_sum = cutlass.Float32(1.0) / row_sum
    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
        dim = lane + j * 32
        output[token, head, dim] = cutlass.BFloat16(r_out[j] * inv_sum)


@cute.jit
def _launch_simt(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    sm_scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    num_tokens = q_nope.layout.shape[0]
    grid = (cute.ceil_div(num_tokens * _HEADS, _HEADS_PER_CTA), 1, 1)
    _dsa_simt_kernel(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        sm_scale,
    ).launch(
        grid=grid,
        block=(_THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


@functools.lru_cache(maxsize=32)
def _compile_simt(
    num_tokens: int,
    num_pages: int,
    device_index: int,
):
    # Compilation needs representative argument layouts.  These tiny placeholders
    # are used only to specialize the ABI; the returned function receives the
    # actual workload tensors on every invocation.
    device = torch.device("cuda", device_index)
    q_nope = torch.empty(
        (num_tokens, _HEADS, _NOPE_DIM), dtype=torch.bfloat16, device=device
    )
    q_pe = torch.empty(
        (num_tokens, _HEADS, _PE_DIM), dtype=torch.bfloat16, device=device
    )
    ckv = torch.empty((num_pages, 64, _NOPE_DIM), dtype=torch.bfloat16, device=device)
    kpe = torch.empty((num_pages, 64, _PE_DIM), dtype=torch.bfloat16, device=device)
    indices = torch.empty((num_tokens, _TOPK), dtype=torch.int32, device=device)
    output = torch.empty_like(q_nope)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    args = tuple(_as_cute(x) for x in (q_nope, q_pe, ckv, kpe, indices, output))
    return cute.compile(
        _launch_simt,
        *args,
        cutlass.Float32(1.0),
        stream,
        options="--generate-line-info",
    )


def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Compute sparse MLA decode attention and return BF16 ``[T, 16, 512]``."""
    output = torch.empty_like(q_nope)
    device_index = q_nope.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compile_simt(q_nope.shape[0], ckv_cache.shape[0], device_index)
    stream = cuda.CUstream(torch.cuda.current_stream(q_nope.device).cuda_stream)
    args = tuple(
        _as_cute(x)
        for x in (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output)
    )
    compiled(*args, cutlass.Float32(sm_scale), stream)
    return output
