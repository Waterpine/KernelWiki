"""CuTe-DSL sparse MLA decode kernel for NVIDIA Blackwell.

The implementation intentionally consumes the native, separate compressed-KV
and RoPE caches.  One CTA computes one (query token, attention head) output.
Scores and the softmax normalization are accumulated in FP32; the compressed
cache is used as both the latent key and latent value, as required by MLA.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import make_ptr


_NUM_HEADS = 16
_LATENT_DIM = 512
_ROPE_DIM = 64
_TOPK = 2048
_PAGE_SIZE = 64
_THREADS = 256
_WARPS = _THREADS // 32
_SMEM_BYTES = _TOPK * 4 * 2 + 16 * 4


@cute.kernel
def _dsa_simt_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    sm_scale: Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(tidx // 32)
    token = block_idx // _NUM_HEADS
    head = block_idx % _NUM_HEADS

    smem = cutlass.utils.SmemAllocator()
    scores = smem.allocate_tensor(
        Float32, cute.make_layout((_TOPK,), stride=(1,)), 16
    )
    indices = smem.allocate_tensor(
        Int32, cute.make_layout((_TOPK,), stride=(1,)), 16
    )
    reduction = smem.allocate_tensor(
        Float32, cute.make_layout((16,), stride=(1,)), 16
    )

    # The query is reused for every selected token.  Keeping it in registers
    # avoids repeatedly issuing the same global loads in the score loop.
    q_frag = cute.make_rmem_tensor((_LATENT_DIM // 32 + _ROPE_DIM // 32,), Float32)
    for i in range(_LATENT_DIM // 32):
        q_frag[i] = Float32(q_nope[token, head, lane + i * 32])
    for i in range(_ROPE_DIM // 32):
        q_frag[_LATENT_DIM // 32 + i] = Float32(
            q_pe[token, head, lane + i * 32]
        )

    score_range = iket.range_start("score_gather_dot")

    # Eight warps independently compute one score at a time.  The index load
    # is warp-uniform, then retained in shared memory for the value pass.
    for it in range(_TOPK // _WARPS):
        pos = warp + it * _WARPS
        index = Int32(-1)
        if lane == 0:
            index = sparse_indices[token, pos]
            indices[pos] = index
        index = cute.arch.shuffle_sync(index, 0)

        score = Float32(0.0)
        if index >= 0:
            page = index // _PAGE_SIZE
            page_offset = index % _PAGE_SIZE
            for i in range(_LATENT_DIM // 32):
                dim = lane + i * 32
                score += q_frag[i] * Float32(ckv_cache[page, page_offset, dim])
            for i in range(_ROPE_DIM // 32):
                dim = lane + i * 32
                score += q_frag[_LATENT_DIM // 32 + i] * Float32(
                    kpe_cache[page, page_offset, dim]
                )

        for offset in [16, 8, 4, 2, 1]:
            score += cute.arch.shuffle_sync_bfly(
                score, offset=offset, mask=-1, mask_and_clamp=31
            )
        if lane == 0:
            scores[pos] = score * sm_scale if index >= 0 else -Float32.inf

    cute.arch.barrier()
    iket.range_end(softmax_range)

    value_range = iket.range_start("value_gather_accumulate")
    iket.range_end(score_range)

    softmax_range = iket.range_start("softmax")

    # Block-wide maximum over the 2048 score slots.
    local_max = -Float32.inf
    for i in range(_TOPK // _THREADS):
        local_max = cute.arch.fmax(local_max, scores[tidx + i * _THREADS])
    for offset in [16, 8, 4, 2, 1]:
        local_max = cute.arch.fmax(
            local_max,
            cute.arch.shuffle_sync_bfly(
                local_max, offset=offset, mask=-1, mask_and_clamp=31
            ),
        )
    if lane == 0:
        reduction[warp] = local_max
    cute.arch.barrier()

    if warp == 0:
        block_max = -Float32.inf
        if lane < _WARPS:
            block_max = reduction[lane]
        for offset in [16, 8, 4, 2, 1]:
            block_max = cute.arch.fmax(
                block_max,
                cute.arch.shuffle_sync_bfly(
                    block_max, offset=offset, mask=-1, mask_and_clamp=31
                ),
            )
        if lane == 0:
            reduction[_WARPS] = block_max
    cute.arch.barrier()
    block_max = reduction[_WARPS]

    # Exponentiate in FP32 and compute the block-wide denominator.
    local_sum = Float32(0.0)
    for i in range(_TOPK // _THREADS):
        pos = tidx + i * _THREADS
        weight = Float32(0.0)
        if scores[pos] != -Float32.inf:
            weight = cute.exp(scores[pos] - block_max)
        scores[pos] = weight
        local_sum += weight
    for offset in [16, 8, 4, 2, 1]:
        local_sum += cute.arch.shuffle_sync_bfly(
            local_sum, offset=offset, mask=-1, mask_and_clamp=31
        )
    if lane == 0:
        reduction[warp] = local_sum
    cute.arch.barrier()

    if warp == 0:
        block_sum = Float32(0.0)
        if lane < _WARPS:
            block_sum = reduction[lane]
        for offset in [16, 8, 4, 2, 1]:
            block_sum += cute.arch.shuffle_sync_bfly(
                block_sum, offset=offset, mask=-1, mask_and_clamp=31
            )
        if lane == 0:
            reduction[0] = block_sum
    cute.arch.barrier()
    block_sum = reduction[0]

    for i in range(_TOPK // _THREADS):
        pos = tidx + i * _THREADS
        normalized = Float32(0.0)
        if block_sum > Float32(0.0):
            normalized = scores[pos] / block_sum
        scores[pos] = normalized
    cute.arch.barrier()

    # Dimension-parallel value pass.  Each thread produces two adjacent
    # 256-wide halves, yielding coalesced cache loads and output stores.
    acc0 = Float32(0.0)
    acc1 = Float32(0.0)
    for pos in range(_TOPK):
        index = indices[pos]
        if index >= 0:
            page = index // _PAGE_SIZE
            page_offset = index % _PAGE_SIZE
            weight = scores[pos]
            acc0 += weight * Float32(ckv_cache[page, page_offset, tidx])
            acc1 += weight * Float32(
                ckv_cache[page, page_offset, tidx + _THREADS]
            )

    output[token, head, tidx] = BFloat16(acc0)
    output[token, head, tidx + _THREADS] = BFloat16(acc1)
    iket.range_end(value_range)


@cute.jit
def _launch(
    q_nope_ptr: cute.Pointer,
    q_pe_ptr: cute.Pointer,
    ckv_ptr: cute.Pointer,
    kpe_ptr: cute.Pointer,
    indices_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    num_tokens: Int32,
    num_pages: Int32,
    sm_scale: Float32,
    stream: cuda.CUstream,
):
    q_nope = cute.make_tensor(
        q_nope_ptr,
        cute.make_layout((num_tokens, _NUM_HEADS, _LATENT_DIM), stride=(8192, 512, 1)),
    )
    q_pe = cute.make_tensor(
        q_pe_ptr,
        cute.make_layout((num_tokens, _NUM_HEADS, _ROPE_DIM), stride=(1024, 64, 1)),
    )
    ckv_cache = cute.make_tensor(
        ckv_ptr,
        cute.make_layout((num_pages, _PAGE_SIZE, _LATENT_DIM), stride=(32768, 512, 1)),
    )
    kpe_cache = cute.make_tensor(
        kpe_ptr,
        cute.make_layout((num_pages, _PAGE_SIZE, _ROPE_DIM), stride=(4096, 64, 1)),
    )
    sparse_indices = cute.make_tensor(
        indices_ptr,
        cute.make_layout((num_tokens, _TOPK), stride=(_TOPK, 1)),
    )
    output = cute.make_tensor(
        output_ptr,
        cute.make_layout((num_tokens, _NUM_HEADS, _LATENT_DIM), stride=(8192, 512, 1)),
    )

    _dsa_dense_kernel(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        sm_scale,
    ).launch(
        grid=(num_tokens * _NUM_HEADS, 1, 1),
        block=(_THREADS, 1, 1),
        smem=_SMEM_BYTES,
        stream=stream,
    )
    _dsa_ragged_kernel(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        sm_scale,
    ).launch(
        grid=(num_tokens * _NUM_HEADS, 1, 1),
        block=(_THREADS, 1, 1),
        smem=_SMEM_BYTES,
        stream=stream,
    )


_compiled = None


def _compiled_kernel(stream: cuda.CUstream):
    global _compiled
    if _compiled is None:
        bf16_ptr = make_ptr(
            BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16
        )
        i32_ptr = make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
        _compiled = cute.compile(
            _launch,
            bf16_ptr,
            bf16_ptr,
            bf16_ptr,
            bf16_ptr,
            i32_ptr,
            bf16_ptr,
            Int32(1),
            Int32(1),
            Float32(1.0),
            stream,
        )
    return _compiled


def _ptr(dtype, tensor: torch.Tensor):
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute sparse MLA decode attention and return BF16 ``[T,16,512]``."""
    output = torch.empty_like(q_nope)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = _compiled_kernel(stream)
    kernel(
        _ptr(BFloat16, q_nope),
        _ptr(BFloat16, q_pe),
        _ptr(BFloat16, ckv_cache),
        _ptr(BFloat16, kpe_cache),
        _ptr(Int32, sparse_indices),
        _ptr(BFloat16, output),
        Int32(q_nope.shape[0]),
        Int32(ckv_cache.shape[0]),
        Float32(float(sm_scale)),
        stream,
    )
    return output
