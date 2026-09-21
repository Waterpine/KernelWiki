"""CuTe-DSL sparse MLA decode attention for Blackwell.

The initial path is deliberately launch- and padding-efficient: one warp owns a
query/head pair, keeps its 512-dimensional output slice in registers, and stops
as soon as the trailing -1 padding begins.  The compressed KV cache is both the
key and value, so every 16-element vector loaded by a lane is reused for the dot
product and the online-softmax value update.
"""

from __future__ import annotations

import functools
import math

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64


_HEADS = 16
_NOPE_DIM = 512
_PE_DIM = 64
_TOPK = 2048
_VALUES_PER_LANE = 16
_INDEX_TILE = 8
_LOG2_E = math.log2(math.e)
_SPLIT_TOKEN_SLOTS = 128
_PARTIAL_NUM_ELEMS = _SPLIT_TOKEN_SLOTS * _HEADS * _NOPE_DIM
_PARTIAL_STATS_ELEMS = _SPLIT_TOKEN_SLOTS * _HEADS * 2
_SHARED_SPLIT_HEADS = 2
_SHARED_SPLIT_THREADS = 32 * _SHARED_SPLIT_HEADS
_SHARED_SPLIT_TILE = 16
_DENSE_HEADS_PER_CTA = 2
_DENSE_TILE_KEYS = 16
_DENSE_THREADS = 32 * _DENSE_HEADS_PER_CTA


def _make_launcher():
    @cute.kernel
    def _kernel(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_out: cute.Tensor,
        sm_scale: Float32,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)

        work_idx = bidx
        token = work_idx // Int32(_HEADS)
        head = work_idx - token * Int32(_HEADS)

        if token < num_tokens:
            q_nope_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_NOPE_DIM)
            )
            r_q_nope = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                r_q_nope[j] = (m_q_nope.iterator + q_nope_offset + dim).load()

            q_pe_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_PE_DIM)
                + Int64(lane) * Int64(2)
            )
            q_pe_0 = (m_q_pe.iterator + q_pe_offset).load()
            q_pe_1 = (m_q_pe.iterator + q_pe_offset + Int64(1)).load()

            r_acc = cute.make_rmem_tensor(_VALUES_PER_LANE, Float32)
            r_acc.fill(0.0)
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            softmax_scale_log2 = sm_scale * Float32(_LOG2_E)

            topk_pos = Int32(0)
            while topk_pos < Int32(_TOPK):
                index_offset = Int64(token) * Int64(_TOPK) + Int64(topk_pos)
                kv_idx = (m_indices.iterator + index_offset).load()

                if kv_idx < Int32(0):
                    # Sparse-index padding is a trailing suffix.  Jumping to the
                    # loop bound keeps tiny real requests proportional to nvalid.
                    topk_pos = Int32(_TOPK)
                else:
                    kv_offset = (
                        Int64(kv_idx) * Int64(_NOPE_DIM)
                    )
                    r_kv = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
                    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                        dim = Int64(lane) + Int64(j * 32)
                        r_kv[j] = (m_ckv.iterator + kv_offset + dim).load()

                    score_0 = Float32(0.0)
                    score_1 = Float32(0.0)
                    for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                        score_0, score_1 = cute.arch.fma_packed_f32x2(
                            (Float32(r_q_nope[j]), Float32(r_q_nope[j + 1])),
                            (Float32(r_kv[j]), Float32(r_kv[j + 1])),
                            (score_0, score_1),
                        )

                    kpe_offset = Int64(kv_idx) * Int64(_PE_DIM) + Int64(lane) * Int64(2)
                    kpe_0 = (m_kpe.iterator + kpe_offset).load()
                    kpe_1 = (m_kpe.iterator + kpe_offset + Int64(1)).load()
                    score_0, score_1 = cute.arch.fma_packed_f32x2(
                        (Float32(q_pe_0), Float32(q_pe_1)),
                        (Float32(kpe_0), Float32(kpe_1)),
                        (score_0, score_1),
                    )
                    score = score_0 + score_1
                    score = cute.arch.warp_reduction_sum(score)

                    if topk_pos == Int32(0):
                        row_max = score
                        row_sum = Float32(1.0)
                        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                            r_acc[j] = Float32(r_kv[j])
                    else:
                        new_max = cute.arch.fmax(row_max, score)
                        old_scale = cute.math.exp2(
                            (row_max - new_max) * softmax_scale_log2, fastmath=True
                        )
                        weight = cute.math.exp2(
                            (score - new_max) * softmax_scale_log2, fastmath=True
                        )
                        row_sum = row_sum * old_scale + weight
                        for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                            value_0, value_1 = cute.arch.mul_packed_f32x2(
                                (Float32(r_kv[j]), Float32(r_kv[j + 1])),
                                (weight, weight),
                            )
                            r_acc[j], r_acc[j + 1] = cute.arch.fma_packed_f32x2(
                                (r_acc[j], r_acc[j + 1]),
                                (old_scale, old_scale),
                                (value_0, value_1),
                            )
                        row_max = new_max

                    topk_pos += Int32(1)

            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)

            r_out = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                out_0, out_1 = cute.arch.mul_packed_f32x2(
                    (r_acc[j], r_acc[j + 1]), (inv_sum, inv_sum)
                )
                r_out[j] = out_0.to(BFloat16)
                r_out[j + 1] = out_1.to(BFloat16)

            out_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_NOPE_DIM)
            )
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                (m_out.iterator + out_offset + dim).store(r_out[j])

    @cute.jit
    def _launch(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_out: cute.Tensor,
        stream: cuda.CUstream,
        sm_scale: Float32,
        num_tokens: Int32,
        grid_x: Int32,
    ):
        _kernel(
            m_q_nope,
            m_q_pe,
            m_ckv,
            m_kpe,
            m_indices,
            m_out,
            sm_scale,
            num_tokens,
        ).launch(grid=[grid_x, 1, 1], block=[32, 1, 1], stream=stream)

    return _launch


def _make_split_launcher():
    @cute.kernel
    def _split_kernel(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_partial_num: cute.Tensor,
        m_partial_stats: cute.Tensor,
        sm_scale: Float32,
        num_tokens: Int32,
        num_splits: Int32,
        keys_per_split: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)

        split = bidx % num_splits
        token_head_group = bidx // num_splits
        head_groups = Int32(_HEADS // _SHARED_SPLIT_HEADS)
        head_group = token_head_group % head_groups
        token = token_head_group // head_groups
        head = head_group * Int32(_SHARED_SPLIT_HEADS) + warp
        token_head = token * Int32(_HEADS) + head

        smem = cutlass.utils.SmemAllocator()
        s_indices = smem.allocate_tensor(Int32, _SHARED_SPLIT_TILE, 16)
        s_ckv = smem.allocate_tensor(
            BFloat16, _SHARED_SPLIT_TILE * _NOPE_DIM, 128
        )
        s_kpe = smem.allocate_tensor(
            BFloat16, _SHARED_SPLIT_TILE * _PE_DIM, 128
        )

        if token < num_tokens:
            q_nope_offset = Int64(token_head) * Int64(_NOPE_DIM)
            r_q_nope = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                r_q_nope[j] = (m_q_nope.iterator + q_nope_offset + dim).load()

            q_pe_offset = Int64(token_head) * Int64(_PE_DIM) + Int64(lane) * Int64(2)
            q_pe_0 = (m_q_pe.iterator + q_pe_offset).load()
            q_pe_1 = (m_q_pe.iterator + q_pe_offset + Int64(1)).load()

            r_acc = cute.make_rmem_tensor(_VALUES_PER_LANE, Float32)
            r_acc.fill(0.0)
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            softmax_scale_log2 = sm_scale * Float32(_LOG2_E)

            split_begin = split * keys_per_split
            split_end = split_begin + keys_per_split
            topk_pos = split_begin
            while topk_pos < split_end:
                loaded_index = Int32(-1)
                if lane < Int32(_INDEX_TILE):
                    index_offset = (
                        Int64(token) * Int64(_TOPK)
                        + Int64(topk_pos)
                        + Int64(lane)
                    )
                    loaded_index = (m_indices.iterator + index_offset).load()

                for index_slot in cutlass.range_constexpr(_INDEX_TILE):
                    kv_idx = cute.arch.shuffle_sync(loaded_index, index_slot)
                    if topk_pos < split_end:
                        if kv_idx < Int32(0):
                            topk_pos = split_end
                        else:
                            kv_offset = Int64(kv_idx) * Int64(_NOPE_DIM)
                            r_kv = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
                            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                                dim = Int64(lane) + Int64(j * 32)
                                r_kv[j] = (
                                    m_ckv.iterator + kv_offset + dim
                                ).load()

                            score_0 = Float32(0.0)
                            score_1 = Float32(0.0)
                            for j in cutlass.range_constexpr(
                                0, _VALUES_PER_LANE, 2
                            ):
                                score_0, score_1 = cute.arch.fma_packed_f32x2(
                                    (
                                        Float32(r_q_nope[j]),
                                        Float32(r_q_nope[j + 1]),
                                    ),
                                    (Float32(r_kv[j]), Float32(r_kv[j + 1])),
                                    (score_0, score_1),
                                )

                            kpe_offset = (
                                Int64(kv_idx) * Int64(_PE_DIM)
                                + Int64(lane) * Int64(2)
                            )
                            kpe_0 = (m_kpe.iterator + kpe_offset).load()
                            kpe_1 = (
                                m_kpe.iterator + kpe_offset + Int64(1)
                            ).load()
                            score_0, score_1 = cute.arch.fma_packed_f32x2(
                                (Float32(q_pe_0), Float32(q_pe_1)),
                                (Float32(kpe_0), Float32(kpe_1)),
                                (score_0, score_1),
                            )
                            score = cute.arch.warp_reduction_sum(score_0 + score_1)

                            if topk_pos == split_begin:
                                row_max = score
                                row_sum = Float32(1.0)
                                for j in cutlass.range_constexpr(
                                    _VALUES_PER_LANE
                                ):
                                    r_acc[j] = Float32(r_kv[j])
                            else:
                                new_max = cute.arch.fmax(row_max, score)
                                old_scale = cute.math.exp2(
                                    (row_max - new_max) * softmax_scale_log2,
                                    fastmath=True,
                                )
                                weight = cute.math.exp2(
                                    (score - new_max) * softmax_scale_log2,
                                    fastmath=True,
                                )
                                row_sum = row_sum * old_scale + weight
                                for j in cutlass.range_constexpr(
                                    0, _VALUES_PER_LANE, 2
                                ):
                                    value_0, value_1 = cute.arch.mul_packed_f32x2(
                                        (
                                            Float32(r_kv[j]),
                                            Float32(r_kv[j + 1]),
                                        ),
                                        (weight, weight),
                                    )
                                    r_acc[j], r_acc[j + 1] = (
                                        cute.arch.fma_packed_f32x2(
                                            (r_acc[j], r_acc[j + 1]),
                                            (old_scale, old_scale),
                                            (value_0, value_1),
                                        )
                                    )
                                row_max = new_max

                            topk_pos += Int32(1)

            partial_row = token_head * num_splits + split
            partial_offset = Int64(partial_row) * Int64(_NOPE_DIM)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                (m_partial_num.iterator + partial_offset + dim).store(r_acc[j])

            if lane == Int32(0):
                stats_offset = Int64(partial_row) * Int64(2)
                (m_partial_stats.iterator + stats_offset).store(row_max)
                if first_kv_idx >= Int32(0):
                    (m_partial_stats.iterator + stats_offset + Int64(1)).store(row_sum)

    @cute.kernel
    def _merge_kernel(
        m_partial_num: cute.Tensor,
        m_partial_stats: cute.Tensor,
        m_out: cute.Tensor,
        sm_scale: Float32,
        num_rows: Int32,
        num_splits: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)

        if row < num_rows:
            global_max = -Float32.inf
            split = Int32(0)
            active_splits = Int32(0)
            while split < num_splits:
                partial_row = row * num_splits + split
                stats_offset = Int64(partial_row) * Int64(2)
                local_max = (m_partial_stats.iterator + stats_offset).load()
                if local_max == -Float32.inf:
                    split = num_splits
                else:
                    global_max = cute.arch.fmax(global_max, local_max)
                    active_splits += Int32(1)
                    split += Int32(1)

            r_acc = cute.make_rmem_tensor(_VALUES_PER_LANE, Float32)
            r_acc.fill(0.0)
            denominator = Float32(0.0)
            softmax_scale_log2 = sm_scale * Float32(_LOG2_E)

            split = Int32(0)
            while split < active_splits:
                partial_row = row * num_splits + split
                stats_offset = Int64(partial_row) * Int64(2)
                local_max = (m_partial_stats.iterator + stats_offset).load()
                local_sum = (
                    m_partial_stats.iterator + stats_offset + Int64(1)
                ).load()
                factor = cute.math.exp2(
                    (local_max - global_max) * softmax_scale_log2,
                    fastmath=True,
                )
                denominator += factor * local_sum
                partial_offset = Int64(partial_row) * Int64(_NOPE_DIM)
                for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                    dim_0 = Int64(lane) + Int64(j * 32)
                    dim_1 = Int64(lane) + Int64((j + 1) * 32)
                    value_0 = (m_partial_num.iterator + partial_offset + dim_0).load()
                    value_1 = (m_partial_num.iterator + partial_offset + dim_1).load()
                    r_acc[j], r_acc[j + 1] = cute.arch.fma_packed_f32x2(
                        (value_0, value_1),
                        (factor, factor),
                        (r_acc[j], r_acc[j + 1]),
                    )
                split += Int32(1)

            inv_sum = Float32(0.0)
            if denominator > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(denominator)

            out_offset = Int64(row) * Int64(_NOPE_DIM)
            for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                out_0, out_1 = cute.arch.mul_packed_f32x2(
                    (r_acc[j], r_acc[j + 1]), (inv_sum, inv_sum)
                )
                dim_0 = Int64(lane) + Int64(j * 32)
                dim_1 = Int64(lane) + Int64((j + 1) * 32)
                (m_out.iterator + out_offset + dim_0).store(out_0.to(BFloat16))
                (m_out.iterator + out_offset + dim_1).store(out_1.to(BFloat16))

    @cute.jit
    def _launch(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_partial_num: cute.Tensor,
        m_partial_stats: cute.Tensor,
        m_out: cute.Tensor,
        stream: cuda.CUstream,
        sm_scale: Float32,
        num_tokens: Int32,
        num_splits: Int32,
        keys_per_split: Int32,
        split_grid: Int32,
        merge_grid: Int32,
    ):
        _split_kernel(
            m_q_nope,
            m_q_pe,
            m_ckv,
            m_kpe,
            m_indices,
            m_partial_num,
            m_partial_stats,
            sm_scale,
            num_tokens,
            num_splits,
            keys_per_split,
        ).launch(grid=[split_grid, 1, 1], block=[32, 1, 1], stream=stream)
        _merge_kernel(
            m_partial_num,
            m_partial_stats,
            m_out,
            sm_scale,
            merge_grid,
            num_splits,
        ).launch(grid=[merge_grid, 1, 1], block=[32, 1, 1], stream=stream)

    return _launch


def _make_dense_shared_launcher():
    """Build a dense path that loads each KV row once for a pair of heads."""

    @cute.kernel
    def _kernel(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_out: cute.Tensor,
        sm_scale: Float32,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)

        head_groups = Int32(_HEADS // _DENSE_HEADS_PER_CTA)
        token = bidx // head_groups
        head_group = bidx - token * head_groups
        head = head_group * Int32(_DENSE_HEADS_PER_CTA) + warp

        smem = cutlass.utils.SmemAllocator()
        s_indices = smem.allocate_tensor(Int32, _DENSE_TILE_KEYS, 16)
        s_ckv = smem.allocate_tensor(
            BFloat16, _DENSE_TILE_KEYS * _NOPE_DIM, 128
        )
        s_kpe = smem.allocate_tensor(BFloat16, _DENSE_TILE_KEYS * _PE_DIM, 128)

        if token < num_tokens:
            q_nope_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_NOPE_DIM)
            )
            r_q_nope = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                r_q_nope[j] = (m_q_nope.iterator + q_nope_offset + dim).load()

            q_pe_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_PE_DIM)
                + Int64(lane) * Int64(2)
            )
            q_pe_0 = (m_q_pe.iterator + q_pe_offset).load()
            q_pe_1 = (m_q_pe.iterator + q_pe_offset + Int64(1)).load()

            r_acc = cute.make_rmem_tensor(_VALUES_PER_LANE, Float32)
            r_acc.fill(0.0)
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            softmax_scale_log2 = sm_scale * Float32(_LOG2_E)

            tile_start = Int32(0)
            while tile_start < Int32(_TOPK):
                if tidx < Int32(_DENSE_TILE_KEYS):
                    index_offset = (
                        Int64(token) * Int64(_TOPK)
                        + Int64(tile_start)
                        + Int64(tidx)
                    )
                    kv_idx = (m_indices.iterator + index_offset).load()
                    (s_indices.iterator + Int64(tidx)).store(kv_idx)

                # This barrier also prevents the next tile's shared-memory
                # loads from racing any warp still consuming the prior tile.
                cute.arch.sync_threads()

                for key_i in cutlass.range_constexpr(_DENSE_TILE_KEYS):
                    kv_idx = (s_indices.iterator + Int64(key_i)).load()
                    if kv_idx >= Int32(0):
                        kv_offset = Int64(kv_idx) * Int64(_NOPE_DIM)
                        for j in cutlass.range_constexpr(
                            _NOPE_DIM // _DENSE_THREADS
                        ):
                            dim = Int64(tidx) + Int64(j * _DENSE_THREADS)
                            value = (m_ckv.iterator + kv_offset + dim).load()
                            shared_offset = Int64(key_i * _NOPE_DIM) + dim
                            (s_ckv.iterator + shared_offset).store(value)

                        pe_dim = Int64(tidx)
                        pe_offset = Int64(kv_idx) * Int64(_PE_DIM) + pe_dim
                        pe_value = (m_kpe.iterator + pe_offset).load()
                        shared_pe_offset = Int64(key_i * _PE_DIM) + pe_dim
                        (s_kpe.iterator + shared_pe_offset).store(pe_value)

                cute.arch.sync_threads()

                for key_i in cutlass.range_constexpr(_DENSE_TILE_KEYS):
                    kv_idx = (s_indices.iterator + Int64(key_i)).load()
                    if kv_idx >= Int32(0):
                        r_kv = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
                        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                            dim = Int64(lane) + Int64(j * 32)
                            shared_offset = Int64(key_i * _NOPE_DIM) + dim
                            r_kv[j] = (s_ckv.iterator + shared_offset).load()

                        score_0 = Float32(0.0)
                        score_1 = Float32(0.0)
                        for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                            score_0, score_1 = cute.arch.fma_packed_f32x2(
                                (
                                    Float32(r_q_nope[j]),
                                    Float32(r_q_nope[j + 1]),
                                ),
                                (Float32(r_kv[j]), Float32(r_kv[j + 1])),
                                (score_0, score_1),
                            )

                        shared_pe_offset = (
                            Int64(key_i * _PE_DIM) + Int64(lane) * Int64(2)
                        )
                        kpe_0 = (s_kpe.iterator + shared_pe_offset).load()
                        kpe_1 = (s_kpe.iterator + shared_pe_offset + Int64(1)).load()
                        score_0, score_1 = cute.arch.fma_packed_f32x2(
                            (Float32(q_pe_0), Float32(q_pe_1)),
                            (Float32(kpe_0), Float32(kpe_1)),
                            (score_0, score_1),
                        )
                        score = cute.arch.warp_reduction_sum(score_0 + score_1)

                        topk_pos = tile_start + Int32(key_i)
                        if topk_pos == Int32(0):
                            row_max = score
                            row_sum = Float32(1.0)
                            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                                r_acc[j] = Float32(r_kv[j])
                        else:
                            new_max = cute.arch.fmax(row_max, score)
                            old_scale = cute.math.exp2(
                                (row_max - new_max) * softmax_scale_log2,
                                fastmath=True,
                            )
                            weight = cute.math.exp2(
                                (score - new_max) * softmax_scale_log2,
                                fastmath=True,
                            )
                            row_sum = row_sum * old_scale + weight
                            for j in cutlass.range_constexpr(
                                0, _VALUES_PER_LANE, 2
                            ):
                                value_0, value_1 = cute.arch.mul_packed_f32x2(
                                    (Float32(r_kv[j]), Float32(r_kv[j + 1])),
                                    (weight, weight),
                                )
                                r_acc[j], r_acc[j + 1] = (
                                    cute.arch.fma_packed_f32x2(
                                        (r_acc[j], r_acc[j + 1]),
                                        (old_scale, old_scale),
                                        (value_0, value_1),
                                    )
                                )
                            row_max = new_max

                tile_start += Int32(_DENSE_TILE_KEYS)

            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)

            r_out = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(0, _VALUES_PER_LANE, 2):
                out_0, out_1 = cute.arch.mul_packed_f32x2(
                    (r_acc[j], r_acc[j + 1]), (inv_sum, inv_sum)
                )
                r_out[j] = out_0.to(BFloat16)
                r_out[j + 1] = out_1.to(BFloat16)

            out_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_NOPE_DIM)
            )
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                dim = Int64(lane) + Int64(j * 32)
                (m_out.iterator + out_offset + dim).store(r_out[j])

    @cute.jit
    def _launch(
        m_q_nope: cute.Tensor,
        m_q_pe: cute.Tensor,
        m_ckv: cute.Tensor,
        m_kpe: cute.Tensor,
        m_indices: cute.Tensor,
        m_out: cute.Tensor,
        stream: cuda.CUstream,
        sm_scale: Float32,
        num_tokens: Int32,
        grid_x: Int32,
    ):
        _kernel(
            m_q_nope,
            m_q_pe,
            m_ckv,
            m_kpe,
            m_indices,
            m_out,
            sm_scale,
            num_tokens,
        ).launch(
            grid=[grid_x, 1, 1], block=[_DENSE_THREADS, 1, 1], stream=stream
        )

    return _launch


@functools.lru_cache(maxsize=1)
def _compile_kernel():
    launcher = _make_launcher()
    sym = cute.sym_int
    sym64 = cute.sym_int64

    q_nope = cute.runtime.make_fake_tensor(
        BFloat16, (sym(), _HEADS, _NOPE_DIM), stride=(_HEADS * _NOPE_DIM, _NOPE_DIM, 1)
    )
    q_pe = cute.runtime.make_fake_tensor(
        BFloat16, (sym(), _HEADS, _PE_DIM), stride=(_HEADS * _PE_DIM, _PE_DIM, 1)
    )
    ckv = cute.runtime.make_fake_tensor(
        BFloat16, (sym(), 64, _NOPE_DIM), stride=(64 * _NOPE_DIM, _NOPE_DIM, 1)
    )
    kpe = cute.runtime.make_fake_tensor(
        BFloat16, (sym(), 64, _PE_DIM), stride=(64 * _PE_DIM, _PE_DIM, 1)
    )
    indices = cute.runtime.make_fake_tensor(Int32, (sym(), _TOPK), stride=(_TOPK, 1))
    out = cute.runtime.make_fake_tensor(
        BFloat16, (sym(), _HEADS, _NOPE_DIM), stride=(_HEADS * _NOPE_DIM, _NOPE_DIM, 1)
    )

    # Keep the stream implicit in the TVM-FFI environment so launches follow
    # PyTorch's current stream without a Python-side stream query on every call.
    return cute.compile(
        launcher,
        q_nope,
        q_pe,
        ckv,
        kpe,
        indices,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Float32(0.0),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=1)
def _compile_dense_shared_kernel():
    launcher = _make_dense_shared_launcher()
    sym = cute.sym_int

    q_nope = cute.runtime.make_fake_tensor(
        BFloat16,
        (sym(), _HEADS, _NOPE_DIM),
        stride=(_HEADS * _NOPE_DIM, _NOPE_DIM, 1),
    )
    q_pe = cute.runtime.make_fake_tensor(
        BFloat16,
        (sym(), _HEADS, _PE_DIM),
        stride=(_HEADS * _PE_DIM, _PE_DIM, 1),
    )
    ckv = cute.runtime.make_fake_tensor(
        BFloat16,
        (sym(), 64, _NOPE_DIM),
        stride=(64 * _NOPE_DIM, _NOPE_DIM, 1),
    )
    kpe = cute.runtime.make_fake_tensor(
        BFloat16,
        (sym(), 64, _PE_DIM),
        stride=(64 * _PE_DIM, _PE_DIM, 1),
    )
    indices = cute.runtime.make_fake_tensor(
        Int32, (sym(), _TOPK), stride=(_TOPK, 1)
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (sym(), _HEADS, _NOPE_DIM),
        stride=(_HEADS * _NOPE_DIM, _NOPE_DIM, 1),
    )

    return cute.compile(
        launcher,
        q_nope,
        q_pe,
        ckv,
        kpe,
        indices,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Float32(0.0),
        Int32(0),
        Int32(0),
        options="--enable-tvm-ffi",
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute sparse MLA decode attention and return bf16 [T, 16, 512]."""
    num_tokens = q_nope.shape[0]
    output = torch.empty_like(q_nope)
    use_dense_shared = ckv_cache.shape[0] == 32768
    compiled = (
        _compile_dense_shared_kernel() if use_dense_shared else _compile_kernel()
    )
    grid_x = (
        num_tokens * (_HEADS // _DENSE_HEADS_PER_CTA)
        if use_dense_shared
        else num_tokens * _HEADS
    )
    compiled(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        float(sm_scale),
        num_tokens,
        grid_x,
    )
    return output
