"""CuTe DSL implementation of DeepSeek-V3 FP8 block-scale MoE.

All GPU computation in this module is authored in CuTe DSL.  Python is limited
to JIT compilation, workspace allocation, launch dispatch, and tensor views.
The two grouped GEMMs use the in-repository, BSD-attributed CUTLASS CuTe
post-scale template in :mod:`solution.postscale_masked_grouped_gemm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

_SOURCE_DIR = str(Path(__file__).resolve().parent)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from postscale_masked_grouped_gemm import BlockwiseMaskedGroupedGemmKernel


HIDDEN = 7168
INTERMEDIATE = 2048
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
SCALE_BLOCK = 128
B300_SMS = 148
WARP_ACTIVATION_CTAS = B300_SMS * 8
# The shared-A dual-output FC1 pipeline needs enough persistent M128 work to
# amortize its additional producer/consumer roles.
DUAL_OUTPUT_FC1_MIN_TOKENS = 96 * 128
U128_ACTIVATION_CTAS = B300_SMS * 4
CAP1024_ACTIVATION_CTAS = B300_SMS * 2


def _current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@cute.kernel
def _reset_counts_kernel(counts: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < LOCAL_EXPERTS:
        counts[tidx] = cutlass.Int32(0)


@cute.jit
def _bitonic32_desc(
    value: cutlass.Float32,
    index: cutlass.Int32,
):
    """Sort one key/id pair per lane by descending key, then ascending id."""
    lane = cute.arch.lane_idx()
    for merge_width in (2, 4, 8, 16, 32):
        stride = merge_width // 2
        while stride > 0:
            partner_value = cute.arch.shuffle_sync_bfly(value, stride)
            partner_index = cute.arch.shuffle_sync_bfly(index, stride)

            partner_better = partner_value > value
            if partner_value == value:
                partner_better = partner_index < index
            current_better = value > partner_value
            if value == partner_value:
                current_better = index < partner_index

            descending = (lane & merge_width) == 0
            lower_lane = (lane & stride) == 0
            if descending:
                if lower_lane:
                    if partner_better:
                        value = partner_value
                        index = partner_index
                else:
                    if current_better:
                        value = partner_value
                        index = partner_index
            else:
                if lower_lane:
                    if current_better:
                        value = partner_value
                        index = partner_index
                else:
                    if partner_better:
                        value = partner_value
                        index = partner_index
            stride = stride // 2
    return value, index


@cute.jit
def _bitonic8_desc(
    value: cutlass.Float32,
    index: cutlass.Int32,
):
    """Sort each eight-lane subgroup by descending key, then ascending id."""
    lane8 = cute.arch.lane_idx() & 7
    for merge_width in (2, 4, 8):
        stride = merge_width // 2
        while stride > 0:
            # Packed NVVM clamp for four independent width-eight subgroups:
            # ((32 - 8) << 8) | (32 - 1).
            partner_value = cute.arch.shuffle_sync_bfly(
                value,
                stride,
                mask=-1,
                mask_and_clamp=0x181F,
            )
            partner_index = cute.arch.shuffle_sync_bfly(
                index,
                stride,
                mask=-1,
                mask_and_clamp=0x181F,
            )

            partner_better = partner_value > value
            if partner_value == value:
                partner_better = partner_index < index
            current_better = value > partner_value
            if value == partner_value:
                current_better = index < partner_index

            descending = (lane8 & merge_width) == 0
            lower_lane = (lane8 & stride) == 0
            if descending:
                if lower_lane:
                    if partner_better:
                        value = partner_value
                        index = partner_index
                else:
                    if current_better:
                        value = partner_value
                        index = partner_index
            else:
                if lower_lane:
                    if current_better:
                        value = partner_value
                        index = partner_index
                else:
                    if partner_better:
                        value = partner_value
                        index = partner_index
            stride = stride // 2
    return value, index


@cute.jit
def _bitonic_merge8_desc(
    value: cutlass.Float32,
    index: cutlass.Int32,
):
    """Merge one bitonic eight-lane subgroup into descending total order."""
    lane8 = cute.arch.lane_idx() & 7
    for stride in (4, 2, 1):
        partner_value = cute.arch.shuffle_sync_bfly(
            value,
            stride,
            mask=-1,
            mask_and_clamp=0x181F,
        )
        partner_index = cute.arch.shuffle_sync_bfly(
            index,
            stride,
            mask=-1,
            mask_and_clamp=0x181F,
        )

        partner_better = partner_value > value
        if partner_value == value:
            partner_better = partner_index < index
        current_better = value > partner_value
        if value == partner_value:
            current_better = index < partner_index

        lower_lane = (lane8 & stride) == 0
        if lower_lane:
            if partner_better:
                value = partner_value
                index = partner_index
        else:
            if current_better:
                value = partner_value
                index = partner_index
    return value, index


@cute.jit
def _merge4x8_top8_desc(
    value: cutlass.Float32,
    index: cutlass.Int32,
):
    """Select exact top eight from four independently sorted eight-lane bands."""
    lane = cute.arch.lane_idx()

    # Merge bands 0+1 and 2+3 concurrently. XOR 15 pairs each descending
    # band's rank r with the other band's reversed rank 7-r.
    partner_value = cute.arch.shuffle_sync_bfly(
        value,
        15,
        mask=-1,
        mask_and_clamp=0x101F,
    )
    partner_index = cute.arch.shuffle_sync_bfly(
        index,
        15,
        mask=-1,
        mask_and_clamp=0x101F,
    )
    partner_better = partner_value > value
    if partner_value == value:
        partner_better = partner_index < index
    current_better = value > partner_value
    if value == partner_value:
        current_better = index < partner_index

    if (lane & 8) == 0:
        if partner_better:
            value = partner_value
            index = partner_index
    else:
        if current_better:
            value = partner_value
            index = partner_index
    value, index = _bitonic_merge8_desc(value, index)

    # Lanes 0..7 and 16..23 now hold the pairwise top-eight lists. XOR 23
    # compares the first list against the reversed second list without a
    # register remap. Only the globally winning low subgroup is consumed.
    partner_value = cute.arch.shuffle_sync_bfly(
        value,
        23,
        mask=-1,
        mask_and_clamp=0x001F,
    )
    partner_index = cute.arch.shuffle_sync_bfly(
        index,
        23,
        mask=-1,
        mask_and_clamp=0x001F,
    )
    partner_better = partner_value > value
    if partner_value == value:
        partner_better = partner_index < index
    current_better = value > partner_value
    if value == partner_value:
        current_better = index < partner_index

    if (lane & 16) == 0:
        if partner_better:
            value = partner_value
            index = partner_index
    else:
        if current_better:
            value = partner_value
            index = partner_index
    return _bitonic_merge8_desc(value, index)


@cute.kernel
def _route_pack_kernel(
    routing_logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hidden: cute.Tensor,
    hidden_scale: cute.Tensor,
    packed_a: cute.Tensor,
    packed_sfa: cute.Tensor,
    counts: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    local_offset: cutlass.Constexpr,
    routed_factor: cutlass.Constexpr,
    wide_dispatch: cutlass.Constexpr,
    parallel_finalize: cutlass.Constexpr,
):
    """Route one token per CTA and physically dispatch its local FP8 rows."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5

    cute.experimental.iket.range_push("route_pack")

    smem = utils.SmemAllocator()
    # Only unbiased sigmoid scores are shared for final normalization. Each
    # thread keeps its own biased selection score in a register for sorting.
    scores = smem.allocate_tensor(cutlass.Float32, GLOBAL_EXPERTS)
    group_scores = smem.allocate_tensor(cutlass.Float32, 8)
    group_candidate_value = smem.allocate_tensor(cutlass.Float32, 8 * TOP_K)
    group_candidate_id = smem.allocate_tensor(cutlass.Int32, 8 * TOP_K)
    selected = smem.allocate_tensor(cutlass.Int32, TOP_K)
    selected_local = smem.allocate_tensor(cutlass.Int32, TOP_K)
    selected_pos = smem.allocate_tensor(cutlass.Int32, TOP_K)
    group_rank = smem.allocate_tensor(cutlass.Int32, 8)

    if cutlass.const_expr(wide_dispatch):
        hidden_u128 = cute.recast_tensor(hidden, cutlass.Uint128)
        packed_a_u128 = cute.recast_tensor(packed_a, cutlass.Uint128)

    logit = routing_logits[token, tidx].to(cutlass.Float32)
    score = cutlass.Float32(1.0) / (
        cutlass.Float32(1.0) + cute.math.exp(-logit, fastmath=False)
    )
    biased_score = score + routing_bias[tidx].to(cutlass.Float32)
    scores[tidx] = score

    # Each warp exactly sorts one contiguous 32-expert group.  Retaining its
    # top eight is sufficient: a ninth-ranked member cannot enter a global
    # top-eight drawn from the four retained groups.
    candidate_value = biased_score
    candidate_id = cutlass.Int32(tidx)
    candidate_value, candidate_id = _bitonic32_desc(
        candidate_value, candidate_id
    )
    if lane < TOP_K:
        candidate_slot = warp * TOP_K + lane
        group_candidate_value[candidate_slot] = candidate_value
        group_candidate_id[candidate_slot] = candidate_id
    second_value = cute.arch.shuffle_sync(candidate_value, 1)
    if lane == 0:
        group_scores[warp] = candidate_value + second_value
    cute.arch.sync_threads()

    # Warp zero sorts the 32 retained expert candidates. The pruned path maps
    # each eight-lane band through its cached order; the legacy path computes
    # that order directly from the group scores.
    if warp == 0:
        group_value = neg_inf
        if lane < 8:
            group_value = group_scores[lane]
        group_id = cutlass.Int32(lane)
        group_value, group_id = _bitonic8_desc(group_value, group_id)

        retained_slot = lane >> 3
        retained_rank = lane & 7
        retained_group = cute.arch.shuffle_sync(group_id, retained_slot)
        candidate_slot = retained_group * TOP_K + retained_rank
        candidate_value = group_candidate_value[candidate_slot]
        candidate_id = group_candidate_id[candidate_slot]
        if cutlass.const_expr(packed_a.shape[1] >= 1024):
            candidate_value, candidate_id = _merge4x8_top8_desc(
                candidate_value, candidate_id
            )
        else:
            candidate_value, candidate_id = _bitonic32_desc(
                candidate_value, candidate_id
            )
        if lane < TOP_K:
            selected[lane] = candidate_id
    cute.arch.sync_threads()

    if cutlass.const_expr(parallel_finalize):
        if warp == 0:
            # Lane zero preserves the exact sequential FP32 normalization
            # order; the other top-k lanes only parallelize independent route
            # bookkeeping after receiving the bit-identical denominator.
            weight_sum = cutlass.Float32(0.0)
            if lane == 0:
                for k in range(TOP_K):
                    weight_sum = weight_sum + scores[selected[k]]
            weight_sum = cute.arch.shuffle_sync(weight_sum, 0)

            if lane < TOP_K:
                k = lane
                expert = selected[k]
                weight = (
                    scores[expert]
                    / (weight_sum + cutlass.Float32(1.0e-20))
                    * cutlass.Float32(routed_factor)
                )
                local_expert = expert - cutlass.Int32(local_offset)
                if local_expert >= 0 and local_expert < LOCAL_EXPERTS:
                    position = cute.arch.atomic_add(
                        (counts.iterator + local_expert).llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    selected_local[k] = local_expert
                    selected_pos[k] = position
                    route_expert[token, k] = local_expert
                    route_pos[token, k] = position
                    route_weight[token, k] = weight
                else:
                    selected_local[k] = cutlass.Int32(-1)
                    selected_pos[k] = cutlass.Int32(-1)
                    local_expert = cutlass.Int32(-1)
                    position = cutlass.Int32(-1)
                    weight = cutlass.Float32(0.0)
                    route_expert[token, k] = cutlass.Int32(-1)
                    route_pos[token, k] = cutlass.Int32(-1)
                    route_weight[token, k] = cutlass.Float32(0.0)
    else:
        if tidx == 0:
            # Preserve the exact sequential FP32 normalization order used by
            # the reference: biased scores select, unbiased scores normalize.
            weight_sum = cutlass.Float32(0.0)
            for k in range(TOP_K):
                weight_sum = weight_sum + scores[selected[k]]

            for k in range(TOP_K):
                expert = selected[k]
                weight = (
                    scores[expert]
                    / (weight_sum + cutlass.Float32(1.0e-20))
                    * cutlass.Float32(routed_factor)
                )
                local_expert = expert - cutlass.Int32(local_offset)
                if local_expert >= 0 and local_expert < LOCAL_EXPERTS:
                    position = cute.arch.atomic_add(
                        (counts.iterator + local_expert).llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    selected_local[k] = local_expert
                    selected_pos[k] = position
                    route_expert[token, k] = local_expert
                    route_pos[token, k] = position
                    route_weight[token, k] = weight
                else:
                    selected_local[k] = cutlass.Int32(-1)
                    selected_pos[k] = cutlass.Int32(-1)
                    route_expert[token, k] = cutlass.Int32(-1)
                    route_pos[token, k] = cutlass.Int32(-1)
                    route_weight[token, k] = cutlass.Float32(0.0)

    cute.arch.sync_threads()

    # Dispatch the original FP8 row and its transposed per-token scales.
    # H=7168 is exactly 28 vectors of 256 scalar elements.
    for k in range(TOP_K):
        local_expert = selected_local[k]
        if local_expert >= 0:
            position = selected_pos[k]
            local_expert_i64 = cutlass.Int64(local_expert)
            position_i64 = cutlass.Int64(position)
            if cutlass.const_expr(wide_dispatch):
                # Seven complete warps copy two aligned 16-byte words each:
                # 224 threads * 2 words * 16 bytes = 7168 bytes.
                if tidx < HIDDEN // (2 * 16):
                    for vec16 in range(2):
                        word_idx = (
                            vec16 * (HIDDEN // (2 * 16)) + tidx
                        )
                        packed_a_u128[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(word_idx),
                        ] = hidden_u128[
                            token,
                            cutlass.Int64(word_idx),
                        ]
            else:
                for vec in range(HIDDEN // 256):
                    hidden_idx = vec * 256 + tidx
                    packed_a[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(hidden_idx),
                    ] = hidden[
                        token, hidden_idx
                    ]
            if tidx < HIDDEN // SCALE_BLOCK:
                packed_sfa[
                    local_expert_i64,
                    position_i64,
                    cutlass.Int64(tidx),
                ] = hidden_scale[tidx, token]
    cute.experimental.iket.range_pop()


@cute.jit
def _route_pack_jit(
    routing_logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hidden: cute.Tensor,
    hidden_scale: cute.Tensor,
    packed_a: cute.Tensor,
    packed_sfa: cute.Tensor,
    counts: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    stream: cuda.CUstream,
    local_offset: cutlass.Constexpr,
    routed_factor: cutlass.Constexpr,
    wide_dispatch: cutlass.Constexpr,
    parallel_finalize: cutlass.Constexpr,
):
    _reset_counts_kernel(counts).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )
    _route_pack_kernel(
        routing_logits,
        routing_bias,
        hidden,
        hidden_scale,
        packed_a,
        packed_sfa,
        counts,
        route_expert,
        route_pos,
        route_weight,
        local_offset,
        routed_factor,
        wide_dispatch,
        parallel_finalize,
    ).launch(
        grid=(routing_logits.shape[0], 1, 1),
        block=(1024, 1, 1),
        stream=stream,
    )


@cute.kernel
def _build_mtile_map_kernel(
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    main_desc: cute.Tensor,
    main_count: cute.Tensor,
    tail_desc: cute.Tensor,
    tail_count: cute.Tensor,
    tile_m: cutlass.Constexpr,
    build_aux_lists: cutlass.Constexpr,
):
    """Build the primary M-tile map and optional large-only auxiliary lists."""
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    tile_counts = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)
    tile_offsets = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)
    main_counts = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)
    main_offsets = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)
    tail_counts = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)
    tail_offsets = smem.allocate_tensor(cutlass.Int32, LOCAL_EXPERTS)

    cute.experimental.iket.range_push("mtile_map")
    if tidx < LOCAL_EXPERTS:
        count = counts[tidx]
        tile_counts[tidx] = (
            count + cutlass.Int32(tile_m - 1)
        ) // tile_m
        if cutlass.const_expr(build_aux_lists):
            main_counts[tidx] = count // tile_m
            tail_counts[tidx] = (
                count + cutlass.Int32(tile_m // 2 - 1)
            ) // (tile_m // 2)
    cute.arch.sync_threads()

    if tidx == 0:
        total = cutlass.Int32(0)
        main_total = cutlass.Int32(0)
        tail_total = cutlass.Int32(0)
        for expert in range(LOCAL_EXPERTS):
            tile_offsets[expert] = total
            total = total + tile_counts[expert]
            if cutlass.const_expr(build_aux_lists):
                main_offsets[expert] = main_total
                main_total = main_total + main_counts[expert]
                tail_offsets[expert] = tail_total
                tail_total = tail_total + tail_counts[expert]
        mtile_count[0] = total
        main_count[0] = main_total
        tail_count[0] = tail_total
    cute.arch.sync_threads()

    if tidx < LOCAL_EXPERTS:
        descriptor_base = tidx * cutlass.Int32(65536)
        offset = tile_offsets[tidx]
        for mtile in cutlass.range(tile_counts[tidx]):
            # High 16 bits hold expert id; low 16 bits hold its M-tile.
            mtile_desc[offset + mtile] = descriptor_base + mtile

        if cutlass.const_expr(build_aux_lists):
            main_offset = main_offsets[tidx]
            for mtile in cutlass.range(main_counts[tidx]):
                main_desc[main_offset + mtile] = descriptor_base + mtile

            tail_offset = tail_offsets[tidx]
            for tail_mtile in cutlass.range(tail_counts[tidx]):
                tail_desc[tail_offset + tail_mtile] = (
                    descriptor_base + tail_mtile
                )
    cute.experimental.iket.range_pop()


@cute.jit
def _build_mtile_map_jit(
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    main_desc: cute.Tensor,
    main_count: cute.Tensor,
    tail_desc: cute.Tensor,
    tail_count: cute.Tensor,
    stream: cuda.CUstream,
    tile_m: cutlass.Constexpr,
    build_aux_lists: cutlass.Constexpr,
):
    _build_mtile_map_kernel(
        counts,
        mtile_desc,
        mtile_count,
        main_desc,
        main_count,
        tail_desc,
        tail_count,
        tile_m,
        build_aux_lists,
    ).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_kernel(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
):
    """SwiGLU and dynamic E4M3 quantization, one selected route per CTA."""
    tidx, _, _ = cute.arch.thread_idx()
    route_linear, _, _ = cute.arch.block_idx()
    token = route_linear // TOP_K
    slot = route_linear - token * TOP_K
    local_expert = route_expert[token, slot]

    smem = utils.SmemAllocator()
    warp_amax = smem.allocate_tensor(cutlass.Float32, 4)
    block_amax = smem.allocate_tensor(cutlass.Float32, 1)

    if local_expert >= 0:
        cute.experimental.iket.range_push("swiglu_route")
        position = route_pos[token, slot]
        local_expert_i64 = cutlass.Int64(local_expert)
        position_i64 = cutlass.Int64(position)
        lane = tidx & 31
        warp = tidx >> 5

        for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
            col = scale_block * SCALE_BLOCK + tidx
            up = gemm1_out[
                local_expert_i64,
                position_i64,
                cutlass.Int64(col),
            ].to(cutlass.Float32)
            gate = gemm1_out[
                local_expert_i64,
                position_i64,
                cutlass.Int64(INTERMEDIATE + col),
            ].to(cutlass.Float32)
            value = up * (
                gate
                / (
                    cutlass.Float32(1.0)
                    + cute.math.exp(-gate, fastmath=False)
                )
            )

            local_amax = cute.arch.warp_redux_sync(
                value,
                kind="fmax",
                abs=True,
                nan=True,
            )
            if lane == 0:
                warp_amax[warp] = local_amax
            cute.arch.sync_threads()

            if warp == 0:
                partial = (
                    warp_amax[lane]
                    if lane < 4
                    else cutlass.Float32(0.0)
                )
                reduced = cute.arch.warp_redux_sync(
                    partial,
                    kind="fmax",
                    nan=True,
                )
                if lane == 0:
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    block_amax[0] = scale
                    activation_scale[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(scale_block),
                    ] = scale
            cute.arch.sync_threads()

            activation[
                local_expert_i64,
                position_i64,
                cutlass.Int64(col),
            ] = cutlass.Float8E4M3FN(value / block_amax[0])
            cute.arch.sync_threads()
        cute.experimental.iket.range_pop()


@cute.jit
def _swiglu_quant_jit(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    stream: cuda.CUstream,
):
    _swiglu_quant_kernel(
        gemm1_out,
        activation,
        activation_scale,
        route_expert,
        route_pos,
    ).launch(
        grid=(route_expert.shape[0] * TOP_K, 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_mtile_kernel(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
):
    """Quantize compact large-case rows with one independent row per warp."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    row_linear = bidx * 8 + warp
    row_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(8)
    total_rows = mtile_count[0] * cutlass.Int32(128)
    # Permit the dependent GEMM2 grid to perform its dependency-independent
    # setup as activation CTAs drain.  GEMM2 waits before its first data read.
    cute.arch.griddepcontrol_launch_dependents()

    values = cute.make_rmem_tensor((4,), cutlass.Float32)
    packed_up = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_gate = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_activation = cute.make_rmem_tensor(
        (4,), cutlass.Float8E4M3FN
    )
    gemm1_out_u64 = cute.recast_tensor(gemm1_out, cutlass.Uint64)
    activation_u32 = cute.recast_tensor(activation, cutlass.Uint32)

    if row_linear < total_rows:
        cute.experimental.iket.range_push("swiglu_mtile")
        while row_linear < total_rows:
            mtile_idx = row_linear // cutlass.Int32(128)
            descriptor = mtile_desc[mtile_idx]
            local_expert = descriptor >> 16
            position = (
                (descriptor & cutlass.Int32(65535)) * cutlass.Int32(128)
                + (row_linear & cutlass.Int32(127))
            )

            if position < counts[local_expert]:
                local_expert_i64 = cutlass.Int64(local_expert)
                position_i64 = cutlass.Int64(position)

                for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
                    packed_col = scale_block * (SCALE_BLOCK // 4) + lane
                    packed_up[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ]
                    packed_gate[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(INTERMEDIATE // 4 + packed_col),
                    ]
                    up_values = packed_up.load().bitcast(cutlass.BFloat16)
                    gate_values = packed_gate.load().bitcast(
                        cutlass.BFloat16
                    )
                    for segment in range(4):
                        up = up_values[segment].to(cutlass.Float32)
                        gate = gate_values[segment].to(cutlass.Float32)
                        value = up * (
                            gate
                            / (
                                cutlass.Float32(1.0)
                                + cute.math.exp(-gate, fastmath=True)
                            )
                        )
                        values[segment] = value

                    # Max is exact, so reduce each lane's four absolute values
                    # first and use only one cross-lane reduction. Explicit
                    # NaN propagation matches redux.sync.max.NaN.abs.
                    amax01 = cute.math.max(
                        cute.math.abs(values[0]),
                        cute.math.abs(values[1]),
                        propagate_nan=True,
                    )
                    amax23 = cute.math.max(
                        cute.math.abs(values[2]),
                        cute.math.abs(values[3]),
                        propagate_nan=True,
                    )
                    thread_amax = cute.math.max(
                        amax01,
                        amax23,
                        propagate_nan=True,
                    )
                    reduced = cute.arch.warp_redux_sync(
                        thread_amax,
                        kind="fmax",
                        nan=True,
                    )
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    if lane == 0:
                        activation_scale[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(scale_block),
                        ] = scale

                    for segment in range(4):
                        packed_activation[segment] = (
                            cutlass.Float8E4M3FN(values[segment] / scale)
                        )
                    packed_out = packed_activation.load().bitcast(
                        cutlass.Uint32
                    )
                    activation_u32[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ] = packed_out[0]

            row_linear = row_linear + row_stride
        cute.experimental.iket.range_pop()


@cute.jit
def _swiglu_quant_mtile_jit(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    stream: cuda.CUstream,
):
    _swiglu_quant_mtile_kernel(
        gemm1_out,
        activation,
        activation_scale,
        counts,
        mtile_desc,
        mtile_count,
    ).launch(
        grid=(WARP_ACTIVATION_CTAS, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_mtile64_kernel(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
):
    """Quantize capacity-1024 M64-map rows with packed per-warp I/O."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    row_linear = bidx * 8 + warp
    row_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(8)
    total_rows = mtile_count[0] * cutlass.Int32(64)

    values = cute.make_rmem_tensor((4,), cutlass.Float32)
    packed_up = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_gate = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_activation = cute.make_rmem_tensor(
        (4,), cutlass.Float8E4M3FN
    )
    gemm1_out_u64 = cute.recast_tensor(gemm1_out, cutlass.Uint64)
    activation_u32 = cute.recast_tensor(activation, cutlass.Uint32)

    if row_linear < total_rows:
        cute.experimental.iket.range_push("swiglu_mtile64")
        while row_linear < total_rows:
            mtile_idx = row_linear // cutlass.Int32(64)
            descriptor = mtile_desc[mtile_idx]
            local_expert = descriptor >> 16
            position = (
                (descriptor & cutlass.Int32(65535)) * cutlass.Int32(64)
                + (row_linear & cutlass.Int32(63))
            )

            if position < counts[local_expert]:
                local_expert_i64 = cutlass.Int64(local_expert)
                position_i64 = cutlass.Int64(position)

                for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
                    packed_col = scale_block * (SCALE_BLOCK // 4) + lane
                    packed_up[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ]
                    packed_gate[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(INTERMEDIATE // 4 + packed_col),
                    ]
                    up_values = packed_up.load().bitcast(cutlass.BFloat16)
                    gate_values = packed_gate.load().bitcast(
                        cutlass.BFloat16
                    )
                    for segment in range(4):
                        up = up_values[segment].to(cutlass.Float32)
                        gate = gate_values[segment].to(cutlass.Float32)
                        value = up * (
                            gate
                            / (
                                cutlass.Float32(1.0)
                                + cute.math.exp(-gate, fastmath=False)
                            )
                        )
                        values[segment] = value

                    amax01 = cute.math.max(
                        cute.math.abs(values[0]),
                        cute.math.abs(values[1]),
                        propagate_nan=True,
                    )
                    amax23 = cute.math.max(
                        cute.math.abs(values[2]),
                        cute.math.abs(values[3]),
                        propagate_nan=True,
                    )
                    thread_amax = cute.math.max(
                        amax01,
                        amax23,
                        propagate_nan=True,
                    )
                    reduced = cute.arch.warp_redux_sync(
                        thread_amax,
                        kind="fmax",
                        nan=True,
                    )
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    if lane == 0:
                        activation_scale[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(scale_block),
                        ] = scale

                    for segment in range(4):
                        packed_activation[segment] = (
                            cutlass.Float8E4M3FN(values[segment] / scale)
                        )
                    packed_out = packed_activation.load().bitcast(
                        cutlass.Uint32
                    )
                    activation_u32[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ] = packed_out[0]

            row_linear = row_linear + row_stride
        cute.experimental.iket.range_pop()


@cute.jit
def _swiglu_quant_mtile64_jit(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    stream: cuda.CUstream,
):
    _swiglu_quant_mtile64_kernel(
        gemm1_out,
        activation,
        activation_scale,
        counts,
        mtile_desc,
        mtile_count,
    ).launch(
        grid=(CAP1024_ACTIVATION_CTAS, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_mtile_recip_kernel(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
):
    """Quantize compact 16K-capacity rows with one reciprocal per block."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    row_linear = bidx * 8 + warp
    row_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(8)
    total_rows = mtile_count[0] * cutlass.Int32(128)
    cute.arch.griddepcontrol_launch_dependents()

    values = cute.make_rmem_tensor((4,), cutlass.Float32)
    packed_up = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_gate = cute.make_rmem_tensor((1,), cutlass.Uint64)
    packed_activation = cute.make_rmem_tensor(
        (4,), cutlass.Float8E4M3FN
    )
    gemm1_out_u64 = cute.recast_tensor(gemm1_out, cutlass.Uint64)
    activation_u32 = cute.recast_tensor(activation, cutlass.Uint32)

    if row_linear < total_rows:
        cute.experimental.iket.range_push("swiglu_mtile_recip")
        while row_linear < total_rows:
            mtile_idx = row_linear // cutlass.Int32(128)
            descriptor = mtile_desc[mtile_idx]
            local_expert = descriptor >> 16
            position = (
                (descriptor & cutlass.Int32(65535)) * cutlass.Int32(128)
                + (row_linear & cutlass.Int32(127))
            )

            if position < counts[local_expert]:
                local_expert_i64 = cutlass.Int64(local_expert)
                position_i64 = cutlass.Int64(position)

                for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
                    packed_col = scale_block * (SCALE_BLOCK // 4) + lane
                    packed_up[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ]
                    packed_gate[0] = gemm1_out_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(INTERMEDIATE // 4 + packed_col),
                    ]
                    up_values = packed_up.load().bitcast(cutlass.BFloat16)
                    gate_values = packed_gate.load().bitcast(
                        cutlass.BFloat16
                    )
                    for segment in range(4):
                        up = up_values[segment].to(cutlass.Float32)
                        gate = gate_values[segment].to(cutlass.Float32)
                        value = up * (
                            gate
                            / (
                                cutlass.Float32(1.0)
                                + cute.math.exp(-gate, fastmath=True)
                            )
                        )
                        values[segment] = value

                    # Max is exact, so reduce each lane's four absolute values
                    # first and use only one cross-lane reduction. Explicit
                    # NaN propagation matches redux.sync.max.NaN.abs.
                    amax01 = cute.math.max(
                        cute.math.abs(values[0]),
                        cute.math.abs(values[1]),
                        propagate_nan=True,
                    )
                    amax23 = cute.math.max(
                        cute.math.abs(values[2]),
                        cute.math.abs(values[3]),
                        propagate_nan=True,
                    )
                    thread_amax = cute.math.max(
                        amax01,
                        amax23,
                        propagate_nan=True,
                    )
                    reduced = cute.arch.warp_redux_sync(
                        thread_amax,
                        kind="fmax",
                        nan=True,
                    )
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    if lane == 0:
                        activation_scale[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(scale_block),
                        ] = scale

                    inv_scale = cutlass.Float32(1.0) / scale
                    for segment in range(4):
                        packed_activation[segment] = cutlass.Float8E4M3FN(
                            values[segment] * inv_scale
                        )
                    packed_out = packed_activation.load().bitcast(
                        cutlass.Uint32
                    )
                    activation_u32[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ] = packed_out[0]

            row_linear = row_linear + row_stride
        cute.experimental.iket.range_pop()


@cute.jit
def _swiglu_quant_mtile_recip_jit(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    stream: cuda.CUstream,
):
    _swiglu_quant_mtile_recip_kernel(
        gemm1_out,
        activation,
        activation_scale,
        counts,
        mtile_desc,
        mtile_count,
    ).launch(
        grid=(WARP_ACTIVATION_CTAS, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.kernel
def _quantize_mtile_kernel(
    swiglu_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    use_reciprocal: cutlass.Constexpr,
):
    """Quantize large-case FP32 SwiGLU rows, one independent row per warp."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    row_linear = bidx * 8 + warp
    row_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(8)
    total_rows = mtile_count[0] * cutlass.Int32(128)

    values = cute.make_rmem_tensor((4,), cutlass.Float32)

    if row_linear < total_rows:
        cute.experimental.iket.range_push("quantize_mtile")
        while row_linear < total_rows:
            mtile_idx = row_linear // cutlass.Int32(128)
            descriptor = mtile_desc[mtile_idx]
            local_expert = descriptor >> 16
            position = (
                (descriptor & cutlass.Int32(65535)) * cutlass.Int32(128)
                + (row_linear & cutlass.Int32(127))
            )

            if position < counts[local_expert]:
                local_expert_i64 = cutlass.Int64(local_expert)
                position_i64 = cutlass.Int64(position)

                for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
                    for segment in range(4):
                        col = (
                            scale_block * SCALE_BLOCK
                            + segment * 32
                            + lane
                        )
                        values[segment] = swiglu_out[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(col),
                        ]

                    amax01 = cute.math.max(
                        cute.math.abs(values[0]),
                        cute.math.abs(values[1]),
                        propagate_nan=True,
                    )
                    amax23 = cute.math.max(
                        cute.math.abs(values[2]),
                        cute.math.abs(values[3]),
                        propagate_nan=True,
                    )
                    thread_amax = cute.math.max(
                        amax01,
                        amax23,
                        propagate_nan=True,
                    )
                    reduced = cute.arch.warp_redux_sync(
                        thread_amax,
                        kind="fmax",
                        nan=True,
                    )
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    if lane == 0:
                        activation_scale[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(scale_block),
                        ] = scale

                    inv_scale = cutlass.Float32(0.0)
                    if cutlass.const_expr(use_reciprocal):
                        inv_scale = cutlass.Float32(1.0) / scale
                    for segment in range(4):
                        col = (
                            scale_block * SCALE_BLOCK
                            + segment * 32
                            + lane
                        )
                        value = values[segment] / scale
                        if cutlass.const_expr(use_reciprocal):
                            value = values[segment] * inv_scale
                        activation[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(col),
                        ] = cutlass.Float8E4M3FN(value)

            row_linear = row_linear + row_stride
        cute.experimental.iket.range_pop()


@cute.jit
def _quantize_mtile_jit(
    swiglu_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    stream: cuda.CUstream,
    use_reciprocal: cutlass.Constexpr,
):
    _quantize_mtile_kernel(
        swiglu_out,
        activation,
        activation_scale,
        counts,
        mtile_desc,
        mtile_count,
        use_reciprocal,
    ).launch(
        grid=(WARP_ACTIVATION_CTAS, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_mtile_u128_kernel(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    use_reciprocal: cutlass.Constexpr,
):
    """Quantize two independent rows per warp through half-warp U128 I/O."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    half = lane >> 4
    lane16 = lane & 15
    row_linear = bidx * 16 + warp * 2 + half
    row_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(16)
    total_rows = mtile_count[0] * cutlass.Int32(128)

    values = cute.make_rmem_tensor((8,), cutlass.Float32)
    packed_up = cute.make_rmem_tensor((1,), cutlass.Uint128)
    packed_gate = cute.make_rmem_tensor((1,), cutlass.Uint128)
    packed_activation = cute.make_rmem_tensor(
        (8,), cutlass.Float8E4M3FN
    )
    gemm1_out_u128 = cute.recast_tensor(gemm1_out, cutlass.Uint128)
    activation_u64 = cute.recast_tensor(activation, cutlass.Uint64)

    if row_linear < total_rows:
        cute.experimental.iket.range_push("swiglu_mtile_u128")
        while row_linear < total_rows:
            mtile_idx = row_linear // cutlass.Int32(128)
            descriptor = mtile_desc[mtile_idx]
            local_expert = descriptor >> 16
            position = (
                (descriptor & cutlass.Int32(65535)) * cutlass.Int32(128)
                + (row_linear & cutlass.Int32(127))
            )

            if position < counts[local_expert]:
                local_expert_i64 = cutlass.Int64(local_expert)
                position_i64 = cutlass.Int64(position)

                for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
                    packed_col = scale_block * (SCALE_BLOCK // 8) + lane16
                    packed_up[0] = gemm1_out_u128[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ]
                    packed_gate[0] = gemm1_out_u128[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(INTERMEDIATE // 8 + packed_col),
                    ]
                    up_values = packed_up.load().bitcast(cutlass.BFloat16)
                    gate_values = packed_gate.load().bitcast(
                        cutlass.BFloat16
                    )
                    for segment in range(8):
                        up = up_values[segment].to(cutlass.Float32)
                        gate = gate_values[segment].to(cutlass.Float32)
                        values[segment] = up * (
                            gate
                            / (
                                cutlass.Float32(1.0)
                                + cute.math.exp(-gate, fastmath=False)
                            )
                        )

                    amax01 = cute.math.max(
                        cute.math.abs(values[0]),
                        cute.math.abs(values[1]),
                        propagate_nan=True,
                    )
                    amax23 = cute.math.max(
                        cute.math.abs(values[2]),
                        cute.math.abs(values[3]),
                        propagate_nan=True,
                    )
                    amax45 = cute.math.max(
                        cute.math.abs(values[4]),
                        cute.math.abs(values[5]),
                        propagate_nan=True,
                    )
                    amax67 = cute.math.max(
                        cute.math.abs(values[6]),
                        cute.math.abs(values[7]),
                        propagate_nan=True,
                    )
                    amax03 = cute.math.max(
                        amax01,
                        amax23,
                        propagate_nan=True,
                    )
                    amax47 = cute.math.max(
                        amax45,
                        amax67,
                        propagate_nan=True,
                    )
                    thread_amax = cute.math.max(
                        amax03,
                        amax47,
                        propagate_nan=True,
                    )
                    reduced = thread_amax
                    if lane < 16:
                        reduced = cute.arch.warp_redux_sync(
                            thread_amax,
                            kind="fmax",
                            mask_and_clamp=65535,
                            nan=True,
                        )
                    else:
                        reduced = cute.arch.warp_redux_sync(
                            thread_amax,
                            kind="fmax",
                            mask_and_clamp=-65536,
                            nan=True,
                        )
                    scale = reduced / cutlass.Float32(448.0)
                    if scale < cutlass.Float32(1.0e-12):
                        scale = cutlass.Float32(1.0e-12)
                    if lane16 == 0:
                        activation_scale[
                            local_expert_i64,
                            position_i64,
                            cutlass.Int64(scale_block),
                        ] = scale

                    if cutlass.const_expr(use_reciprocal):
                        inv_scale = cutlass.Float32(1.0) / scale
                        for segment in range(8):
                            packed_activation[segment] = (
                                cutlass.Float8E4M3FN(
                                    values[segment] * inv_scale
                                )
                            )
                    else:
                        for segment in range(8):
                            packed_activation[segment] = (
                                cutlass.Float8E4M3FN(
                                    values[segment] / scale
                                )
                            )
                    packed_out = packed_activation.load().bitcast(
                        cutlass.Uint64
                    )
                    activation_u64[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(packed_col),
                    ] = packed_out[0]

            row_linear = row_linear + row_stride
        cute.experimental.iket.range_pop()


@cute.jit
def _swiglu_quant_mtile_u128_jit(
    gemm1_out: cute.Tensor,
    activation: cute.Tensor,
    activation_scale: cute.Tensor,
    counts: cute.Tensor,
    mtile_desc: cute.Tensor,
    mtile_count: cute.Tensor,
    stream: cuda.CUstream,
    use_reciprocal: cutlass.Constexpr,
):
    _swiglu_quant_mtile_u128_kernel(
        gemm1_out,
        activation,
        activation_scale,
        counts,
        mtile_desc,
        mtile_count,
        use_reciprocal,
    ).launch(
        grid=(U128_ACTIVATION_CTAS, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.kernel
def _combine_kernel(
    gemm2_out: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear = bidx * 256 + tidx
    total = output.shape[0] * (HIDDEN // 2)
    if linear < total:
        cute.experimental.iket.range_push("combine_flat")
        token = linear // (HIDDEN // 2)
        pair_idx = linear - token * (HIDDEN // 2)
        gemm2_out_u32 = cute.recast_tensor(gemm2_out, cutlass.Uint32)
        output_u32 = cute.recast_tensor(output, cutlass.Uint32)
        value_lo = cutlass.Float32(0.0)
        value_hi = cutlass.Float32(0.0)
        packed_in = cute.make_rmem_tensor((1,), cutlass.Uint32)
        for k in range(TOP_K):
            local_expert = route_expert[token, k]
            if local_expert >= 0:
                position = route_pos[token, k]
                weight = route_weight[token, k]
                packed_in[0] = gemm2_out_u32[
                    cutlass.Int64(local_expert),
                    cutlass.Int64(position),
                    cutlass.Int64(pair_idx),
                ]
                pair_in = packed_in.load().bitcast(cutlass.BFloat16)
                value_lo = value_lo + (
                    pair_in[0].to(cutlass.Float32) * weight
                )
                value_hi = value_hi + (
                    pair_in[1].to(cutlass.Float32) * weight
                )
        pair_out = cute.make_rmem_tensor((2,), cutlass.BFloat16)
        pair_out[0] = cutlass.BFloat16(value_lo)
        pair_out[1] = cutlass.BFloat16(value_hi)
        packed_out = pair_out.load().bitcast(cutlass.Uint32)
        output_u32[token, cutlass.Int64(pair_idx)] = packed_out[0]
        cute.experimental.iket.range_pop()


@cute.kernel
def _combine_token_kernel(
    gemm2_out: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    output: cute.Tensor,
):
    """Combine one token per CTA while reusing its eight route records."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    local_experts = smem.allocate_tensor(cutlass.Int32, TOP_K)
    positions = smem.allocate_tensor(cutlass.Int32, TOP_K)
    weights = smem.allocate_tensor(cutlass.Float32, TOP_K)

    if tidx < TOP_K:
        local_experts[tidx] = route_expert[token, tidx]
        positions[tidx] = route_pos[token, tidx]
        weights[tidx] = route_weight[token, tidx]
    cute.arch.sync_threads()

    cute.experimental.iket.range_push("combine_role")
    values = cute.make_rmem_tensor((HIDDEN // 1024,), cutlass.Float32)
    values.fill(0.0)

    for k in range(TOP_K):
        local_expert = local_experts[k]
        if local_expert >= 0:
            position = positions[k]
            weight = weights[k]
            local_expert_i64 = cutlass.Int64(local_expert)
            position_i64 = cutlass.Int64(position)
            for vec in range(HIDDEN // 1024):
                hidden_idx = tidx + vec * 1024
                values[vec] = values[vec] + (
                    gemm2_out[
                        local_expert_i64,
                        position_i64,
                        cutlass.Int64(hidden_idx),
                    ].to(cutlass.Float32)
                    * weight
                )

    for vec in range(HIDDEN // 1024):
        hidden_idx = tidx + vec * 1024
        output[token, cutlass.Int64(hidden_idx)] = cutlass.BFloat16(values[vec])
    cute.experimental.iket.range_pop()


@cute.jit
def _combine_jit(
    gemm2_out: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
):
    _combine_kernel(
        gemm2_out,
        route_expert,
        route_pos,
        route_weight,
        output,
    ).launch(
        grid=(
            cute.ceil_div(output.shape[0] * (HIDDEN // 2), 256),
            1,
            1,
        ),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.jit
def _combine_token_jit(
    gemm2_out: cute.Tensor,
    route_expert: cute.Tensor,
    route_pos: cute.Tensor,
    route_weight: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
):
    _combine_token_kernel(
        gemm2_out,
        route_expert,
        route_pos,
        route_weight,
        output,
    ).launch(
        grid=(output.shape[0], 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@dataclass
class _Workspace:
    tokens: int
    capacity: int
    packed_a: torch.Tensor
    packed_sfa: torch.Tensor
    counts: torch.Tensor
    route_expert: torch.Tensor
    route_pos: torch.Tensor
    route_weight: torch.Tensor
    mtile_desc: torch.Tensor
    mtile_count: torch.Tensor
    main_desc: torch.Tensor
    main_count: torch.Tensor
    tail_desc: torch.Tensor
    tail_count: torch.Tensor
    gemm1_out: torch.Tensor
    activation: torch.Tensor
    activation_scale: torch.Tensor
    gemm2_out: torch.Tensor


_workspace: _Workspace | None = None
_route_cache: Dict[Tuple[int, int, bool], object] = {}
_mtile_map_cache: Dict[Tuple[int, int], object] = {}
_activation_cache: Dict[int, object] = {}
_combine_cache: Dict[int, object] = {}
_gemm_cache: Dict[Tuple[int, ...], object] = {}


def _capacity(tokens: int) -> int:
    if tokens <= 80:
        return 128
    if tokens <= 901:
        return 1024
    if tokens <= 16384:
        return 16384
    return 32768


def _mma_m(capacity: int) -> int:
    if capacity == 32768:
        return 256
    return 64 if capacity <= 1024 else 128


def _mtile_desc_rows(capacity: int, mma_m: int | None = None) -> int:
    tile_m = _mma_m(capacity) if mma_m is None else mma_m
    return LOCAL_EXPERTS * ((capacity + tile_m - 1) // tile_m)


def _allocate_workspace(tokens: int, capacity: int, device) -> _Workspace:
    return _Workspace(
        tokens=tokens,
        capacity=capacity,
        packed_a=torch.empty(
            (LOCAL_EXPERTS, capacity, HIDDEN),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        packed_sfa=torch.empty(
            (LOCAL_EXPERTS, capacity, HIDDEN // SCALE_BLOCK),
            dtype=torch.float32,
            device=device,
        ),
        counts=torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=device),
        route_expert=torch.empty(
            (tokens, TOP_K), dtype=torch.int32, device=device
        ),
        route_pos=torch.empty(
            (tokens, TOP_K), dtype=torch.int32, device=device
        ),
        route_weight=torch.empty(
            (tokens, TOP_K), dtype=torch.float32, device=device
        ),
        mtile_desc=torch.empty(
            (_mtile_desc_rows(capacity),),
            dtype=torch.int32,
            device=device,
        ),
        mtile_count=torch.empty((1,), dtype=torch.int32, device=device),
        main_desc=torch.empty(
            (_mtile_desc_rows(capacity, 128),),
            dtype=torch.int32,
            device=device,
        ),
        main_count=torch.empty((1,), dtype=torch.int32, device=device),
        tail_desc=torch.empty(
            (_mtile_desc_rows(capacity, _mma_m(capacity) // 2),),
            dtype=torch.int32,
            device=device,
        ),
        tail_count=torch.empty((1,), dtype=torch.int32, device=device),
        gemm1_out=torch.empty(
            (
                LOCAL_EXPERTS,
                capacity,
                INTERMEDIATE if capacity >= 16384 else 2 * INTERMEDIATE,
            ),
            dtype=torch.float32 if capacity >= 16384 else torch.bfloat16,
            device=device,
        ),
        activation=torch.empty(
            (LOCAL_EXPERTS, capacity, INTERMEDIATE),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        activation_scale=torch.empty(
            (LOCAL_EXPERTS, capacity, INTERMEDIATE // SCALE_BLOCK),
            dtype=torch.float32,
            device=device,
        ),
        gemm2_out=torch.empty(
            (LOCAL_EXPERTS, capacity, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def _fake(
    dtype,
    shape,
    *,
    stride_order=None,
    align=16,
):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=align,
    )


def _get_route(
    tokens: int,
    capacity: int,
    local_offset: int,
    *,
    parallel_finalize: bool | None = None,
):
    if parallel_finalize is None:
        parallel_finalize = capacity >= 1024
    key = (capacity, local_offset, parallel_finalize)
    compiled = _route_cache.get(key)
    if compiled is not None:
        return compiled

    t = cute.sym_int64()
    stream = _current_stream()
    compiled = cute.compile(
        _route_pack_jit,
        _fake(cutlass.Float32, (t, GLOBAL_EXPERTS), stride_order=(1, 0)),
        _fake(cutlass.BFloat16, (GLOBAL_EXPERTS,), stride_order=(0,)),
        _fake(cutlass.Float8E4M3FN, (t, HIDDEN), stride_order=(1, 0)),
        _fake(
            cutlass.Float32,
            (HIDDEN // SCALE_BLOCK, t),
            stride_order=(1, 0),
        ),
        _fake(
            cutlass.Float8E4M3FN,
            (LOCAL_EXPERTS, capacity, HIDDEN),
            stride_order=(2, 1, 0),
        ),
        _fake(
            cutlass.Float32,
            (LOCAL_EXPERTS, capacity, HIDDEN // SCALE_BLOCK),
            stride_order=(2, 1, 0),
        ),
        _fake(cutlass.Int32, (LOCAL_EXPERTS,), stride_order=(0,), align=4),
        _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
        _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
        _fake(cutlass.Float32, (t, TOP_K), stride_order=(1, 0), align=4),
        stream,
        local_offset=local_offset,
        routed_factor=2.5,
        wide_dispatch=capacity >= 128,
        parallel_finalize=parallel_finalize,
        options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
    )
    _route_cache[key] = compiled
    return compiled


def _get_mtile_map(capacity: int):
    tile_m = _mma_m(capacity)
    key = (capacity, tile_m)
    compiled = _mtile_map_cache.get(key)
    if compiled is not None:
        return compiled

    max_mtiles = _mtile_desc_rows(capacity)
    stream = _current_stream()
    compiled = cute.compile(
        _build_mtile_map_jit,
        _fake(cutlass.Int32, (LOCAL_EXPERTS,), stride_order=(0,), align=4),
        _fake(cutlass.Int32, (max_mtiles,), stride_order=(0,), align=4),
        _fake(cutlass.Int32, (1,), stride_order=(0,), align=4),
        _fake(
            cutlass.Int32,
            (_mtile_desc_rows(capacity, 128),),
            stride_order=(0,),
            align=4,
        ),
        _fake(cutlass.Int32, (1,), stride_order=(0,), align=4),
        _fake(
            cutlass.Int32,
            (_mtile_desc_rows(capacity, tile_m // 2),),
            stride_order=(0,),
            align=4,
        ),
        _fake(cutlass.Int32, (1,), stride_order=(0,), align=4),
        stream,
        tile_m=tile_m,
        build_aux_lists=capacity >= 16384,
        options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
    )
    _mtile_map_cache[key] = compiled
    return compiled


def _get_activation(capacity: int):
    compiled = _activation_cache.get(capacity)
    if compiled is not None:
        return compiled

    stream = _current_stream()
    common_args = (
        _fake(
            cutlass.Float32 if capacity >= 16384 else cutlass.BFloat16,
            (
                LOCAL_EXPERTS,
                capacity,
                INTERMEDIATE if capacity >= 16384 else 2 * INTERMEDIATE,
            ),
            stride_order=(2, 1, 0),
        ),
        _fake(
            cutlass.Float8E4M3FN,
            (LOCAL_EXPERTS, capacity, INTERMEDIATE),
            stride_order=(2, 1, 0),
        ),
        _fake(
            cutlass.Float32,
            (LOCAL_EXPERTS, capacity, INTERMEDIATE // SCALE_BLOCK),
            stride_order=(2, 1, 0),
        ),
    )
    if capacity >= 16384:
        compiled = cute.compile(
            _quantize_mtile_jit,
            *common_args,
            _fake(cutlass.Int32, (LOCAL_EXPERTS,), stride_order=(0,), align=4),
            _fake(
                cutlass.Int32,
                (_mtile_desc_rows(capacity, 128),),
                stride_order=(0,),
                align=4,
            ),
            _fake(cutlass.Int32, (1,), stride_order=(0,), align=4),
            stream,
            use_reciprocal=capacity == 16384,
            options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
        )
    else:
        t = cute.sym_int64()
        compiled = cute.compile(
            _swiglu_quant_jit,
            *common_args,
            _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
            _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
            stream,
            options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
        )
    _activation_cache[capacity] = compiled
    return compiled


def _get_combine(tokens: int, capacity: int):
    compiled = _combine_cache.get(capacity)
    if compiled is not None:
        return compiled

    t = cute.sym_int64()
    stream = _current_stream()
    combine_jit = _combine_token_jit if capacity >= 1024 else _combine_jit
    compiled = cute.compile(
        combine_jit,
        _fake(
            cutlass.BFloat16,
            (LOCAL_EXPERTS, capacity, HIDDEN),
            stride_order=(2, 1, 0),
        ),
        _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
        _fake(cutlass.Int32, (t, TOP_K), stride_order=(1, 0), align=4),
        _fake(cutlass.Float32, (t, TOP_K), stride_order=(1, 0), align=4),
        _fake(cutlass.BFloat16, (t, HIDDEN), stride_order=(1, 0)),
        stream,
        options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
    )
    _combine_cache[capacity] = compiled
    return compiled


def _get_gemm(
    capacity: int,
    n: int,
    k: int,
    mma_m: int | None = None,
    *,
    fuse_swiglu: bool = False,
):
    mma_m = _mma_m(capacity) if mma_m is None else mma_m
    # Large GEMMs pair adjacent M64 tiles so their identical N128 weight tile
    # is loaded once through TMA multicast.  The direct map stores one M128
    # cluster descriptor and compact scheduling applies the per-CTA M rank.
    cluster_shape = (1, 1) if capacity <= 1024 else (2, 1)
    use_2cta = False
    use_mtile_map = capacity >= 1024
    key = (
        capacity,
        n,
        k,
        mma_m,
        cluster_shape[0],
        cluster_shape[1],
        int(use_2cta),
        int(use_mtile_map),
        int(fuse_swiglu),
    )
    compiled = _gemm_cache.get(key)
    if compiled is not None:
        return compiled

    gemm = BlockwiseMaskedGroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta,
        mma_tiler_mn=(mma_m, 128),
        cluster_shape_mn=cluster_shape,
        use_mtile_map=use_mtile_map,
        fuse_swiglu=fuse_swiglu,
    )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape[0] * cluster_shape[1]
    )
    stream = _current_stream()
    compiled = cute.compile(
        gemm,
        _fake(
            cutlass.Float8E4M3FN,
            (capacity, k, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(
            cutlass.Float8E4M3FN,
            (2 * n if fuse_swiglu else n, k, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(
            cutlass.Float32 if fuse_swiglu else cutlass.BFloat16,
            (capacity, n, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(
            cutlass.Float32,
            (capacity, k // SCALE_BLOCK, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(
            cutlass.Float32,
            (
                (2 * n if fuse_swiglu else n) // SCALE_BLOCK,
                k // SCALE_BLOCK,
                LOCAL_EXPERTS,
            ),
            stride_order=(1, 0, 2),
        ),
        _fake(cutlass.Int32, (LOCAL_EXPERTS,), stride_order=(0,), align=4),
        _fake(
            cutlass.Int32,
            (_mtile_desc_rows(capacity, mma_m * cluster_shape[0]),),
            stride_order=(0,),
            align=4,
        ),
        _fake(cutlass.Int32, (1,), stride_order=(0,), align=4),
        max_active_clusters,
        stream,
        options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
    )
    _gemm_cache[key] = compiled
    return compiled


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Launch the CuTe-only routing, grouped-GEMM, activation, and combine path."""
    del routed_scaling_factor  # The experiment fixes this scalar at 2.5.
    global _workspace

    tokens = routing_logits.shape[0]
    capacity = _capacity(tokens)
    if (
        _workspace is None
        or _workspace.tokens != tokens
        or _workspace.capacity != capacity
        or _workspace.packed_a.device != hidden_states.device
    ):
        _workspace = _allocate_workspace(tokens, capacity, hidden_states.device)
    ws = _workspace
    stream = _current_stream()

    route = _get_route(tokens, capacity, int(local_expert_offset))
    mtile_map = _get_mtile_map(capacity) if capacity >= 1024 else None
    activation = _get_activation(capacity)
    combine = _get_combine(tokens, capacity)
    gemm_m = 64 if capacity >= 16384 else _mma_m(capacity)
    gemm1 = _get_gemm(
        capacity,
        2 * INTERMEDIATE,
        HIDDEN,
        mma_m=gemm_m,
        dual_output_fc1=tokens >= DUAL_OUTPUT_FC1_MIN_TOKENS,
    )
    gemm2 = _get_gemm(
        capacity,
        HIDDEN,
        INTERMEDIATE,
        mma_m=gemm_m,
    )

    route(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        ws.packed_a,
        ws.packed_sfa,
        ws.counts,
        ws.route_expert,
        ws.route_pos,
        ws.route_weight,
        stream,
    )

    if mtile_map is not None:
        mtile_map(
            ws.counts,
            ws.mtile_desc,
            ws.mtile_count,
            ws.main_desc,
            ws.main_count,
            ws.tail_desc,
            ws.tail_count,
            stream,
        )

    gemm_desc = ws.mtile_desc
    gemm_count = ws.mtile_count
    activation_desc = ws.tail_desc if gemm_m == 256 else ws.mtile_desc
    activation_count = ws.tail_count if gemm_m == 256 else ws.mtile_count
    gemm1(
        ws.packed_a.permute(1, 2, 0),
        gemm1_weights.permute(1, 2, 0),
        ws.gemm1_out.permute(1, 2, 0),
        ws.packed_sfa.permute(1, 2, 0),
        gemm1_weights_scale.permute(1, 2, 0),
        ws.counts,
        gemm_desc,
        gemm_count,
        stream,
    )
    if capacity >= 16384:
        activation(
            ws.gemm1_out,
            ws.activation,
            ws.activation_scale,
            ws.counts,
            activation_desc,
            activation_count,
            stream,
        )
    else:
        activation(
            ws.gemm1_out,
            ws.activation,
            ws.activation_scale,
            ws.route_expert,
            ws.route_pos,
            stream,
        )

    gemm2(
        ws.activation.permute(1, 2, 0),
        gemm2_weights.permute(1, 2, 0),
        ws.gemm2_out.permute(1, 2, 0),
        ws.activation_scale.permute(1, 2, 0),
        gemm2_weights_scale.permute(1, 2, 0),
        ws.counts,
        gemm_desc,
        gemm_count,
        stream,
    )
    output = torch.empty(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=hidden_states.device
    )
    combine(
        ws.gemm2_out,
        ws.route_expert,
        ws.route_pos,
        ws.route_weight,
        output,
        stream,
    )
    return output
