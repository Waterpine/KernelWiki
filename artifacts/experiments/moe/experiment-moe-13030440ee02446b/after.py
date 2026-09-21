"""DeepSeek-V3 FP8 block-scale MoE for Blackwell, written in CuTe DSL.

The implementation deliberately keeps PyTorch on the host side only for owning
workspace tensors.  Routing, packing, both GEMMs, SwiGLU/requantization, and the
weighted combine are GPU work implemented in CuTe DSL.

The grouped GEMM primitive is instantiated from the sibling CUTLASS CuTe
template vendored in ``blockwise_gemm.py``.  That primitive uses TMA,
warp-specialized tcgen05 MMA, TMEM accumulation, and a persistent scheduler.
Packing and weighted combine use shape-specialized 128-bit vector copies.
Activation-scale workspaces are K-block-major for coalesced GEMM scale loads.
Large SwiGLU uses two 16-lane block groups with 128-bit contiguous loads.
SwiGLU and requantization use fast FP32 reciprocal primitives.
Large-sequence routing uses a 32-token, one-warp-per-token fast-math path.
The launch-bound CTA-per-token router shares its approximate sigmoid and
normalization sequence while retaining the exact stable selection network;
the 901-token medium router keeps exact arithmetic.
Its retained-group candidates are merged by one stable warp sort.
Expert-prefix initialization is striped into coalesced eight-thread groups.
Routing retains each local route's atomic expert rank; the final routing CTA
builds the aligned prefix with one warp per expert, eliminating separate
prefix and mapping kernels.
For throughput shapes, the exact packed-row count remains device-resident:
capacity-backed row kernels and persistent GEMM schedulers consume it directly,
and one compiled executor enqueues the complete six-kernel pipeline.
Device-resident gather/compute chains use programmatic dependent launch to
pre-admit each successor during the producer tail while preserving dependency
waits.
Launch-bound shapes retain narrow, exact-extent gather/GEMM/SwiGLU interfaces.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr


_SOLUTION_DIR = str(Path(__file__).resolve().parent)
if _SOLUTION_DIR not in sys.path:
    sys.path.insert(0, _SOLUTION_DIR)

from blockwise_gemm import BlockwiseContiguousGroupedGemmKernel


# Fixed DeepSeek-V3 geometry.
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
BLOCK = 128
H_BLOCKS = H // BLOCK
I_BLOCKS = I // BLOCK
W13_BLOCKS = (2 * I) // BLOCK
THREADS = 256
GATHER_THREADS_SMALL = 512
GATHER_THREADS_LARGE = 128
GATHER_SMALL_THRESHOLD = 2048
GATHER_VECTOR_BITS = 128
GEMM1_CLUSTER_N_THRESHOLD = 4096
# Keep launch-bound shapes on the exact host extent; throughput shapes use the
# device-resident persistent scheduler validated by the full 20-workload suite.
DEVICE_EXTENT_THRESHOLD = GEMM1_CLUSTER_N_THRESHOLD
# CUPTI-selected small regime: halve per-expert padding and use one-CTA tiles.
SMALL_ROW_ALIGNMENT = 64
GEMM1_TINY_THRESHOLD = 128
GEMM1_TINY_CLUSTER = (1, 2)
GEMM1_MEDIUM_CLUSTER = (1, 1)
GEMM1_LARGE_CLUSTER = (2, 2)
ROUTING_FAST_THRESHOLD = 4096
ROUTING_TOKENS_PER_CTA = 32
# CUPTI-selected launch regimes: tiny shapes favor width; denser shapes favor
# resident CTA count, while the large stream is bandwidth/throughput limited.
GEMM2_TINY_THRESHOLD = 128
SWIGLU_THREADS_TINY = 512
SWIGLU_THREADS_SMALL = 256
SWIGLU_THREADS_LARGE = 128
SWIGLU_TINY_THRESHOLD = 7
SWIGLU_LARGE_THRESHOLD = 4096
COMBINE_THREADS_SMALL = 256
COMBINE_THREADS_LARGE = 160
COMBINE_SMALL_THRESHOLD = 2048
COMBINE_VECTOR_BITS = 128
COMBINE_VECTOR_VALUES_SMALL = 16
COMBINE_VECTOR_VALUES_LARGE = 16
NEG_INF = -3.402823466e38
INT_MAX = 2147483647
FP8_MAX = 448.0


@cute.jit
def _warp_argmax(
    value: Float32,
    index: Int32,
    width: cutlass.Constexpr[int],
) -> Tuple[Float32, Int32]:
    """Deterministic (value, -index) argmax within power-of-two warp groups."""
    for offset in (16, 8, 4, 2, 1):
        if cutlass.const_expr(offset < width):
            other_value = cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=-1, mask_and_clamp=31
            )
            other_index = cute.arch.shuffle_sync_bfly(
                index, offset=offset, mask=-1, mask_and_clamp=31
            )
            take_other = (other_value > value) | (
                (other_value == value) & (other_index < index)
            )
            if take_other:
                value = other_value
                index = other_index
    return value, index


@cute.jit
def _warp_sum(
    value: Float32,
    width: cutlass.Constexpr[int],
) -> Float32:
    for offset in (16, 8, 4, 2, 1):
        if cutlass.const_expr(offset < width):
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=-1, mask_and_clamp=31
            )
    return value


@cute.jit
def _warp_sort_desc(
    value: Float32,
    index: Int32,
    lane: Int32,
    width: cutlass.Constexpr[int],
) -> Tuple[Float32, Int32]:
    """CUPTI-selected stable warp sort by (value desc, index asc)."""
    for span in (2, 4, 8, 16, 32):
        if cutlass.const_expr(span <= width):
            stride = span // 2
            while stride > 0:
                other_value = cute.arch.shuffle_sync_bfly(
                    value, offset=stride, mask=-1, mask_and_clamp=31
                )
                other_index = cute.arch.shuffle_sync_bfly(
                    index, offset=stride, mask=-1, mask_and_clamp=31
                )
                other_better = (other_value > value) | (
                    (other_value == value) & (other_index < index)
                )
                current_better = (value > other_value) | (
                    (value == other_value) & (index < other_index)
                )
                low_lane = (lane & stride) == 0
                descending_span = (lane & span) == 0
                want_better = low_lane == descending_span
                take_other = other_better
                if not want_better:
                    take_other = current_better
                if take_other:
                    value = other_value
                    index = other_index
                stride = stride // 2
    return value, index


@cute.jit
def _warp_bitonic_merge_desc_8(
    value: Float32,
    index: Int32,
    lane_in_8: Int32,
) -> Tuple[Float32, Int32]:
    """Merge one eight-lane bitonic sequence into stable descending order."""
    for stride in (4, 2, 1):
        other_value = cute.arch.shuffle_sync_bfly(
            value, offset=stride, mask=-1, mask_and_clamp=31
        )
        other_index = cute.arch.shuffle_sync_bfly(
            index, offset=stride, mask=-1, mask_and_clamp=31
        )
        other_better = (other_value > value) | (
            (other_value == value) & (other_index < index)
        )
        current_better = (value > other_value) | (
            (value == other_value) & (index < other_index)
        )
        if (lane_in_8 & stride) == 0:
            if other_better:
                value = other_value
                index = other_index
        else:
            if current_better:
                value = other_value
                index = other_index
    return value, index


@cute.jit
def _warp_merge_top8_desc_4x8(
    value: Float32,
    index: Int32,
    lane: Int32,
) -> Tuple[Float32, Int32]:
    """Merge four presorted eight-lane lists and retain a sorted top eight."""
    lane_in_8 = lane & 7
    lane_in_16 = lane & 15

    # Merge each pair of sorted eight-element lists, retaining its best eight
    # in the low half of the corresponding sixteen-lane group.
    pair_base = lane - lane_in_16
    reverse_source = pair_base + 15 - lane_in_16
    other_value = cute.arch.shuffle_sync(
        value, offset=reverse_source, mask=-1, mask_and_clamp=31
    )
    other_index = cute.arch.shuffle_sync(
        index, offset=reverse_source, mask=-1, mask_and_clamp=31
    )
    if lane_in_16 < 8:
        other_better = (other_value > value) | (
            (other_value == value) & (other_index < index)
        )
        if other_better:
            value = other_value
            index = other_index
    value, index = _warp_bitonic_merge_desc_8(
        value, index, lane_in_8
    )

    # Lanes 0--7 and 16--23 now hold the two sorted top-eight lists; the
    # remaining merge is sufficient because callers never consume the tail.
    other_value = cute.arch.shuffle_sync(
        value, offset=23 - lane_in_8, mask=-1, mask_and_clamp=31
    )
    other_index = cute.arch.shuffle_sync(
        index, offset=23 - lane_in_8, mask=-1, mask_and_clamp=31
    )
    if lane < 8:
        other_better = (other_value > value) | (
            (other_value == value) & (other_index < index)
        )
        if other_better:
            value = other_value
            index = other_index
    value, index = _warp_bitonic_merge_desc_8(
        value, index, lane_in_8
    )
    return value, index


@cute.jit
def _warp_top8_desc_32(
    value: Float32,
    index: Int32,
    lane: Int32,
) -> Tuple[Float32, Int32]:
    """Stable 32-to-8 selection network; only lanes zero through seven survive."""
    value, index = _warp_sort_desc(
        value, index, lane & 7, cutlass.const_expr(8)
    )
    return _warp_merge_top8_desc_4x8(value, index, lane)


@cute.jit
def _warp_top4_desc_8(
    value: Float32,
    index: Int32,
    lane: Int32,
) -> Tuple[Float32, Int32]:
    """Stable four-stage eight-to-four selection; winners occupy lanes 0--3."""
    value, index = _warp_sort_desc(
        value, index, lane, cutlass.const_expr(4)
    )
    other_value = cute.arch.shuffle_sync_bfly(
        value, offset=4, mask=-1, mask_and_clamp=31
    )
    other_index = cute.arch.shuffle_sync_bfly(
        index, offset=4, mask=-1, mask_and_clamp=31
    )
    if lane < 4:
        other_better = (other_value > value) | (
            (other_value == value) & (other_index < index)
        )
        if other_better:
            value = other_value
            index = other_index
    return value, index


@cute.jit
def _finish_routing_prefix(
    tidx: Int32,
    last_ticket: cute.Tensor,
    expert_counts: cute.Tensor,
    offsets: cute.Tensor,
    packed_group_index: cute.Tensor,
    total_m: cute.Tensor,
    row_alignment: cutlass.Constexpr[int],
):
    """Let the final routing CTA build the aligned expert prefix in place."""
    # The CTA barrier orders every routing atomic before the elected thread's
    # device-scope publication of this block's completion ticket.
    cute.arch.sync_threads()

    grid_x, _, _ = cute.arch.grid_dim()
    if tidx == 0:
        cute.arch.fence_acq_rel_gpu()
        last_ticket[0] = cute.arch.atomic_add(
            offsets.iterator,
            Int32(1),
            sem="acq_rel",
            scope="gpu",
        )
    cute.arch.sync_threads()

    if last_ticket[0] == grid_x - 1:
        if tidx == 0:
            running = Int32(0)
            offsets[0] = running
            for expert in cutlass.range_constexpr(E_LOCAL):
                count = expert_counts[expert]
                padded = (
                    (count + (row_alignment - 1)) // row_alignment
                ) * row_alignment
                running = running + padded
                offsets[expert + 1] = running
            total_m[0] = running
            packed_group_index[cute.size(packed_group_index) - 1] = running
        cute.arch.sync_threads()

        # Eight threads stripe each expert's aligned row range.
        if tidx < THREADS:
            expert = tidx >> 3
            expert_lane = tidx & 7
            begin = offsets[expert]
            end = offsets[expert + 1]
            row = begin + expert_lane
            while row < end:
                packed_group_index[row] = expert
                row = row + 8


@cute.kernel
def _routing_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    top_ids: cute.Tensor,
    top_weights: cute.Tensor,
    expanded_to_row: cute.Tensor,
    expert_counts: cute.Tensor,
    offsets: cute.Tensor,
    expert_routes: cute.Tensor,
    packed_group_index: cute.Tensor,
    total_m: cute.Tensor,
    num_tokens: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    use_fast_math: cutlass.Constexpr[bool],
    row_alignment: cutlass.Constexpr[int],
):
    """One CTA per token: sigmoid, group-top2/top4, global top8, normalize."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    expert = tidx

    smem = cutlass.utils.SmemAllocator()
    s_candidate_value = smem.allocate_tensor(
        Float32, cute.make_layout((8 * TOP_K,), stride=(1,)), 16
    )
    s_candidate_index = smem.allocate_tensor(
        Int32, cute.make_layout((8 * TOP_K,), stride=(1,)), 16
    )
    s_selected_group = smem.allocate_tensor(
        Int32, cute.make_layout((4,), stride=(1,)), 16
    )
    s_last_ticket = smem.allocate_tensor(
        Int32, cute.make_layout((1,), stride=(1,)), 16
    )

    logit = logits[token, expert]
    raw_score = cute.arch.rcp_approx(
        Float32(1.0) + cute.exp(-logit, fastmath=True)
    )
    biased_score = raw_score + bias[expert].to(Float32)

    # One stable sort produces every group's top eight. This is both the group
    # top-two score and the complete candidate set for the global merge.
    group_value, group_index = _warp_sort_desc(
        biased_score, expert, lane, cutlass.const_expr(32)
    )
    if lane < TOP_K:
        candidate_slot = warp * TOP_K + lane
        s_candidate_value[candidate_slot] = group_value
        s_candidate_index[candidate_slot] = group_index
    cute.arch.barrier()

    # Warp 0 selects four of the eight group scores.
    if warp == 0:
        group_value = Float32(NEG_INF)
        group_index = Int32(INT_MAX)
        if lane < 8:
            group_value = (
                s_candidate_value[lane * TOP_K]
                + s_candidate_value[lane * TOP_K + 1]
            )
            group_index = lane
        _, selected_group = _warp_sort_desc(
            group_value,
            packed_group_index,
            lane,
            cutlass.const_expr(8),
        )
        if lane < 4:
            s_selected_group[lane] = selected_group
    cute.arch.barrier()

    # Warp 0 gathers exactly the 32 candidates from the four retained groups.
    # A single stable sort replaces eight rounds of warp-wide argmax.
    if warp == 0:
        selected_group = s_selected_group[lane // TOP_K]
        candidate_pos = selected_group * TOP_K + (lane & (TOP_K - 1))
        candidate_value = s_candidate_value[candidate_pos]
        candidate_index = s_candidate_index[candidate_pos]
        selected_value, selected_expert = _warp_merge_top8_desc_4x8(
            candidate_value, candidate_index, lane
        )
        selected_raw = Float32(0.0)
        if lane < TOP_K:
            selected_raw = selected_value - bias[selected_expert].to(Float32)

        # The first eight lanes write routing outputs and initialize the
        # inverse permutation map used by the final combine.
        raw_sum = _warp_sum(selected_raw, cutlass.const_expr(8))
        inv_raw_sum = cute.arch.rcp_approx(raw_sum)
        if lane < TOP_K:
            out_idx = token * TOP_K + lane
            weight = selected_raw * routed_scale * inv_raw_sum
            top_ids[out_idx] = selected_expert
            top_weights[out_idx] = weight
            expanded_to_row[out_idx] = Int32(-1)
            local_expert = selected_expert - local_offset
            if (local_expert >= 0) & (local_expert < E_LOCAL):
                rank = cute.arch.atomic_add(
                    expert_counts.iterator + local_expert,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                expanded_to_row[out_idx] = rank
                expert_routes[local_expert * num_tokens + rank] = out_idx
    _finish_routing_prefix(
        tidx,
        s_last_ticket,
        expert_counts,
        offsets,
        packed_group_index,
        total_m,
        row_alignment,
    )
    _finish_routing_prefix(
        tidx,
        s_last_ticket,
        expert_counts,
        offsets,
        expert_write,
        permuted_to_expanded,
        group_index,
        total_m,
        row_alignment,
    )


@cute.kernel
def _routing_warp_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    top_ids: cute.Tensor,
    top_weights: cute.Tensor,
    expanded_to_row: cute.Tensor,
    expert_counts: cute.Tensor,
    offsets: cute.Tensor,
    expert_write: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    packed_group_index: cute.Tensor,
    total_m: cute.Tensor,
    num_tokens: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    row_alignment: cutlass.Constexpr[int],
):
    """One warp per token: sequential groups, warp-local exact selection."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    token = bidx * ROUTING_TOKENS_PER_CTA + warp

    smem = cutlass.utils.SmemAllocator()
    s_candidate_value = smem.allocate_tensor(
        Float32,
        cute.make_layout(
            (ROUTING_TOKENS_PER_CTA, 8 * TOP_K),
            stride=(8 * TOP_K, 1),
        ),
        16,
    )
    s_candidate_index = smem.allocate_tensor(
        Int32,
        cute.make_layout(
            (ROUTING_TOKENS_PER_CTA, 8 * TOP_K),
            stride=(8 * TOP_K, 1),
        ),
        16,
    )
    s_group_score = smem.allocate_tensor(
        Float32,
        cute.make_layout(
            (ROUTING_TOKENS_PER_CTA, 8), stride=(8, 1)
        ),
        16,
    )
    s_last_ticket = smem.allocate_tensor(
        Int32, cute.make_layout((1,), stride=(1,)), 16
    )

    if token < num_tokens:
        for group in cutlass.range_constexpr(8):
            expert = group * 32 + lane
            logit = logits[token, expert]
            raw_score = cute.arch.rcp_approx(
                Float32(1.0) + cute.exp(-logit, fastmath=True)
            )
            biased_score = raw_score + bias[expert].to(Float32)
            group_value, group_index = _warp_top8_desc_32(
                biased_score, expert, lane
            )
            second_value = cute.arch.shuffle_sync(
                group_value, offset=1, mask=-1, mask_and_clamp=31
            )
            if lane < TOP_K:
                candidate_slot = group * TOP_K + lane
                s_candidate_value[warp, candidate_slot] = group_value
                s_candidate_index[warp, candidate_slot] = group_index
            if lane == 0:
                s_group_score[warp, group] = group_value + second_value

        cute.arch.sync_warp()

        group_value = Float32(NEG_INF)
        group_index = Int32(INT_MAX)
        if lane < 8:
            group_value = s_group_score[warp, lane]
            group_index = lane
        # Candidate gathering only needs the winning group set, not its order.
        _, selected_group = _warp_top4_desc_8(
            group_value, group_index, lane
        )
        selected_group = cute.arch.shuffle_sync(
            selected_group,
            offset=lane // TOP_K,
            mask=-1,
            mask_and_clamp=31,
        )

        candidate_pos = selected_group * TOP_K + (lane & (TOP_K - 1))
        candidate_value = s_candidate_value[warp, candidate_pos]
        candidate_index = s_candidate_index[warp, candidate_pos]
        selected_value, selected_expert = _warp_sort_desc(
            candidate_value,
            candidate_index,
            lane,
            cutlass.const_expr(32),
        )
        selected_raw = Float32(0.0)
        if lane < TOP_K:
            selected_raw = selected_value - bias[selected_expert].to(Float32)

        raw_sum = _warp_sum(selected_raw, cutlass.const_expr(8))
        inv_raw_sum = cute.arch.rcp_approx(raw_sum)
        if lane < TOP_K:
            out_idx = token * TOP_K + lane
            top_ids[out_idx] = selected_expert
            top_weights[out_idx] = (
                selected_raw * routed_scale * inv_raw_sum
            )
            expanded_to_row[out_idx] = Int32(-1)
            local_expert = selected_expert - local_offset
            if (local_expert >= 0) & (local_expert < E_LOCAL):
                cute.arch.atomic_add(
                    expert_counts.iterator + local_expert,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )


@cute.jit
def _launch_routing(
    logits_ptr: cute.Pointer,
    bias_ptr: cute.Pointer,
    top_ids_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_routes_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    use_fast_math: cutlass.Constexpr[bool],
    row_alignment: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    logits = cute.make_tensor(
        logits_ptr,
        cute.make_layout((num_tokens, E_GLOBAL), stride=(E_GLOBAL, 1)),
    )
    bias = cute.make_tensor(
        bias_ptr, cute.make_layout((E_GLOBAL,), stride=(1,))
    )

    offsets = cute.make_tensor(
        offsets_ptr, cute.make_layout((E_LOCAL + 1,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    top_weights = cute.make_tensor(
        top_weights_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    top_ids = cute.make_tensor(
        top_ids_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    expanded_to_row = cute.make_tensor(
        expanded_to_row_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    expert_counts = cute.make_tensor(
        expert_counts_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    offsets = cute.make_tensor(
        offsets_ptr, cute.make_layout((E_LOCAL + 1,), stride=(1,))
    )
    expert_routes = cute.make_tensor(
        expert_routes_ptr,
        cute.make_layout((E_LOCAL * num_tokens,), stride=(1,)),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    if cutlass.const_expr(use_fast_math):
        _routing_warp_kernel(
            logits,
            bias,
            top_ids,
            top_weights,
            expanded_to_row,
            expert_counts,
            offsets,
            expert_routes,
            group_index,
            total_m,
            num_tokens,
            local_offset,
            routed_scale,
            row_alignment,
        ).launch(
            grid=[cute.ceil_div(num_tokens, ROUTING_TOKENS_PER_CTA), 1, 1],
            block=[ROUTING_TOKENS_PER_CTA * 32, 1, 1],
            # Two 64-entry candidate tables plus eight group scores per warp.
            smem=ROUTING_TOKENS_PER_CTA * 544 + 16,
            stream=stream,
        )
    else:
        _routing_kernel(
            logits,
            bias,
            top_ids,
            top_weights,
            expanded_to_row,
            expert_counts,
            offsets,
            expert_routes,
            group_index,
            total_m,
            num_tokens,
            local_offset,
            routed_scale,
            use_fast_math,
            row_alignment,
        ).launch(
            grid=[num_tokens, 1, 1],
            block=[THREADS, 1, 1],
            smem=1040,
            stream=stream,
        )


@cute.kernel
def _prefix_kernel(
    expert_counts: cute.Tensor,
    offsets: cute.Tensor,
    expert_write: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    group_index: cute.Tensor,
    total_m: cute.Tensor,
    row_alignment: cutlass.Constexpr[int],
):
    """Build aligned expert offsets from routing counts and initialize maps."""
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        running = Int32(0)
        offsets[0] = running
        for expert in cutlass.range_constexpr(E_LOCAL):
            count = expert_counts[expert]
            padded = (
                (count + (row_alignment - 1)) // row_alignment
            ) * row_alignment
            running = running + padded
            offsets[expert + 1] = running
            expert_write[expert] = Int32(0)
        total_m[0] = running
        # The conservative mapping allocation always leaves a final sentinel
        # row. Large persistent GEMMs consume the exact M extent from here
        # without adding another runtime kernel argument.
        group_index[cute.size(group_index) - 1] = running
    cute.arch.barrier()

    # Eight threads stripe each expert's aligned row range.
    expert = tidx >> 3
    expert_lane = tidx & 7
    begin = offsets[expert]
    end = offsets[expert + 1]
    row = begin + expert_lane
    while row < end:
        permuted_to_expanded[row] = Int32(-1)
        group_index[row] = expert
        row = row + 8


@cute.jit
def _launch_prefix(
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_write_ptr: cute.Pointer,
    permuted_to_expanded_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    max_m: Int32,
    row_alignment: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    expert_counts = cute.make_tensor(
        expert_counts_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    offsets = cute.make_tensor(
        offsets_ptr, cute.make_layout((E_LOCAL + 1,), stride=(1,))
    )
    expert_write = cute.make_tensor(
        expert_write_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    permuted_to_expanded = cute.make_tensor(
        expert_routes_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    _prefix_kernel(
        expert_counts,
        offsets,
        expert_write,
        permuted_to_expanded,
        group_index,
        total_m,
        row_alignment,
    ).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_route_prefix(
    logits_ptr: cute.Pointer,
    bias_ptr: cute.Pointer,
    top_ids_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_routes_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    use_fast_math: cutlass.Constexpr[bool],
    row_alignment: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Route, build the aligned prefix in the final CTA, and retain ranks."""
    _launch_routing(
        logits_ptr,
        bias_ptr,
        top_ids_ptr,
        top_weights_ptr,
        expanded_to_row_ptr,
        expert_counts_ptr,
        offsets_ptr,
        expert_routes_ptr,
        group_index_ptr,
        total_m_ptr,
        num_tokens,
        max_m,
        local_offset,
        routed_scale,
        use_fast_math,
        row_alignment,
        stream,
    )


@cute.kernel
def _hybrid_native_remap_kernel(
    expert_counts: cute.Tensor,
    offsets: cute.Tensor,
    group_index: cute.Tensor,
    total_m: cute.Tensor,
    full_capacity: Int32,
):
    """Build one expert id per cooperative/body tile and one per tail tile."""
    tidx, _, _ = cute.arch.thread_idx()
    tail_mapping_base = full_capacity + 1

    if tidx == 0:
        running_full = Int32(0)
        running_tail = Int32(0)
        offsets[0] = Int32(0)
        offsets[E_LOCAL + 1] = Int32(0)
        for expert in cutlass.range_constexpr(E_LOCAL):
            count = expert_counts[expert]
            full_count = (count // (2 * BLOCK)) * (2 * BLOCK)
            tail_count = count - full_count
            tail_padded = (
                (tail_count + (BLOCK - 1)) // BLOCK
            ) * BLOCK
            running_full = running_full + full_count
            running_tail = running_tail + tail_padded
            offsets[expert + 1] = running_full
            offsets[E_LOCAL + 2 + expert] = running_tail
        total_m[0] = running_full
        total_m[1] = running_tail
        # Each persistent scheduler reads its exact extent from the final
        # element of its otherwise row-indexed mapping view.
        group_index[full_capacity] = running_full
        group_index[
            tail_mapping_base + NATIVE_TAIL_CAPACITY
        ] = running_tail
    cute.arch.barrier()

    if tidx < E_LOCAL:
        expert = tidx
        row = offsets[expert]
        full_end = offsets[expert + 1]
        while row < full_end:
            group_index[row] = expert
            # The cooperative instruction spans M256, but each partner CTA
            # queries the mapping at its own M128 scheduler coordinate.
            row = row + BLOCK

        tail_local = offsets[E_LOCAL + 1 + expert]
        tail_local_end = offsets[E_LOCAL + 2 + expert]
        while tail_local < tail_local_end:
            group_index[tail_mapping_base + tail_local] = expert
            tail_local = tail_local + BLOCK


@cute.jit
def _launch_hybrid_native_remap(
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    max_m: Int32,
    stream: cuda.CUstream,
):
    expert_counts = cute.make_tensor(
        expert_counts_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    offsets = cute.make_tensor(
        offsets_ptr,
        cute.make_layout((2 * (E_LOCAL + 1),), stride=(1,)),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((max_m + 2,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((2,), stride=(1,))
    )
    _hybrid_native_remap_kernel(
        expert_counts,
        offsets,
        group_index,
        total_m,
        max_m - NATIVE_TAIL_CAPACITY,
    ).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _build_mapping_kernel(
    top_ids: cute.Tensor,
    offsets: cute.Tensor,
    expert_write: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    expanded_to_row: cute.Tensor,
    num_pairs: Int32,
    local_offset: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    pair_idx = bidx * THREADS + tidx
    if pair_idx < num_pairs:
        expert = top_ids[pair_idx]
        local_expert = expert - local_offset
        if (local_expert >= 0) & (local_expert < E_LOCAL):
            rank = cute.arch.atomic_add(
                expert_write.iterator + local_expert,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            row = offsets[local_expert] + rank
            permuted_to_expanded[row] = pair_idx
            expanded_to_row[pair_idx] = row


@cute.jit
def _launch_build_mapping(
    top_ids_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_write_ptr: cute.Pointer,
    permuted_to_expanded_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    local_offset: Int32,
    stream: cuda.CUstream,
):
    num_pairs = num_tokens * TOP_K
    top_ids = cute.make_tensor(
        top_ids_ptr, cute.make_layout((num_pairs,), stride=(1,))
    )
    offsets = cute.make_tensor(
        offsets_ptr, cute.make_layout((E_LOCAL + 1,), stride=(1,))
    )
    expert_write = cute.make_tensor(
        expert_write_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    permuted_to_expanded = cute.make_tensor(
        expert_routes_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    expanded_to_row = cute.make_tensor(
        expanded_to_row_ptr, cute.make_layout((num_pairs,), stride=(1,))
    )
    _build_mapping_kernel(
        top_ids,
        offsets,
        expert_write,
        permuted_to_expanded,
        expanded_to_row,
        num_pairs,
        local_offset,
    ).launch(
        grid=[cute.ceil_div(num_pairs, THREADS), 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _gather_kernel(
    hidden: cute.Tensor,
    hidden_scale: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    packed_hidden: cute.Tensor,
    packed_scale: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    thread_count: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    expanded = permuted_to_expanded[row]
    token = Int32(-1)
    if expanded >= 0:
        token = expanded // TOP_K

    # Move 16 adjacent FP8 values per thread with a single 128-bit load/store.
    # Padded expert rows retain the same explicit zero-fill semantics.
    safe_token = token
    if safe_token < 0:
        safe_token = Int32(0)
    g_src = cute.local_tile(
        hidden, tiler=(1, H), coord=(safe_token, 0)
    )[0, None]
    g_dst = cute.local_tile(
        packed_hidden, tiler=(1, H), coord=(row, 0)
    )[0, None]
    thr_copy = tiled_copy.get_slice(tidx)
    t_src = thr_copy.partition_S(g_src)
    t_dst = thr_copy.partition_S(g_dst)
    r_value = cute.make_fragment_like(t_src)
    r_value.fill(0)
    for i in range(cute.size(r_value, mode=[1])):
        col = (i * thread_count + tidx) * 16
        if col < H:
            if token >= 0:
                cute.autovec_copy(t_src[None, i], r_value[None, i])
            cute.autovec_copy(r_value[None, i], t_dst[None, i])

    if tidx < H_BLOCKS:
        scale = Float32(1.0)
        if token >= 0:
            scale = hidden_scale[tidx, token]
        packed_scale[row, tidx] = scale


@cute.jit
def _launch_gather(
    hidden_ptr: cute.Pointer,
    hidden_scale_ptr: cute.Pointer,
    permuted_to_expanded_ptr: cute.Pointer,
    packed_hidden_ptr: cute.Pointer,
    packed_scale_ptr: cute.Pointer,
    num_tokens: Int32,
    m: Int32,
    thread_count: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    hidden = cute.make_tensor(
        hidden_ptr, cute.make_layout((num_tokens, H), stride=(H, 1))
    )
    hidden_scale = cute.make_tensor(
        hidden_scale_ptr,
        cute.make_layout((H_BLOCKS, num_tokens), stride=(num_tokens, 1)),
    )
    permuted_to_expanded = cute.make_tensor(
        permuted_to_expanded_ptr, cute.make_layout((m,), stride=(1,))
    )
    packed_hidden = cute.make_tensor(
        packed_hidden_ptr, cute.make_layout((m, H), stride=(H, 1))
    )
    packed_scale = cute.make_tensor(
        packed_scale_ptr,
        cute.make_layout((m, H_BLOCKS), stride=(1, m)),
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float8E4M3FN,
        num_bits_per_copy=GATHER_VECTOR_BITS,
    )
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout(thread_count),
        cute.make_layout(16),
    )
    _gather_kernel(
        hidden,
        hidden_scale,
        permuted_to_expanded,
        packed_hidden,
        packed_scale,
        tiled_copy,
        thread_count,
        vector_values,
    ).launch(
        grid=[m, 1, 1],
        block=[thread_count, 1, 1],
        stream=stream,
    )


@cute.kernel
def _gather_exact_kernel(
    hidden: cute.Tensor,
    hidden_scale: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    packed_hidden: cute.Tensor,
    packed_scale: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    thread_count: cutlass.Constexpr[int],
):
    """Original one-CTA-per-row gather for exact host-sized shapes."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    expanded = permuted_to_expanded[row]
    token = Int32(-1)
    if expanded >= 0:
        token = expanded // TOP_K

    safe_token = token
    if safe_token < 0:
        safe_token = Int32(0)
    g_src = cute.local_tile(
        hidden, tiler=(1, H), coord=(safe_token, 0)
    )[0, None]
    g_dst = cute.local_tile(
        packed_hidden, tiler=(1, H), coord=(row, 0)
    )[0, None]
    thr_copy = tiled_copy.get_slice(tidx)
    t_src = thr_copy.partition_S(g_src)
    t_dst = thr_copy.partition_S(g_dst)
    r_value = cute.make_fragment_like(t_src)
    r_value.fill(0)
    for i in range(cute.size(r_value, mode=[1])):
        col = (i * thread_count + tidx) * 16
        if col < H:
            if token >= 0:
                cute.autovec_copy(t_src[None, i], r_value[None, i])
            cute.autovec_copy(r_value[None, i], t_dst[None, i])

    if tidx < H_BLOCKS:
        scale = Float32(1.0)
        if token >= 0:
            scale = hidden_scale[tidx, token]
        packed_scale[row, tidx] = scale


@cute.jit
def _launch_gather_exact(
    hidden_ptr: cute.Pointer,
    hidden_scale_ptr: cute.Pointer,
    permuted_to_expanded_ptr: cute.Pointer,
    packed_hidden_ptr: cute.Pointer,
    packed_scale_ptr: cute.Pointer,
    num_tokens: Int32,
    m: Int32,
    thread_count: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    hidden = cute.make_tensor(
        hidden_ptr, cute.make_layout((num_tokens, H), stride=(H, 1))
    )
    hidden_scale = cute.make_tensor(
        hidden_scale_ptr,
        cute.make_layout((H_BLOCKS, num_tokens), stride=(num_tokens, 1)),
    )
    permuted_to_expanded = cute.make_tensor(
        permuted_to_expanded_ptr, cute.make_layout((m,), stride=(1,))
    )
    packed_hidden = cute.make_tensor(
        packed_hidden_ptr, cute.make_layout((m, H), stride=(H, 1))
    )
    packed_scale = cute.make_tensor(
        packed_scale_ptr,
        cute.make_layout((m, H_BLOCKS), stride=(1, m)),
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float8E4M3FN,
        num_bits_per_copy=GATHER_VECTOR_BITS,
    )
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout(thread_count),
        cute.make_layout(16),
    )
    _gather_exact_kernel(
        hidden,
        hidden_scale,
        permuted_to_expanded,
        packed_hidden,
        packed_scale,
        tiled_copy,
        thread_count,
    ).launch(
        grid=[m, 1, 1],
        block=[thread_count, 1, 1],
        stream=stream,
    )


@cute.kernel
def _gather_real_routes_kernel(
    hidden: cute.Tensor,
    hidden_scale: cute.Tensor,
    expanded_to_row: cute.Tensor,
    packed_hidden: cute.Tensor,
    packed_scale: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    thread_count: cutlass.Constexpr[int],
):
    """Pack only real local routes; aligned padding rows are never observed."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    s_rows = smem.allocate_tensor(
        Int32, cute.make_layout((TOP_K,), stride=(1,)), 16
    )
    if tidx < TOP_K:
        s_rows[tidx] = expanded_to_row[token * TOP_K + tidx]
    cute.arch.barrier()

    g_src = cute.local_tile(
        hidden, tiler=(1, H), coord=(token, 0)
    )[0, None]
    thr_copy = tiled_copy.get_slice(tidx)
    t_src = thr_copy.partition_S(g_src)
    r_value = cute.make_fragment_like(t_src)
    for slot in cutlass.range_constexpr(TOP_K):
        row = s_rows[slot]
        if row >= 0:
            g_dst = cute.local_tile(
                packed_hidden, tiler=(1, H), coord=(row, 0)
            )[0, None]
            t_dst = thr_copy.partition_S(g_dst)
            for wave in range(cute.size(r_value, mode=[1])):
                col = (wave * thread_count + tidx) * 16
                if col < H:
                    cute.autovec_copy(t_src[None, wave], r_value[None, wave])
                    cute.autovec_copy(r_value[None, wave], t_dst[None, wave])

            if tidx < H_BLOCKS:
                packed_scale[row, tidx] = hidden_scale[tidx, token]


@cute.jit
def _launch_gather_real_routes(
    hidden_ptr: cute.Pointer,
    hidden_scale_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    packed_hidden_ptr: cute.Pointer,
    packed_scale_ptr: cute.Pointer,
    num_tokens: Int32,
    m: Int32,
    thread_count: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    hidden = cute.make_tensor(
        hidden_ptr, cute.make_layout((num_tokens, H), stride=(H, 1))
    )
    hidden_scale = cute.make_tensor(
        hidden_scale_ptr,
        cute.make_layout((H_BLOCKS, num_tokens), stride=(num_tokens, 1)),
    )
    expanded_to_row = cute.make_tensor(
        expanded_to_row_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    packed_hidden = cute.make_tensor(
        packed_hidden_ptr, cute.make_layout((m, H), stride=(H, 1))
    )
    packed_scale = cute.make_tensor(
        packed_scale_ptr,
        cute.make_layout((m, H_BLOCKS), stride=(H_BLOCKS, 1)),
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float8E4M3FN,
        num_bits_per_copy=GATHER_VECTOR_BITS,
    )
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout(thread_count),
        cute.make_layout(16),
    )
    _gather_real_routes_kernel(
        hidden,
        hidden_scale,
        expanded_to_row,
        packed_hidden,
        packed_scale,
        tiled_copy,
        thread_count,
    ).launch(
        grid=[num_tokens, 1, 1],
        block=[thread_count, 1, 1],
        smem=32,
        stream=stream,
    )


@cute.jit
def _launch_device_pipeline(
    logits_ptr: cute.Pointer,
    bias_ptr: cute.Pointer,
    top_ids_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_routes_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    hidden_ptr: cute.Pointer,
    hidden_scale_ptr: cute.Pointer,
    packed_hidden_ptr: cute.Pointer,
    packed_scale_ptr: cute.Pointer,
    gemm1_weight_ptr: cute.Pointer,
    gemm1_weight_scale_ptr: cute.Pointer,
    gemm1_out_ptr: cute.Pointer,
    quantized_ptr: cute.Pointer,
    quantized_scale_ptr: cute.Pointer,
    gemm2_weight_ptr: cute.Pointer,
    gemm2_weight_scale_ptr: cute.Pointer,
    gemm2_out_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    row_ctas: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    gemm1_max_active_clusters: cutlass.Constexpr[int],
    gemm2_max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Enqueue the complete device-sized large MoE pipeline."""
    _launch_route_prefix(
        logits_ptr,
        bias_ptr,
        top_ids_ptr,
        top_weights_ptr,
        expanded_to_row_ptr,
        expert_counts_ptr,
        offsets_ptr,
        permuted_to_expanded_ptr,
        group_index_ptr,
        total_m_ptr,
        num_tokens,
        max_m,
        local_offset,
        routed_scale,
        True,
        BLOCK,
        stream,
    )
    _launch_gather_native(
        hidden_ptr,
        hidden_scale_ptr,
        permuted_to_expanded_ptr,
        expert_counts_ptr,
        offsets_ptr,
        group_index_ptr,
        packed_hidden_ptr,
        packed_scale_ptr,
        total_m_ptr,
        num_tokens,
        max_m,
        row_ctas,
        GATHER_THREADS_LARGE,
        True,
        stream,
    )
    _launch_gemm1(
        packed_hidden_ptr,
        gemm1_weight_ptr,
        gemm1_out_ptr,
        packed_scale_ptr,
        gemm1_weight_scale_ptr,
        group_index_ptr,
        max_m,
        gemm1_max_active_clusters,
        2,
        stream,
    )
    _launch_swiglu_quant(
        gemm1_out_ptr,
        quantized_ptr,
        quantized_scale_ptr,
        total_m_ptr,
        max_m,
        row_ctas,
        SWIGLU_THREADS_LARGE,
        True,
        stream,
    )
    _launch_gemm2(
        quantized_ptr,
        gemm2_weight_ptr,
        gemm2_out_ptr,
        quantized_scale_ptr,
        gemm2_weight_scale_ptr,
        group_index_ptr,
        max_m,
        gemm2_max_active_clusters,
        2,
        stream,
    )
    _launch_combine(
        gemm2_out_ptr,
        top_weights_ptr,
        top_ids_ptr,
        expanded_to_row_ptr,
        offsets_ptr,
        output_ptr,
        expert_counts_ptr,
        num_tokens,
        max_m,
        local_offset,
        COMBINE_THREADS_LARGE,
        COMBINE_VECTOR_VALUES_LARGE,
        stream,
    )


@cute.jit
def _launch_native_device_pipeline(
    logits_ptr: cute.Pointer,
    bias_ptr: cute.Pointer,
    top_ids_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    expert_counts_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_routes_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    hidden_ptr: cute.Pointer,
    hidden_scale_ptr: cute.Pointer,
    packed_hidden_ptr: cute.Pointer,
    packed_scale_ptr: cute.Pointer,
    packed_native_scale_ptr: cute.Pointer,
    gemm1_weight_ptr: cute.Pointer,
    gemm1_weight_scale_ptr: cute.Pointer,
    gemm1_out_ptr: cute.Pointer,
    quantized_ptr: cute.Pointer,
    quantized_scale_ptr: cute.Pointer,
    quantized_native_scale_ptr: cute.Pointer,
    gemm2_weight_ptr: cute.Pointer,
    gemm2_weight_scale_ptr: cute.Pointer,
    gemm2_out_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    row_ctas: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    requant_ctas: Int32,
    gemm1_max_active_clusters: cutlass.Constexpr[int],
    gemm2_max_active_clusters: cutlass.Constexpr[int],
    route_fast_math: cutlass.Constexpr[bool],
    row_alignment: cutlass.Constexpr[int],
    gather_threads: cutlass.Constexpr[int],
    swiglu_threads: cutlass.Constexpr[int],
    combine_threads: cutlass.Constexpr[int],
    combine_bf16: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Enqueue the large-shape native-MXFP8 MoE pipeline."""
    _launch_route_prefix(
        logits_ptr,
        bias_ptr,
        top_ids_ptr,
        top_weights_ptr,
        expanded_to_row_ptr,
        expert_counts_ptr,
        offsets_ptr,
        expert_routes_ptr,
        group_index_ptr,
        total_m_ptr,
        num_tokens,
        max_m,
        local_offset,
        routed_scale,
        route_fast_math,
        row_alignment,
        stream,
    )
    _launch_gather(
        hidden_ptr,
        hidden_scale_ptr,
        expert_routes_ptr,
        expert_counts_ptr,
        offsets_ptr,
        group_index_ptr,
        packed_hidden_ptr,
        packed_native_scale_ptr,
        total_m_ptr,
        num_tokens,
        max_m,
        row_ctas,
        stream,
    )
    _launch_native_gemm1(
        packed_hidden_ptr,
        gemm1_weight_ptr,
        gemm1_out_ptr,
        packed_native_scale_ptr,
        gemm1_weight_scale_ptr,
        group_index_ptr,
        max_m,
        gemm1_max_active_clusters,
        stream,
    )
    _launch_swiglu_quant_native(
        gemm1_out_ptr,
        quantized_ptr,
        quantized_native_scale_ptr,
        total_m_ptr,
        max_m,
        row_ctas,
        stream,
    )
    _launch_native_gemm2(
        quantized_ptr,
        gemm2_weight_ptr,
        gemm2_out_ptr,
        quantized_native_scale_ptr,
        gemm2_weight_scale_ptr,
        output_ptr,
        packed_scale_ptr,
        top_weights_ptr,
        expert_counts_ptr,
        group_index_ptr,
        num_tokens,
        max_m,
        gemm2_max_active_clusters,
        stream,
    )


@cute.kernel
def _swiglu_quant_kernel(
    gemm1: cute.Tensor,
    quantized: cute.Tensor,
    quantized_scale: cute.Tensor,
    total_m: cute.Tensor,
    thread_count: cutlass.Constexpr[int],
):
    """Fused SwiGLU and 1x128 E4M3 block quantization."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    lane = tidx & 31
    warp = tidx >> 5

    warps_per_cta = thread_count // 32
    valid_m = total_m[0]
    while row < valid_m:
        values = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), Float32
        )
        for wave in cutlass.range_constexpr(I_BLOCKS // warps_per_cta):
            scale_block = warp + wave * warps_per_cta
            local_max = Float32(0.0)
            for item in cutlass.range_constexpr(4):
                col = scale_block * BLOCK + lane + item * 32
                first = gemm1[row, col].to(Float32)
                second = gemm1[row, I + col].to(Float32)
                activated = second * cute.arch.rcp_approx(
                    Float32(1.0) + cute.exp(-second, fastmath=True)
                )
                value = first * activated
                values[item] = value
                local_max = cute.arch.fmax(local_max, cute.math.absf(value))

            block_max = cute.arch.warp_reduction_max(local_max)
            scale = block_max / Float32(FP8_MAX)
            inv_scale = cute.arch.rcp_approx(scale)
            if block_max == 0.0:
                scale = Float32(1.0)
                inv_scale = Float32(1.0)
            if lane == 0:
                quantized_scale[row, scale_block] = scale

            for item in cutlass.range_constexpr(4):
                col = scale_block * BLOCK + lane + item * 32
                quantized[row, col] = (values[item] * inv_scale).to(
                    cutlass.Float8E4M3FN
                )
        row = row + grid_x


@cute.jit
def _launch_swiglu_quant(
    gemm1_ptr: cute.Pointer,
    quantized_ptr: cute.Pointer,
    quantized_scale_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    m: Int32,
    row_ctas: Int32,
    thread_count: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    gemm1 = cute.make_tensor(
        gemm1_ptr, cute.make_layout((m, 2 * I), stride=(2 * I, 1))
    )
    quantized = cute.make_tensor(
        quantized_ptr, cute.make_layout((m, I), stride=(I, 1))
    )
    quantized_scale = cute.make_tensor(
        quantized_scale_ptr,
        cute.make_layout((m, I_BLOCKS), stride=(1, m)),
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    _swiglu_quant_kernel(
        gemm1, quantized, quantized_scale, total_m, thread_count
    ).launch(
        grid=[row_ctas, 1, 1],
        block=[thread_count, 1, 1],
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_exact_kernel(
    gemm1: cute.Tensor,
    quantized: cute.Tensor,
    quantized_scale: cute.Tensor,
    thread_count: cutlass.Constexpr[int],
):
    """Original one-CTA-per-row SwiGLU path for exact host-sized shapes."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    values = cute.make_rmem_tensor(
        cute.make_layout((4,), stride=(1,)), Float32
    )

    warps_per_cta = thread_count // 32
    for wave in cutlass.range_constexpr(I_BLOCKS // warps_per_cta):
        scale_block = warp + wave * warps_per_cta
        local_max = Float32(0.0)
        for item in cutlass.range_constexpr(4):
            col = scale_block * BLOCK + lane + item * 32
            first = gemm1[row, col].to(Float32)
            second = gemm1[row, I + col].to(Float32)
            activated = second * cute.arch.rcp_approx(
                Float32(1.0) + cute.exp(-second, fastmath=True)
            )
            value = first * activated
            values[item] = value
            local_max = cute.arch.fmax(local_max, cute.math.absf(value))

        block_max = cute.arch.warp_reduction_max(local_max)
        scale = block_max / Float32(FP8_MAX)
        inv_scale = cute.arch.rcp_approx(scale)
        if block_max == 0.0:
            scale = Float32(1.0)
            inv_scale = Float32(1.0)
        if lane == 0:
            quantized_scale[row, scale_block] = scale

        for item in cutlass.range_constexpr(4):
            col = scale_block * BLOCK + lane + item * 32
            quantized[row, col] = (values[item] * inv_scale).to(
                cutlass.Float8E4M3FN
            )


@cute.jit
def _launch_swiglu_quant_exact(
    gemm1_ptr: cute.Pointer,
    quantized_ptr: cute.Pointer,
    quantized_scale_ptr: cute.Pointer,
    m: Int32,
    thread_count: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    gemm1 = cute.make_tensor(
        gemm1_ptr, cute.make_layout((m, 2 * I), stride=(2 * I, 1))
    )
    quantized = cute.make_tensor(
        quantized_ptr, cute.make_layout((m, I), stride=(I, 1))
    )
    quantized_scale = cute.make_tensor(
        quantized_scale_ptr,
        cute.make_layout((m, I_BLOCKS), stride=(1, m)),
    )
    _swiglu_quant_exact_kernel(
        gemm1, quantized, quantized_scale, thread_count
    ).launch(
        grid=[m, 1, 1],
        block=[thread_count, 1, 1],
        stream=stream,
    )


@cute.kernel
def _combine_kernel(
    gemm2: cute.Tensor,
    top_weights: cute.Tensor,
    expanded_to_row: cute.Tensor,
    offsets: cute.Tensor,
    total_m: cute.Tensor,
    output: cute.Tensor,
    expert_counts: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    thread_count: cutlass.Constexpr[int],
    vector_values: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    if (token == 0) & (tidx < E_LOCAL):
        expert_counts[tidx] = Int32(0)
    if (token == 0) & (tidx == 0):
        total_m[0] = Int32(-1)

    smem = cutlass.utils.SmemAllocator()
    s_rows = smem.allocate_tensor(
        Int32, cute.make_layout((TOP_K,), stride=(1,)), 16
    )
    s_weights = smem.allocate_tensor(
        Float32, cute.make_layout((TOP_K,), stride=(1,)), 16
    )
    s_count = smem.allocate_tensor(
        Int32, cute.make_layout((1,), stride=(1,)), 16
    )

    # Only local routes have packed GEMM2 rows. Compact those once per token
    # instead of making every output element re-walk eight mostly invalid
    # routes from global memory.
    if tidx == 0:
        local_count = Int32(0)
        for slot in cutlass.range_constexpr(TOP_K):
            expanded = token * TOP_K + slot
            encoded_rank = expanded_to_row[expanded]
            if encoded_rank >= 0:
                local_expert = encoded_rank >> ROUTE_RANK_BITS
                rank = encoded_rank & ROUTE_RANK_MASK
                s_rows[local_count] = offsets[local_expert] + rank
                s_weights[local_count] = top_weights[expanded]
                local_count = local_count + 1
        s_count[0] = local_count
    cute.arch.barrier()

    local_count = s_count[0]
    local_rows = cute.make_rmem_tensor(
        cute.make_layout((TOP_K,), stride=(1,)), Int32
    )
    local_weights = cute.make_rmem_tensor(
        cute.make_layout((TOP_K,), stride=(1,)), Float32
    )
    for slot in cutlass.range_constexpr(TOP_K):
        local_rows[slot] = Int32(0)
        local_weights[slot] = Float32(0.0)
        if slot < local_count:
            local_rows[slot] = s_rows[slot]
            local_weights[slot] = s_weights[slot]

    # The throughput executor uses 128-bit BF16 route loads and output stores.
    # Both vectorized launch regimes accumulate in BF16. Their packed BF16x2
    # FMAs convert less data and need fewer registers than FP32 accumulation.
    if cutlass.const_expr(
        (thread_count == COMBINE_THREADS_LARGE)
        | (thread_count == COMBINE_THREADS_SMALL)
    ):
        g_dst = cute.local_tile(
            output, tiler=(1, H), coord=(token, 0)
        )[0, None]
        thr_copy = tiled_copy.get_slice(tidx)
        t_dst = thr_copy.partition_S(g_dst)
        for wave in range(cute.size(t_dst, mode=[1])):
            col = (
                (wave * thread_count + tidx) * vector_values
            )
            if col < H:
                combine_acc_dtype = Float32
                if cutlass.const_expr(
                    (thread_count == COMBINE_THREADS_LARGE)
                    | (thread_count == COMBINE_THREADS_SMALL)
                ):
                    combine_acc_dtype = cutlass.BFloat16
                r_acc = cute.make_rmem_tensor(
                    t_dst[None, wave].shape, combine_acc_dtype
                )
                r_acc.fill(0)
                r_src = cute.make_rmem_tensor(
                    t_dst[None, wave].shape, cutlass.BFloat16
                )
                r_out = cute.make_rmem_tensor(
                    t_dst[None, wave].shape, cutlass.BFloat16
                )
                for slot in cutlass.range_constexpr(TOP_K):
                    if slot < local_count:
                        g_src = cute.local_tile(
                            gemm2,
                            tiler=(1, H),
                            coord=(local_rows[slot], 0),
                        )[0, None]
                        t_src = thr_copy.partition_S(g_src)
                        cute.autovec_copy(
                            t_src[None, wave], r_src
                        )
                        for item in cutlass.range_constexpr(
                            vector_values
                        ):
                            r_acc[item] = (
                                r_acc[item]
                                + r_src[item].to(combine_acc_dtype)
                                * local_weights[slot].to(combine_acc_dtype)
                            )
                for item in cutlass.range_constexpr(
                    vector_values
                ):
                    r_out[item] = r_acc[item].to(cutlass.BFloat16)
                cute.autovec_copy(r_out, t_dst[None, wave])
    else:
        col = tidx
        acc = Float32(0.0)
        while col < H:
            acc = Float32(0.0)
            for slot in cutlass.range_constexpr(TOP_K):
                if slot < local_count:
                    acc = (
                        acc
                        + gemm2[local_rows[slot], col].to(Float32)
                        * local_weights[slot]
                    )
            output[token, col] = acc.to(cutlass.BFloat16)
            col = col + thread_count


@cute.jit
def _launch_combine(
    gemm2_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    expert_counts_ptr: cute.Pointer,
    num_tokens: Int32,
    m_storage: Int32,
    thread_count: cutlass.Constexpr[int],
    vector_values: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    gemm2 = cute.make_tensor(
        gemm2_ptr, cute.make_layout((m_storage, H), stride=(H, 1))
    )
    top_weights = cute.make_tensor(
        top_weights_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    expanded_to_row = cute.make_tensor(
        expanded_to_row_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    output = cute.make_tensor(
        output_ptr, cute.make_layout((num_tokens, H), stride=(H, 1))
    )
    expert_counts = cute.make_tensor(
        expert_counts_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=COMBINE_VECTOR_BITS,
    )
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout(thread_count),
        cute.make_layout(vector_values),
    )
    _combine_kernel(
        gemm2,
        top_weights,
        expanded_to_row,
        offsets,
        total_m,
        output,
        expert_counts,
        tiled_copy,
        thread_count,
    ).launch(
        grid=[num_tokens, 1, 1],
        block=[thread_count, 1, 1],
        smem=128,
        stream=stream,
    )


# Independent instances are important because the CUTLASS template materializes
# N/K-dependent layouts into instance attributes during JIT compilation.
_GEMM1_TINY = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=GEMM1_TINY_CLUSTER,
)
_GEMM1_MEDIUM = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=GEMM1_MEDIUM_CLUSTER,
    # At 901 tokens, N-major traversal reuses each packed A tile across N.
    scheduler_raster_along_m=False,
)
_GEMM1_LARGE = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=True,
    mma_tiler_mn=(128, 256),
    cluster_shape_mn=GEMM1_LARGE_CLUSTER,
    # Four-cluster N swizzle is the CUPTI-selected large G1 raster.
    scheduler_swizzle_size=2,
    # Four TMEM stages are the measured balance for the 56-block H reduction.
    num_acc_stage_override=2,
)
_GEMM2_TINY = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=(1, 2),
)
_GEMM2_SMALL = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=(1, 1),
    num_acc_stage_override=4,
    use_dynamic_m=True,
    scale_granularity_n=128,
)
_GEMM2_LARGE = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=True,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(2, 1),
    # The shorter 16-block I reduction needs only three buffered partials.
    num_acc_stage_override=3,
    use_dynamic_m=True,
)


@cute.jit
def _launch_gemm1(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    m: Int32,
    max_active_clusters: cutlass.Constexpr[int],
    tile_mode: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, H, 1), stride=(H, 1, m * H)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (2 * I, H, E_LOCAL),
            stride=(H, 1, (2 * I) * H),
        ),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout(
            (m, 2 * I, 1), stride=(2 * I, 1, m * 2 * I)
        ),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, H_BLOCKS, 1), stride=(1, m, m * H_BLOCKS)
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (W13_BLOCKS, H_BLOCKS, E_LOCAL),
            stride=(H_BLOCKS, 1, W13_BLOCKS * H_BLOCKS),
        ),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((m,), stride=(1,))
    )
    # Keep the device extent as a direct kernel operand for throughput shapes;
    # exact host-sized shapes compile the branch away.
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    if cutlass.const_expr(tile_mode == 2):
        _GEMM1_LARGE(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
            False,
            False,
            True,
            False,
            False,
            True,
        )
    else:
        _GEMM2_TINY(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
        )
    elif cutlass.const_expr(tile_mode == 1):
        _GEMM1_MEDIUM(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
        )
    else:
        _GEMM1_TINY(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
        )


@cute.jit
def _launch_gemm1_split(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    finalize_output_ptr: cute.Pointer,
    finalize_routes_ptr: cute.Pointer,
    finalize_weights_ptr: cute.Pointer,
    finalize_counts_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    num_tokens: Int32,
    m: Int32,
    max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Run independent N=2048 up and gate halves with retained N=128 tiles."""
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, H, 1), stride=(H, 1, m * H)),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, H_BLOCKS, 1), stride=(H_BLOCKS, 1, m * H_BLOCKS)
        ),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((m,), stride=(1,))
    )
    half_b_layout = cute.make_layout(
        (I, H, E_LOCAL),
        stride=(H, 1, (2 * I) * H),
    )
    half_c_layout = cute.make_layout(
        (m, I, 1),
        stride=(2 * I, 1, m * 2 * I),
    )
    half_sfb_layout = cute.make_layout(
        (I_BLOCKS, H_BLOCKS, E_LOCAL),
        stride=(H_BLOCKS, 1, W13_BLOCKS * H_BLOCKS),
    )

    b_up = cute.make_tensor(b_ptr, half_b_layout)
    b_gate = cute.make_tensor(b_ptr + I * H, half_b_layout)
    c_up = cute.make_tensor(c_ptr, half_c_layout)
    c_gate = cute.make_tensor(c_ptr + I, half_c_layout)
    sfb_up = cute.make_tensor(sfb_ptr, half_sfb_layout)
    sfb_gate = cute.make_tensor(
        sfb_ptr + I_BLOCKS * H_BLOCKS, half_sfb_layout
    )

    _GEMM1_LARGE_UP(
        a,
        b_up,
        c_up,
        sfa,
        sfb_up,
        group_index,
        max_active_clusters,
        stream,
    )
    _GEMM1_LARGE_GATE(
        a,
        b_gate,
        c_gate,
        sfa,
        sfb_gate,
        group_index,
        max_active_clusters,
        stream,
    )


@cute.jit
def _launch_gemm2(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    m: Int32,
    max_active_clusters: cutlass.Constexpr[int],
    tile_mode: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, I, 1), stride=(I, 1, m * I)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (H, I, E_LOCAL), stride=(I, 1, H * I)
        ),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout((m, H, 1), stride=(H, 1, m * H)),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, I_BLOCKS, 1), stride=(1, m, m * I_BLOCKS)
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (H_BLOCKS, I_BLOCKS, E_LOCAL),
            stride=(I_BLOCKS, 1, H_BLOCKS * I_BLOCKS),
        ),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((m,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    if cutlass.const_expr(tile_mode == 2):
        _GEMM2_LARGE(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
        )
    elif cutlass.const_expr(tile_mode == 1):
        _GEMM2_SMALL(
            a,
            b,
            c,
            sfa,
            sfb,
            group_index,
            total_m,
            max_active_clusters,
            stream,
        )


@cute.jit
def _launch_native_gemm1(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    m: Int32,
    max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, H, 1), stride=(H, 1, m * H)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (2 * I, H, E_LOCAL),
            stride=(H, 1, (2 * I) * H),
        ),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout(
            (m, 2 * I, 1), stride=(2 * I, 1, m * 2 * I)
        ),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, H // 32, 1),
            stride=(H // 32, 1, m * (H // 32)),
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (2 * I, H // 32, E_LOCAL),
            stride=(
                H // 32,
                1,
                (2 * I) * (H // 32),
            ),
        ),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((m,), stride=(1,))
    )
    _GEMM1_NATIVE(
        a,
        b,
        c,
        sfa,
        sfb,
        group_index,
        c,
        sfa,
        sfa,
        group_index,
        max_active_clusters,
        stream,
    )


@cute.jit
def _launch_native_gemm2(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    m: Int32,
    max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, I, 1), stride=(I, 1, m * I)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (H, I, E_LOCAL), stride=(I, 1, H * I)
        ),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout((m, H, 1), stride=(H, 1, m * H)),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, I // 32, 1),
            stride=(I // 32, 1, m * (I // 32)),
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (H, I // 32, E_LOCAL),
            stride=(I // 32, 1, H * (I // 32)),
        ),
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((m,), stride=(1,))
    )
    finalize_output = cute.make_tensor(
        finalize_output_ptr,
        cute.make_layout((num_tokens, H), stride=(H, 1)),
    )
    finalize_routes = cute.make_tensor(
        finalize_routes_ptr,
        cute.make_layout((m,), stride=(1,)),
    )
    finalize_weights = cute.make_tensor(
        finalize_weights_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    finalize_counts = cute.make_tensor(
        finalize_counts_ptr,
        cute.make_layout((E_LOCAL,), stride=(1,)),
    )
    _GEMM2_NATIVE(
        a,
        b,
        c,
        sfa,
        sfb,
        group_index,
        finalize_output,
        finalize_routes,
        finalize_weights,
        finalize_counts,
        max_active_clusters,
        stream,
    )


@cute.jit
def _launch_native_gemm1_hybrid(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    max_m: Int32,
    full_max_active_clusters: cutlass.Constexpr[int],
    tail_max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    full_capacity = max_m - NATIVE_TAIL_CAPACITY
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (2 * I, H, E_LOCAL),
            stride=(H, 1, (2 * I) * H),
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (2 * I, H // 32, E_LOCAL),
            stride=(H // 32, 1, (2 * I) * (H // 32)),
        ),
    )

    a_full = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (full_capacity, H, 1),
            stride=(H, 1, full_capacity * H),
        ),
    )
    c_full = cute.make_tensor(
        c_ptr,
        cute.make_layout(
            (full_capacity, 2 * I, 1),
            stride=(2 * I, 1, full_capacity * 2 * I),
        ),
    )
    sfa_full = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (full_capacity, H // 32, 1),
            stride=(
                H // 32,
                1,
                full_capacity * (H // 32),
            ),
        ),
    )
    group_index_full = cute.make_tensor(
        group_index_ptr,
        cute.make_layout((full_capacity + 1,), stride=(1,)),
    )
    _GEMM1_NATIVE(
        a_full,
        b,
        c_full,
        sfa_full,
        sfb,
        group_index_full,
        full_max_active_clusters,
        stream,
    )

    a_tail = cute.make_tensor(
        a_ptr + full_capacity * H,
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, H, 1),
            stride=(H, 1, NATIVE_TAIL_CAPACITY * H),
        ),
    )
    c_tail = cute.make_tensor(
        c_ptr + full_capacity * (2 * I),
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, 2 * I, 1),
            stride=(
                2 * I,
                1,
                NATIVE_TAIL_CAPACITY * 2 * I,
            ),
        ),
    )
    sfa_tail = cute.make_tensor(
        sfa_ptr + full_capacity * (H // 32),
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, H // 32, 1),
            stride=(
                H // 32,
                1,
                NATIVE_TAIL_CAPACITY * (H // 32),
            ),
        ),
    )
    group_index_tail = cute.make_tensor(
        group_index_ptr + full_capacity + 1,
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY + 1,), stride=(1,)
        ),
    )
    _GEMM1_NATIVE_TAIL(
        a_tail,
        b,
        c_tail,
        sfa_tail,
        sfb,
        group_index_tail,
        tail_max_active_clusters,
        stream,
    )


@cute.jit
def _launch_native_gemm2_hybrid(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    max_m: Int32,
    full_max_active_clusters: cutlass.Constexpr[int],
    tail_max_active_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    full_capacity = max_m - NATIVE_TAIL_CAPACITY
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (H, I, E_LOCAL), stride=(I, 1, H * I)
        ),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (H, I // 32, E_LOCAL),
            stride=(I // 32, 1, H * (I // 32)),
        ),
    )

    a_full = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (full_capacity, I, 1),
            stride=(I, 1, full_capacity * I),
        ),
    )
    c_full = cute.make_tensor(
        c_ptr,
        cute.make_layout(
            (full_capacity, H, 1),
            stride=(H, 1, full_capacity * H),
        ),
    )
    sfa_full = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (full_capacity, I // 32, 1),
            stride=(
                I // 32,
                1,
                full_capacity * (I // 32),
            ),
        ),
    )
    group_index_full = cute.make_tensor(
        group_index_ptr,
        cute.make_layout((full_capacity + 1,), stride=(1,)),
    )
    _GEMM2_NATIVE(
        a_full,
        b,
        c_full,
        sfa_full,
        sfb,
        group_index_full,
        full_max_active_clusters,
        stream,
    )

    a_tail = cute.make_tensor(
        a_ptr + full_capacity * I,
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, I, 1),
            stride=(I, 1, NATIVE_TAIL_CAPACITY * I),
        ),
    )
    c_tail = cute.make_tensor(
        c_ptr + full_capacity * H,
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, H, 1),
            stride=(H, 1, NATIVE_TAIL_CAPACITY * H),
        ),
    )
    sfa_tail = cute.make_tensor(
        sfa_ptr + full_capacity * (I // 32),
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY, I // 32, 1),
            stride=(
                I // 32,
                1,
                NATIVE_TAIL_CAPACITY * (I // 32),
            ),
        ),
    )
    group_index_tail = cute.make_tensor(
        group_index_ptr + full_capacity + 1,
        cute.make_layout(
            (NATIVE_TAIL_CAPACITY + 1,), stride=(1,)
        ),
    )
    _GEMM2_NATIVE_TAIL(
        a_tail,
        b,
        c_tail,
        sfa_tail,
        sfb,
        group_index_tail,
        tail_max_active_clusters,
        stream,
    )


@dataclass
class _MetaWorkspace:
    max_m: int
    top_ids: torch.Tensor
    top_weights: torch.Tensor
    expanded_to_row: torch.Tensor
    offsets: torch.Tensor
    expert_counts: torch.Tensor
    expert_routes: torch.Tensor
    group_index: torch.Tensor
    total_m: torch.Tensor
    total_m_host: torch.Tensor
    output: torch.Tensor


@dataclass
class _ComputeWorkspace:
    packed_hidden: torch.Tensor
    packed_scale: torch.Tensor
    packed_native_scale: torch.Tensor | None
    gemm1: torch.Tensor
    quantized: torch.Tensor
    quantized_scale: torch.Tensor
    quantized_native_scale: torch.Tensor | None
    gemm2: torch.Tensor


@dataclass
class _NativeWeightWorkspace:
    gemm1: torch.Tensor
    gemm1_scale: torch.Tensor
    gemm2: torch.Tensor
    gemm2_scale: torch.Tensor
    sources: Tuple[weakref.ReferenceType, ...] | None = None


_META_CACHE: Dict[Tuple[int, int], _MetaWorkspace] = {}
_COMPUTE_CACHE: Dict[Tuple[int, int, int], _ComputeWorkspace] = {}
_NATIVE_WEIGHT_CACHE: Dict[int, _NativeWeightWorkspace] = {}
_COMPILED_OPS: Dict[str, object] = {}
_MAX_ACTIVE_CLUSTERS: Dict[int, int] = {}


def _device_key(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _get_meta(num_tokens: int, device: torch.device) -> _MetaWorkspace:
    key = (_device_key(device), num_tokens)
    workspace = _META_CACHE.get(key)
    if workspace is not None:
        return workspace

    # At most 8*T real local rows plus up to 127 padding rows per expert.
    max_m = ((TOP_K * num_tokens + E_LOCAL * (BLOCK - 1) + BLOCK - 1) // BLOCK) * BLOCK
    workspace = _MetaWorkspace(
        max_m=max_m,
        top_ids=torch.empty(
            (num_tokens * TOP_K,), dtype=torch.int32, device=device
        ),
        top_weights=torch.empty(
            (num_tokens * TOP_K,), dtype=torch.float32, device=device
        ),
        expanded_to_row=torch.empty(
            (num_tokens * TOP_K,), dtype=torch.int32, device=device
        ),
        # offsets[0] is also the routing completion ticket between calls.
        offsets=torch.zeros(
            (2 * (E_LOCAL + 1),), dtype=torch.int32, device=device
        ),
        expert_counts=torch.zeros((E_LOCAL,), dtype=torch.int32, device=device),
        expert_routes=torch.empty(
            (max(max_m, E_LOCAL * num_tokens),),
            dtype=torch.int32,
            device=device,
        ),
        group_index=torch.empty((max_m,), dtype=torch.int32, device=device),
        total_m=torch.full(
            (1,), -1, dtype=torch.int32, device=device
        ),
        total_m_host=torch.empty((1,), dtype=torch.int32, pin_memory=True),
        output=torch.empty((num_tokens, H), dtype=torch.bfloat16, device=device),
    )
    _META_CACHE[key] = workspace
    return workspace


def _get_compute(
    num_tokens: int, m: int, device: torch.device
) -> _ComputeWorkspace:
    key = (_device_key(device), num_tokens, m)
    workspace = _COMPUTE_CACHE.get(key)
    if workspace is not None:
        return workspace

    storage_m = max(m, 1)
    use_native = num_tokens > NATIVE_MXFP8_THRESHOLD
    workspace = _ComputeWorkspace(
        packed_hidden=torch.empty(
            (storage_m, H), dtype=torch.float8_e4m3fn, device=device
        ),
        packed_scale=torch.empty(
            (storage_m, H_BLOCKS), dtype=torch.float32, device=device
        ),
        packed_native_scale=(
            torch.empty(
                (storage_m * H // 32,),
                dtype=torch.uint8,
                device=device,
            )
            if use_native
            else None
        ),
        gemm1=torch.empty(
            (storage_m, 2 * I), dtype=torch.bfloat16, device=device
        ),
        quantized=torch.empty(
            (storage_m, I), dtype=torch.float8_e4m3fn, device=device
        ),
        quantized_scale=torch.empty(
            (storage_m, I_BLOCKS), dtype=torch.float32, device=device
        ),
        quantized_native_scale=(
            torch.empty(
                (storage_m * I // 32,),
                dtype=torch.uint8,
                device=device,
            )
            if use_native
            else None
        ),
        gemm2=torch.empty(
            (storage_m, H), dtype=torch.bfloat16, device=device
        ),
    )
    _COMPUTE_CACHE[key] = workspace
    return workspace


def _get_native_weights(device: torch.device) -> _NativeWeightWorkspace:
    key = _device_key(device)
    workspace = _NATIVE_WEIGHT_CACHE.get(key)
    if workspace is not None:
        return workspace
    workspace = _NativeWeightWorkspace(
        gemm1=torch.empty(
            (E_LOCAL, 2 * I, H),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        gemm1_scale=torch.empty(
            (E_LOCAL * (2 * I) * H // 32,),
            dtype=torch.uint8,
            device=device,
        ),
        gemm2=torch.empty(
            (E_LOCAL, H, I),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        gemm2_scale=torch.empty(
            (E_LOCAL * H * I // 32,),
            dtype=torch.uint8,
            device=device,
        ),
    )
    _NATIVE_WEIGHT_CACHE[key] = workspace
    return workspace


def _ptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=align,
    )


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _max_active_clusters(cluster_size: int) -> int:
    cached = _MAX_ACTIVE_CLUSTERS.get(cluster_size)
    if cached is None:
        info = cutlass.utils.HardwareInfo()
        cached = info.get_max_active_clusters(cluster_size)
        _MAX_ACTIVE_CLUSTERS[cluster_size] = cached
    return cached


def _invoke_compiled(
    name: str,
    jit_function,
    compile_args: tuple,
    runtime_args: tuple | None = None,
):
    """Compile a CuTe host function once, then use its direct executor.

    Calling a ``@cute.jit`` host function directly is convenient during
    development, but it retains substantial Python/JIT materialization cost.
    The benchmark calls ``run`` many times, so all steady-state launches go
    through an explicitly compiled executor.
    """
    compiled = _COMPILED_OPS.get(name)
    if compiled is None:
        compiled = cute.compile(jit_function, *compile_args)
        _COMPILED_OPS[name] = compiled
    if runtime_args is None:
        runtime_args = compile_args
    return compiled(*runtime_args)


def _invoke_requant(
    name: str,
    source: torch.Tensor,
    source_scale: torch.Tensor,
    destination: torch.Tensor,
    native_scale: torch.Tensor,
    valid_m: torch.Tensor,
    m: int,
    k: int,
    groups: int,
    scale_per_row: bool,
    dynamic_m: bool,
    interleave_gate_up: bool,
    grid_ctas: int,
    enable_pdl: bool,
    stream: cuda.CUstream,
):
    compile_args = (
        _ptr(cutlass.Float8E4M3FN, source),
        _ptr(cutlass.Float32, source_scale),
        _ptr(cutlass.Float8E4M3FN, destination),
        _ptr(cutlass.Float8E8M0FNU, native_scale),
        _ptr(cutlass.Int32, valid_m),
        m,
        k,
        groups,
        scale_per_row,
        dynamic_m,
        interleave_gate_up,
        grid_ctas,
        enable_pdl,
        stream,
    )
    runtime_args = (
        *compile_args[:6],
        grid_ctas,
        stream,
    )
    return _invoke_compiled(
        name,
        launch_requant,
        compile_args,
        runtime_args,
    )


def _prepare_native_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    valid_m: torch.Tensor,
    stream: cuda.CUstream,
) -> _NativeWeightWorkspace:
    workspace = _get_native_weights(gemm1_weights.device)
    sources = (
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    if workspace.sources is not None and all(
        source_ref() is source
        for source_ref, source in zip(
            workspace.sources, sources, strict=True
        )
    ):
        return workspace

    grid_ctas = 4 * _max_active_clusters(1)
    # Pair 32-channel gate/up chunks inside each native N256 accumulator tile.
    _invoke_requant(
        "native_weight_gemm1",
        gemm1_weights,
        gemm1_weights_scale,
        workspace.gemm1,
        workspace.gemm1_scale,
        valid_m,
        2 * I,
        H,
        E_LOCAL,
        False,
        False,
        True,
        grid_ctas,
        False,
        stream,
    )
    _invoke_requant(
        "native_weight_gemm2",
        gemm2_weights,
        gemm2_weights_scale,
        workspace.gemm2,
        workspace.gemm2_scale,
        valid_m,
        H,
        I,
        E_LOCAL,
        False,
        False,
        False,
        grid_ctas,
        False,
        stream,
    )
    workspace.sources = tuple(weakref.ref(source) for source in sources)
    return workspace


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
):
    """Run the fused-routing, FP8 block-scale local MoE."""
    num_tokens = routing_logits.shape[0]
    device = routing_logits.device
    meta = _get_meta(num_tokens, device)
    stream = _stream()

    route_fast_math = num_tokens > ROUTING_FAST_THRESHOLD
    row_alignment = (
        SMALL_ROW_ALIGNMENT
        if num_tokens <= GEMM1_CLUSTER_N_THRESHOLD
        else BLOCK
    )
    gather_threads = (
        GATHER_THREADS_SMALL
        if num_tokens <= GATHER_SMALL_THRESHOLD
        else GATHER_THREADS_LARGE
    )
    device_extent = num_tokens > DEVICE_EXTENT_THRESHOLD
    exact_m = 0

    if device_extent:
        m = meta.max_m
        row_ctas = num_tokens + E_LOCAL
        compute = _get_compute(num_tokens, m, device)
        gemm1_mac = _max_active_clusters(
            GEMM1_LARGE_CLUSTER[0] * GEMM1_LARGE_CLUSTER[1]
        )
        gemm2_mac = _max_active_clusters(2)
        device_pipeline_compile_args = (
            _ptr(cutlass.Float32, routing_logits),
            _ptr(cutlass.BFloat16, routing_bias),
            _ptr(cutlass.Int32, meta.top_ids),
            _ptr(cutlass.Float32, meta.top_weights),
            _ptr(cutlass.Int32, meta.expanded_to_row),
            _ptr(cutlass.Int32, meta.expert_counts),
            _ptr(cutlass.Int32, meta.offsets),
            _ptr(cutlass.Int32, meta.expert_routes),
            _ptr(cutlass.Int32, meta.group_index),
            _ptr(cutlass.Int32, meta.finalize_routes),
            _ptr(cutlass.Int32, meta.total_m),
            _ptr(cutlass.Float8E4M3FN, hidden_states),
            _ptr(cutlass.Float32, hidden_states_scale),
            _ptr(cutlass.Float8E4M3FN, compute.packed_hidden),
            _ptr(cutlass.Float32, compute.packed_scale),
            _ptr(cutlass.Float8E4M3FN, gemm1_weights),
            _ptr(cutlass.Float32, gemm1_weights_scale),
            _ptr(cutlass.BFloat16, compute.gemm1),
            _ptr(cutlass.Float8E4M3FN, compute.quantized),
            _ptr(cutlass.Float32, compute.quantized_scale),
            _ptr(cutlass.Float8E4M3FN, gemm2_weights),
            _ptr(cutlass.Float32, gemm2_weights_scale),
            _ptr(cutlass.BFloat16, compute.gemm2),
            _ptr(cutlass.BFloat16, meta.output),
            num_tokens,
            m,
            row_ctas,
            int(local_expert_offset),
            float(routed_scaling_factor),
            gemm1_mac,
            gemm2_mac,
            stream,
        )
        device_pipeline_runtime_args = (
            *device_pipeline_compile_args[:29],
            stream,
        )
        _invoke_compiled(
            "device_pipeline",
            _launch_device_pipeline,
            device_pipeline_compile_args,
            device_pipeline_runtime_args,
        )
        return meta.output
    else:
        front_compile_args = (
            _ptr(cutlass.Float32, routing_logits),
            _ptr(cutlass.BFloat16, routing_bias),
            _ptr(cutlass.Int32, meta.top_ids),
            _ptr(cutlass.Float32, meta.top_weights),
            _ptr(cutlass.Int32, meta.expanded_to_row),
            _ptr(cutlass.Int32, meta.expert_counts),
            _ptr(cutlass.Int32, meta.offsets),
            _ptr(cutlass.Int32, meta.expert_routes),
            _ptr(cutlass.Int32, meta.group_index),
            _ptr(cutlass.Int32, meta.total_m),
            num_tokens,
            meta.max_m,
            int(local_expert_offset),
            float(routed_scaling_factor),
            route_fast_math,
            row_alignment,
            stream,
        )
        front_runtime_args = (
            *front_compile_args[:14],
            stream,
        )
        _invoke_compiled(
            f"front_{'fast' if route_fast_math else 'exact'}_{row_alignment}",
            _launch_route_prefix,
            front_compile_args,
            front_runtime_args,
        )

        # Launch-bound shapes retain the exact host extent.
        cuda.cuMemcpyDtoHAsync(
            meta.total_m_host.data_ptr(),
            cuda.CUdeviceptr(meta.total_m.data_ptr()),
            4,
            stream,
        )
        cuda.cuStreamSynchronize(stream)
        exact_m = ctypes.c_int32.from_address(meta.total_m_host.data_ptr()).value
        m = max(exact_m, 1)
        row_ctas = m
        compute = _get_compute(num_tokens, m, device)

    if device_extent or exact_m > 0:
        if not device_extent:
            gather_compile_args = (
                _ptr(cutlass.Float8E4M3FN, hidden_states),
                _ptr(cutlass.Float32, hidden_states_scale),
                _ptr(cutlass.Int32, meta.expert_routes),
                _ptr(cutlass.Float8E4M3FN, compute.packed_hidden),
                _ptr(cutlass.Float32, compute.packed_scale),
                num_tokens,
                m,
                gather_threads,
                stream,
            )
            gather_runtime_args = (
                *gather_compile_args[:7],
                stream,
            )
            _invoke_compiled(
                f"gather_exact_{gather_threads}",
                _launch_gather_exact,
                gather_compile_args,
                gather_runtime_args,
            )

        gemm1_tile_mode = (
            2
            if num_tokens > GEMM1_CLUSTER_N_THRESHOLD
            else (0 if num_tokens <= GEMM1_TINY_THRESHOLD else 1)
        )
        gemm1_cluster_size = (
            GEMM1_LARGE_CLUSTER[0] * GEMM1_LARGE_CLUSTER[1]
            if gemm1_tile_mode == 2
            else (
                GEMM1_MEDIUM_CLUSTER[0] * GEMM1_MEDIUM_CLUSTER[1]
                if gemm1_tile_mode == 1
                else GEMM1_TINY_CLUSTER[0] * GEMM1_TINY_CLUSTER[1]
            )
        )
        gemm1_mac = _max_active_clusters(gemm1_cluster_size)
        gemm1_compile_args = (
            _ptr(cutlass.Float8E4M3FN, compute.packed_hidden),
            _ptr(cutlass.Float8E4M3FN, gemm1_weights),
            _ptr(cutlass.BFloat16, compute.gemm1),
            _ptr(cutlass.Float32, compute.packed_scale),
            _ptr(cutlass.Float32, gemm1_weights_scale),
            _ptr(cutlass.Int32, meta.group_index),
            m,
            gemm1_mac,
            gemm1_tile_mode,
            stream,
        )
        gemm1_runtime_args = (
            *gemm1_compile_args[:7],
            stream,
        )
        _invoke_compiled(
            f"gemm1_{('tiny', 'medium', 'large')[gemm1_tile_mode]}",
            _launch_gemm1,
            gemm1_compile_args,
            gemm1_runtime_args,
        )

        swiglu_threads = (
            SWIGLU_THREADS_TINY
            if num_tokens <= SWIGLU_TINY_THRESHOLD
            else (
                SWIGLU_THREADS_SMALL
                if num_tokens <= SWIGLU_LARGE_THRESHOLD
                else SWIGLU_THREADS_LARGE
            )
        )
        swiglu_compile_args = (
            _ptr(cutlass.BFloat16, compute.gemm1),
            _ptr(cutlass.Float8E4M3FN, compute.quantized),
            _ptr(cutlass.Float32, compute.quantized_scale),
            m,
            swiglu_threads,
            stream,
        )
        swiglu_runtime_args = (
            *swiglu_compile_args[:4],
            stream,
        )
        _invoke_compiled(
            f"swiglu_exact_{swiglu_threads}",
            _launch_swiglu_quant_exact,
            swiglu_compile_args,
            swiglu_runtime_args,
        )

        gemm2_tile_mode = (
            2
            if num_tokens > GEMM1_CLUSTER_N_THRESHOLD
            else (0 if num_tokens <= GEMM2_TINY_THRESHOLD else 1)
        )
        gemm2_cluster_size = 1 if gemm2_tile_mode == 1 else 2
        gemm2_mac = _max_active_clusters(gemm2_cluster_size)
        gemm2_compile_args = (
            _ptr(cutlass.Float8E4M3FN, compute.quantized),
            _ptr(cutlass.Float8E4M3FN, gemm2_weights),
            _ptr(cutlass.BFloat16, compute.gemm2),
            _ptr(cutlass.Float32, compute.quantized_scale),
            _ptr(cutlass.Float32, gemm2_weights_scale),
            _ptr(cutlass.Int32, meta.group_index),
            m,
            gemm2_mac,
            gemm2_tile_mode,
            stream,
        )
        gemm2_runtime_args = (
            *gemm2_compile_args[:7],
            stream,
        )
        _invoke_compiled(
            f"gemm2_{('tiny', 'small', 'large')[gemm2_tile_mode]}",
            _launch_gemm2,
            gemm2_compile_args,
            gemm2_runtime_args,
        )

    combine_threads = (
        COMBINE_THREADS_SMALL
        if num_tokens <= COMBINE_SMALL_THRESHOLD
        else COMBINE_THREADS_LARGE
    )
    combine_vector_values = (
        COMBINE_VECTOR_VALUES_SMALL
        if num_tokens <= COMBINE_SMALL_THRESHOLD
        else COMBINE_VECTOR_VALUES_LARGE
    )
    combine_compile_args = (
        _ptr(cutlass.BFloat16, compute.gemm2),
        _ptr(cutlass.Float32, meta.top_weights),
        _ptr(cutlass.Int32, meta.expanded_to_row),
        _ptr(cutlass.Int32, meta.offsets),
        _ptr(cutlass.Int32, meta.total_m),
        _ptr(cutlass.BFloat16, meta.output),
        _ptr(cutlass.Int32, meta.expert_counts),
        num_tokens,
        m,
        combine_threads,
        combine_vector_values,
        False,
        stream,
    )
    combine_runtime_args = (
        *combine_compile_args[:9],
        stream,
    )
    _invoke_compiled(
        f"combine_{combine_threads}x{combine_vector_values}",
        _launch_combine,
        combine_compile_args,
        combine_runtime_args,
    )
    return meta.output
