"""DeepSeek-V3 FP8 block-scale MoE for Blackwell, written in CuTe DSL.

The implementation deliberately keeps PyTorch on the host side only for owning
workspace tensors.  Routing, packing, both GEMMs, SwiGLU/requantization, and the
weighted combine are GPU work implemented in CuTe DSL.

The grouped GEMM primitive is instantiated from the sibling CUTLASS CuTe
template vendored in ``blockwise_gemm.py``.  That primitive uses TMA,
warp-specialized tcgen05 MMA, TMEM accumulation, and a persistent scheduler.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
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


@cute.kernel
def _routing_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    top_ids: cute.Tensor,
    top_weights: cute.Tensor,
    expanded_to_row: cute.Tensor,
    local_offset: Int32,
    routed_scale: Float32,
):
    """One CTA per token: sigmoid, group-top2/top4, global top8, normalize."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    expert = tidx

    smem = cutlass.utils.SmemAllocator()
    s_warp_value = smem.allocate_tensor(
        Float32, cute.make_layout((8,), stride=(1,)), 16
    )
    s_warp_index = smem.allocate_tensor(
        Int32, cute.make_layout((8,), stride=(1,)), 16
    )
    s_group_keep = smem.allocate_tensor(
        Int32, cute.make_layout((8,), stride=(1,)), 16
    )
    s_top_ids = smem.allocate_tensor(
        Int32, cute.make_layout((TOP_K,), stride=(1,)), 16
    )
    s_top_raw = smem.allocate_tensor(
        Float32, cute.make_layout((TOP_K,), stride=(1,)), 16
    )

    if tidx < 8:
        s_group_keep[tidx] = Int32(0)

    logit = logits[token, expert]
    raw_score = Float32(1.0) / (
        Float32(1.0) + cute.exp(-logit, fastmath=False)
    )
    biased_score = raw_score + bias[expert].to(Float32)

    # Each warp is exactly one 32-expert routing group.
    group_first, group_first_idx = _warp_argmax(
        biased_score, expert, cutlass.const_expr(32)
    )
    second_candidate = biased_score
    if expert == group_first_idx:
        second_candidate = Float32(NEG_INF)
    group_second, _ = _warp_argmax(
        second_candidate, expert, cutlass.const_expr(32)
    )
    if lane == 0:
        s_warp_value[warp] = group_first + group_second
    cute.arch.barrier()

    # Warp 0 selects four of the eight group scores.
    if warp == 0:
        group_value = Float32(NEG_INF)
        group_index = Int32(INT_MAX)
        if lane < 8:
            group_value = s_warp_value[lane]
            group_index = lane
        for _ in cutlass.range_constexpr(4):
            _, selected_group = _warp_argmax(
                group_value, group_index, cutlass.const_expr(8)
            )
            if lane == selected_group:
                s_group_keep[lane] = Int32(1)
                group_value = Float32(NEG_INF)
                group_index = Int32(INT_MAX)
    cute.arch.barrier()

    candidate = Float32(NEG_INF)
    if s_group_keep[warp] != 0:
        candidate = biased_score

    # Hierarchical CTA argmax, repeated eight times.  The deterministic expert
    # id tie-break mirrors a stable top-k and avoids duplicate winners.
    for slot in cutlass.range_constexpr(TOP_K):
        warp_value, warp_index = _warp_argmax(
            candidate, expert, cutlass.const_expr(32)
        )
        if lane == 0:
            s_warp_value[warp] = warp_value
            s_warp_index[warp] = warp_index
        cute.arch.barrier()

        if warp == 0:
            cta_value = Float32(NEG_INF)
            cta_index = Int32(INT_MAX)
            if lane < 8:
                cta_value = s_warp_value[lane]
                cta_index = s_warp_index[lane]
            _, winner = _warp_argmax(
                cta_value, cta_index, cutlass.const_expr(8)
            )
            if lane == 0:
                s_top_ids[slot] = winner
        cute.arch.barrier()

        winner = s_top_ids[slot]
        if expert == winner:
            s_top_raw[slot] = raw_score
            candidate = Float32(NEG_INF)
        cute.arch.barrier()

    # The first eight lanes write routing outputs and initialize the inverse
    # permutation map used by the final combine.
    if warp == 0:
        selected_raw = Float32(0.0)
        if lane < TOP_K:
            selected_raw = s_top_raw[lane]
        raw_sum = _warp_sum(selected_raw, cutlass.const_expr(8))
        if lane < TOP_K:
            out_idx = token * TOP_K + lane
            top_ids[out_idx] = s_top_ids[lane]
            top_weights[out_idx] = (
                selected_raw * routed_scale / raw_sum
            )
            expanded_to_row[out_idx] = Int32(-1)


@cute.jit
def _launch_routing(
    logits_ptr: cute.Pointer,
    bias_ptr: cute.Pointer,
    top_ids_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    num_tokens: Int32,
    local_offset: Int32,
    routed_scale: Float32,
    stream: cuda.CUstream,
):
    logits = cute.make_tensor(
        logits_ptr,
        cute.make_layout((num_tokens, E_GLOBAL), stride=(E_GLOBAL, 1)),
    )
    bias = cute.make_tensor(
        bias_ptr, cute.make_layout((E_GLOBAL,), stride=(1,))
    )
    top_ids = cute.make_tensor(
        top_ids_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    top_weights = cute.make_tensor(
        top_weights_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    expanded_to_row = cute.make_tensor(
        expanded_to_row_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    _routing_kernel(
        logits,
        bias,
        top_ids,
        top_weights,
        expanded_to_row,
        local_offset,
        routed_scale,
    ).launch(
        grid=[num_tokens, 1, 1],
        block=[THREADS, 1, 1],
        smem=512,
        stream=stream,
    )


@cute.kernel
def _prefix_kernel(
    top_ids: cute.Tensor,
    offsets: cute.Tensor,
    expert_write: cute.Tensor,
    permuted_to_expanded: cute.Tensor,
    group_index: cute.Tensor,
    total_m: cute.Tensor,
    num_pairs: Int32,
    local_offset: Int32,
):
    """Histogram local routes, build aligned expert offsets, initialize maps."""
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    counts = smem.allocate_tensor(
        Int32, cute.make_layout((E_LOCAL,), stride=(1,)), 16
    )

    if tidx < E_LOCAL:
        counts[tidx] = Int32(0)
    cute.arch.barrier()

    pair_idx = tidx
    while pair_idx < num_pairs:
        expert = top_ids[pair_idx]
        local_expert = expert - local_offset
        if (local_expert >= 0) & (local_expert < E_LOCAL):
            cute.arch.atomic_add(
                counts.iterator + local_expert,
                Int32(1),
                sem="relaxed",
                scope="cta",
            )
        pair_idx = pair_idx + THREADS
    cute.arch.barrier()

    if tidx == 0:
        running = Int32(0)
        offsets[0] = running
        for expert in cutlass.range_constexpr(E_LOCAL):
            count = counts[expert]
            padded = ((count + (BLOCK - 1)) // BLOCK) * BLOCK
            running = running + padded
            offsets[expert + 1] = running
            expert_write[expert] = Int32(0)
        total_m[0] = running
    cute.arch.barrier()

    if tidx < E_LOCAL:
        begin = offsets[tidx]
        end = offsets[tidx + 1]
        row = begin
        while row < end:
            permuted_to_expanded[row] = Int32(-1)
            group_index[row] = tidx
            row = row + 1


@cute.jit
def _launch_prefix(
    top_ids_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    expert_write_ptr: cute.Pointer,
    permuted_to_expanded_ptr: cute.Pointer,
    group_index_ptr: cute.Pointer,
    total_m_ptr: cute.Pointer,
    num_tokens: Int32,
    max_m: Int32,
    local_offset: Int32,
    stream: cuda.CUstream,
):
    top_ids = cute.make_tensor(
        top_ids_ptr,
        cute.make_layout((num_tokens * TOP_K,), stride=(1,)),
    )
    offsets = cute.make_tensor(
        offsets_ptr, cute.make_layout((E_LOCAL + 1,), stride=(1,))
    )
    expert_write = cute.make_tensor(
        expert_write_ptr, cute.make_layout((E_LOCAL,), stride=(1,))
    )
    permuted_to_expanded = cute.make_tensor(
        permuted_to_expanded_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    group_index = cute.make_tensor(
        group_index_ptr, cute.make_layout((max_m,), stride=(1,))
    )
    total_m = cute.make_tensor(
        total_m_ptr, cute.make_layout((1,), stride=(1,))
    )
    _prefix_kernel(
        top_ids,
        offsets,
        expert_write,
        permuted_to_expanded,
        group_index,
        total_m,
        num_tokens * TOP_K,
        local_offset,
    ).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        smem=256,
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
        permuted_to_expanded_ptr, cute.make_layout((max_m,), stride=(1,))
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
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    expanded = permuted_to_expanded[row]
    token = Int32(-1)
    if expanded >= 0:
        token = expanded // TOP_K

    col = tidx
    while col < H:
        value = cutlass.Float8E4M3FN(0.0)
        if token >= 0:
            value = hidden[token, col]
        packed_hidden[row, col] = value
        col = col + THREADS

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
        cute.make_layout((m, H_BLOCKS), stride=(H_BLOCKS, 1)),
    )
    _gather_kernel(
        hidden,
        hidden_scale,
        permuted_to_expanded,
        packed_hidden,
        packed_scale,
    ).launch(
        grid=[m, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _swiglu_quant_kernel(
    gemm1: cute.Tensor,
    quantized: cute.Tensor,
    quantized_scale: cute.Tensor,
):
    """Fused SwiGLU and 1x128 E4M3 block quantization."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5

    values = cute.make_rmem_tensor(
        cute.make_layout((4,), stride=(1,)), Float32
    )

    # Eight warps cover eight scale blocks at a time.
    for wave in cutlass.range_constexpr(2):
        scale_block = warp + wave * 8
        local_max = Float32(0.0)
        for item in cutlass.range_constexpr(4):
            col = scale_block * BLOCK + lane + item * 32
            first = gemm1[row, col].to(Float32)
            second = gemm1[row, I + col].to(Float32)
            activated = second / (
                Float32(1.0) + cute.exp(-second, fastmath=False)
            )
            value = first * activated
            values[item] = value
            local_max = cute.arch.fmax(local_max, cute.math.absf(value))

        block_max = cute.arch.warp_reduction_max(local_max)
        scale = block_max / Float32(FP8_MAX)
        if block_max == 0.0:
            scale = Float32(1.0)
        if lane == 0:
            quantized_scale[row, scale_block] = scale

        for item in cutlass.range_constexpr(4):
            col = scale_block * BLOCK + lane + item * 32
            quantized[row, col] = (values[item] / scale).to(
                cutlass.Float8E4M3FN
            )


@cute.jit
def _launch_swiglu_quant(
    gemm1_ptr: cute.Pointer,
    quantized_ptr: cute.Pointer,
    quantized_scale_ptr: cute.Pointer,
    m: Int32,
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
        cute.make_layout((m, I_BLOCKS), stride=(I_BLOCKS, 1)),
    )
    _swiglu_quant_kernel(gemm1, quantized, quantized_scale).launch(
        grid=[m, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _combine_kernel(
    gemm2: cute.Tensor,
    top_weights: cute.Tensor,
    expanded_to_row: cute.Tensor,
    output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    col = tidx
    while col < H:
        acc = Float32(0.0)
        for slot in cutlass.range_constexpr(TOP_K):
            expanded = token * TOP_K + slot
            row = expanded_to_row[expanded]
            if row >= 0:
                acc = acc + gemm2[row, col].to(Float32) * top_weights[expanded]
        output[token, col] = acc.to(cutlass.BFloat16)
        col = col + THREADS


@cute.jit
def _launch_combine(
    gemm2_ptr: cute.Pointer,
    top_weights_ptr: cute.Pointer,
    expanded_to_row_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    num_tokens: Int32,
    m_storage: Int32,
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
    _combine_kernel(gemm2, top_weights, expanded_to_row, output).launch(
        grid=[num_tokens, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


# Independent instances are important because the CUTLASS template materializes
# N/K-dependent layouts into instance attributes during JIT compilation.
_GEMM1 = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
)
_GEMM2 = BlockwiseContiguousGroupedGemmKernel(
    cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
)


@cute.jit
def _launch_gemm1(
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
            (m, H_BLOCKS, 1), stride=(H_BLOCKS, 1, m * H_BLOCKS)
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
    _GEMM1(
        a,
        b,
        c,
        sfa,
        sfb,
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
            (m, I_BLOCKS, 1), stride=(I_BLOCKS, 1, m * I_BLOCKS)
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
    _GEMM2(
        a,
        b,
        c,
        sfa,
        sfb,
        group_index,
        max_active_clusters,
        stream,
    )


@dataclass
class _MetaWorkspace:
    max_m: int
    top_ids: torch.Tensor
    top_weights: torch.Tensor
    expanded_to_row: torch.Tensor
    offsets: torch.Tensor
    expert_write: torch.Tensor
    permuted_to_expanded: torch.Tensor
    group_index: torch.Tensor
    total_m: torch.Tensor
    output: torch.Tensor


@dataclass
class _ComputeWorkspace:
    packed_hidden: torch.Tensor
    packed_scale: torch.Tensor
    gemm1: torch.Tensor
    quantized: torch.Tensor
    quantized_scale: torch.Tensor
    gemm2: torch.Tensor


_META_CACHE: Dict[Tuple[int, int], _MetaWorkspace] = {}
_COMPUTE_CACHE: Dict[Tuple[int, int, int], _ComputeWorkspace] = {}
_MAX_ACTIVE_CLUSTERS: int | None = None


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
        offsets=torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device),
        expert_write=torch.empty((E_LOCAL,), dtype=torch.int32, device=device),
        permuted_to_expanded=torch.empty(
            (max_m,), dtype=torch.int32, device=device
        ),
        group_index=torch.empty((max_m,), dtype=torch.int32, device=device),
        total_m=torch.empty((1,), dtype=torch.int32, device=device),
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
    workspace = _ComputeWorkspace(
        packed_hidden=torch.empty(
            (storage_m, H), dtype=torch.float8_e4m3fn, device=device
        ),
        packed_scale=torch.empty(
            (storage_m, H_BLOCKS), dtype=torch.float32, device=device
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
        gemm2=torch.empty(
            (storage_m, H), dtype=torch.bfloat16, device=device
        ),
    )
    _COMPUTE_CACHE[key] = workspace
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


def _max_active_clusters() -> int:
    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        info = cutlass.utils.HardwareInfo()
        _MAX_ACTIVE_CLUSTERS = info.get_max_active_clusters(1)
    return _MAX_ACTIVE_CLUSTERS


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

    _launch_routing(
        _ptr(cutlass.Float32, routing_logits),
        _ptr(cutlass.BFloat16, routing_bias),
        _ptr(cutlass.Int32, meta.top_ids),
        _ptr(cutlass.Float32, meta.top_weights),
        _ptr(cutlass.Int32, meta.expanded_to_row),
        num_tokens,
        int(local_expert_offset),
        float(routed_scaling_factor),
        stream,
    )
    _launch_prefix(
        _ptr(cutlass.Int32, meta.top_ids),
        _ptr(cutlass.Int32, meta.offsets),
        _ptr(cutlass.Int32, meta.expert_write),
        _ptr(cutlass.Int32, meta.permuted_to_expanded),
        _ptr(cutlass.Int32, meta.group_index),
        _ptr(cutlass.Int32, meta.total_m),
        num_tokens,
        meta.max_m,
        int(local_expert_offset),
        stream,
    )

    # Only one scalar crosses to the host.  It selects the exact aligned GEMM
    # extent, avoiding the 8x worst-case workspace from entering either GEMM.
    m = int(meta.total_m.item())
    compute = _get_compute(num_tokens, m, device)

    if m > 0:
        _launch_build_mapping(
            _ptr(cutlass.Int32, meta.top_ids),
            _ptr(cutlass.Int32, meta.offsets),
            _ptr(cutlass.Int32, meta.expert_write),
            _ptr(cutlass.Int32, meta.permuted_to_expanded),
            _ptr(cutlass.Int32, meta.expanded_to_row),
            num_tokens,
            meta.max_m,
            int(local_expert_offset),
            stream,
        )
        _launch_gather(
            _ptr(cutlass.Float8E4M3FN, hidden_states),
            _ptr(cutlass.Float32, hidden_states_scale),
            _ptr(cutlass.Int32, meta.permuted_to_expanded),
            _ptr(cutlass.Float8E4M3FN, compute.packed_hidden),
            _ptr(cutlass.Float32, compute.packed_scale),
            num_tokens,
            m,
            stream,
        )

        mac = _max_active_clusters()
        _launch_gemm1(
            _ptr(cutlass.Float8E4M3FN, compute.packed_hidden),
            _ptr(cutlass.Float8E4M3FN, gemm1_weights),
            _ptr(cutlass.BFloat16, compute.gemm1),
            _ptr(cutlass.Float32, compute.packed_scale),
            _ptr(cutlass.Float32, gemm1_weights_scale),
            _ptr(cutlass.Int32, meta.group_index),
            m,
            mac,
            stream,
        )
        _launch_swiglu_quant(
            _ptr(cutlass.BFloat16, compute.gemm1),
            _ptr(cutlass.Float8E4M3FN, compute.quantized),
            _ptr(cutlass.Float32, compute.quantized_scale),
            m,
            stream,
        )
        _launch_gemm2(
            _ptr(cutlass.Float8E4M3FN, compute.quantized),
            _ptr(cutlass.Float8E4M3FN, gemm2_weights),
            _ptr(cutlass.BFloat16, compute.gemm2),
            _ptr(cutlass.Float32, compute.quantized_scale),
            _ptr(cutlass.Float32, gemm2_weights_scale),
            _ptr(cutlass.Int32, meta.group_index),
            m,
            mac,
            stream,
        )

    _launch_combine(
        _ptr(cutlass.BFloat16, compute.gemm2),
        _ptr(cutlass.Float32, meta.top_weights),
        _ptr(cutlass.Int32, meta.expanded_to_row),
        _ptr(cutlass.BFloat16, meta.output),
        num_tokens,
        max(m, 1),
        stream,
    )
    return meta.output
