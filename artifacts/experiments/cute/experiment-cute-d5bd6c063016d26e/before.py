"""CuTe DSL implementation of DeepSeek-V3 FP8 block-scale MoE.

All GPU computation in this module is authored in CuTe DSL.  Python is limited
to JIT compilation, workspace allocation, launch dispatch, and tensor views.
The two grouped GEMMs use the in-repository, BSD-attributed CUTLASS CuTe
post-scale template in :mod:`solution.postscale_masked_grouped_gemm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from solution.postscale_masked_grouped_gemm import (
    BlockwiseMaskedGroupedGemmKernel,
)


HIDDEN = 7168
INTERMEDIATE = 2048
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
SCALE_BLOCK = 128


def _current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@cute.kernel
def _reset_counts_kernel(counts: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < LOCAL_EXPERTS:
        counts[tidx] = cutlass.Int32(0)


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
):
    """Route one token per CTA and physically dispatch its local FP8 rows."""
    tidx, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    # First half is unbiased sigmoid, second half is biased selection score.
    scores = smem.allocate_tensor(cutlass.Float32, GLOBAL_EXPERTS * 2)
    group_scores = smem.allocate_tensor(cutlass.Float32, 8)
    group_keep = smem.allocate_tensor(cutlass.Int32, 8)
    selected = smem.allocate_tensor(cutlass.Int32, TOP_K)
    selected_local = smem.allocate_tensor(cutlass.Int32, TOP_K)
    selected_pos = smem.allocate_tensor(cutlass.Int32, TOP_K)
    selected_weight = smem.allocate_tensor(cutlass.Float32, TOP_K)

    logit = routing_logits[token, tidx].to(cutlass.Float32)
    score = cutlass.Float32(1.0) / (
        cutlass.Float32(1.0) + cute.math.exp(-logit, fastmath=False)
    )
    scores[tidx] = score
    scores[GLOBAL_EXPERTS + tidx] = (
        score + routing_bias[tidx].to(cutlass.Float32)
    )
    cute.arch.sync_threads()

    if tidx == 0:
        neg_inf = cutlass.Float32(-3.402823466e38)

        # Group score is the sum of the top two biased scores in each
        # contiguous 32-expert group.
        for group in range(8):
            top1 = neg_inf
            top2 = neg_inf
            for lane in range(32):
                value = scores[GLOBAL_EXPERTS + group * 32 + lane]
                if value > top1:
                    top2 = top1
                    top1 = value
                elif value > top2:
                    top2 = value
            group_scores[group] = top1 + top2
            group_keep[group] = cutlass.Int32(0)

        # Exact cutoff ties choose the lower group id.  This matches the
        # official B300 torch.topk membership on every provided workload.
        for _ in range(4):
            best_score = neg_inf
            best_group = cutlass.Int32(8)
            for group in range(8):
                if group_keep[group] == 0:
                    value = group_scores[group]
                    if value > best_score:
                        best_score = value
                        best_group = cutlass.Int32(group)
                    elif value == best_score:
                        if cutlass.Int32(group) < best_group:
                            best_group = cutlass.Int32(group)
            group_keep[best_group] = cutlass.Int32(1)

        # Select eight experts from the four retained groups.  Mutating the
        # biased scratch score prevents duplicate selections.
        weight_sum = cutlass.Float32(0.0)
        for k in range(TOP_K):
            best_score = neg_inf
            best_expert = cutlass.Int32(GLOBAL_EXPERTS)
            for expert in range(GLOBAL_EXPERTS):
                if group_keep[expert // 32] != 0:
                    value = scores[GLOBAL_EXPERTS + expert]
                    if value > best_score:
                        best_score = value
                        best_expert = cutlass.Int32(expert)
                    elif value == best_score:
                        if cutlass.Int32(expert) < best_expert:
                            best_expert = cutlass.Int32(expert)
            selected[k] = best_expert
            scores[GLOBAL_EXPERTS + best_expert] = neg_inf
            weight_sum = weight_sum + scores[best_expert]

        for k in range(TOP_K):
            expert = selected[k]
            weight = (
                scores[expert]
                / (weight_sum + cutlass.Float32(1.0e-20))
                * cutlass.Float32(routed_factor)
            )
            local_expert = expert - cutlass.Int32(local_offset)
            selected_weight[k] = weight
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
            for vec in range(HIDDEN // 256):
                hidden_idx = vec * 256 + tidx
                packed_a[local_expert, position, hidden_idx] = hidden[
                    token, hidden_idx
                ]
            if tidx < HIDDEN // SCALE_BLOCK:
                packed_sfa[local_expert, position, tidx] = hidden_scale[
                    tidx, token
                ]


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
    ).launch(
        grid=(routing_logits.shape[0], 1, 1),
        block=(256, 1, 1),
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
        position = route_pos[token, slot]
        lane = tidx & 31
        warp = tidx >> 5

        for scale_block in range(INTERMEDIATE // SCALE_BLOCK):
            col = scale_block * SCALE_BLOCK + tidx
            up = gemm1_out[local_expert, position, col].to(cutlass.Float32)
            gate = gemm1_out[
                local_expert, position, INTERMEDIATE + col
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
                        local_expert, position, scale_block
                    ] = scale
            cute.arch.sync_threads()

            activation[local_expert, position, col] = cutlass.Float8E4M3FN(
                value / block_amax[0]
            )
            cute.arch.sync_threads()


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
    total = output.shape[0] * HIDDEN
    if linear < total:
        token = linear // HIDDEN
        hidden_idx = linear - token * HIDDEN
        value = cutlass.Float32(0.0)
        for k in range(TOP_K):
            local_expert = route_expert[token, k]
            if local_expert >= 0:
                position = route_pos[token, k]
                value = value + (
                    gemm2_out[local_expert, position, hidden_idx].to(
                        cutlass.Float32
                    )
                    * route_weight[token, k]
                )
        output[token, hidden_idx] = cutlass.BFloat16(value)


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
        grid=(cute.ceil_div(output.shape[0] * HIDDEN, 256), 1, 1),
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
    gemm1_out: torch.Tensor
    activation: torch.Tensor
    activation_scale: torch.Tensor
    gemm2_out: torch.Tensor


_workspace: _Workspace | None = None
_route_cache: Dict[Tuple[int, int], object] = {}
_activation_cache: Dict[int, object] = {}
_combine_cache: Dict[int, object] = {}
_gemm_cache: Dict[Tuple[int, int, int], object] = {}


def _capacity(tokens: int) -> int:
    if tokens <= 80:
        return 128
    if tokens <= 901:
        return 1024
    if tokens <= 16384:
        return 16384
    return 32768


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
        gemm1_out=torch.empty(
            (LOCAL_EXPERTS, capacity, 2 * INTERMEDIATE),
            dtype=torch.bfloat16,
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


def _get_route(tokens: int, capacity: int, local_offset: int):
    key = (capacity, local_offset)
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
        options="--opt-level 2 --enable-tvm-ffi --generate-line-info",
    )
    _route_cache[key] = compiled
    return compiled


def _get_activation(capacity: int):
    compiled = _activation_cache.get(capacity)
    if compiled is not None:
        return compiled

    t = cute.sym_int64()
    stream = _current_stream()
    compiled = cute.compile(
        _swiglu_quant_jit,
        _fake(
            cutlass.BFloat16,
            (LOCAL_EXPERTS, capacity, 2 * INTERMEDIATE),
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
    compiled = cute.compile(
        _combine_jit,
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


def _get_gemm(capacity: int, n: int, k: int):
    key = (capacity, n, k)
    compiled = _gemm_cache.get(key)
    if compiled is not None:
        return compiled

    gemm = BlockwiseMaskedGroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)
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
            (n, k, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(
            cutlass.BFloat16,
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
            (n // SCALE_BLOCK, k // SCALE_BLOCK, LOCAL_EXPERTS),
            stride_order=(1, 0, 2),
        ),
        _fake(cutlass.Int32, (LOCAL_EXPERTS,), stride_order=(0,), align=4),
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
    activation = _get_activation(capacity)
    combine = _get_combine(tokens, capacity)
    gemm1 = _get_gemm(capacity, 2 * INTERMEDIATE, HIDDEN)
    gemm2 = _get_gemm(capacity, HIDDEN, INTERMEDIATE)

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

    gemm1(
        ws.packed_a.permute(1, 2, 0),
        gemm1_weights.permute(1, 2, 0),
        ws.gemm1_out.permute(1, 2, 0),
        ws.packed_sfa.permute(1, 2, 0),
        gemm1_weights_scale.permute(1, 2, 0),
        ws.counts,
        stream,
    )

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
