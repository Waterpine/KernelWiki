"""Correctness-first CuTe DSL implementation of the DeepSeek-V3 FP8 MoE.

The projection kernels in this first implementation intentionally use CUDA-core
dot products.  Keeping the block-scale accumulation explicit gives us a small,
auditable seed that can be compared with the PyTorch definition before the dot
product body is replaced by a tcgen05/TMEM pipeline.
"""

from __future__ import annotations

import functools
import operator

import torch

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32


HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
BLOCK_SIZE = 128


class _MoeSeedKernel:
    """Three-launch CuTe implementation: route, W13/SwiGLU, then W2/combine."""

    threads = 256
    output_tile = 256

    @cute.jit
    def _sigmoid(self, x: Float32):
        return Float32(1.0) / (
            Float32(1.0) + cute.math.exp(-x, fastmath=False)
        )

    @cute.jit
    def _group_score(
        self,
        routing_logits: cute.Tensor,
        routing_bias: cute.Tensor,
        token: Int32,
        group: Int32,
    ):
        """Sum of the largest two biased sigmoid scores in one 32-wide group."""
        best = Float32(-3.402823466e38)
        second = Float32(-3.402823466e38)
        expert = group * Int32(32)
        group_end = expert + Int32(32)
        row = token * Int32(NUM_GLOBAL_EXPERTS)
        while expert < group_end:
            raw = self._sigmoid(routing_logits[row + expert].to(Float32))
            score = raw + routing_bias[expert].to(Float32)
            if score > best:
                second = best
                best = score
            elif score > second:
                second = score
            expert += Int32(1)
        return best + second

    @cute.jit
    def __call__(
        self,
        routing_logits: cute.Tensor,
        routing_bias: cute.Tensor,
        hidden_states: cute.Tensor,
        hidden_states_scale: cute.Tensor,
        gemm1_weights: cute.Tensor,
        gemm1_weights_scale: cute.Tensor,
        gemm2_weights: cute.Tensor,
        gemm2_weights_scale: cute.Tensor,
        route_ids: cute.Tensor,
        route_weights: cute.Tensor,
        intermediate: cute.Tensor,
        output: cute.Tensor,
        seq_len: Int32,
        local_expert_offset: Int32,
        routed_scaling_factor: Float32,
        stream: cuda.CUstream,
    ):
        self._routing_kernel(
            routing_logits,
            routing_bias,
            route_ids,
            route_weights,
            routed_scaling_factor,
        ).launch(
            grid=[seq_len, 1, 1],
            block=[32, 1, 1],
            stream=stream,
        )

        self._gemm1_kernel(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            route_ids,
            intermediate,
            seq_len,
            local_expert_offset,
        ).launch(
            grid=[seq_len * Int32(TOP_K), INTERMEDIATE_SIZE // self.output_tile, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

        self._gemm2_kernel(
            intermediate,
            gemm2_weights,
            gemm2_weights_scale,
            route_ids,
            route_weights,
            output,
            local_expert_offset,
        ).launch(
            grid=[seq_len, HIDDEN_SIZE // self.output_tile, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _routing_kernel(
        self,
        routing_logits: cute.Tensor,
        routing_bias: cute.Tensor,
        route_ids: cute.Tensor,
        route_weights: cute.Tensor,
        routed_scaling_factor: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(bidx)

        # One lane performs the small 256-way selection.  Strict comparisons
        # while scanning ascending IDs implement CUDA topk's lowest-ID tie rule.
        if tid == Int32(0):
            route_base = token * Int32(TOP_K)

            # Temporarily store the four selected group IDs in the weight
            # scratch.  route_ids is overwritten one slot at a time below, so
            # it cannot also hold the group mask during expert selection.
            for rank in cutlass.range_constexpr(4):
                best_group = Int32(-1)
                best_group_score = Float32(-3.402823466e38)
                for group_py in cutlass.range_constexpr(8):
                    group = Int32(group_py)
                    already_selected = False
                    for previous in cutlass.range_constexpr(rank):
                        if (
                            route_weights[
                                route_base + Int32(previous)
                            ].to(Int32)
                            == group
                        ):
                            already_selected = True
                    if not already_selected:
                        group_score = self._group_score(
                            routing_logits, routing_bias, token, group
                        )
                        if group_score > best_group_score:
                            best_group_score = group_score
                            best_group = group
                route_weights[route_base + Int32(rank)] = Float32(best_group)

            # Select the eight largest biased expert scores in those groups.
            # Previously selected IDs are read back from global scratch; this
            # avoids a large per-thread local array and preserves exact ties.
            for rank in cutlass.range_constexpr(TOP_K):
                best_expert = Int32(-1)
                best_score = Float32(-3.402823466e38)
                expert = Int32(0)
                while expert < Int32(NUM_GLOBAL_EXPERTS):
                    group = expert // Int32(32)
                    kept_group = False
                    for group_slot in cutlass.range_constexpr(4):
                        if (
                            route_weights[
                                route_base + Int32(group_slot)
                            ].to(Int32)
                            == group
                        ):
                            kept_group = True
                    already_selected = False
                    for previous in cutlass.range_constexpr(rank):
                        if route_ids[route_base + Int32(previous)] == expert:
                            already_selected = True
                    if kept_group:
                        if not already_selected:
                            raw = self._sigmoid(
                                routing_logits[
                                    token * Int32(NUM_GLOBAL_EXPERTS) + expert
                                ].to(Float32)
                            )
                            score = raw + routing_bias[expert].to(Float32)
                            if score > best_score:
                                best_score = score
                                best_expert = expert
                    expert += Int32(1)
                route_ids[route_base + Int32(rank)] = best_expert

            weight_sum = Float32(0.0)
            for rank in cutlass.range_constexpr(TOP_K):
                expert = route_ids[route_base + Int32(rank)]
                raw = self._sigmoid(
                    routing_logits[
                        token * Int32(NUM_GLOBAL_EXPERTS) + expert
                    ].to(Float32)
                )
                route_weights[route_base + Int32(rank)] = raw
                weight_sum += raw

            normalization = routed_scaling_factor / (
                weight_sum + Float32(1.0e-20)
            )
            for rank in cutlass.range_constexpr(TOP_K):
                idx = route_base + Int32(rank)
                route_weights[idx] = (
                    route_weights[idx].to(Float32) * normalization
                )

    @cute.kernel
    def _gemm1_kernel(
        self,
        hidden_states: cute.Tensor,
        hidden_states_scale: cute.Tensor,
        gemm1_weights: cute.Tensor,
        gemm1_weights_scale: cute.Tensor,
        route_ids: cute.Tensor,
        intermediate: cute.Tensor,
        seq_len: Int32,
        local_expert_offset: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        pair_idx_raw, tile_idx_raw, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid % Int32(32)
        warp = tid // Int32(32)
        pair_idx = Int32(pair_idx_raw)
        tile_idx = Int32(tile_idx_raw)
        token = pair_idx // Int32(TOP_K)
        slot = pair_idx - token * Int32(TOP_K)
        global_expert = route_ids[pair_idx]
        local_expert = global_expert - local_expert_offset

        if local_expert >= Int32(0):
            if local_expert < Int32(NUM_LOCAL_EXPERTS):
                n = tile_idx * Int32(self.output_tile) + warp
                tile_end = (tile_idx + Int32(1)) * Int32(self.output_tile)
                while n < tile_end:
                    first_acc = Float32(0.0)
                    second_acc = Float32(0.0)
                    k_block = Int32(0)
                    while k_block < Int32(HIDDEN_SIZE // BLOCK_SIZE):
                        first_partial = Float32(0.0)
                        second_partial = Float32(0.0)
                        for lane_part in cutlass.range_constexpr(4):
                            k = (
                                k_block * Int32(BLOCK_SIZE)
                                + lane
                                + Int32(lane_part * 32)
                            )
                            a = hidden_states[
                                token * Int32(HIDDEN_SIZE) + k
                            ].to(Float32)
                            first_w_idx = (
                                (local_expert * Int32(2 * INTERMEDIATE_SIZE) + n)
                                * Int32(HIDDEN_SIZE)
                                + k
                            )
                            second_w_idx = (
                                (
                                    local_expert * Int32(2 * INTERMEDIATE_SIZE)
                                    + n
                                    + Int32(INTERMEDIATE_SIZE)
                                )
                                * Int32(HIDDEN_SIZE)
                                + k
                            )
                            first_partial += (
                                a * gemm1_weights[first_w_idx].to(Float32)
                            )
                            second_partial += (
                                a * gemm1_weights[second_w_idx].to(Float32)
                            )

                        first_partial = cute.arch.warp_reduction(
                            first_partial, operator.add
                        )
                        second_partial = cute.arch.warp_reduction(
                            second_partial, operator.add
                        )
                        a_scale = hidden_states_scale[
                            k_block * seq_len + token
                        ].to(Float32)
                        first_scale_idx = (
                            (
                                local_expert * Int32(2 * INTERMEDIATE_SIZE // BLOCK_SIZE)
                                + n // Int32(BLOCK_SIZE)
                            )
                            * Int32(HIDDEN_SIZE // BLOCK_SIZE)
                            + k_block
                        )
                        second_scale_idx = (
                            (
                                local_expert * Int32(2 * INTERMEDIATE_SIZE // BLOCK_SIZE)
                                + (n + Int32(INTERMEDIATE_SIZE))
                                // Int32(BLOCK_SIZE)
                            )
                            * Int32(HIDDEN_SIZE // BLOCK_SIZE)
                            + k_block
                        )
                        first_acc += (
                            first_partial
                            * a_scale
                            * gemm1_weights_scale[first_scale_idx].to(Float32)
                        )
                        second_acc += (
                            second_partial
                            * a_scale
                            * gemm1_weights_scale[second_scale_idx].to(Float32)
                        )
                        k_block += Int32(1)

                    if lane == Int32(0):
                        silu_second = second_acc / (
                            Float32(1.0)
                            + cute.math.exp(-second_acc, fastmath=False)
                        )
                        intermediate[
                            pair_idx * Int32(INTERMEDIATE_SIZE) + n
                        ] = BFloat16(silu_second * first_acc)
                    n += Int32(8)

    @cute.kernel
    def _gemm2_kernel(
        self,
        intermediate: cute.Tensor,
        gemm2_weights: cute.Tensor,
        gemm2_weights_scale: cute.Tensor,
        route_ids: cute.Tensor,
        route_weights: cute.Tensor,
        output: cute.Tensor,
        local_expert_offset: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_raw, tile_idx_raw, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid % Int32(32)
        warp = tid // Int32(32)
        token = Int32(token_raw)
        tile_idx = Int32(tile_idx_raw)
        h = tile_idx * Int32(self.output_tile) + warp
        tile_end = (tile_idx + Int32(1)) * Int32(self.output_tile)

        while h < tile_end:
            output_acc = Float32(0.0)
            route_base = token * Int32(TOP_K)
            for slot in cutlass.range_constexpr(TOP_K):
                pair_idx = route_base + Int32(slot)
                global_expert = route_ids[pair_idx]
                local_expert = global_expert - local_expert_offset
                if local_expert >= Int32(0):
                    if local_expert < Int32(NUM_LOCAL_EXPERTS):
                        route_acc = Float32(0.0)
                        k_block = Int32(0)
                        while k_block < Int32(INTERMEDIATE_SIZE // BLOCK_SIZE):
                            partial = Float32(0.0)
                            for lane_part in cutlass.range_constexpr(4):
                                k = (
                                    k_block * Int32(BLOCK_SIZE)
                                    + lane
                                    + Int32(lane_part * 32)
                                )
                                a = intermediate[
                                    pair_idx * Int32(INTERMEDIATE_SIZE) + k
                                ].to(Float32)
                                w_idx = (
                                    (local_expert * Int32(HIDDEN_SIZE) + h)
                                    * Int32(INTERMEDIATE_SIZE)
                                    + k
                                )
                                partial += (
                                    a * gemm2_weights[w_idx].to(Float32)
                                )
                            partial = cute.arch.warp_reduction(partial, operator.add)
                            scale_idx = (
                                (
                                    local_expert * Int32(HIDDEN_SIZE // BLOCK_SIZE)
                                    + h // Int32(BLOCK_SIZE)
                                )
                                * Int32(INTERMEDIATE_SIZE // BLOCK_SIZE)
                                + k_block
                            )
                            route_acc += (
                                partial
                                * gemm2_weights_scale[scale_idx].to(Float32)
                            )
                            k_block += Int32(1)
                        output_acc += (
                            route_acc
                            * route_weights[pair_idx].to(Float32)
                        )

            if lane == Int32(0):
                output[token * Int32(HIDDEN_SIZE) + h] = BFloat16(output_acc)
            h += Int32(8)


@functools.cache
def _compiled_kernel():
    """Compile one dynamic-sequence TVM-FFI specialization."""
    seq_len = cute.sym_int32(divisibility=1)

    def fake(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=align,
            use_32bit_stride=True,
        )

    routing_logits = fake(cutlass.Float32, (seq_len * NUM_GLOBAL_EXPERTS,))
    routing_bias = fake(cutlass.BFloat16, (NUM_GLOBAL_EXPERTS,))
    hidden_states = fake(cutlass.Float8E4M3FN, (seq_len * HIDDEN_SIZE,))
    hidden_states_scale = fake(
        cutlass.Float32, (seq_len * (HIDDEN_SIZE // BLOCK_SIZE),)
    )
    gemm1_weights = fake(
        cutlass.Float8E4M3FN,
        (NUM_LOCAL_EXPERTS * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE,),
    )
    gemm1_weights_scale = fake(
        cutlass.Float32,
        (
            NUM_LOCAL_EXPERTS
            * (2 * INTERMEDIATE_SIZE // BLOCK_SIZE)
            * (HIDDEN_SIZE // BLOCK_SIZE),
        ),
    )
    gemm2_weights = fake(
        cutlass.Float8E4M3FN,
        (NUM_LOCAL_EXPERTS * HIDDEN_SIZE * INTERMEDIATE_SIZE,),
    )
    gemm2_weights_scale = fake(
        cutlass.Float32,
        (
            NUM_LOCAL_EXPERTS
            * (HIDDEN_SIZE // BLOCK_SIZE)
            * (INTERMEDIATE_SIZE // BLOCK_SIZE),
        ),
    )
    route_ids = fake(cutlass.Int32, (seq_len * TOP_K,))
    route_weights = fake(cutlass.Float32, (seq_len * TOP_K,))
    intermediate = fake(
        cutlass.BFloat16, (seq_len * TOP_K * INTERMEDIATE_SIZE,)
    )
    output = fake(cutlass.BFloat16, (seq_len * HIDDEN_SIZE,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        _MoeSeedKernel(),
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        route_ids,
        route_weights,
        intermediate,
        output,
        Int32(1),
        Int32(0),
        Float32(2.5),
        stream,
        options="--enable-tvm-ffi",
    )


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
    """Compute this rank's 32-expert contribution and return ``[T, 7168]`` BF16."""
    seq_len = int(hidden_states.shape[0])
    device = hidden_states.device
    route_ids = torch.empty((seq_len * TOP_K,), dtype=torch.int32, device=device)
    route_weights = torch.empty(
        (seq_len * TOP_K,), dtype=torch.float32, device=device
    )
    intermediate = torch.empty(
        (seq_len * TOP_K * INTERMEDIATE_SIZE,),
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty(
        (seq_len * HIDDEN_SIZE,), dtype=torch.bfloat16, device=device
    )

    kernel = _compiled_kernel()
    kernel(
        routing_logits.reshape(-1),
        routing_bias.reshape(-1),
        hidden_states.reshape(-1),
        hidden_states_scale.reshape(-1),
        gemm1_weights.reshape(-1),
        gemm1_weights_scale.reshape(-1),
        gemm2_weights.reshape(-1),
        gemm2_weights_scale.reshape(-1),
        route_ids,
        route_weights,
        intermediate,
        output,
        seq_len,
        int(local_expert_offset),
        float(routed_scaling_factor),
    )
    return output.view(seq_len, HIDDEN_SIZE)
