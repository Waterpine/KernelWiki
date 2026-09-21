from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


NUM_Q_HEADS = 4
NUM_V_HEADS = 8
HEAD_DIM = 128
WARPS_PER_BLOCK = 8
THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32
ROWS_PER_BLOCK = WARPS_PER_BLOCK
ROW_TILES = HEAD_DIM // ROWS_PER_BLOCK
PARAMS_PER_TOKEN = 2 * NUM_V_HEADS


@cute.kernel
def _gate_kernel(
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    b: cute.Tensor,
    params: cute.Tensor,
    total_tokens: cutlass.Constexpr[int],
):
    """Generate decay and update gates entirely in CuTe DSL."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear_idx = bidx * 128 + tidx
    num_values = total_tokens * NUM_V_HEADS

    if linear_idx < num_values:
        token_idx = linear_idx // NUM_V_HEADS
        value_head = linear_idx % NUM_V_HEADS

        x = cutlass.Float32(a[token_idx, value_head]) + cutlass.Float32(
            dt_bias[value_head]
        )
        if x <= 20.0:
            softplus_x = cute.log(
                cutlass.Float32(1.0) + cute.exp(x, fastmath=False),
                fastmath=False,
            )
        else:
            softplus_x = x

        log_decay = (
            -cute.exp(cutlass.Float32(A_log[value_head]), fastmath=False)
            * softplus_x
        )
        decay = cute.exp(log_decay, fastmath=False)
        beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp(-cutlass.Float32(b[token_idx, value_head]), fastmath=False)
        )

        params[token_idx, value_head] = decay
        params[token_idx, NUM_V_HEADS + value_head] = beta


@cute.kernel
def _recurrent_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state: cute.Tensor,
    params: cute.Tensor,
    cu_seqlens: cute.Tensor,
    output: cute.Tensor,
    new_state: cute.Tensor,
    scale: cutlass.Constexpr[float],
):
    """Warp-per-state-row recurrent GDN prefill.

    Each warp owns one complete k-last state row. Four FP32 elements per lane
    remain in registers for the whole sequence, so rows and sequences are
    independent and require no inter-CTA synchronization.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    row_tile = bidx % ROW_TILES
    head_seq_idx = bidx // ROW_TILES
    value_head = head_seq_idx % NUM_V_HEADS
    seq_idx = head_seq_idx // NUM_V_HEADS
    qk_head = value_head // (NUM_V_HEADS // NUM_Q_HEADS)
    value_row = row_tile * ROWS_PER_BLOCK + warp_idx

    r_state = cute.make_rmem_tensor(
        cute.make_layout((4,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((4,), stride=(1,)), cutlass.Float32
    )

    for i in cutlass.range_constexpr(4):
        k_col = lane_id + i * 32
        r_state[i] = cutlass.Float32(
            state[seq_idx, value_head, value_row, k_col]
        )

    seq_start = cutlass.Int32(0)
    seq_end = cutlass.Int32(0)
    if lane_id == 0:
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
    seq_start = cute.arch.shuffle_sync(seq_start, 0)
    seq_end = cute.arch.shuffle_sync(seq_end, 0)

    token_idx = seq_start
    while token_idx < seq_end:
        decay = cutlass.Float32(0.0)
        beta = cutlass.Float32(0.0)
        value = cutlass.Float32(0.0)
        if lane_id == 0:
            decay = cutlass.Float32(params[token_idx, value_head])
            beta = cutlass.Float32(
                params[token_idx, NUM_V_HEADS + value_head]
            )
            value = cutlass.Float32(v[token_idx, value_head, value_row])
        decay = cute.arch.shuffle_sync(decay, 0)
        beta = cute.arch.shuffle_sync(beta, 0)
        value = cute.arch.shuffle_sync(value, 0)

        retrieved = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(4):
            k_col = lane_id + i * 32
            k_value = cutlass.Float32(k[token_idx, qk_head, k_col])
            r_k[i] = k_value
            r_state[i] = r_state[i] * decay
            retrieved += r_state[i] * k_value

        for offset in [16, 8, 4, 2, 1]:
            retrieved += cute.arch.shuffle_sync_bfly(
                retrieved, offset=offset, mask=-1, mask_and_clamp=31
            )

        delta = beta * (value - retrieved)
        out_value = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(4):
            k_col = lane_id + i * 32
            r_state[i] = r_state[i] + r_k[i] * delta
            out_value += r_state[i] * cutlass.Float32(
                q[token_idx, qk_head, k_col]
            )

        for offset in [16, 8, 4, 2, 1]:
            out_value += cute.arch.shuffle_sync_bfly(
                out_value, offset=offset, mask=-1, mask_and_clamp=31
            )

        if lane_id == 0:
            output[token_idx, value_head, value_row] = cutlass.BFloat16(
                out_value * scale
            )
        token_idx += 1

    for i in cutlass.range_constexpr(4):
        k_col = lane_id + i * 32
        new_state[seq_idx, value_head, value_row, k_col] = r_state[i]


_gate_kernel.set_name_prefix("gdn_prefill_gate")
_recurrent_kernel.set_name_prefix("gdn_prefill_recurrent")


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    b: cute.Tensor,
    cu_seqlens: cute.Tensor,
    output: cute.Tensor,
    new_state: cute.Tensor,
    params: cute.Tensor,
    scale: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    total_tokens = cute.size(q, mode=[0])
    num_seqs = cute.size(state, mode=[0])

    _gate_kernel(A_log, a, dt_bias, b, params, total_tokens).launch(
        grid=(cute.ceil_div(total_tokens * NUM_V_HEADS, 128), 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )
    _recurrent_kernel(
        q,
        k,
        v,
        state,
        params,
        cu_seqlens,
        output,
        new_state,
        scale,
    ).launch(
        grid=(num_seqs * NUM_V_HEADS * ROW_TILES, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )


@functools.cache
def _compile_cache(
    total_tokens: int,
    num_seqs: int,
    input_dtype: torch.dtype,
    scale: float,
):
    return {}


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Compile, launch, and return the CuTe-DSL GDN prefill outputs."""
    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(HEAD_DIM)
    scale = float(scale)

    total_tokens = int(q.shape[0])
    num_seqs = int(state.shape[0])
    output = torch.empty_like(v)
    new_state = torch.empty_like(state)
    params = torch.empty(
        (total_tokens, PARAMS_PER_TOKEN),
        dtype=torch.float32,
        device=q.device,
    )

    cache = _compile_cache(total_tokens, num_seqs, q.dtype, scale)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if "compiled" not in cache:
        compiled = cute.compile(
            _launch,
            from_dlpack(q, assumed_align=16),
            from_dlpack(k, assumed_align=16),
            from_dlpack(v, assumed_align=16),
            from_dlpack(state, assumed_align=16),
            from_dlpack(A_log, assumed_align=16),
            from_dlpack(a, assumed_align=16),
            from_dlpack(dt_bias, assumed_align=16),
            from_dlpack(b, assumed_align=16),
            from_dlpack(cu_seqlens, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            from_dlpack(new_state, assumed_align=16),
            from_dlpack(params, assumed_align=16),
            scale=scale,
            stream=stream,
            options="--enable-tvm-ffi --generate-line-info --opt-level 3",
        )
        cache["compiled"] = compiled

    cache["compiled"](
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        output,
        new_state,
        params,
        stream,
    )
    return output, new_state
