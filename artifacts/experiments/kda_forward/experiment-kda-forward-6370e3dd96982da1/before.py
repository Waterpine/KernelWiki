"""CuTe-DSL Kimi Delta Attention forward kernel.

The first implementation deliberately uses the exact recurrent formulation.  A
CTA owns a 64-row value shard of one (sequence, head) state and retains that
128x64 shard in registers for the complete sequence.  Splitting V in two gives
the fixed-length workloads enough CTAs to cover Blackwell while avoiding any
cross-CTA communication: every value row of the recurrent state is independent.

All device math in this module is authored in CuTe DSL.  The Python wrapper only
allocates the output, specializes/caches the CuTe compilation, and launches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


HEAD_DIM = 128
VALUE_TILE = 64
THREADS = 256
THREADS_PER_VALUE = 4
STATE_VALUES_PER_THREAD = HEAD_DIM // THREADS_PER_VALUE
VALUE_SPLITS = HEAD_DIM // VALUE_TILE


@dataclass(frozen=True)
class _CompileKey:
    total_tokens: int
    heads: int
    sequences: int
    varlen: bool


class _RecurrentKDA:
    def __init__(self, total_tokens: int, heads: int, sequences: int, varlen: bool):
        self.total_tokens = total_tokens
        self.heads = heads
        self.sequences = sequences
        self.varlen = varlen

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        output: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            output,
            scale,
        ).launch(
            grid=(self.sequences, self.heads, VALUE_SPLITS),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        output: cute.Tensor,
        scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        seq_idx, head_idx, value_split = cute.arch.block_idx()
        lane = tidx % 32
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        value_local = tidx // THREADS_PER_VALUE
        key_lane = tidx % THREADS_PER_VALUE
        value_idx = value_split * VALUE_TILE + value_local

        smem = cutlass.utils.SmemAllocator()
        vec_layout = cute.make_layout((HEAD_DIM,), stride=(1,))
        s_q = smem.allocate_tensor(cutlass.Float32, vec_layout, 128)
        s_k = smem.allocate_tensor(cutlass.Float32, vec_layout, 128)
        s_decay = smem.allocate_tensor(cutlass.Float32, vec_layout, 128)
        s_delta = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((VALUE_TILE,), stride=(1,)), 128
        )
        s_scalar = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16
        )

        # One thread owns one value row and every fourth key channel.  Keeping
        # these FP32 elements live across the time loop is the key optimization
        # of this recurrent implementation.
        r_state = cute.make_rmem_tensor((STATE_VALUES_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(STATE_VALUES_PER_THREAD):
            key_idx = key_lane + i * THREADS_PER_VALUE
            r_state[i] = cutlass.Float32(
                initial_state[seq_idx, head_idx, value_idx, key_idx]
            )

        if cutlass.const_expr(self.varlen):
            bos = cutlass.Int32(cu_seqlens[seq_idx])
            eos = cutlass.Int32(cu_seqlens[seq_idx + 1])
        else:
            bos = cutlass.Int32(0)
            eos = cutlass.Int32(self.total_tokens)

        # This parameter is constant for the complete CTA.  Replication avoids
        # a shared-memory dependency in the channel-gate path.
        a_scale = cute.exp(cutlass.Float32(a_log[head_idx]))

        for token in range(bos, eos):
            # Warp 0 loads and normalizes q/k.  Four coalesced passes cover 128
            # channels; the other warps overlap address/arithmetic setup.
            if warp == 0:
                q0 = cutlass.Float32(q[0, token, head_idx, lane])
                q1 = cutlass.Float32(q[0, token, head_idx, lane + 32])
                q2 = cutlass.Float32(q[0, token, head_idx, lane + 64])
                q3 = cutlass.Float32(q[0, token, head_idx, lane + 96])
                k0 = cutlass.Float32(k[0, token, head_idx, lane])
                k1 = cutlass.Float32(k[0, token, head_idx, lane + 32])
                k2 = cutlass.Float32(k[0, token, head_idx, lane + 64])
                k3 = cutlass.Float32(k[0, token, head_idx, lane + 96])

                q_sq = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
                k_sq = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3
                q_sq = cute.arch.warp_reduction_sum(q_sq)
                k_sq = cute.arch.warp_reduction_sum(k_sq)
                q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6)) * scale
                k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6))

                s_q[lane] = q0 * q_inv
                s_q[lane + 32] = q1 * q_inv
                s_q[lane + 64] = q2 * q_inv
                s_q[lane + 96] = q3 * q_inv
                s_k[lane] = k0 * k_inv
                s_k[lane + 32] = k1 * k_inv
                s_k[lane + 64] = k2 * k_inv
                s_k[lane + 96] = k3 * k_inv

            # The decay is channel-wise.  Only half the CTA is needed and each
            # lane writes a unique, contiguous channel.
            if tidx < HEAD_DIM:
                gate_x = a_scale * (
                    cutlass.Float32(g[0, token, head_idx, tidx])
                    + cutlass.Float32(dt_bias[head_idx, tidx])
                )
                gate_sigmoid = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-gate_x)
                )
                s_decay[tidx] = cute.exp(cutlass.Float32(-5.0) * gate_sigmoid)

            if tidx == 0:
                beta_x = cutlass.Float32(beta[0, token, head_idx])
                s_scalar[0] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-beta_x)
                )

            cute.arch.barrier()

            # Apply channel decay and retrieve the value currently associated
            # with this key.  Four adjacent lanes form one value-row reduction.
            retrieved = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(STATE_VALUES_PER_THREAD):
                key_idx = key_lane + i * THREADS_PER_VALUE
                state_i = r_state[i] * s_decay[key_idx]
                r_state[i] = state_i
                retrieved += state_i * s_k[key_idx]

            retrieved += cute.arch.shuffle_sync_bfly(
                retrieved, offset=1, mask=-1, mask_and_clamp=3
            )
            retrieved += cute.arch.shuffle_sync_bfly(
                retrieved, offset=2, mask=-1, mask_and_clamp=3
            )
            if key_lane == 0:
                s_delta[value_local] = (
                    cutlass.Float32(v[0, token, head_idx, value_idx]) - retrieved
                ) * s_scalar[0]

            cute.arch.barrier()

            # Delta update and query projection.  State stays FP32 in registers;
            # only the final output is rounded to BF16.
            delta_v = s_delta[value_local]
            out_v = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(STATE_VALUES_PER_THREAD):
                key_idx = key_lane + i * THREADS_PER_VALUE
                state_i = r_state[i] + delta_v * s_k[key_idx]
                r_state[i] = state_i
                out_v += state_i * s_q[key_idx]

            out_v += cute.arch.shuffle_sync_bfly(
                out_v, offset=1, mask=-1, mask_and_clamp=3
            )
            out_v += cute.arch.shuffle_sync_bfly(
                out_v, offset=2, mask=-1, mask_and_clamp=3
            )
            if key_lane == 0:
                output[0, token, head_idx, value_idx] = cutlass.BFloat16(out_v)

            # Synchronizes the register-state readers before warp 0 overwrites
            # q/k and the first 128 threads overwrite decay for the next token.
            cute.arch.barrier()


_compiled: dict[_CompileKey, object] = {}


def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _compiled_kernel(
    key: _CompileKey,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    stream: cuda.CUstream,
):
    compiled = _compiled.get(key)
    if compiled is None:
        op = _RecurrentKDA(
            key.total_tokens, key.heads, key.sequences, key.varlen
        )
        compiled = cute.compile(
            op,
            _as_cute(q),
            _as_cute(k),
            _as_cute(v),
            _as_cute(g),
            _as_cute(beta),
            _as_cute(a_log),
            _as_cute(dt_bias),
            _as_cute(initial_state),
            _as_cute(cu_seqlens),
            _as_cute(output),
            cutlass.Float32(HEAD_DIM**-0.5),
            stream,
        )
        _compiled[key] = compiled
    return compiled


@torch.no_grad()
def run(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    cu_seqlens=None,
):
    """Return packed-varlen KDA output with shape ``[1, T, H, 128]``."""
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM:
        raise ValueError("This kernel requires packed B=1 inputs with D=128")

    total_tokens = q.shape[1]
    heads = q.shape[2]
    varlen = cu_seqlens is not None
    sequences = int(cu_seqlens.numel() - 1) if varlen else 1
    output = torch.empty_like(v)

    # A non-varlen compile still receives a well-typed placeholder tensor; the
    # corresponding constexpr path never reads it.
    cu = (
        cu_seqlens
        if varlen
        else torch.empty((1,), dtype=torch.int64, device=q.device)
    )
    dt = dt_bias.reshape(heads, HEAD_DIM)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = _CompileKey(total_tokens, heads, sequences, varlen)
    compiled = _compiled_kernel(
        key,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt,
        initial_state,
        cu,
        output,
        stream,
    )
    compiled(
        _as_cute(q),
        _as_cute(k),
        _as_cute(v),
        _as_cute(g),
        _as_cute(beta),
        _as_cute(A_log),
        _as_cute(dt),
        _as_cute(initial_state),
        _as_cute(cu),
        _as_cute(output),
        cutlass.Float32(float(scale)),
        stream,
    )
    return output
