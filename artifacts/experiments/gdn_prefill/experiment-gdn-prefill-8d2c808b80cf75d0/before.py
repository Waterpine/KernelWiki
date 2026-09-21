"""CuTe-DSL Gated Delta Net prefill kernel for Blackwell.

The recurrent state has independent value columns.  A CTA owns 32 columns of
one (sequence, value-head) state, with four columns per warp.  Each thread keeps
16 K-dimension values in registers for the entire sequence.  This makes the
work proportional to each sequence's true length instead of the batch maximum.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute import EnableTVMFFI
from cutlass.cute.runtime import from_dlpack


_COMPILED: Dict[Tuple[int, int, torch.dtype, float], object] = {}

_HEAD_DIM = 128
_Q_HEADS = 4
_V_HEADS = 8
_V_PER_WARP = 4
_WARPS = 8
_V_PER_CTA = _V_PER_WARP * _WARPS
_K_PER_LANE = _HEAD_DIM // (32 // _V_PER_WARP)
_THREADS = 32 * _WARPS


class _GdnRecurrent:
    def __init__(self, total_tokens: int, num_sequences: int, scale: float):
        self.total_tokens = total_tokens
        self.num_sequences = num_sequences
        self.scale = scale

    @cute.kernel
    def _prepare_gates(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        decay: cute.Tensor,
        beta: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * _THREADS + tidx
        if linear < self.total_tokens * _V_HEADS:
            head = linear % _V_HEADS
            x = cutlass.Float32(a[linear]) + cutlass.Float32(dt_bias[head])
            softplus = cutlass.Float32(0.0)
            if x <= 20.0:
                softplus = cutlass.Float32(cute.log(1.0 + cute.exp(x)))
            else:
                softplus = x

            rate = cute.exp(cutlass.Float32(a_log[head]))
            decay[linear] = cute.exp(-rate * softplus)

            raw_beta = cutlass.Float32(b[linear])
            beta[linear] = 1.0 / (1.0 + cute.exp(-raw_beta))

    @cute.kernel
    def _recurrent(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        initial_state: cute.Tensor,
        decay: cute.Tensor,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        v_tile, value_head, sequence = cute.arch.block_idx()
        lane = tidx % 32
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        v_local = lane % _V_PER_WARP
        k_local = lane // _V_PER_WARP
        value_col = v_tile * _V_PER_CTA + warp * _V_PER_WARP + v_local
        qk_head = value_head // 2

        state_regs = cute.make_rmem_tensor(
            cute.make_layout((_K_PER_LANE,), stride=(1,)), cutlass.Float32
        )
        for i in range(_K_PER_LANE):
            k_idx = i * (32 // _V_PER_WARP) + k_local
            state_regs[i] = cutlass.Float32(
                initial_state[(sequence, value_head, value_col, k_idx)]
            )

        start = cutlass.Int32(cu_seqlens[sequence])
        end = cutlass.Int32(cu_seqlens[sequence + 1])
        length = end - start

        for token_offset in cutlass.range(length, unroll=1):
            token = start + token_offset
            gate = cutlass.Float32(decay[(token, value_head)])
            update_gate = cutlass.Float32(beta[(token, value_head)])

            old_value = cutlass.Float32(0.0)
            for i in range(_K_PER_LANE):
                k_idx = i * (32 // _V_PER_WARP) + k_local
                state_value = state_regs[i] * gate
                state_regs[i] = state_value
                old_value += state_value * cutlass.Float32(k[(token, qk_head, k_idx)])

            for offset in (16, 8, 4):
                old_value += cute.arch.shuffle_sync_bfly(
                    old_value, offset=offset, mask=-1, mask_and_clamp=31
                )

            delta = cutlass.Float32(0.0)
            if k_local == 0:
                delta = update_gate * (
                    cutlass.Float32(v[(token, value_head, value_col)]) - old_value
                )
            delta = cute.arch.shuffle_sync(
                delta, v_local, mask=-1, mask_and_clamp=31
            )

            out_value = cutlass.Float32(0.0)
            for i in range(_K_PER_LANE):
                k_idx = i * (32 // _V_PER_WARP) + k_local
                state_value = state_regs[i] + cutlass.Float32(
                    k[(token, qk_head, k_idx)]
                ) * delta
                state_regs[i] = state_value
                out_value += state_value * cutlass.Float32(
                    q[(token, qk_head, k_idx)]
                )

            for offset in (16, 8, 4):
                out_value += cute.arch.shuffle_sync_bfly(
                    out_value, offset=offset, mask=-1, mask_and_clamp=31
                )

            if k_local == 0:
                output[(token, value_head, value_col)] = cutlass.BFloat16(
                    out_value * self.scale
                )

        for i in range(_K_PER_LANE):
            k_idx = i * (32 // _V_PER_WARP) + k_local
            final_state[(sequence, value_head, value_col, k_idx)] = state_regs[i]

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        state_iter: cute.Pointer,
        a_iter: cute.Pointer,
        b_iter: cute.Pointer,
        a_log_iter: cute.Pointer,
        dt_bias_iter: cute.Pointer,
        cu_iter: cute.Pointer,
        decay_iter: cute.Pointer,
        beta_iter: cute.Pointer,
        output_iter: cute.Pointer,
        final_state_iter: cute.Pointer,
        stream: cuda.CUstream,
    ):
        q = cute.make_tensor(
            q_iter,
            cute.make_layout(
                (self.total_tokens, _Q_HEADS, _HEAD_DIM),
                stride=(_Q_HEADS * _HEAD_DIM, _HEAD_DIM, 1),
            ),
        )
        k = cute.make_tensor(
            k_iter,
            cute.make_layout(
                (self.total_tokens, _Q_HEADS, _HEAD_DIM),
                stride=(_Q_HEADS * _HEAD_DIM, _HEAD_DIM, 1),
            ),
        )
        v = cute.make_tensor(
            v_iter,
            cute.make_layout(
                (self.total_tokens, _V_HEADS, _HEAD_DIM),
                stride=(_V_HEADS * _HEAD_DIM, _HEAD_DIM, 1),
            ),
        )
        state_layout = cute.make_layout(
            (self.num_sequences, _V_HEADS, _HEAD_DIM, _HEAD_DIM),
            stride=(
                _V_HEADS * _HEAD_DIM * _HEAD_DIM,
                _HEAD_DIM * _HEAD_DIM,
                _HEAD_DIM,
                1,
            ),
        )
        initial_state = cute.make_tensor(state_iter, state_layout)
        final_state = cute.make_tensor(final_state_iter, state_layout)

        flat_gate_layout = cute.make_layout(
            (self.total_tokens * _V_HEADS,), stride=(1,)
        )
        a = cute.make_tensor(a_iter, flat_gate_layout)
        b = cute.make_tensor(b_iter, flat_gate_layout)
        decay_flat = cute.make_tensor(decay_iter, flat_gate_layout)
        beta_flat = cute.make_tensor(beta_iter, flat_gate_layout)
        decay = cute.make_tensor(
            decay_iter,
            cute.make_layout(
                (self.total_tokens, _V_HEADS), stride=(_V_HEADS, 1)
            ),
        )
        beta = cute.make_tensor(
            beta_iter,
            cute.make_layout(
                (self.total_tokens, _V_HEADS), stride=(_V_HEADS, 1)
            ),
        )
        a_log = cute.make_tensor(
            a_log_iter, cute.make_layout((_V_HEADS,), stride=(1,))
        )
        dt_bias = cute.make_tensor(
            dt_bias_iter, cute.make_layout((_V_HEADS,), stride=(1,))
        )
        cu_seqlens = cute.make_tensor(
            cu_iter,
            cute.make_layout((self.num_sequences + 1,), stride=(1,)),
        )
        output = cute.make_tensor(
            output_iter,
            cute.make_layout(
                (self.total_tokens, _V_HEADS, _HEAD_DIM),
                stride=(_V_HEADS * _HEAD_DIM, _HEAD_DIM, 1),
            ),
        )

        self._prepare_gates(a, b, a_log, dt_bias, decay_flat, beta_flat).launch(
            grid=(
                cute.ceil_div(self.total_tokens * _V_HEADS, _THREADS),
                1,
                1,
            ),
            block=(_THREADS, 1, 1),
            stream=stream,
        )
        self._recurrent(
            q,
            k,
            v,
            initial_state,
            decay,
            beta,
            cu_seqlens,
            output,
            final_state,
        ).launch(
            grid=(_HEAD_DIM // _V_PER_CTA, _V_HEADS, self.num_sequences),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


def _as_cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True)


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Run GDN prefill and return (bf16 output, fp32 final state)."""
    total_tokens = q.shape[0]
    num_sequences = state.shape[0]
    scale_value = float(scale) if scale else 1.0 / (_HEAD_DIM**0.5)

    output = torch.empty_like(v)
    final_state = torch.empty_like(state)
    decay = torch.empty(
        (total_tokens, _V_HEADS), dtype=torch.float32, device=q.device
    )
    beta = torch.empty_like(decay)

    key = (total_tokens, num_sequences, q.dtype, scale_value)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _COMPILED.get(key)
    if compiled is None:
        op = _GdnRecurrent(total_tokens, num_sequences, scale_value)
        tensors = [
            _as_cute_tensor(x)
            for x in (
                q,
                k,
                v,
                state,
                a,
                b,
                A_log,
                dt_bias,
                cu_seqlens,
                decay,
                beta,
                output,
                final_state,
            )
        ]
        compiled = cute.compile[EnableTVMFFI](
            op, *(tensor.iterator for tensor in tensors), stream=stream
        )
        _COMPILED[key] = compiled

    compiled(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        state.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        cu_seqlens.data_ptr(),
        decay.data_ptr(),
        beta.data_ptr(),
        output.data_ptr(),
        final_state.data_ptr(),
        stream=stream,
    )
    return output, final_state
