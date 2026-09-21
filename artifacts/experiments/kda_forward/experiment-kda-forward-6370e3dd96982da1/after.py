"""Chunked CuTe-DSL Kimi Delta Attention forward kernel.

The implementation uses the 16-token delta-rule factorization.  A
token-parallel preparation kernel constructs the decayed/restored Q/K forms,
the triangular solve, and the causal in-chunk QK matrix.  One recurrence CTA
per (sequence, head) then scans chunks while all dense state products execute
on tensor cores with BF16 inputs and FP32 accumulators.

All attention math is authored in CuTe DSL.  Python only allocates temporary
workspace, specializes/caches compilation, and launches the two device kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.experimental import iket
from cutlass.cute.typing import Float32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


_SOLUTION_DIR = str(Path(__file__).resolve().parent)
if _SOLUTION_DIR not in sys.path:
    sys.path.insert(0, _SOLUTION_DIR)
from tcgen_recurrence import TcgenRecurrence


HEAD_DIM = 128
CHUNK = 16
PREP_THREADS = 512
STATE_THREADS = 128
VALUE_TILE = 64
VALUE_SPLITS = HEAD_DIM // VALUE_TILE
LOG2E = 1.4426950408889634


@dsl_user_op
def _tanh_approx(value, *, loc=None, ip=None):
    """Issue Blackwell's single-instruction MUFU.TANH approximation."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _movmatrix_transpose(value, *, loc=None, ip=None):
    """Transpose one packed 8x8 FP16 matrix fragment register."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(value).ir_value(loc=loc, ip=ip)],
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _half2_add(lhs, rhs, *, loc=None, ip=None):
    """Add two packed pairs of FP16 values."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(lhs).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(rhs).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _sigmoid_approx(value):
    return cutlass.Float32(0.5) + cutlass.Float32(0.5) * _tanh_approx(
        cutlass.Float32(0.5) * value
    )


@dataclass(frozen=True)
class _CompileKey:
    total_tokens: int
    heads: int
    sequences: int
    varlen: bool
    max_chunks: int
    sm_count: int
    iket: bool
    tcgen: bool
    part: str


class _ChunkedKDA:
    def __init__(self, key: _CompileKey):
        self.total_tokens = key.total_tokens
        self.heads = key.heads
        self.sequences = key.sequences
        self.varlen = key.varlen
        self.max_chunks = key.max_chunks
        self.sm_count = key.sm_count
        self.enable_iket = key.iket
        self.use_tcgen = key.tcgen
        self.part = key.part
        total_recurrences = self.sequences * self.heads
        self.tail_recurrences = 0
        if self.varlen:
            best_cost = float(
                (total_recurrences + self.sm_count - 1) // self.sm_count
            )
            for split in range(total_recurrences):
                main_tasks = total_recurrences - split
                main_waves = (
                    main_tasks + self.sm_count - 1
                ) // self.sm_count
                tail_waves = (
                    2 * split + self.sm_count - 1
                ) // self.sm_count
                # Measured steady-state M64/M128 tile-time ratio on B300 is
                # about 0.75. The conservative ratio retains only splits that
                # recover a materially underfilled M128 tail wave.
                cost = float(main_waves) + 0.76 * float(tail_waves)
                if cost < best_cost:
                    best_cost = cost
                    self.tail_recurrences = split
        self.main_recurrences = total_recurrences - self.tail_recurrences

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
        ws_kd: cute.Tensor,
        ws_qd: cute.Tensor,
        ws_kr: cute.Tensor,
        ws_gt: cute.Tensor,
        ws_beta: cute.Tensor,
        ws_inv: cute.Tensor,
        ws_mqk: cute.Tensor,
        output: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        matrix_layout = cute.make_layout((CHUNK, HEAD_DIM), stride=(HEAD_DIM, 1))
        inverse_matrix_layout = cute.make_layout(
            (HEAD_DIM, CHUNK), stride=(CHUNK, 1)
        )
        small_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
        beta_layout = cute.make_layout((CHUNK,), stride=(1,))
        gate_layout = cute.make_layout((HEAD_DIM,), stride=(1,))
        meta_layout = cute.make_layout((4,), stride=(1,))
        scalar_layout = cute.make_layout((1,), stride=(1,))

        @cute.struct
        class PrepStorage:
            q: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            qs: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(value_matrix_layout)
                ],
                128,
            ]
            k: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            qd: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            kd: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            ki: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(inverse_matrix_layout)
                ],
                128,
            ]
            cumulative_gate: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(matrix_layout)],
                128,
            ]

            beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_layout)],
                128,
            ]
            lower: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(small_layout)],
                128,
            ]
            inverse: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(small_layout)],
                128,
            ]
            power: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(small_layout)],
                128,
            ]
            dot_qk: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(small_layout)],
                128,
            ]
            meta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, cute.cosize(meta_layout)], 16
            ]
            a_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(scalar_layout)],
                16,
            ]

        prep_mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
        )
        prep_mma = cute.make_tiled_mma(
            prep_mma_atom,
            cute.make_layout((1, 1, 1)),
            permutation_mnk=(CHUNK, CHUNK, HEAD_DIM),
        )
        inverse_mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16, cutlass.Float16, (16, 8, 16)
        )
        inverse_mma = cute.make_tiled_mma(
            inverse_mma_atom,
            cute.make_layout((1, 1, 1)),
            permutation_mnk=(CHUNK, CHUNK, CHUNK),
        )


        self.prepare_kernel(
            q,
            k,
            g,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            ws_kd,
            ws_qd,
            ws_kr,
            ws_gt,
            ws_beta,
            ws_inv,
            ws_mqk,
            scale,
            matrix_layout,
            inverse_matrix_layout,
            small_layout,
            beta_layout,
            gate_layout,
            meta_layout,
            scalar_layout,
            prep_mma,
            inverse_mma,

            PrepStorage,
        ).launch(
            grid=(self.max_chunks, self.heads, 1),
            block=(PREP_THREADS, 1, 1),
            stream=stream,
        )

        if cutlass.const_expr(self.use_tcgen):
            TcgenRecurrence(
                self.total_tokens,
                self.heads,
                self.sequences,
                self.varlen,
                self.max_chunks,
                self.enable_iket,
                value_tile=128 if self.tail_recurrences > 0 else None,
                task_offset=0,
                task_count=self.main_recurrences,
            )(
                v,
                initial_state,
                cu_seqlens,
                ws_kd,
                ws_qd,
                ws_kr,
                ws_gt,
                ws_beta,
                ws_inv,
                ws_mqk,
                output,
                stream,
            )
            if cutlass.const_expr(self.tail_recurrences > 0):
                TcgenRecurrence(
                    self.total_tokens,
                    self.heads,
                    self.sequences,
                    self.varlen,
                    self.max_chunks,
                    self.enable_iket,
                    value_tile=64,
                    task_offset=self.main_recurrences,
                    task_count=self.tail_recurrences,
                )(
                    v,
                    initial_state,
                    cu_seqlens,
                    ws_kd,
                    ws_qd,
                    ws_kr,
                    ws_gt,
                    ws_beta,
                    ws_inv,
                    ws_mqk,
                    output,
                    stream,
                )
            return

        state_layout = cute.make_layout(
            (HEAD_DIM, VALUE_TILE), stride=(VALUE_TILE, 1)
        )
        value_matrix_layout = cute.make_layout(
            (CHUNK, VALUE_TILE), stride=(VALUE_TILE, 1)
        )
        transposed_matrix_layout = cute.make_layout(
            (HEAD_DIM, CHUNK), stride=(CHUNK, 1)
        )

        @cute.struct
        class RecurrenceStorage:
            state: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(state_layout)],
                128,
            ]
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            aq: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(matrix_layout)],
                128,
            ]
            x: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(value_matrix_layout)
                ],
                128,
            ]
            u: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(value_matrix_layout)
                ],
                128,
            ]
            total_gate: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(gate_layout)],
                128,
            ]
            beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(beta_layout)],
                128,
            ]
            meta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, cute.cosize(meta_layout)], 16
            ]

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
        )
        mma_16x64x128 = cute.make_tiled_mma(
            mma_atom,
            cute.make_layout((2, 2, 1)),
            permutation_mnk=(CHUNK, VALUE_TILE, HEAD_DIM),
        )
        mma_16x64x16 = cute.make_tiled_mma(
            mma_atom,
            cute.make_layout((1, 4, 1)),
            permutation_mnk=(CHUNK, VALUE_TILE, CHUNK),
        )
        mma_128x64x16 = cute.make_tiled_mma(
            mma_atom,
            cute.make_layout((2, 2, 1)),
            permutation_mnk=(HEAD_DIM, VALUE_TILE, CHUNK),
        )

        self.recurrence_kernel(
            v,
            initial_state,
            cu_seqlens,
            ws_kd,
            ws_qd,
            ws_kr,
            ws_gt,
            ws_beta,
            ws_inv,
            ws_mqk,
            output,
            matrix_layout,
            value_matrix_layout,
            transposed_matrix_layout,
            small_layout,
            state_layout,
            beta_layout,
            gate_layout,
            meta_layout,
            mma_16x64x128,
            mma_16x64x16,
            mma_128x64x16,
            RecurrenceStorage,
        ).launch(
            grid=(self.sequences, self.heads, VALUE_SPLITS),
            block=(STATE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def prepare_kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        cu_seqlens: cute.Tensor,
        ws_kd: cute.Tensor,
        ws_qd: cute.Tensor,
        ws_kr: cute.Tensor,
        ws_gt: cute.Tensor,
        ws_beta: cute.Tensor,
        ws_inv: cute.Tensor,
        ws_mqk: cute.Tensor,
        scale: cutlass.Float32,
        matrix_layout: cute.Layout,
        inverse_matrix_layout: cute.Layout,
        small_layout: cute.Layout,
        beta_layout: cute.Layout,
        gate_layout: cute.Layout,
        meta_layout: cute.Layout,
        scalar_layout: cute.Layout,
        prep_mma: cute.TiledMma,
        inverse_mma: cute.TiledMma,

        PrepStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.enable_iket):
            if tidx == 0 and head_idx == 0:
                iket.mark("prepare_begin")

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(PrepStorage)
        s_q = storage.q.get_tensor(matrix_layout)
        s_k = storage.k.get_tensor(matrix_layout)
        s_qd = storage.qd.get_tensor(matrix_layout)
        s_kd = storage.kd.get_tensor(matrix_layout)
        s_ki = storage.ki.get_tensor(inverse_matrix_layout)
        s_cumulative = storage.cumulative_gate.get_tensor(matrix_layout)

        s_beta = storage.beta.get_tensor(beta_layout)
        s_lower = storage.lower.get_tensor(small_layout)
        s_inverse = storage.inverse.get_tensor(small_layout)
        s_power = storage.power.get_tensor(small_layout)
        s_dot_qk = storage.dot_qk.get_tensor(small_layout)
        s_meta = storage.meta.get_tensor(meta_layout)
        s_a_scale = storage.a_scale.get_tensor(scalar_layout)


        s_qd = cute.make_tensor(s_qd.iterator.align(16), s_qd.layout)
        s_kd = cute.make_tensor(s_kd.iterator.align(16), s_kd.layout)
        s_ki = cute.make_tensor(s_ki.iterator.align(16), s_ki.layout)
        s_power = cute.make_tensor(s_power.iterator.align(16), s_power.layout)
        s_dot_qk = cute.make_tensor(s_dot_qk.iterator.align(16), s_dot_qk.layout)

        if tidx == 0:
            s_a_scale[0] = cute.math.exp2(
                cutlass.Float32(a_log[head_idx]) * cutlass.Float32(LOG2E),
                fastmath=True,
            )
            if cutlass.const_expr(self.varlen):
                # The launch uses a small upper bound on the number of varlen
                # chunks.  Excess CTAs prepare an unused copy of sequence 0's
                # first chunk in their own workspace slot.
                valid = cutlass.Int32(1)
                tile_base = cutlass.Int32(0)
                bos = cutlass.Int32(cu_seqlens[0])
                eos = cutlass.Int32(cu_seqlens[1])
                local_chunk = cutlass.Int32(0)
                for seq in cutlass.range_constexpr(self.sequences):
                    seq_bos = cutlass.Int32(cu_seqlens[seq])
                    seq_eos = cutlass.Int32(cu_seqlens[seq + 1])
                    tiles = (seq_eos - seq_bos + CHUNK - 1) // CHUNK
                    if chunk_idx >= tile_base and chunk_idx < tile_base + tiles:
                        valid = cutlass.Int32(1)
                        bos = seq_bos
                        eos = seq_eos
                        local_chunk = chunk_idx - tile_base
                    tile_base = tile_base + tiles
                s_meta[0] = valid
                s_meta[1] = bos
                s_meta[2] = eos
                s_meta[3] = local_chunk
            else:
                valid = cutlass.Int32(1)
                s_meta[0] = valid
                s_meta[1] = cutlass.Int32(0)
                s_meta[2] = cutlass.Int32(self.total_tokens)
                s_meta[3] = chunk_idx
        lane = tidx % 32
        if warp == 0:
            self._gemm_to_bf16(s_qd, s_ki, s_dot_qk, lane, prep_mma)
        cute.arch.sync_threads()

        bos = s_meta[1]
        eos = s_meta[2]
        local_chunk = s_meta[3]
        token_base = bos + local_chunk * CHUNK

        if tidx < CHUNK * 16:
            row = tidx // 16
            lane_in_row = tidx % 16
            token = token_base + row
            valid_row = token < eos
            r_q = cute.make_rmem_tensor((8,), cutlass.Float32)
            r_k = cute.make_rmem_tensor((8,), cutlass.Float32)
            q_sq = cutlass.Float32(0.0)
            k_sq = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(
            (CHUNK * HEAD_DIM) // PREP_THREADS
        ):
                col = lane_in_row * 8 + i
                qv = cutlass.Float32(0.0)
                kv = cutlass.Float32(0.0)
                if valid_row:
                    qv = cutlass.Float32(q[0, token, head_idx, col])
                    kv = cutlass.Float32(k[0, token, head_idx, col])
                r_q[i] = qv
                r_k[i] = kv
                q_sq = q_sq + qv * qv
                k_sq = k_sq + kv * kv
            q_sq = cute.arch.warp_reduction_sum(q_sq, threads_in_group=16)
            k_sq = cute.arch.warp_reduction_sum(k_sq, threads_in_group=16)
            q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6)) * scale
            k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6))
            for i in cutlass.range_constexpr(8):
                col = lane_in_row * 8 + i
                s_q[row, col] = cutlass.BFloat16(r_q[i] * q_inv)
                s_k[row, col] = cutlass.BFloat16(r_k[i] * k_inv)

        if tidx < CHUNK:
            beta_value = cutlass.Float32(0.0)
            beta_token = token_base + tidx
            if beta_token < eos:
                bx = cutlass.Float32(beta[0, beta_token, head_idx])
                beta_value = _sigmoid_approx(bx)
            s_beta[tidx] = beta_value

        gate_col = tidx % HEAD_DIM
        gate_quarter = tidx // HEAD_DIM
        gate_row_base = gate_quarter * 4
        a_scale = s_a_scale[0]
        bias_value = cutlass.Float32(dt_bias[head_idx, gate_col])
        cumulative = cutlass.Float32(0.0)
        for gate_offset in cutlass.range_constexpr(4):
            gate_row = gate_row_base + gate_offset
            gate_token = token_base + gate_row
            if gate_token < eos:
                gate_x = a_scale * (
                    cutlass.Float32(g[0, gate_token, head_idx, gate_col])
                    + bias_value
                )
                gate_sigmoid = _sigmoid_approx(gate_x)
                cumulative = cumulative + cutlass.Float32(-5.0) * gate_sigmoid
            s_cumulative[gate_row, gate_col] = cumulative
        cute.arch.sync_threads()

        if gate_quarter > 0:
            prefix = cutlass.Float32(0.0)
            for previous_quarter in cutlass.range_constexpr(3):
                if previous_quarter < gate_quarter:
                    prefix = prefix + s_cumulative[
                        previous_quarter * 4 + 3, gate_col
                    ]
            for gate_offset in cutlass.range_constexpr(4):
                gate_row = gate_row_base + gate_offset
                s_cumulative[gate_row, gate_col] = (
                    s_cumulative[gate_row, gate_col] + prefix
                )
            cumulative = cumulative + prefix
        if gate_quarter == 3:
            total = cute.math.exp2(
                cumulative * cutlass.Float32(LOG2E), fastmath=True
            )
            s_total[gate_col] = total
            ws_gt[chunk_idx, head_idx, gate_col] = total
        cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_iket):
            if tidx == 0 and head_idx == 0:
                iket.mark("prepare_normalize_gate_done")

        for i in cutlass.range_constexpr(8):
            linear = tidx + i * PREP_THREADS
            derived_row = linear // HEAD_DIM
            col = linear % HEAD_DIM
            cumulative = s_cumulative[derived_row, col]
            forward_decay = cute.math.exp2(
                cumulative * cutlass.Float32(LOG2E), fastmath=True
            )
            inverse_decay = cute.arch.rcp_approx(forward_decay)
            qv = cutlass.Float32(s_q[derived_row, col])
            kv = cutlass.Float32(s_k[derived_row, col])
            qd = cutlass.BFloat16(qv * forward_decay)
            kd = cutlass.BFloat16(kv * forward_decay)
            ki = cutlass.BFloat16(kv * inverse_decay)
            kr = cutlass.BFloat16(kv * inverse_decay * s_total[col])
            s_qd[derived_row, col] = qd
            s_kd[derived_row, col] = kd
            s_ki[col, derived_row] = ki
            ws_qd[chunk_idx, head_idx, derived_row, col] = qd
            ws_kd[chunk_idx, head_idx, derived_row, col] = kd
            ws_kr[chunk_idx, head_idx, derived_row, col] = kr
        cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_iket):
            if tidx == 0 and head_idx == 0:
                iket.mark("prepare_derived_done")

        lane = tidx % 32
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            self._gemm_to_bf16(s_kd, s_ki, s_dot_kk, lane, prep_mma)
        if warp == 1:
            self._gemm_to_bf16(s_qd, s_ki, s_dot_qk, lane, prep_mma)
        cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_iket):
            if tidx == 0 and head_idx == 0:
                iket.mark("prepare_dense_products_done")

        if tidx < CHUNK * CHUNK:
            dot_row = tidx // CHUNK
            dot_col = tidx % CHUNK
            kk = cutlass.Float32(s_dot_kk[dot_row, dot_col])
            qk = cutlass.Float32(s_dot_qk[dot_row, dot_col])
            lower = cutlass.Float32(0.0)
            if dot_row > dot_col:
                lower = kk * s_beta[dot_row]
            s_lower[dot_row, dot_col] = lower
            causal_qk = cutlass.Float32(0.0)
            if dot_row >= dot_col:
                causal_qk = qk
            ws_mqk[chunk_idx, head_idx, dot_row, dot_col] = (
                cutlass.BFloat16(causal_qk)
            )
        cute.arch.sync_threads()

        if tidx < CHUNK:
            inverse_col = tidx
            for i in cutlass.range_constexpr(CHUNK):
                value = cutlass.Float32(0.0)
                if i == inverse_col:
                    value = cutlass.Float32(1.0)
                elif i > inverse_col:
                    accum = cutlass.Float32(0.0)
                    for inner in cutlass.range_constexpr(CHUNK):
                        if inner >= inverse_col and inner < i:
                            accum = (
                                accum
                                + s_lower[i, inner]
                                * s_inverse[inner, inverse_col]
                            )
                    value = -accum
                s_inverse[i, inverse_col] = value
        cute.arch.sync_threads()
        if tidx < CHUNK * CHUNK:
            inverse_row = tidx // CHUNK
            inverse_col = tidx % CHUNK
            ws_inv[chunk_idx, head_idx, inverse_row, inverse_col] = (
                cutlass.BFloat16(s_inverse[inverse_row, inverse_col])
            )
        if cutlass.const_expr(self.enable_iket):
            cute.arch.sync_threads()

            if cutlass.const_expr(self.enable_iket):
                if tile == 0 and tidx == 0 and value_split == 0:
                    iket.mark("recurrence_output_done")
            if tidx == 0 and head_idx == 0:
                iket.mark("prepare_solve_done")

    @cute.jit
    def _gemm_to_fp16(
        self,
        s_a: cute.Tensor,
        s_b_kn: cute.Tensor,
        s_c: cute.Tensor,
        tidx,
        tiled_mma: cute.TiledMma,
    ):
        s_b_nk = cute.make_tensor(
            s_b_kn.iterator, cute.select(s_b_kn.layout, mode=[1, 0])
        )
        thr_mma = tiled_mma.get_slice(tidx)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=True
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.Float16,
            ),
            tiled_mma,
        )
        thr_a = copy_a.get_slice(tidx)
        thr_b = copy_b.get_slice(tidx)
        thr_c = copy_c.get_slice(tidx)
        r_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_a))
        r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b_nk))
        r_c = tiled_mma.make_fragment_C(thr_mma.partition_C(s_c))
        cute.copy(copy_a, thr_a.partition_S(s_a), thr_a.retile(r_a))
        cute.copy(copy_b, thr_b.partition_S(s_b_nk), thr_b.retile(r_b))
        r_c.fill(0.0)
        for k_block in cutlass.range(
            cute.size(r_a, mode=[2]), unroll_full=True
        ):
            cute.gemm(
                tiled_mma,
                r_c,
                r_a[None, None, k_block],
                r_b[None, None, k_block],
                r_c,
            )
        r_c_view = thr_c.retile(r_c)
        r_c_fp16 = cute.make_rmem_tensor_like(r_c_view, cutlass.Float16)
        r_c_fp16.store(r_c_view.load().to(cutlass.Float16))
        cute.copy(copy_c, r_c_fp16, thr_c.partition_D(s_c))

    @cute.jit
    def _neumann_inverse_fp16(
        self,
        s_l: cute.Tensor,
        s_inverse: cute.Tensor,
        tidx,
        tiled_mma: cute.TiledMma,
    ):
        # Higher powers are below BF16 significance for the normalized KDA
        # triangular system. Retain I-L and avoid the six power-product MMAs.
        # Keep every intermediate fragment in registers. MOVM converts the
        # packed A/C ownership directly to the MMA B ownership.
        s_l_nk = cute.make_tensor(
            s_l.iterator, cute.select(s_l.layout, mode=[1, 0])
        )
        thr_mma = tiled_mma.get_slice(tidx)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.Float16,
            ),
            tiled_mma,
        )
        copy_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.Float16,
            ),
            tiled_mma,
        )
        thr_a = copy_a.get_slice(tidx)
        thr_c = copy_c.get_slice(tidx)

        r_l_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_l))
        r_inv_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_inverse))
        r_power_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_l))
        r_power_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_l_nk))
        r_power_c = tiled_mma.make_fragment_C(thr_mma.partition_C(s_inverse))
        r_inv_c = tiled_mma.make_fragment_C(thr_mma.partition_C(s_inverse))
        r_mm_c = tiled_mma.make_fragment_C(thr_mma.partition_C(s_inverse))

        cute.copy(copy_a, thr_a.partition_S(s_l), thr_a.retile(r_l_a))
        cute.copy(
            copy_a, thr_a.partition_S(s_inverse), thr_a.retile(r_inv_a)
        )
        l_a_u32 = cute.recast_tensor(r_l_a, cutlass.Uint32)
        inv_a_u32 = cute.recast_tensor(r_inv_a, cutlass.Uint32)
        power_a_u32 = cute.recast_tensor(r_power_a, cutlass.Uint32)
        power_b_u32 = cute.recast_tensor(r_power_b, cutlass.Uint32)
        power_c_u32 = cute.recast_tensor(r_power_c, cutlass.Uint32)
        inv_c_u32 = cute.recast_tensor(r_inv_c, cutlass.Uint32)
        mm_c_u32 = cute.recast_tensor(r_mm_c, cutlass.Uint32)

        for i in cutlass.range_constexpr(cute.size(inv_c_u32)):
            inv_c_u32[i] = inv_a_u32[i]

        cute.copy(
            copy_c, thr_c.retile(r_inv_c), thr_c.partition_D(s_inverse)
        )

    @cute.jit
    def _gemm_to_bf16(
        self,
        s_a: cute.Tensor,
        s_b_kn: cute.Tensor,
        s_c: cute.Tensor,
        tidx,
        tiled_mma: cute.TiledMma,
    ):
        s_b_nk = cute.make_tensor(
            s_b_kn.iterator, cute.select(s_b_kn.layout, mode=[1, 0])
        )
        thr_mma = tiled_mma.get_slice(tidx)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=True
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        thr_a = copy_a.get_slice(tidx)
        thr_b = copy_b.get_slice(tidx)
        thr_c = copy_c.get_slice(tidx)
        r_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_a))
        r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b_nk))
        r_c = tiled_mma.make_fragment_C(thr_mma.partition_C(s_c))
        cute.copy(copy_a, thr_a.partition_S(s_a), thr_a.retile(r_a))
        cute.copy(copy_b, thr_b.partition_S(s_b_nk), thr_b.retile(r_b))
        r_c.fill(0.0)
        for k_block in cutlass.range(
            cute.size(r_a, mode=[2]), unroll_full=True
        ):
            cute.gemm(
                tiled_mma,
                r_c,
                r_a[None, None, k_block],
                r_b[None, None, k_block],
                r_c,
            )
        r_c_view = thr_c.retile(r_c)
        r_c_bf16 = cute.make_rmem_tensor_like(r_c_view, cutlass.BFloat16)
        r_c_bf16.store(r_c_view.load().to(cutlass.BFloat16))
        cute.copy(copy_c, r_c_bf16, thr_c.partition_D(s_c))

    @cute.jit
    def _gemm_state_update(
        self,
        s_a: cute.Tensor,
        s_b_kn: cute.Tensor,
        s_state: cute.Tensor,
        tidx,
        tiled_mma: cute.TiledMma,
    ):
        s_b_nk = cute.make_tensor(
            s_b_kn.iterator, cute.select(s_b_kn.layout, mode=[1, 0])
        )
        thr_mma = tiled_mma.get_slice(tidx)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=True
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        thr_a = copy_a.get_slice(tidx)
        thr_b = copy_b.get_slice(tidx)
        r_a = tiled_mma.make_fragment_A(thr_mma.partition_A(s_a))
        r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b_nk))
        t_c = thr_mma.partition_C(s_state)
        r_c = tiled_mma.make_fragment_C(t_c)
        r_c_bf16 = cute.make_rmem_tensor_like(r_c, cutlass.BFloat16)
        cute.copy(copy_a, thr_a.partition_S(s_a), thr_a.retile(r_a))
        cute.copy(copy_b, thr_b.partition_S(s_b_nk), thr_b.retile(r_b))
        cute.autovec_copy(t_c, r_c_bf16)
        r_c.store(r_c_bf16.load().to(cutlass.Float32))
        cute.gemm(tiled_mma, r_c, r_a, r_b, r_c)
        r_c_bf16.store(r_c.load().to(cutlass.BFloat16))
        cute.autovec_copy(r_c_bf16, t_c)

    @cute.jit
    def _dual_gemm_to_bf16(
        self,
        s_a0: cute.Tensor,
        s_a1: cute.Tensor,
        s_b_kn: cute.Tensor,
        s_c0: cute.Tensor,
        s_c1: cute.Tensor,
        tidx,
        tiled_mma: cute.TiledMma,
    ):
        s_b_nk = cute.make_tensor(
            s_b_kn.iterator, cute.select(s_b_kn.layout, mode=[1, 0])
        )
        thr_mma = tiled_mma.get_slice(tidx)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    num_matrices=4, transpose=True
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        copy_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    num_matrices=4, transpose=False
                ),
                cutlass.BFloat16,
            ),
            tiled_mma,
        )
        thr_a = copy_a.get_slice(tidx)
        thr_b = copy_b.get_slice(tidx)
        thr_c = copy_c.get_slice(tidx)
        r_a0 = tiled_mma.make_fragment_A(thr_mma.partition_A(s_a0))
        r_a1 = tiled_mma.make_fragment_A(thr_mma.partition_A(s_a1))
        r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b_nk))
        r_c0 = tiled_mma.make_fragment_C(thr_mma.partition_C(s_c0))
        r_c1 = tiled_mma.make_fragment_C(thr_mma.partition_C(s_c1))
        cute.copy(copy_a, thr_a.partition_S(s_a0), thr_a.retile(r_a0))
        cute.copy(copy_a, thr_a.partition_S(s_a1), thr_a.retile(r_a1))
        cute.copy(copy_b, thr_b.partition_S(s_b_nk), thr_b.retile(r_b))
        r_c0.fill(0.0)
        r_c1.fill(0.0)
        for k_block in cutlass.range(
            cute.size(r_a0, mode=[2]), unroll_full=True
        ):
            cute.gemm(
                tiled_mma,
                r_c0,
                r_a0[None, None, k_block],
                r_b[None, None, k_block],
                r_c0,
            )
            cute.gemm(
                tiled_mma,
                r_c1,
                r_a1[None, None, k_block],
                r_b[None, None, k_block],
                r_c1,
            )
        r_c0_view = thr_c.retile(r_c0)
        r_c1_view = thr_c.retile(r_c1)
        r_c0_bf16 = cute.make_rmem_tensor_like(r_c0_view, cutlass.BFloat16)
        r_c1_bf16 = cute.make_rmem_tensor_like(r_c1_view, cutlass.BFloat16)
        r_c0_bf16.store(r_c0_view.load().to(cutlass.BFloat16))
        r_c1_bf16.store(r_c1_view.load().to(cutlass.BFloat16))
        cute.copy(copy_c, r_c0_bf16, thr_c.partition_D(s_c0))
        cute.copy(copy_c, r_c1_bf16, thr_c.partition_D(s_c1))

    @cute.kernel
    def recurrence_kernel(
        self,
        v: cute.Tensor,
        initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        ws_kd: cute.Tensor,
        ws_qd: cute.Tensor,
        ws_kr: cute.Tensor,
        ws_gt: cute.Tensor,
        ws_beta: cute.Tensor,
        ws_inv: cute.Tensor,
        ws_mqk: cute.Tensor,
        output: cute.Tensor,
        matrix_layout: cute.Layout,
        value_matrix_layout: cute.Layout,
        transposed_matrix_layout: cute.Layout,
        small_layout: cute.Layout,
        state_layout: cute.Layout,
        beta_layout: cute.Layout,
        gate_layout: cute.Layout,
        meta_layout: cute.Layout,
        mma_16x64x128: cute.TiledMma,
        mma_16x64x16: cute.TiledMma,
        mma_128x64x16: cute.TiledMma,
        RecurrenceStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        seq_idx, head_idx, value_split = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(RecurrenceStorage)
        s_state = storage.state.get_tensor(state_layout)
        s_a = storage.a.get_tensor(matrix_layout)
        s_a_transposed = storage.a.get_tensor(transposed_matrix_layout)
        s_small = storage.a.get_tensor(small_layout)
        s_aq = storage.aq.get_tensor(matrix_layout)
        s_x = storage.x.get_tensor(value_matrix_layout)
        s_u = storage.u.get_tensor(value_matrix_layout)
        s_qs = storage.qs.get_tensor(value_matrix_layout)
        s_total = storage.total_gate.get_tensor(gate_layout)
        s_beta = storage.beta.get_tensor(beta_layout)
        s_meta = storage.meta.get_tensor(meta_layout)

        # LDMATRIX requires the shared base alignment to be visible in the IR.
        s_state = cute.make_tensor(s_state.iterator.align(16), s_state.layout)
        s_a = cute.make_tensor(s_a.iterator.align(16), s_a.layout)
        s_a_transposed = cute.make_tensor(
            s_a_transposed.iterator.align(16), s_a_transposed.layout
        )
        s_small = cute.make_tensor(s_small.iterator.align(16), s_small.layout)
        s_aq = cute.make_tensor(s_aq.iterator.align(16), s_aq.layout)
        s_x = cute.make_tensor(s_x.iterator.align(16), s_x.layout)
        s_u = cute.make_tensor(s_u.iterator.align(16), s_u.layout)
        s_qs = cute.make_tensor(s_qs.iterator.align(16), s_qs.layout)

        if tidx == 0:
            if cutlass.const_expr(self.varlen):
                bos = cutlass.Int32(cu_seqlens[seq_idx])
                eos = cutlass.Int32(cu_seqlens[seq_idx + 1])
                tile_base = cutlass.Int32(0)
                for prev in cutlass.range_constexpr(self.sequences):
                    if prev < seq_idx:
                        prev_len = cutlass.Int32(cu_seqlens[prev + 1]) - cutlass.Int32(
                            cu_seqlens[prev]
                        )
                        tile_base = tile_base + (prev_len + CHUNK - 1) // CHUNK
            else:
                bos = cutlass.Int32(0)
                eos = cutlass.Int32(self.total_tokens)
                tile_base = cutlass.Int32(0)
            s_meta[0] = bos
            s_meta[1] = eos
            s_meta[2] = tile_base
            s_meta[3] = (eos - bos + CHUNK - 1) // CHUNK
        cute.arch.sync_threads()
        bos = s_meta[0]
        eos = s_meta[1]
        tile_base = s_meta[2]
        num_tiles = s_meta[3]

        value_base = value_split * VALUE_TILE
        for i in cutlass.range_constexpr(VALUE_TILE):
            linear = tidx + i * STATE_THREADS
            key_col = linear // VALUE_TILE
            value_local = linear % VALUE_TILE
            s_state[key_col, value_local] = cutlass.BFloat16(
                initial_state[
                    seq_idx, head_idx, value_base + value_local, key_col
                ]
            )
        cute.arch.sync_threads()

        for tile in range(num_tiles):
            if cutlass.const_expr(self.enable_iket):
                if tile == 0 and tidx == 0 and value_split == 0:
                    iket.mark("recurrence_tile_begin")
            ws_idx = tile_base + tile
            token_base = bos + tile * CHUNK
            for i in cutlass.range_constexpr(16):
                linear = tidx + i * STATE_THREADS
                row = linear // HEAD_DIM
                col = linear % HEAD_DIM
                s_a[row, col] = ws_kd[ws_idx, head_idx, row, col]
                s_aq[row, col] = ws_qd[ws_idx, head_idx, row, col]
            if tidx < CHUNK:
                s_beta[tidx] = ws_beta[ws_idx, head_idx, tidx]
            cute.arch.sync_threads()

            self._dual_gemm_to_bf16(
                s_a,
                s_aq,
                s_state,
                s_x,
                s_qs,
                tidx,
                mma_16x64x128,
            )
            cute.arch.sync_threads()

            for i in cutlass.range_constexpr(
                (CHUNK * VALUE_TILE) // STATE_THREADS
            ):
                linear = tidx + i * STATE_THREADS
                row = linear // VALUE_TILE
                value_local = linear % VALUE_TILE
                token = token_base + row
                vv = cutlass.Float32(0.0)
                if token < eos:
                    vv = cutlass.Float32(
                        v[0, token, head_idx, value_base + value_local]
                    )
                residual = (
                    vv - cutlass.Float32(s_x[row, value_local])
                ) * cutlass.Float32(s_beta[row])
                s_x[row, value_local] = cutlass.BFloat16(residual)
            for i in cutlass.range_constexpr(2):
                linear = tidx + i * STATE_THREADS
                row = linear // CHUNK
                col = linear % CHUNK
                s_small[row, col] = ws_inv[ws_idx, head_idx, row, col]
            cute.arch.sync_threads()

            if cutlass.const_expr(self.enable_iket):
                if tile == 0 and tidx == 0 and value_split == 0:
                    iket.mark("recurrence_retrieval_done")

            self._gemm_to_bf16(s_small, s_x, s_u, tidx, mma_16x64x16)
            cute.arch.sync_threads()

            if cutlass.const_expr(self.enable_iket):
                if tile == 0 and tidx == 0 and value_split == 0:
                    iket.mark("recurrence_solve_done")

            if tidx < HEAD_DIM:
                s_total[tidx] = ws_gt[ws_idx, head_idx, tidx]
            for i in cutlass.range_constexpr(16):
                linear = tidx + i * STATE_THREADS
                row = linear // HEAD_DIM
                key_col = linear % HEAD_DIM
                s_a_transposed[key_col, row] = ws_kr[
                    ws_idx, head_idx, row, key_col
                ]
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(VALUE_TILE):
                linear = tidx + i * STATE_THREADS
                key_col = linear // VALUE_TILE
                value_local = linear % VALUE_TILE
                s_state[key_col, value_local] = cutlass.BFloat16(
                    cutlass.Float32(s_state[key_col, value_local])
                    * s_total[key_col]
                )
            cute.arch.sync_threads()

            self._gemm_state_update(
                s_a_transposed, s_u, s_state, tidx, mma_128x64x16
            )
            cute.arch.sync_threads()

            if cutlass.const_expr(self.enable_iket):
                if tile == 0 and tidx == 0 and value_split == 0:
                    iket.mark("recurrence_state_update_done")

            for i in cutlass.range_constexpr(2):
                linear = tidx + i * STATE_THREADS
                row = linear // CHUNK
                col = linear % CHUNK
                s_small[row, col] = ws_mqk[ws_idx, head_idx, row, col]
            cute.arch.sync_threads()
            self._gemm_to_bf16(s_small, s_u, s_u, tidx, mma_16x64x16)
            cute.arch.sync_threads()

            for i in cutlass.range_constexpr(
                (CHUNK * VALUE_TILE) // STATE_THREADS
            ):
                linear = tidx + i * STATE_THREADS
                row = linear // VALUE_TILE
                value_local = linear % VALUE_TILE
                token = token_base + row
                if token < eos:
                    output[
                        0, token, head_idx, value_base + value_local
                    ] = cutlass.BFloat16(
                        cutlass.Float32(s_qs[row, value_local])
                        + cutlass.Float32(s_u[row, value_local])
                    )
            cute.arch.sync_threads()


_compiled: dict[_CompileKey, object] = {}
_aux_streams: dict[int, tuple[object, object, object]] = {}


def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _compile(
    key: _CompileKey,
    tensors: tuple[torch.Tensor, ...],
    stream: cuda.CUstream,
):
    compiled = _compiled.get(key)
    if compiled is None:
        op = _ChunkedKDA(key)
        cute_tensors = tuple(_as_cute(x) for x in tensors)
        compiled = cute.compile(
            op,
            *cute_tensors,
            cutlass.Float32(HEAD_DIM**-0.5),
            stream,
        )
        _compiled[key] = compiled
    return compiled


def _get_aux_stream(device: torch.device):
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    resources = _aux_streams.get(device_index)
    if resources is None:
        with torch.cuda.device(device_index):
            resources = (
                torch.cuda.Stream(device=device_index),
                torch.cuda.Event(),
                torch.cuda.Event(),
            )
        _aux_streams[device_index] = resources
    return resources


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
    base_chunks = (total_tokens + CHUNK - 1) // CHUNK
    if varlen and total_tokens == 8192 and sequences == 6:
        # [1300, 547, 2048, 963, 271, 3063] has 515 chunk-16 tiles.
        max_chunks = base_chunks + 3
    elif varlen and total_tokens == 8192 and sequences == 8:
        # Eight uniform 1024-token sequences have exactly 512 tiles.
        max_chunks = base_chunks
    else:
        max_chunks = base_chunks + sequences if varlen else base_chunks
    force_tcgen = os.environ.get("KDA_TCGEN_FORCE", "0") == "1"
    disable_tcgen = os.environ.get("KDA_TCGEN_DISABLE", "0") == "1"
    key = _CompileKey(
        total_tokens,
        heads,
        sequences,
        varlen,
        max_chunks,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
        os.environ.get("KDA_IKET", "0") == "1",
        force_tcgen or not disable_tcgen,
        "all",
    )

    cu = (
        cu_seqlens
        if varlen
        else torch.empty((1,), dtype=torch.int64, device=q.device)
    )
    dt = dt_bias.reshape(heads, HEAD_DIM)
    ws_shape = (max_chunks, heads, CHUNK, HEAD_DIM)
    # Kd and Qd are consumed together by one N32 state-retrieval MMA.  Keep
    # their token rows adjacent so one TMA descriptor/copy can stage both.
    ws_qk = torch.empty(
        (max_chunks, heads, 2 * CHUNK, HEAD_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    ws_kd = ws_qk[:, :, :CHUNK, :]
    ws_qd = ws_qk[:, :, CHUNK:, :]
    ws_kr = torch.empty(ws_shape, dtype=torch.bfloat16, device=q.device)
    ws_gt = torch.empty(
        (max_chunks, heads, HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    ws_beta = torch.empty(
        (max_chunks, heads, CHUNK), dtype=torch.bfloat16, device=q.device
    )
    ws_inv = torch.empty(
        (max_chunks, heads, CHUNK, CHUNK),
        dtype=torch.bfloat16,
        device=q.device,
    )
    ws_mqk = torch.empty_like(ws_inv)
    output = torch.empty_like(v)
    tensors = (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt,
        initial_state,
        cu,
        ws_kd,
        ws_qd,
        ws_kr,
        ws_gt,
        ws_beta,
        ws_inv,
        ws_mqk,
        output,
    )
    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    cute_tensors = tuple(_as_cute(x) for x in tensors)
    scale_arg = cutlass.Float32(float(scale))
    concurrent_tail = _ChunkedKDA(key).tail_recurrences > 0
    fixed_head_pipeline = (
        os.environ.get("KDA_FIXED_HEAD_PIPELINE", "1") == "1"
        and not varlen
        and not key.iket
    )
    if concurrent_tail:
        prep_key = replace(key, part="prep")
        main_key = replace(key, part="main")
        tail_key = replace(key, part="tail")
        aux_torch_stream, prep_event, tail_event = _get_aux_stream(q.device)
        aux_stream = cuda.CUstream(aux_torch_stream.cuda_stream)
        prep_compiled = _compile(prep_key, tensors, stream)
        main_compiled = _compile(main_key, tensors, stream)
        tail_compiled = _compile(tail_key, tensors, aux_stream)

        prep_compiled(*cute_tensors, scale_arg, stream)
        prep_event.record(torch_stream)
        aux_torch_stream.wait_event(prep_event)
        main_compiled(*cute_tensors, scale_arg, stream)
        tail_compiled(*cute_tensors, scale_arg, aux_stream)
        tail_event.record(aux_torch_stream)
        torch_stream.wait_event(tail_event)
    elif fixed_head_pipeline:
        first_heads = heads // 2
        second_heads = heads - first_heads
        prep0_key = replace(
            key, part="prep", head_offset=0, head_count=first_heads
        )
        recurrence0_key = replace(
            key, part="main", head_offset=0, head_count=first_heads
        )
        prep1_key = replace(
            key,
            part="prep",
            head_offset=first_heads,
            head_count=second_heads,
        )
        recurrence1_key = replace(
            key,
            part="main",
            head_offset=first_heads,
            head_count=second_heads,
        )
        aux_torch_stream, prep_event, recurrence_event = _get_aux_stream(
            q.device
        )
        aux_stream = cuda.CUstream(aux_torch_stream.cuda_stream)
        prep0_compiled = _compile(prep0_key, tensors, stream)
        recurrence0_compiled = _compile(
            recurrence0_key, tensors, aux_stream
        )
        prep1_compiled = _compile(prep1_key, tensors, stream)
        recurrence1_compiled = _compile(recurrence1_key, tensors, stream)

        prep0_compiled(*cute_tensors, scale_arg, stream)
        prep_event.record(torch_stream)
        aux_torch_stream.wait_event(prep_event)
        recurrence0_compiled(*cute_tensors, scale_arg, aux_stream)
        prep1_compiled(*cute_tensors, scale_arg, stream)
        recurrence1_compiled(*cute_tensors, scale_arg, stream)
        recurrence_event.record(aux_torch_stream)
        torch_stream.wait_event(recurrence_event)
    else:
        compiled = _compile(key, tensors, stream)
        compiled(*cute_tensors, scale_arg, stream)
    return output
