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
_LOG2_E = math.log2(math.e)


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
                + Int64(lane) * Int64(_VALUES_PER_LANE)
            )
            g_q_nope = cute.make_tensor(
                m_q_nope.iterator + q_nope_offset,
                cute.make_layout(_VALUES_PER_LANE),
            )
            r_q_nope = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            cute.autovec_copy(g_q_nope, r_q_nope)

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
                        + Int64(lane) * Int64(_VALUES_PER_LANE)
                    )
                    g_kv = cute.make_tensor(
                        m_ckv.iterator + kv_offset,
                        cute.make_layout(_VALUES_PER_LANE),
                    )
                    r_kv = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
                    cute.autovec_copy(g_kv, r_kv)

                    score = Float32(0.0)
                    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                        score += Float32(r_q_nope[j]) * Float32(r_kv[j])

                    kpe_offset = Int64(kv_idx) * Int64(_PE_DIM) + Int64(lane) * Int64(2)
                    kpe_0 = (m_kpe.iterator + kpe_offset).load()
                    kpe_1 = (m_kpe.iterator + kpe_offset + Int64(1)).load()
                    score += Float32(q_pe_0) * Float32(kpe_0)
                    score += Float32(q_pe_1) * Float32(kpe_1)
                    score = cute.arch.warp_reduction_sum(score) * sm_scale

                    if topk_pos == Int32(0):
                        row_max = score
                        row_sum = Float32(1.0)
                        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                            r_acc[j] = Float32(r_kv[j])
                    else:
                        new_max = cute.arch.fmax(row_max, score)
                        old_scale = cute.math.exp2(
                            (row_max - new_max) * Float32(_LOG2_E), fastmath=True
                        )
                        weight = cute.math.exp2(
                            (score - new_max) * Float32(_LOG2_E), fastmath=True
                        )
                        row_sum = row_sum * old_scale + weight
                        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                            r_acc[j] = (
                                r_acc[j] * old_scale + weight * Float32(r_kv[j])
                            )
                        row_max = new_max

                    topk_pos += Int32(1)

            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)

            r_out = cute.make_rmem_tensor(_VALUES_PER_LANE, BFloat16)
            for j in cutlass.range_constexpr(_VALUES_PER_LANE):
                r_out[j] = (r_acc[j] * inv_sum).to(BFloat16)

            out_offset = (
                (Int64(token) * Int64(_HEADS) + Int64(head)) * Int64(_NOPE_DIM)
                + Int64(lane) * Int64(_VALUES_PER_LANE)
            )
            g_out = cute.make_tensor(
                m_out.iterator + out_offset,
                cute.make_layout(_VALUES_PER_LANE),
            )
            cute.autovec_copy(r_out, g_out)

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


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute sparse MLA decode attention and return bf16 [T, 16, 512]."""
    num_tokens = q_nope.shape[0]
    output = torch.empty_like(q_nope)
    compiled = _compile_kernel()
    compiled(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        float(sm_scale),
        num_tokens,
        num_tokens * _HEADS,
    )
    return output
