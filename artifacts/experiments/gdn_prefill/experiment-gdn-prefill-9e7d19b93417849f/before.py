"""Gated Delta Net prefill kernel for NVIDIA B300 (SM100/SM103), CuTe-DSL.

Two-kernel chunk-parallel decomposition of the chunked gated delta rule:

  K1 ("local", parallel over every (chunk, v-head) unit):
    - fused gate computation:  lg = -exp(A_log) * softplus(a + dt_bias),
      beta = sigmoid(b), G = cumsum(lg) within the chunk
    - KK = K@K^T, QK = Q@K^T   (tcgen05 bf16 MMAs)
    - M = tril(beta_i * exp(G_i - G_j) * KK, -1);  T = (I+M)^-1 via
      warp-level 32x32 forward substitution + two levels of block combines
      (fp16 MMAs; the combine is  T_next = Td - Td@M@Td  restricted to the
      new off-diagonal blocks)
    - U0 = T@(beta*V), W = T@(beta*gamma*K)
    - att = tril(exp(G_i - G_j) * QK)  (incl. diagonal)
    - writes to global workspace:  O_partial = scale*att@U0 (into O),
      Qt = gamma*Q - att@W,  G = W^T@Ks,  P = U0^T@Ks,  gl = exp(G_last)
      where Ks_j = exp(G_last - G_j) * k_j.
  K2 ("chain", one CTA per (seq, v-head)):
    per chunk:  Y = S~@G^T (one MMA on the critical path),
                S <- gl*S - Y + P   (fp32 master state in TMEM)
                O += scale * (Qt @ S~_old)
    state kept as S~[v,k] (k-last), matching the global state layout.

Everything is computed in one host call; no torch eager kernels are launched.
"""

import functools
import math

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Boolean, Float32
from cutlass.cute.nvgpu import OperandMajorMode, OperandSource

# ---------------------------------------------------------------------------
C = 128          # chunk size
D = 128          # head dim
HV = 8           # value heads
HQ = 4           # q/k heads

ACC = cutlass.Float32
BF16 = cutlass.BFloat16
F16 = cutlass.Float16
F32 = cutlass.Float32

# named barrier ids (0 is reserved for sync_threads)
BAR_CORES = 1        # cores-only internal barrier (128 threads)
BAR_MD = 2           # M + D ready               (cores + mma = 160)
BAR_RT1 = 3          # X1 roundtrip done         (160)
BAR_T64 = 4          # T64d updated              (160)
BAR_RT2 = 5          # X2 roundtrip done         (160)
BAR_TAILS = 6        # T-tmem, R, bgK ready      (160)
BAR_UWK = 7          # U0/W smem copies + Ks     (160)
BAR_CORES2 = 8       # cores-only (cumsum)
# K2:
BAR_S_READY = 2      # S~ produced               (160)
BAR_Z_FREE = 3       # O-epi consumed Z          (160)

THREADS = 192        # warps 0-3 cores, warp 4 mma, warp 5 load


def group(n):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, n)


class GdnKernels:
    def __init__(self):
        self.tiler = (C, C, D)

    # ------------------------------------------------------------------
    # host-side entry: builds layouts/TMA atoms and launches K1 then K2
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,      # [T, 4, 128] bf16
        k_ptr: cute.Pointer,      # [T, 4, 128] bf16
        v_ptr: cute.Pointer,      # [T, 8, 128] bf16
        o_ptr: cute.Pointer,      # [T, 8, 128] bf16 (out)
        state_ptr: cute.Pointer,  # [N, 8, 128, 128] f32 (in, k-last)
        ns_ptr: cute.Pointer,     # [N, 8, 128, 128] f32 (out)
        alog_ptr: cute.Pointer,   # [8] f32
        a_ptr: cute.Pointer,      # [T, 8] bf16
        dtb_ptr: cute.Pointer,    # [8] f32
        b_ptr: cute.Pointer,      # [T, 8] bf16
        cu_seqlens: cute.Tensor,  # [N+1] i64
        gws_ptr: cute.Pointer,    # [slots, 8, 128, 128] bf16 workspace (G^T)
        pws_ptr: cute.Pointer,    # [slots, 8, 128, 128] f32 workspace (P, rows v)
        qtws_ptr: cute.Pointer,   # [slots, 8, 128, 128] bf16 workspace (Qt)
        glws_ptr: cute.Pointer,   # [slots, 8] f32 workspace
        T: Int32,
        N: Int32,
        slots: Int32,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        # ---------------- global tensors ----------------
        # q/k: (row, d, head)   head has stride D in memory
        qk_layout = cute.make_layout((T, D, HQ), stride=(HQ * D, 1, D))
        mQ = cute.make_tensor(q_ptr, qk_layout)
        mK = cute.make_tensor(k_ptr, qk_layout)
        v_layout = cute.make_layout((T, D, HV), stride=(HV * D, 1, D))
        mV = cute.make_tensor(v_ptr, v_layout)
        mO = cute.make_tensor(o_ptr, v_layout)
        st_layout = cute.make_layout(
            (D, D, (HV, N)), stride=(D, 1, (D * D, HV * D * D))
        )
        mState = cute.make_tensor(state_ptr, st_layout)
        mNState = cute.make_tensor(ns_ptr, st_layout)
        ab_layout = cute.make_layout((T, HV), stride=(HV, 1))
        mA = cute.make_tensor(a_ptr, ab_layout)
        mB = cute.make_tensor(b_ptr, ab_layout)
        mAlog = cute.make_tensor(alog_ptr, cute.make_layout((HV,)))
        mDtb = cute.make_tensor(dtb_ptr, cute.make_layout((HV,)))

        ws_layout = cute.make_layout(
            (D, D, (HV, slots)), stride=(D, 1, (D * D, HV * D * D))
        )
        mGws = cute.make_tensor(gws_ptr, ws_layout)
        mPws = cute.make_tensor(pws_ptr, ws_layout)
        mQtws = cute.make_tensor(qtws_ptr, ws_layout)
        mGlws = cute.make_tensor(glws_ptr, cute.make_layout((slots, HV), stride=(HV, 1)))

        # ---------------- MMA objects ----------------
        cg = tcgen05.CtaGroup.ONE
        # bf16, A K-major (smem), B K-major: KK / QK
        mma_kk = sm100_utils.make_trivial_tiled_mma(
            BF16, OperandMajorMode.K, OperandMajorMode.K, ACC, cg, (C, C)
        )
        # fp16 ts: A from TMEM (K-major), B MN-major: X1 = D@M, X2 = Td@M
        mma_f16_ts = sm100_utils.make_trivial_tiled_mma(
            F16, OperandMajorMode.K, OperandMajorMode.MN, ACC, cg, (C, C),
            OperandSource.TMEM,
        )
        # fp16 ss: A K-major, B MN-major: E = X1@D, F = X2@Td
        mma_f16_ss = sm100_utils.make_trivial_tiled_mma(
            F16, OperandMajorMode.K, OperandMajorMode.MN, ACC, cg, (C, C)
        )
        # bf16 ts: A from TMEM (K-major), B MN-major: U0/W (A=T), OL/ATTW (A=att)
        mma_bf16_ts = sm100_utils.make_trivial_tiled_mma(
            BF16, OperandMajorMode.K, OperandMajorMode.MN, ACC, cg, (C, C),
            OperandSource.TMEM,
        )
        # bf16 ss: A MN-major, B MN-major: P = U0^T-role, G = W^T-role vs Ks
        mma_bf16_mn = sm100_utils.make_trivial_tiled_mma(
            BF16, OperandMajorMode.MN, OperandMajorMode.MN, ACC, cg, (C, C)
        )
        # K2: bf16 ss: A K-major, B MN-major (Y = S~ @ Gt)
        mma_k2_y = sm100_utils.make_trivial_tiled_mma(
            BF16, OperandMajorMode.K, OperandMajorMode.MN, ACC, cg, (C, C)
        )
        # K2: bf16 ss: A K-major, B K-major (Z = Qt @ S~)
        mma_k2_z = sm100_utils.make_trivial_tiled_mma(
            BF16, OperandMajorMode.K, OperandMajorMode.K, ACC, cg, (C, C)
        )

        # ---------------- smem layouts ----------------
        sK_lay = sm100_utils.make_smem_layout_a(mma_kk, self.tiler, BF16, 1)
        sV_lay = sm100_utils.make_smem_layout_b(mma_bf16_ts, self.tiler, BF16, 1)
        sM_lay = sm100_utils.make_smem_layout_b(mma_f16_ts, self.tiler, F16, 1)
        sX_lay = sm100_utils.make_smem_layout_a(mma_f16_ss, self.tiler, F16, 1)
        sKs_lay = sm100_utils.make_smem_layout_b(mma_bf16_mn, self.tiler, BF16, 1)
        sUW_lay = sm100_utils.make_smem_layout_b(mma_bf16_ts, self.tiler, BF16, 1)

        # K2
        sSt_lay = sm100_utils.make_smem_layout_a(mma_k2_y, self.tiler, BF16, 1)
        sGt_lay = sm100_utils.make_smem_layout_b(mma_k2_y, self.tiler, BF16, 2)
        sQt_lay = sm100_utils.make_smem_layout_a(mma_k2_z, self.tiler, BF16, 2)

        # ---------------- TMA atoms ----------------
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cg)
        sK_l1 = cute.select(sK_lay, mode=[0, 1, 2])
        tma_k, tv_k = cute.nvgpu.make_tiled_tma_atom_A(
            op, mK, sK_l1, self.tiler, mma_kk, cute.make_layout((1, 1, 1)).shape
        )
        tma_q, tv_q = cute.nvgpu.make_tiled_tma_atom_A(
            op, mQ, sK_l1, self.tiler, mma_kk, cute.make_layout((1, 1, 1)).shape
        )
        sV_l1 = cute.select(sV_lay, mode=[0, 1, 2])
        tma_v, tv_v = cute.nvgpu.make_tiled_tma_atom_B(
            op, mV, sV_l1, self.tiler, mma_bf16_ts, cute.make_layout((1, 1, 1)).shape
        )
        sGt_l1 = cute.select(sGt_lay, mode=[0, 1, 2])
        tma_g, tv_g = cute.nvgpu.make_tiled_tma_atom_B(
            op, mGws, sGt_l1, self.tiler, mma_k2_y, cute.make_layout((1, 1, 1)).shape
        )
        sQt_l1 = cute.select(sQt_lay, mode=[0, 1, 2])
        tma_qt, tv_qt = cute.nvgpu.make_tiled_tma_atom_A(
            op, mQtws, sQt_l1, self.tiler, mma_k2_z, cute.make_layout((1, 1, 1)).shape
        )

        k_bytes = cute.size_in_bytes(BF16, sK_l1)
        v_bytes = cute.size_in_bytes(BF16, sV_l1)
        g_bytes = cute.size_in_bytes(BF16, sGt_l1)

        self.k1(
            mma_kk, mma_f16_ts, mma_f16_ss, mma_bf16_ts, mma_bf16_mn,
            tma_k, tv_k, tma_q, tv_q, tma_v, tv_v,
            mQ, mA, mB, mAlog, mDtb, cu_seqlens,
            mO, mGws, mPws, mQtws, mGlws,
            sK_lay, sV_lay, sM_lay, sX_lay, sKs_lay, sUW_lay,
            T, N, scale, k_bytes, v_bytes,
        ).launch(
            grid=(slots, HV, 1), block=[THREADS, 1, 1], stream=stream,
        )

        self.k2(
            mma_k2_y, mma_k2_z,
            tma_g, tv_g, tma_qt, tv_qt,
            mState, mNState, mO, mPws, mGlws, cu_seqlens,
            sSt_lay, sGt_lay, sQt_lay,
            T, N, scale, g_bytes,
        ).launch(
            grid=(N, HV, 1), block=[THREADS, 1, 1], stream=stream,
        )

    # ------------------------------------------------------------------
    # device helpers
    # ------------------------------------------------------------------
    @cute.jit
    def map_slot(self, cu_seqlens: cute.Tensor, N: Int32, bid: Int32):
        """slot id -> (seq n, chunk c, cu_n, Lseq, found)."""
        sb = Int32(0)
        n_out = Int32(0)
        c_out = Int32(0)
        cu_out = Int32(0)
        L_out = Int32(0)
        found = Boolean(False)
        for m in cutlass.range(N):
            cu0 = Int32(cu_seqlens[m])
            cu1 = Int32(cu_seqlens[m + 1])
            L = cu1 - cu0
            nc = (L + C - 1) // C
            if (not found) and (bid < sb + nc):
                found = Boolean(True)
                n_out = Int32(m)
                c_out = bid - sb
                cu_out = cu0
                L_out = L
            sb += nc
        return n_out, c_out, cu_out, L_out, found

    @cute.jit
    def slot_base(self, cu_seqlens: cute.Tensor, n: Int32):
        sb = Int32(0)
        for m in cutlass.range(n):
            cu0 = Int32(cu_seqlens[m])
            cu1 = Int32(cu_seqlens[m + 1])
            sb += (cu1 - cu0 + C - 1) // C
        return sb

    @cute.jit
    def exec_mma(self, tiled_mma, tAcc, tA, tB, acc: cutlass.Constexpr = False):
        num_kphases = cute.size(tB, mode=[2])
        for kphase in cutlass.range(num_kphases, unroll_all=True):
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE, cutlass.Boolean(kphase != 0 or acc)
            )
            cute.gemm(
                tiled_mma, tAcc,
                tA[None, None, kphase, 0],
                tB[None, None, kphase, 0],
                tAcc,
            )
        return tiled_mma

    # ==================================================================
    # K1: chunk-local kernel
    # ==================================================================
    @cute.kernel
    def k1(
        self,
        mma_kk: cute.TiledMma,
        mma_f16_ts: cute.TiledMma,
        mma_f16_ss: cute.TiledMma,
        mma_bf16_ts: cute.TiledMma,
        mma_bf16_mn: cute.TiledMma,
        tma_k: cute.CopyAtom, mKv: cute.Tensor,
        tma_q: cute.CopyAtom, mQv: cute.Tensor,
        tma_v: cute.CopyAtom, mVv: cute.Tensor,
        mQ: cute.Tensor,
        mA: cute.Tensor, mB: cute.Tensor,
        mAlog: cute.Tensor, mDtb: cute.Tensor,
        cu_seqlens: cute.Tensor,
        mO: cute.Tensor, mGws: cute.Tensor, mPws: cute.Tensor,
        mQtws: cute.Tensor, mGlws: cute.Tensor,
        sK_lay: cute.ComposedLayout, sV_lay: cute.ComposedLayout,
        sM_lay: cute.ComposedLayout, sX_lay: cute.ComposedLayout,
        sKs_lay: cute.ComposedLayout, sUW_lay: cute.ComposedLayout,
        T: Int32, N: Int32, scale: Float32,
        k_bytes: cutlass.Constexpr, v_bytes: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        if warp_idx == 5:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)

        # ---------------- shared memory ----------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorageK1)

        sK = smem.allocate_tensor(BF16, sK_lay.outer, 1024, sK_lay.inner)
        sQ = smem.allocate_tensor(BF16, sK_lay.outer, 1024, sK_lay.inner)
        sV = smem.allocate_tensor(BF16, sV_lay.outer, 1024, sV_lay.inner)
        sM = smem.allocate_tensor(F16, sM_lay.outer, 1024, sM_lay.inner)
        sD = smem.allocate_tensor(F16, sM_lay.outer, 1024, sM_lay.inner)
        sX = smem.allocate_tensor(F16, sX_lay.outer, 1024, sX_lay.inner)

        # overlay views (same bytes, different role):
        # bgK (B MN bf16) over sQ; Ks (B MN bf16) over sD; U0/W (B MN bf16)
        # over sX and sM respectively.
        sBGK = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, dtype=BF16), sUW_lay.outer
        )
        sBGK = cute.make_tensor(sBGK.iterator, sUW_lay)
        sKs = cute.make_tensor(
            cute.recast_ptr(sD.iterator, dtype=BF16), sKs_lay
        )
        sU0 = cute.make_tensor(
            cute.recast_ptr(sX.iterator, dtype=BF16), sUW_lay
        )
        sW = cute.make_tensor(
            cute.recast_ptr(sM.iterator, dtype=BF16), sUW_lay
        )

        # small buffers
        sCum = storage.cum.get_tensor(cute.make_layout((C,)))
        sBeta = storage.beta.get_tensor(cute.make_layout((C,)))
        sIvt = storage.ivt.get_tensor(cute.make_layout((4, 32, 33)))
        sIvtT = storage.ivtT.get_tensor(cute.make_layout((4, 32, 33)))

        # ---------------- pipelines ----------------
        pipe_kq_p, pipe_kq_c = pipeline.PipelineTmaUmma.create(
            num_stages=2, producer_group=group(1), consumer_group=group(1),
            tx_count=k_bytes, barrier_storage=storage.mbar_kq.data_ptr(),
        ).make_participants()
        pipe_v_p, pipe_v_c = pipeline.PipelineTmaAsync.create(
            num_stages=1, producer_group=group(1), consumer_group=group(128),
            tx_count=v_bytes, barrier_storage=storage.mbar_v.data_ptr(),
        ).make_participants()
        ev_p, ev_c = pipeline.PipelineUmmaAsync.create(
            num_stages=4, producer_group=group(1), consumer_group=group(128),
            barrier_storage=storage.mbar_ev.data_ptr(),
        ).make_participants()

        # tmem alloc
        if warp_idx == 4:
            cute.arch.alloc_tmem(cutlass.Int32(512), storage.tmem_buf)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            ACC, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_buf
        )

        # ---------------- work mapping ----------------
        hv = bidy
        hq = hv // 2
        n, c, cu_n, Lseq, found = self.map_slot(cu_seqlens, N, bidx)
        if found:
            slot = bidx
            row0 = cu_n + C * c          # first token row of this chunk
            Lc = cutlass.min(Int32(C), Lseq - C * c)

            # ---------------- fragments / tmem tensors ----------------
            thr_kk = mma_kk.get_slice(0)
            tArK = thr_kk.make_fragment_A(sK)
            tBrK = thr_kk.make_fragment_B(sK)
            tArQ = thr_kk.make_fragment_A(sQ)
            acc_shape = thr_kk.partition_shape_C((C, C))
            tCfake = thr_kk.make_fragment_C(acc_shape)
            tKK = cute.make_tensor(tmem_ptr + 0, tCfake.layout)
            tQK = cute.make_tensor(tmem_ptr + 128, tCfake.layout)
            tSCRATCH = cute.make_tensor(tmem_ptr + 0, tCfake.layout)   # X1/E/X2/F
            tU0 = cute.make_tensor(tmem_ptr + 0, tCfake.layout)
            tW = cute.make_tensor(tmem_ptr + 256, tCfake.layout)
            tOL = cute.make_tensor(tmem_ptr + 0, tCfake.layout)
            tATTW = cute.make_tensor(tmem_ptr + 256, tCfake.layout)
            tP = cute.make_tensor(tmem_ptr + 128, tCfake.layout)
            tG = cute.make_tensor(tmem_ptr + 384, tCfake.layout)

            # fp16 TMEM A operand: D at f32-col 448 (fp16 pair-packed)
            thr_f16ts = mma_f16_ts.get_slice(0)
            tDf = thr_f16ts.make_fragment_A(sX_lay.outer.shape)
            tD_A = cute.make_tensor(
                cute.recast_ptr(tmem_ptr, dtype=F16) + 448 * 2, tDf.layout
            )
            tBrM = thr_f16ts.make_fragment_B(sM)

            thr_f16ss = mma_f16_ss.get_slice(0)
            tArX = thr_f16ss.make_fragment_A(sX)
            tBrD = thr_f16ss.make_fragment_B(sD)

            # bf16 TMEM A operands: T at 448, att at 128
            thr_bf16ts = mma_bf16_ts.get_slice(0)
            tTf = thr_bf16ts.make_fragment_A(sUW_lay.outer.shape)
            tT_A = cute.make_tensor(
                cute.recast_ptr(tmem_ptr, dtype=BF16) + 448 * 2, tTf.layout
            )
            tATT_A = cute.make_tensor(
                cute.recast_ptr(tmem_ptr, dtype=BF16) + 128 * 2, tTf.layout
            )
            tBrV = thr_bf16ts.make_fragment_B(sV)
            tBrBGK = thr_bf16ts.make_fragment_B(sBGK)
            tBrU0 = thr_bf16ts.make_fragment_B(sU0)
            tBrW = thr_bf16ts.make_fragment_B(sW)

            thr_mn = mma_bf16_mn.get_slice(0)
            tArU0mn = thr_mn.make_fragment_A(sU0)
            tArWmn = thr_mn.make_fragment_A(sW)
            tBrKs = thr_mn.make_fragment_B(sKs)

            # ==========================================================
            # LOAD warp
            # ==========================================================
            if warp_idx == 5:
                mK_off = cute.domain_offset((row0, 0, 0), mKv)
                gK = cute.flat_divide(mK_off, cute.select(self.tiler, mode=[0, 2]))
                tSgK = thr_kk.partition_A(gK)
                tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                    tma_k, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3),
                )
                mQ_off = cute.domain_offset((row0, 0, 0), mQv)
                gQ = cute.flat_divide(mQ_off, cute.select(self.tiler, mode=[0, 2]))
                tSgQ = thr_kk.partition_A(gQ)
                tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                    tma_q, 0, cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3),
                )
                mV_off = cute.domain_offset((row0, 0, 0), mVv)
                gV = cute.flat_divide(mV_off, cute.select(self.tiler, mode=[1, 2]))
                tSgV = thr_bf16ts.partition_B(gV)
                tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                    tma_v, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3),
                )
                hk = pipe_kq_p.acquire_and_advance()
                cute.copy(tma_k, tKgK[None, 0, 0, hq], tKsK[None, 0],
                          tma_bar_ptr=hk.barrier)
                hq_ = pipe_kq_p.acquire_and_advance()
                cute.copy(tma_q, tQgQ[None, 0, 0, hq], tQsQ[None, 0],
                          tma_bar_ptr=hq_.barrier)
                hvideo = pipe_v_p.acquire_and_advance()
                cute.copy(tma_v, tVgV[None, 0, 0, hv], tVsV[None, 0],
                          tma_bar_ptr=hvideo.barrier)

            # ==========================================================
            # MMA warp
            # ==========================================================
            if warp_idx == 4:
                hk = pipe_kq_c.wait_and_advance()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_kk, tKK, tArK, tBrK)
                h.commit()
                hqq = pipe_kq_c.wait_and_advance()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_kk, tQK, tArQ, tBrK)
                h.commit()

                # X1 = D @ M
                cute.arch.barrier(barrier_id=BAR_MD, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_f16_ts, tSCRATCH, tD_A, tBrM)
                h.commit()
                # E = X1 @ D
                cute.arch.barrier(barrier_id=BAR_RT1, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_f16_ss, tSCRATCH, tArX, tBrD)
                h.commit()
                # X2 = T64d @ M
                cute.arch.barrier(barrier_id=BAR_T64, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_f16_ts, tSCRATCH, tD_A, tBrM)
                h.commit()
                # F = X2 @ T64d
                cute.arch.barrier(barrier_id=BAR_RT2, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_f16_ss, tSCRATCH, tArX, tBrD)
                h.commit()
                # U0 = T @ R ; W = T @ bgK
                cute.arch.barrier(barrier_id=BAR_TAILS, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_ts, tU0, tT_A, tBrV)
                h.commit()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_ts, tW, tT_A, tBrBGK)
                h.commit()
                # OL = att @ U0 ; ATTW = att @ W ; P = U0^T Ks ; G = W^T Ks
                cute.arch.barrier(barrier_id=BAR_UWK, number_of_threads=160)
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_ts, tOL, tATT_A, tBrU0)
                h.commit()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_ts, tATTW, tATT_A, tBrW)
                h.commit()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_mn, tP, tArU0mn, tBrKs)
                h.commit()
                h = ev_p.acquire_and_advance()
                self.exec_mma(mma_bf16_mn, tG, tArWmn, tBrKs)
                h.commit()

            # ==========================================================
            # CORE warps (0-3)
            # ==========================================================
            if warp_idx < 4:
                lane = tidx % 32
                row = tidx        # 0..127 (thread's token/matrix row)

                # ---- gates ----
                grow = row0 + row
                a_val = Float32(0.0)
                b_val = Float32(0.0)
                if row < Lc:
                    a_val = Float32(mA[grow, hv])
                    b_val = Float32(mB[grow, hv])
                alog = Float32(mAlog[hv])
                dtb = Float32(mDtb[hv])
                x = a_val + dtb
                sp = x
                if x <= 20.0:
                    sp = cute.math.log1p(cute.math.exp(x, fastmath=True))
                lg = -cute.math.exp(alog, fastmath=True) * sp
                bt = 1.0 / (1.0 + cute.math.exp(-b_val, fastmath=True))
                if row >= Lc:
                    lg = Float32(0.0)
                    bt = Float32(0.0)
                sBeta[row] = bt

                # ---- inclusive cumsum of lg over 128 rows ----
                val = lg
                stride_ = 1
                while stride_ < 32:
                    other = cute.arch.shuffle_sync_up(val, offset=stride_)
                    if lane >= stride_:
                        val += other
                    stride_ = stride_ * 2
                if lane == 31:
                    sCum[warp_idx] = val   # temp: per-warp totals in first 4 slots
                cute.arch.barrier(barrier_id=BAR_CORES2, number_of_threads=128)
                pre = Float32(0.0)
                if warp_idx == 1:
                    pre = sCum[0]
                if warp_idx == 2:
                    pre = sCum[0] + sCum[1]
                if warp_idx == 3:
                    pre = sCum[0] + sCum[1] + sCum[2]
                val += pre
                cute.arch.barrier(barrier_id=BAR_CORES2, number_of_threads=128)
                sCum[row] = val
                cute.arch.barrier(barrier_id=BAR_CORES, number_of_threads=128)
                g_row = val                      # G_i for this thread's row
                g_last = sCum[C - 1]

                # ---- consume KK -> build M (smem fp16 MN) + diag blocks ----
                hkk = ev_c.wait_and_advance()
                copy_atom_ld = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), ACC
                )
                tKK0 = cute.make_tensor(
                    tKK.iterator, cute.composition(tKK.layout, cute.make_layout((C, 32)))
                )
                tcp = tcgen05.make_tmem_copy(copy_atom_ld, tKK0)
                thr_cp = tcp.get_slice(row)
                cIn32 = cute.make_identity_tensor((C, 32))
                tTR_src = thr_cp.partition_S(tKK0)
                tTR_coord = thr_cp.partition_D(cIn32)
                tTR_reg = cute.make_rmem_tensor(tTR_coord.shape, ACC)

                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tTR_src.iterator + it * 32, tTR_src.layout)
                    cute.copy(tcp, tsrc, tTR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tTR_reg)):
                        ii = tTR_coord[e][0]
                        jj = tTR_coord[e][1] + it * 32
                        kkv = tTR_reg[e]
                        mval = Float32(0.0)
                        if jj < ii:
                            mval = bt * cute.math.exp(
                                g_row - sCum[jj], fastmath=True
                            ) * kkv
                        sM[ii, jj, 0] = F16(mval)
                        blk = ii // 32
                        if jj >= blk * 32 and jj < blk * 32 + 32 and jj - blk * 32 < 33:
                            sIvt[blk, ii % 32, jj - blk * 32] = mval
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_CORES, number_of_threads=128)
                hkk.release()

                # ---- 32x32 forward substitution (per warp) ----
                # warp w inverts (I + Mblk_w); column per lane in registers.
                tcol = cute.make_rmem_tensor(cute.make_layout((32,)), ACC)
                for r in cutlass.range_constexpr(32):
                    tcol[r] = Float32(0.0)
                if lane < 32:
                    tcol[0] = Float32(1.0) if lane == 0 else Float32(0.0)
                    # tcol[j] = T32[j, lane]
                    for r in cutlass.range_constexpr(1, 32):
                        acc_v = Float32(0.0)
                        for j in cutlass.range_constexpr(0, 32):
                            if j < r:
                                acc_v += sIvt[warp_idx, r, j] * tcol[j]
                        dval = -acc_v
                        if lane == r:
                            dval = Float32(1.0)
                        if lane > r:
                            dval = Float32(0.0)
                        tcol[r] = dval
                    for r in cutlass.range_constexpr(32):
                        sIvtT[warp_idx, lane, r] = tcol[r]  # transposed: [col][row]
                cute.arch.barrier(barrier_id=BAR_CORES, number_of_threads=128)

                # ---- write D: tmem fp16 A-operand + smem fp16 MN ----
                # thread row writes its full 128-wide row (block + zeros)
                blk_r = row // 32
                r_in = row % 32
                copy_atom_st16 = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), ACC
                )
                tDst0 = cute.make_tensor(
                    tmem_ptr + 448,
                    cute.composition(tCfake.layout, cute.make_layout((C, 16))),
                )
                tst = tcgen05.make_tmem_copy(copy_atom_st16, tDst0)
                thr_st = tst.get_slice(row)
                cSt = cute.make_identity_tensor((C, 16))
                tST_dst = thr_st.partition_D(tDst0)
                tST_coord = thr_st.partition_S(cSt)
                tST_reg = cute.make_rmem_tensor(tST_coord.shape, ACC)
                tST_reg_f16 = cute.make_tensor(
                    cute.recast_ptr(tST_reg.iterator, dtype=F16),
                    cute.make_layout((cute.size(tST_coord) * 2,)),
                )
                for it in cutlass.range_constexpr(4):
                    # f32-cols [it*16, it*16+16) == fp16 cols [it*32, it*32+32)
                    for e in cutlass.range_constexpr(cute.size(tST_coord)):
                        ii = tST_coord[e][0]
                        base_j = (tST_coord[e][1] + it * 16) * 2
                        for half in cutlass.range_constexpr(2):
                            jj = base_j + half
                            dv = Float32(0.0)
                            if jj == ii:
                                dv = Float32(1.0)
                            bb = ii // 32
                            if jj >= bb * 32 and jj < bb * 32 + 32:
                                dv = sIvtT[bb, jj - bb * 32, ii - bb * 32]
                            tST_reg_f16[e * 2 + half] = F16(dv)
                    tdst = cute.make_tensor(tST_dst.iterator + it * 16, tST_dst.layout)
                    cute.copy(tst, tST_reg, tdst)
                cute.arch.fence_view_async_tmem_store()
                # smem D (same values)
                for jj in cutlass.range_constexpr(0, C):
                    dv = Float32(0.0)
                    if jj == row:
                        dv = Float32(1.0)
                    if jj >= blk_r * 32 and jj < blk_r * 32 + 32:
                        dv = sIvtT[blk_r, jj - blk_r * 32, r_in]
                    sD[row, jj, 0] = F16(dv)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_MD, number_of_threads=160)

                # ---- consume QK -> att (bf16, in-place tmem) ----
                hqk = ev_c.wait_and_advance()
                tQK0 = cute.make_tensor(
                    tQK.iterator, cute.composition(tQK.layout, cute.make_layout((C, 32)))
                )
                tcpq = tcgen05.make_tmem_copy(copy_atom_ld, tQK0)
                thr_cpq = tcpq.get_slice(row)
                tQR_src = thr_cpq.partition_S(tQK0)
                tQR_coord = thr_cpq.partition_D(cIn32)
                tQR_reg = cute.make_rmem_tensor(tQR_coord.shape, ACC)

                tATTdst0 = cute.make_tensor(
                    tmem_ptr + 128,
                    cute.composition(tCfake.layout, cute.make_layout((C, 16))),
                )
                tstA = tcgen05.make_tmem_copy(copy_atom_st16, tATTdst0)
                thr_stA = tstA.get_slice(row)
                tSA_dst = thr_stA.partition_D(tATTdst0)
                tSA_coord = thr_stA.partition_S(cSt)
                tSA_reg = cute.make_rmem_tensor(tSA_coord.shape, ACC)
                tSA_reg_bf = cute.make_tensor(
                    cute.recast_ptr(tSA_reg.iterator, dtype=BF16),
                    cute.make_layout((cute.size(tSA_coord) * 2,)),
                )
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tQR_src.iterator + it * 32, tQR_src.layout)
                    cute.copy(tcpq, tsrc, tQR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tQR_reg)):
                        ii = tQR_coord[e][0]
                        jj = tQR_coord[e][1] + it * 32
                        qkv = tQR_reg[e]
                        av = Float32(0.0)
                        if jj <= ii:
                            av = cute.math.exp(g_row - sCum[jj], fastmath=True) * qkv
                        tQR_reg[e] = av
                    # repack to bf16 pairs and store to att region (cols it*16..)
                    for e in cutlass.range_constexpr(cute.size(tSA_coord)):
                        tSA_reg_bf[e * 2] = BF16(tQR_reg[e * 2])
                        tSA_reg_bf[e * 2 + 1] = BF16(tQR_reg[e * 2 + 1])
                    tdstA = cute.make_tensor(
                        tSA_dst.iterator + it * 16, tSA_dst.layout
                    )
                    cute.copy(tstA, tSA_reg, tdstA)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier(barrier_id=BAR_CORES, number_of_threads=128)
                hqk.release()

                # ---- bgK = beta*gamma*K (bf16 MN over sQ) ----
                gam_row = cute.math.exp(g_row, fastmath=True)
                bgk_f = bt * gam_row
                for jj in cutlass.range_constexpr(C):
                    sBGK[row, jj, 0] = BF16(bgk_f * Float32(sK[row, jj, 0]))
                # ---- R = beta*V in place ----
                hv_ = pipe_v_c.wait_and_advance()
                for jj in cutlass.range_constexpr(C):
                    sV[row, jj, 0] = BF16(bt * Float32(sV[row, jj, 0]))
                cute.arch.fence_proxy("async.shared", space="cta")
                hv_.release()

                # ---- T-build ping-pong ----
                # X1 roundtrip
                hx1 = ev_c.wait_and_advance()
                tS0 = cute.make_tensor(
                    tSCRATCH.iterator,
                    cute.composition(tSCRATCH.layout, cute.make_layout((C, 32))),
                )
                tcps = tcgen05.make_tmem_copy(copy_atom_ld, tS0)
                thr_cps = tcps.get_slice(row)
                tSR_src = thr_cps.partition_S(tS0)
                tSR_coord = thr_cps.partition_D(cIn32)
                tSR_reg = cute.make_rmem_tensor(tSR_coord.shape, ACC)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tSR_src.iterator + it * 32, tSR_src.layout)
                    cute.copy(tcps, tsrc, tSR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tSR_reg)):
                        ii = tSR_coord[e][0]
                        jj = tSR_coord[e][1] + it * 32
                        sX[ii, jj, 0] = F16(tSR_reg[e])
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_RT1, number_of_threads=160)
                hx1.release()

                # E -> T64d update (blocks (1,0) and (3,2))
                he = ev_c.wait_and_advance()
                in_l0 = (row >= 32 and row < 64) or (row >= 96)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tSR_src.iterator + it * 32, tSR_src.layout)
                    cute.copy(tcps, tsrc, tSR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tSR_reg)):
                        ii = tSR_coord[e][0]
                        jj = tSR_coord[e][1] + it * 32
                        c0 = (ii // 32 - 1) * 32
                        if (ii // 32 == 1 or ii // 32 == 3) and jj >= c0 and jj < c0 + 32:
                            sD[ii, jj, 0] = F16(-tSR_reg[e])
                # tmem D update: rows 32-63 write cols 0-31; rows 96-127 cols 64-95
                if in_l0:
                    itc = Int32(0) if row < 64 else Int32(2)
                    for e in cutlass.range_constexpr(cute.size(tST_coord)):
                        ii = tST_coord[e][0]
                        base_j = tST_coord[e][1] * 2
                        cc0 = (ii // 32 - 1) * 32
                        for half in cutlass.range_constexpr(2):
                            jj = base_j + half
                            dv = Float32(0.0)
                            if jj + itc * 32 >= cc0 and jj + itc * 32 < cc0 + 32:
                                dv = -Float32(
                                    sD[ii, jj + itc * 32, 0]
                                )
                            # note: sD just updated above holds -E in l0 blocks
                            if jj + itc * 32 >= cc0 and jj + itc * 32 < cc0 + 32:
                                tST_reg_f16[e * 2 + half] = F16(dv * -1.0 * -1.0)
                            else:
                                tST_reg_f16[e * 2 + half] = F16(0.0)
                    tdst = cute.make_tensor(tST_dst.iterator + itc * 16, tST_dst.layout)
                    cute.copy(tst, tST_reg, tdst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_T64, number_of_threads=160)
                he.release()

                # X2 roundtrip
                hx2 = ev_c.wait_and_advance()
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tSR_src.iterator + it * 32, tSR_src.layout)
                    cute.copy(tcps, tsrc, tSR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tSR_reg)):
                        ii = tSR_coord[e][0]
                        jj = tSR_coord[e][1] + it * 32
                        sX[ii, jj, 0] = F16(tSR_reg[e])
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_RT2, number_of_threads=160)
                hx2.release()

                # F -> final T (bf16 tmem A at 448)
                hf = ev_c.wait_and_advance()
                tTdst0 = cute.make_tensor(
                    tmem_ptr + 448,
                    cute.composition(tCfake.layout, cute.make_layout((C, 16))),
                )
                tstT = tcgen05.make_tmem_copy(copy_atom_st16, tTdst0)
                thr_stT = tstT.get_slice(row)
                tTT_dst = thr_stT.partition_D(tTdst0)
                tTT_coord = thr_stT.partition_S(cSt)
                tTT_reg = cute.make_rmem_tensor(tTT_coord.shape, ACC)
                tTT_reg_bf = cute.make_tensor(
                    cute.recast_ptr(tTT_reg.iterator, dtype=BF16),
                    cute.make_layout((cute.size(tTT_coord) * 2,)),
                )
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tSR_src.iterator + it * 32, tSR_src.layout)
                    cute.copy(tcps, tsrc, tSR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tTT_coord)):
                        ii = tTT_coord[e][0]
                        base_j = (tTT_coord[e][1] + it * 16) * 2
                        for half in cutlass.range_constexpr(2):
                            jj = base_j + half
                            tv = Float32(0.0)
                            if row >= 64 and jj < 64:
                                tv = -tSR_reg[e * 2 + half]
                            else:
                                if jj <= ii:
                                    tv = Float32(sD[ii, jj, 0])
                            tTT_reg_bf[e * 2 + half] = BF16(tv)
                    tdst = cute.make_tensor(tTT_dst.iterator + it * 16, tTT_dst.layout)
                    cute.copy(tstT, tTT_reg, tdst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier(barrier_id=BAR_CORES, number_of_threads=128)
                hf.release()

                # ---- Ks into sD region (bf16 MN) ----
                ks_f = cute.math.exp(g_last - g_row, fastmath=True)
                for jj in cutlass.range_constexpr(C):
                    sKs[row, jj, 0] = BF16(ks_f * Float32(sK[row, jj, 0]))
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_TAILS, number_of_threads=160)

                # ---- U0 / W copies to smem ----
                hu0 = ev_c.wait_and_advance()
                tU00 = cute.make_tensor(
                    tU0.iterator, cute.composition(tU0.layout, cute.make_layout((C, 32)))
                )
                tcpu = tcgen05.make_tmem_copy(copy_atom_ld, tU00)
                thr_cpu = tcpu.get_slice(row)
                tUR_src = thr_cpu.partition_S(tU00)
                tUR_coord = thr_cpu.partition_D(cIn32)
                tUR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tUR_src.iterator + it * 32, tUR_src.layout)
                    cute.copy(tcpu, tsrc, tUR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tUR_reg)):
                        ii = tUR_coord[e][0]
                        jj = tUR_coord[e][1] + it * 32
                        sU0[ii, jj, 0] = BF16(tUR_reg[e])
                hu0.release()
                hw = ev_c.wait_and_advance()
                tW0 = cute.make_tensor(
                    tW.iterator, cute.composition(tW.layout, cute.make_layout((C, 32)))
                )
                tcpw = tcgen05.make_tmem_copy(copy_atom_ld, tW0)
                thr_cpw = tcpw.get_slice(row)
                tWR_src = thr_cpw.partition_S(tW0)
                tWR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tWR_src.iterator + it * 32, tWR_src.layout)
                    cute.copy(tcpw, tsrc, tWR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tWR_reg)):
                        ii = tUR_coord[e][0]
                        jj = tUR_coord[e][1] + it * 32
                        sW[ii, jj, 0] = BF16(tWR_reg[e])
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_UWK, number_of_threads=160)
                hw.release()

                # ---- epilogues ----
                # O_partial
                hol = ev_c.wait_and_advance()
                tOL0 = cute.make_tensor(
                    tOL.iterator, cute.composition(tOL.layout, cute.make_layout((C, 32)))
                )
                tcpo = tcgen05.make_tmem_copy(copy_atom_ld, tOL0)
                thr_cpo = tcpo.get_slice(row)
                tOR_src = thr_cpo.partition_S(tOL0)
                tOR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                if row < Lc:
                    for it in cutlass.range_constexpr(4):
                        tsrc = cute.make_tensor(
                            tOR_src.iterator + it * 32, tOR_src.layout
                        )
                        cute.copy(tcpo, tsrc, tOR_reg)
                        cute.arch.fence_view_async_tmem_load()
                        for e in cutlass.range_constexpr(cute.size(tOR_reg)):
                            ii = tUR_coord[e][0]
                            jj = tUR_coord[e][1] + it * 32
                            if ii == row:
                                mO[row0 + ii, jj, hv] = BF16(scale * tOR_reg[e])
                hol.release()
                # Qt = gamma*Q - attW
                hatw = ev_c.wait_and_advance()
                tAW0 = cute.make_tensor(
                    tATTW.iterator,
                    cute.composition(tATTW.layout, cute.make_layout((C, 32))),
                )
                tcpaw = tcgen05.make_tmem_copy(copy_atom_ld, tAW0)
                thr_cpaw = tcpaw.get_slice(row)
                tAWR_src = thr_cpaw.partition_S(tAW0)
                tAWR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                qrow_ok = grow < T
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tAWR_src.iterator + it * 32, tAWR_src.layout)
                    cute.copy(tcpaw, tsrc, tAWR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tAWR_reg)):
                        ii = tUR_coord[e][0]
                        jj = tUR_coord[e][1] + it * 32
                        qv = Float32(0.0)
                        if qrow_ok:
                            qv = Float32(mQ[grow, jj, hq])
                        mQtws[ii, jj, (hv, slot)] = BF16(gam_row * qv - tAWR_reg[e])
                hatw.release()
                # P (f32)
                hp = ev_c.wait_and_advance()
                tP0 = cute.make_tensor(
                    tP.iterator, cute.composition(tP.layout, cute.make_layout((C, 32)))
                )
                tcpp = tcgen05.make_tmem_copy(copy_atom_ld, tP0)
                thr_cpp = tcpp.get_slice(row)
                tPR_src = thr_cpp.partition_S(tP0)
                tPR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tPR_src.iterator + it * 32, tPR_src.layout)
                    cute.copy(tcpp, tsrc, tPR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tPR_reg)):
                        ii = tUR_coord[e][0]
                        jj = tUR_coord[e][1] + it * 32
                        mPws[ii, jj, (hv, slot)] = tPR_reg[e]
                hp.release()
                # G (bf16)
                hg = ev_c.wait_and_advance()
                tG0 = cute.make_tensor(
                    tG.iterator, cute.composition(tG.layout, cute.make_layout((C, 32)))
                )
                tcpg = tcgen05.make_tmem_copy(copy_atom_ld, tG0)
                thr_cpg = tcpg.get_slice(row)
                tGR_src = thr_cpg.partition_S(tG0)
                tGR_reg = cute.make_rmem_tensor(tUR_coord.shape, ACC)
                for it in cutlass.range_constexpr(4):
                    tsrc = cute.make_tensor(tGR_src.iterator + it * 32, tGR_src.layout)
                    cute.copy(tcpg, tsrc, tGR_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range_constexpr(cute.size(tGR_reg)):
                        ii = tUR_coord[e][0]
                        jj = tUR_coord[e][1] + it * 32
                        mGws[ii, jj, (hv, slot)] = BF16(tGR_reg[e])
                hg.release()
                if row == 0:
                    mGlws[slot, hv] = cute.math.exp(g_last, fastmath=True)

        # ---------------- tmem dealloc ----------------
        cute.arch.sync_threads()
        if warp_idx == 4:
            cute.arch.dealloc_tmem(tmem_ptr, cutlass.Int32(512))

    # ==================================================================
    # K2: sequential state-chain kernel
    # ==================================================================
    @cute.kernel
    def k2(
        self,
        mma_y: cute.TiledMma,
        mma_z: cute.TiledMma,
        tma_g: cute.CopyAtom, mGv: cute.Tensor,
        tma_qt: cute.CopyAtom, mQtv: cute.Tensor,
        mState: cute.Tensor, mNState: cute.Tensor, mO: cute.Tensor,
        mPws: cute.Tensor, mGlws: cute.Tensor,
        cu_seqlens: cute.Tensor,
        sSt_lay: cute.ComposedLayout, sGt_lay: cute.ComposedLayout,
        sQt_lay: cute.ComposedLayout,
        T: Int32, N: Int32, scale: Float32,
        g_bytes: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidn, bidh, _ = cute.arch.block_idx()

        if warp_idx == 5:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_g)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_qt)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorageK2)
        sSt0 = smem.allocate_tensor(BF16, sSt_lay.outer, 1024, sSt_lay.inner)
        sSt1 = smem.allocate_tensor(BF16, sSt_lay.outer, 1024, sSt_lay.inner)
        sGt = smem.allocate_tensor(BF16, sGt_lay.outer, 1024, sGt_lay.inner)
        sQt = smem.allocate_tensor(BF16, sQt_lay.outer, 1024, sQt_lay.inner)

        pipe_g_p, pipe_g_c = pipeline.PipelineTmaUmma.create(
            num_stages=2, producer_group=group(1), consumer_group=group(1),
            tx_count=g_bytes, barrier_storage=storage.mbar_g.data_ptr(),
        ).make_participants()
        pipe_qt_p, pipe_qt_c = pipeline.PipelineTmaUmma.create(
            num_stages=2, producer_group=group(1), consumer_group=group(1),
            tx_count=g_bytes, barrier_storage=storage.mbar_qt.data_ptr(),
        ).make_participants()
        evy_p, evy_c = pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=group(1), consumer_group=group(128),
            barrier_storage=storage.mbar_evy.data_ptr(),
        ).make_participants()
        evz_p, evz_c = pipeline.PipelineUmmaAsync.create(
            num_stages=2, producer_group=group(1), consumer_group=group(128),
            barrier_storage=storage.mbar_evz.data_ptr(),
        ).make_participants()

        if warp_idx == 4:
            cute.arch.alloc_tmem(cutlass.Int32(512), storage.tmem_buf)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            ACC, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_buf
        )

        n = bidn
        hv = bidh
        cu0 = Int32(cu_seqlens[n])
        cu1 = Int32(cu_seqlens[n + 1])
        L = cu1 - cu0
        nc = (L + C - 1) // C
        base = self.slot_base(cu_seqlens, n)

        row = tidx % 128

        thr_y = mma_y.get_slice(0)
        tASt0 = thr_y.make_fragment_A(sSt0)
        tASt1 = thr_y.make_fragment_A(sSt1)
        tBGt = thr_y.make_fragment_B(sGt)
        acc_shape = thr_y.partition_shape_C((C, C))
        tCfake = thr_y.make_fragment_C(acc_shape)
        tY = cute.make_tensor(tmem_ptr + 0, tCfake.layout)
        tZ = cute.make_tensor(tmem_ptr + 256, tCfake.layout)

        thr_z = mma_z.get_slice(0)
        tAQt = thr_z.make_fragment_A(sQt)
        tBSt0 = thr_z.make_fragment_B(sSt0)
        tBSt1 = thr_z.make_fragment_B(sSt1)

        # S master state in TMEM at 128 (f32), rows = v
        tSmA = cute.make_tensor(tmem_ptr + 128, tCfake.layout)

        if L > 0:
            # ---------------- load warp ----------------
            if warp_idx == 5:
                gG = cute.flat_divide(mGv, cute.select(self.tiler, mode=[1, 2]))
                tSgG = thr_y.partition_B(gG)
                tGsG, tGgG = cute.nvgpu.cpasync.tma_partition(
                    tma_g, 0, cute.make_layout(1),
                    cute.group_modes(sGt, 0, 3), cute.group_modes(tSgG, 0, 3),
                )
                gQt = cute.flat_divide(mQtv, cute.select(self.tiler, mode=[0, 2]))
                tSgQt = thr_z.partition_A(gQt)
                tQsQt, tQgQt = cute.nvgpu.cpasync.tma_partition(
                    tma_qt, 0, cute.make_layout(1),
                    cute.group_modes(sQt, 0, 3), cute.group_modes(tSgQt, 0, 3),
                )
                for cc in cutlass.range(nc):
                    hg = pipe_g_p.acquire_and_advance()
                    cute.copy(tma_g, tGgG[None, 0, 0, (hv, base + cc)],
                              tGsG[None, hg.index], tma_bar_ptr=hg.barrier)
                    hqt = pipe_qt_p.acquire_and_advance()
                    cute.copy(tma_qt, tQgQt[None, 0, 0, (hv, base + cc)],
                              tQsQt[None, hqt.index], tma_bar_ptr=hqt.barrier)

            # ---------------- mma warp ----------------
            if warp_idx == 4:
                for cc in cutlass.range(nc):
                    hg = pipe_g_c.wait_and_advance()
                    cute.arch.barrier(barrier_id=BAR_S_READY, number_of_threads=160)
                    hy = evy_p.acquire_and_advance()
                    if cc % 2 == 0:
                        self.exec_mma(mma_y, tY, tASt0, tBGt[None, None, None, hg.index])
                    else:
                        self.exec_mma(mma_y, tY, tASt1, tBGt[None, None, None, hg.index])
                    hy.commit()
                    hqt = pipe_qt_c.wait_and_advance()
                    cute.arch.barrier(barrier_id=BAR_Z_FREE, number_of_threads=160)
                    hz = evz_p.acquire_and_advance()
                    if cc % 2 == 0:
                        self.exec_mma(
                            mma_z, tZ, tAQt[None, None, None, hqt.index], tBSt0
                        )
                    else:
                        self.exec_mma(
                            mma_z, tZ, tAQt[None, None, None, hqt.index], tBSt1
                        )
                    hz.commit()

            # ---------------- core warps ----------------
            if warp_idx < 4:
                # init: load state -> tmem f32 + S~[0]
                copy_atom_st32 = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), ACC
                )
                tSm0 = cute.make_tensor(
                    tSmA.iterator,
                    cute.composition(tSmA.layout, cute.make_layout((C, 32))),
                )
                tstS = tcgen05.make_tmem_copy(copy_atom_st32, tSm0)
                thr_stS = tstS.get_slice(row)
                cIn32 = cute.make_identity_tensor((C, 32))
                tSS_dst = thr_stS.partition_D(tSm0)
                tSS_coord = thr_stS.partition_S(cIn32)
                tSS_reg = cute.make_rmem_tensor(tSS_coord.shape, ACC)

                copy_atom_ld = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), ACC
                )
                tY0 = cute.make_tensor(
                    tY.iterator, cute.composition(tY.layout, cute.make_layout((C, 32)))
                )
                tcpy = tcgen05.make_tmem_copy(copy_atom_ld, tY0)
                thr_cpy = tcpy.get_slice(row)
                tYR_src = thr_cpy.partition_S(tY0)
                tYR_coord = thr_cpy.partition_D(cIn32)
                tYR_reg = cute.make_rmem_tensor(tYR_coord.shape, ACC)
                tZ0 = cute.make_tensor(
                    tZ.iterator, cute.composition(tZ.layout, cute.make_layout((C, 32)))
                )
                tcpz = tcgen05.make_tmem_copy(copy_atom_ld, tZ0)
                thr_cpz = tcpz.get_slice(row)
                tZR_src = thr_cpz.partition_S(tZ0)
                tZR_reg = cute.make_rmem_tensor(tYR_coord.shape, ACC)

                for it in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(cute.size(tSS_coord)):
                        ii = tSS_coord[e][0]
                        jj = tSS_coord[e][1] + it * 32
                        sv = mState[ii, jj, (hv, n)]
                        tSS_reg[e] = sv
                        if ii == row:
                            sSt0[ii, jj, 0] = BF16(sv)
                    tdst = cute.make_tensor(tSS_dst.iterator + it * 32, tSS_dst.layout)
                    cute.copy(tstS, tSS_reg, tdst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_S_READY, number_of_threads=160)
                cute.arch.barrier(barrier_id=BAR_Z_FREE, number_of_threads=160)

                for cc in cutlass.range(nc):
                    slot = base + cc
                    Lc = cutlass.min(Int32(C), L - C * cc)
                    glv = mGlws[slot, hv]
                    is_last = cc == nc - 1
                    # merge: S = gl*S - Y + P
                    hy = evy_c.wait_and_advance()
                    for it in cutlass.range_constexpr(4):
                        tsrcY = cute.make_tensor(
                            tYR_src.iterator + it * 32, tYR_src.layout
                        )
                        cute.copy(tcpy, tsrcY, tYR_reg)
                        # also load S from tmem: reuse Y loader on tSmA
                        tsrcS = cute.make_tensor(
                            tSmA.iterator + it * 32,
                            cute.composition(
                                tSmA.layout, cute.make_layout((C, 32))
                            ),
                        )
                        # direct: same partition pattern as Y
                        tS_src = thr_cpy.partition_S(
                            cute.make_tensor(
                                tSmA.iterator,
                                cute.composition(
                                    tSmA.layout, cute.make_layout((C, 32))
                                ),
                            )
                        )
                        tS_src_it = cute.make_tensor(
                            tS_src.iterator + it * 32, tS_src.layout
                        )
                        tS_reg = cute.make_rmem_tensor(tYR_coord.shape, ACC)
                        cute.copy(tcpy, tS_src_it, tS_reg)
                        cute.arch.fence_view_async_tmem_load()
                        for e in cutlass.range_constexpr(cute.size(tYR_reg)):
                            ii = tYR_coord[e][0]
                            jj = tYR_coord[e][1] + it * 32
                            pv = mPws[ii, jj, (hv, slot)]
                            snew = glv * tS_reg[e] - tYR_reg[e] + pv
                            tSS_reg[e] = snew
                            if ii == row:
                                if cc % 2 == 0:
                                    sSt1[ii, jj, 0] = BF16(snew)
                                else:
                                    sSt0[ii, jj, 0] = BF16(snew)
                            if is_last:
                                mNState[ii, jj, (hv, n)] = snew
                        tdst = cute.make_tensor(
                            tSS_dst.iterator + it * 32, tSS_dst.layout
                        )
                        cute.copy(tstS, tSS_reg, tdst)
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_proxy("async.shared", space="cta")
                    hy.release()
                    cute.arch.barrier(barrier_id=BAR_S_READY, number_of_threads=160)
                    # O epilogue: O += scale*Z
                    hz = evz_c.wait_and_advance()
                    orow = cu0 + C * cc + row
                    for it in cutlass.range_constexpr(4):
                        tsrcZ = cute.make_tensor(
                            tZR_src.iterator + it * 32, tZR_src.layout
                        )
                        cute.copy(tcpz, tsrcZ, tZR_reg)
                        cute.arch.fence_view_async_tmem_load()
                        if row < Lc:
                            for e in cutlass.range_constexpr(cute.size(tZR_reg)):
                                ii = tYR_coord[e][0]
                                jj = tYR_coord[e][1] + it * 32
                                if ii == row:
                                    prev = Float32(mO[orow, jj, hv])
                                    mO[orow, jj, hv] = BF16(
                                        prev + scale * tZR_reg[e]
                                    )
                    hz.release()
                    cute.arch.barrier(barrier_id=BAR_Z_FREE, number_of_threads=160)
        else:
            # empty sequence: zero the output state
            if warp_idx < 4:
                for jj in cutlass.range_constexpr(C):
                    mNState[row, jj, (hv, bidh * 0 + n)] = Float32(0.0)

        cute.arch.sync_threads()
        if warp_idx == 4:
            cute.arch.dealloc_tmem(tmem_ptr, cutlass.Int32(512))


@cute.struct
class SharedStorageK1:
    mbar_kq: cute.struct.MemRange[cutlass.Int64, 4]
    mbar_v: cute.struct.MemRange[cutlass.Int64, 2]
    mbar_ev: cute.struct.MemRange[cutlass.Int64, 8]
    tmem_buf: cutlass.Int32
    cum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 16]
    beta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 16]
    ivt: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4 * 32 * 33], 16]
    ivtT: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4 * 32 * 33], 16]


@cute.struct
class SharedStorageK2:
    mbar_g: cute.struct.MemRange[cutlass.Int64, 4]
    mbar_qt: cute.struct.MemRange[cutlass.Int64, 4]
    mbar_evy: cute.struct.MemRange[cutlass.Int64, 4]
    mbar_evz: cute.struct.MemRange[cutlass.Int64, 4]
    tmem_buf: cutlass.Int32


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------
_COMPILED = {}


def _get_compiled(args_key, maker):
    if args_key not in _COMPILED:
        _COMPILED[args_key] = maker()
    return _COMPILED[args_key]


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T = q.shape[0]
    N = cu_seqlens.shape[0] - 1
    device = q.device
    slots = (T + C - 1) // C + N

    o = torch.empty_like(v)
    new_state = torch.empty((N, HV, D, D), dtype=torch.float32, device=device)
    gws = torch.empty((slots, HV, D, D), dtype=torch.bfloat16, device=device)
    pws = torch.empty((slots, HV, D, D), dtype=torch.float32, device=device)
    qtws = torch.empty((slots, HV, D, D), dtype=torch.bfloat16, device=device)
    glws = torch.empty((slots, HV), dtype=torch.float32, device=device)

    if cu_seqlens.dtype != torch.int64:
        cu_seqlens = cu_seqlens.to(torch.int64)
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def ptr(t, dtype):
        return from_dlpack(t, assumed_align=16).iterator

    q_p = ptr(q, BF16)
    k_p = ptr(k, BF16)
    v_p = ptr(v, BF16)
    o_p = ptr(o, BF16)
    st_p = ptr(state, F32)
    ns_p = ptr(new_state, F32)
    al_p = ptr(A_log, F32)
    a_p = ptr(a, BF16)
    dtb_p = ptr(dt_bias, F32)
    b_p = ptr(b, BF16)
    cu_t = from_dlpack(cu_seqlens, assumed_align=8)
    g_p = ptr(gws, BF16)
    p_p = ptr(pws, F32)
    qt_p = ptr(qtws, BF16)
    gl_p = ptr(glws, F32)

    def maker():
        kern = GdnKernels()
        return cute.compile(
            kern, q_p, k_p, v_p, o_p, st_p, ns_p, al_p, a_p, dtb_p, b_p,
            cu_t, g_p, p_p, qt_p, gl_p,
            Int32(T), Int32(N), Int32(slots), Float32(scale), stream,
        )

    compiled = _get_compiled("gdn", maker)
    compiled(
        q_p, k_p, v_p, o_p, st_p, ns_p, al_p, a_p, dtb_p, b_p,
        cu_t, g_p, p_p, qt_p, gl_p,
        Int32(T), Int32(N), Int32(slots), Float32(scale), stream,
    )
    return o, new_state
