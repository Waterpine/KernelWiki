"""Latency-optimized small-T Gated Delta Net prefill path for SM100.

The flashinfer baseline (and the big persistent tcgen05 kernel) carry ~11us of
fixed per-call machinery (tensormap updates in GMEM, TMEM alloc, a dozen
mbarrier pipelines).  For the launch-floor workloads (total_seq_len up to a few
hundred tokens) that overhead IS the latency.  This path is a single lean
kernel launch:

  - one CTA per (sequence, value-head, value-slice) recurrent stream
  - power-of-two token/value tiles: 16x16 (2 warps), 32x32 (4 warps), or
    64x64 (8 warps), all using mma.sync m16n8k16 bf16 tensor cores
  - no TMA / TMEM / tensormaps / persistent scheduler; cp.async + ldmatrix
  - fp32 state lives in mma.sync accumulator registers across chunks;
    a bf16 [V,K] copy is kept in SMEM as the B operand of the Q/K-vs-state
    GEMMs (matching the operand precision of the tcgen05 baseline)
  - latency classes issue the initial fp32 state with 128-bit cp.async and
    hide it under first-chunk Q/K/V issue plus gate preparation

Per C-token chunk (C in {16, 32, 64}), with
D[i,j] = exp(G_i - G_j), G = cumsum(g):
  KK   = K K^T                      (mma)
  -M   = -(beta_i D_ij KK_ij)       strictly-lower, staged bf16 in SMEM
  W    = scale * D_ij QK_ij         inclusive-lower, staged bf16 in SMEM
  rhs  = beta (V - exp(G_i) K S^T)  (mma + elementwise, kept in C-fragments)
  u    = (I + M)^{-1} rhs           4 rounds of [mma correction + 16-row
                                    scalar forward substitution in SMEM]
  O    = scale exp(G_i) Q S^T + W u (mma, predicated store)
  S    = exp(G_last) S + U'^T K     (mma into the persistent accumulators),
         U'_i = exp(G_last - G_i) u_i
"""

from __future__ import annotations

import functools

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.base_dsl.address_space import AddressSpace
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

_GMEM = AddressSpace.gmem

# Static problem configuration (asserted by the dispatcher).
HQ = 4
HV = 8
DK = 128
DV = 128
PAD = 8  # bf16 elements of row padding => 16B, keeps ldmatrix conflict-free
Q_ROW = HQ * DK  # token stride of q/k in elements
V_ROW = HV * DV


class GDNSmallKernel:
    def __init__(
        self,
        io_dtype=cutlass.BFloat16,
        *,
        chunk: int = 64,
        value_split: int = 2,
        subst_rows: int = 16,
        fixed_one_chunk: bool = False,
        parallel_gates: bool = True,
        fast_softplus: bool = False,
        poly_softplus: bool = False,
        packed_output: bool = True,
        local_head_params: bool = True,
        fast_sigmoid: bool = True,
        async_state: bool = False,
        softplus_degree: int = 6,
        clear_padded_q: bool = False,
        predicated_loads: bool = False,
        tail_subst: bool = False,
    ):
        if chunk not in (16, 32, 64):
            raise ValueError(f"chunk must be 16, 32, or 64, got {chunk}")
        if value_split not in (2, 4, 8) or DV % value_split:
            raise ValueError(f"unsupported value_split={value_split}")
        if subst_rows not in (6, 8, 16):
            raise ValueError(f"subst_rows must be 6, 8, or 16, got {subst_rows}")
        if subst_rows < 16 and chunk != 16:
            raise ValueError("subst_rows<16 requires chunk=16")
        self.io_dtype = io_dtype
        self.acc_dtype = Float32
        self.chunk = chunk
        self.block = 16
        self.num_blocks = chunk // self.block
        self.value_split = value_split
        self.subst_rows = subst_rows
        self.fixed_one_chunk = fixed_one_chunk
        self.parallel_gates = parallel_gates
        self.fast_softplus = fast_softplus
        self.poly_softplus = poly_softplus
        self.packed_output = packed_output
        self.local_head_params = local_head_params
        self.fast_sigmoid = fast_sigmoid
        self.async_state = async_state
        self.clear_padded_q = clear_padded_q
        self.predicated_loads = predicated_loads
        self.tail_subst = tail_subst
        if softplus_degree not in (4, 6):
            raise ValueError(
                f"softplus_degree must be 4 or 6, got {softplus_degree}"
            )
        self.softplus_degree = softplus_degree
        self.value_tile = DV // value_split
        self.m_warps = max(chunk, self.value_tile) // 16
        self.threads = self.m_warps * 2 * 32
        self.scan_size = max(32, chunk)
        self.scan_stripes = self.scan_size // 32

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (T, HQ, DK) bf16
        mK: cute.Tensor,  # (T, HQ, DK) bf16
        mV: cute.Tensor,  # (T, HV, DV) bf16
        mA: cute.Tensor,  # (T, HV) bf16
        mB: cute.Tensor,  # (T, HV) bf16
        mAlog: cute.Tensor,  # (HV,) f32
        mDtBias: cute.Tensor,  # (HV,) f32
        mO: cute.Tensor,  # (T, HV, DV) bf16
        mSin: cute.Tensor,  # (N, HV, DV, DK) f32
        mSout: cute.Tensor,  # (N, HV, DV, DK) f32
        cu_seqlens: cute.Tensor,  # (N+1,) i64
        scale: Float32,
        stream: cuda.CUstream,
    ):
        dt = self.io_dtype
        chunk = self.chunk
        value_tile = self.value_tile
        threads = self.threads

        sK_layout = cute.make_layout((chunk, DK), stride=(DK + PAD, 1))
        sV_layout = cute.make_layout(
            (chunk, value_tile), stride=(value_tile + PAD, 1)
        )
        sS_layout = cute.make_layout((value_tile, DK), stride=(DK + PAD, 1))
        sSf_layout = cute.make_layout((value_tile, DK), stride=(DK, 1))
        sM_layout = cute.make_layout((chunk, chunk), stride=(chunk + PAD, 1))

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sK_layout)], 128
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sK_layout)], 128
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sV_layout)], 128
            ]
            sS: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sS_layout)], 128
            ]
            sSf: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(sSf_layout)], 128
            ]
            sM: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sM_layout)], 128
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sM_layout)], 128
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sV_layout)], 128
            ]
            sUp: cute.struct.Align[
                cute.struct.MemRange[dt, cute.cosize(sV_layout)], 128
            ]
            sGr: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scan_size], 128
            ]
            sG: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scan_size], 128
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scan_size], 128
            ]
            sLam: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scan_size], 128
            ]
            sLco: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scan_size], 128
            ]

        # One tiled MMA spans both the token and value tiles.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)),
            (self.m_warps, 2, 1),
            permutation_mnk=(self.m_warps * 16, 16, 16),
        )
        # State update is [value_tile, 128].  When value_tile is narrower
        # than the token tile, redistribute the CTA's warps over N instead of
        # leaving an out-of-range M partition.
        state_m_warps = self.value_tile // 16
        state_n_warps = (self.threads // 32) // state_m_warps
        tiled_state_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)),
            (state_m_warps, state_n_warps, 1),
            permutation_mnk=(self.value_tile, state_n_warps * 8, 16),
        )

        # gmem -> smem tiled copy: 16B per thread, (16 rows x 16 col-chunks).
        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dt,
            num_bits_per_copy=128,
        )
        g2s_thr_layout = cute.make_layout(
            (threads // 16, 16), stride=(16, 1)
        )
        g2s_val_layout = cute.make_layout((1, 8))
        tiled_g2s = cute.make_tiled_copy_tv(atom_g2s, g2s_thr_layout, g2s_val_layout)
        g2sv_thr_layout = cute.make_layout(
            (threads // (value_tile // 8), value_tile // 8),
            stride=(value_tile // 8, 1),
        )
        tiled_g2s_v = cute.make_tiled_copy_tv(
            atom_g2s, g2sv_thr_layout, g2s_val_layout
        )
        atom_g2s_f32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=128,
        )
        tiled_g2s_state = cute.make_tiled_copy_tv(
            atom_g2s_f32,
            cute.make_layout((threads // 32, 32), stride=(32, 1)),
            cute.make_layout((1, 4)),
        )
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dt, num_bits_per_copy=128
        )
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g, g2sv_thr_layout, g2s_val_layout
        )

        num_seqs = cu_seqlens.shape[0] - 1
        grid = (HV * self.value_split * num_seqs, 1, 1)

        self.kernel(
            mQ,
            mK,
            mV,
            mA,
            mB,
            mAlog,
            mDtBias,
            mO,
            mSin,
            mSout,
            cu_seqlens,
            scale,
            tiled_mma,
            tiled_state_mma,
            tiled_g2s,
            tiled_g2s_v,
            tiled_g2s_state,
            tiled_s2g,
            sK_layout,
            sV_layout,
            sS_layout,
            sSf_layout,
            sM_layout,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mAlog: cute.Tensor,
        mDtBias: cute.Tensor,
        mO: cute.Tensor,
        mSin: cute.Tensor,
        mSout: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: Float32,
        tiled_mma: cute.TiledMma,
        tiled_state_mma: cute.TiledMma,
        tiled_g2s: cute.TiledCopy,
        tiled_g2s_v: cute.TiledCopy,
        tiled_g2s_state: cute.TiledCopy,
        tiled_s2g: cute.TiledCopy,
        sK_layout: cute.Layout,
        sV_layout: cute.Layout,
        sS_layout: cute.Layout,
        sSf_layout: cute.Layout,
        sM_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        dt = self.io_dtype
        chunk = self.chunk
        block = self.block
        num_blocks = self.num_blocks
        value_split = self.value_split
        value_tile = self.value_tile
        scan_size = self.scan_size
        scan_stripes = self.scan_stripes
        subst_rows = self.subst_rows
        active_rows = subst_rows if subst_rows < block else chunk
        scan_steps = 3 if subst_rows < block else (4 if chunk == block else 5)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp_id = tidx // 32

        seq_idx = bidx // (HV * value_split)
        rem = bidx % (HV * value_split)
        head = rem // value_split
        vslice = rem % value_split
        head_q = head // (HV // HQ)

        bos = Int32(cu_seqlens[seq_idx])
        eos = Int32(cu_seqlens[seq_idx + 1])
        seqlen = eos - bos

        # --------------------------------------------------------------
        # Shared memory
        # --------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sK.get_tensor(sK_layout)
        sQ = storage.sQ.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sS = storage.sS.get_tensor(sS_layout)
        sSf = storage.sSf.get_tensor(sSf_layout)
        sM = storage.sM.get_tensor(sM_layout)
        sW = storage.sW.get_tensor(sM_layout)
        sU = storage.sU.get_tensor(sV_layout)
        sUp = storage.sUp.get_tensor(sV_layout)
        sGr = storage.sGr.get_tensor(cute.make_layout(scan_size))
        sG = storage.sG.get_tensor(cute.make_layout(scan_size))
        sBeta = storage.sBeta.get_tensor(cute.make_layout(scan_size))
        sLam = storage.sLam.get_tensor(cute.make_layout(scan_size))
        sLco = storage.sLco.get_tensor(cute.make_layout(scan_size))

        # Transposed views for ldmatrix.trans operands.
        sUt = cute.make_tensor(
            sU.iterator,
            cute.make_layout((value_tile, chunk), stride=(1, value_tile + PAD)),
        )
        sUpt = cute.make_tensor(
            sUp.iterator,
            cute.make_layout((value_tile, chunk), stride=(1, value_tile + PAD)),
        )
        sKt = cute.make_tensor(
            sK.iterator, cute.make_layout((DK, chunk), stride=(1, DK + PAD))
        )

        # --------------------------------------------------------------
        # MMA partitions
        # --------------------------------------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        thr_state_mma = tiled_state_mma.get_slice(tidx)

        ldsm = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dt
        )
        ldsm_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dt
        )
        copy_A = cute.make_tiled_copy_A(ldsm, tiled_mma)
        copy_B = cute.make_tiled_copy_B(ldsm, tiled_mma)
        copy_A_t = cute.make_tiled_copy_A(ldsm_t, tiled_mma)
        copy_B_t = cute.make_tiled_copy_B(ldsm_t, tiled_mma)
        copy_A_state_t = cute.make_tiled_copy_A(ldsm_t, tiled_state_mma)
        copy_B_state_t = cute.make_tiled_copy_B(ldsm_t, tiled_state_mma)
        thr_copy_A = copy_A.get_slice(tidx)
        thr_copy_B = copy_B.get_slice(tidx)
        thr_copy_A_t = copy_A_t.get_slice(tidx)
        thr_copy_B_t = copy_B_t.get_slice(tidx)
        thr_copy_A_state_t = copy_A_state_t.get_slice(tidx)
        thr_copy_B_state_t = copy_B_state_t.get_slice(tidx)

        # A operands (row-major smem, non-transposed ldmatrix)
        tAK = thr_mma.make_fragment_A(thr_mma.partition_A(sK))
        tAQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tAW = thr_mma.make_fragment_A(thr_mma.partition_A(sW))
        tAM = thr_mma.make_fragment_A(thr_mma.partition_A(sM))
        tAUp = thr_state_mma.make_fragment_A(thr_state_mma.partition_A(sUpt))
        tAK_cv = thr_copy_A.retile(tAK)
        tAQ_cv = thr_copy_A.retile(tAQ)
        tAW_cv = thr_copy_A.retile(tAW)
        tAM_cv = thr_copy_A.retile(tAM)
        tAUp_cv = thr_copy_A_state_t.retile(tAUp)
        tAsK = thr_copy_A.partition_S(sK)
        tAsQ = thr_copy_A.partition_S(sQ)
        tAsW = thr_copy_A.partition_S(sW)
        tAsM = thr_copy_A.partition_S(sM)
        tAsUp = thr_copy_A_state_t.partition_S(sUpt)

        # B operands
        tBK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tBS = thr_mma.make_fragment_B(thr_mma.partition_B(sS))
        tBU = thr_mma.make_fragment_B(thr_mma.partition_B(sUt))
        tBKt = thr_state_mma.make_fragment_B(thr_state_mma.partition_B(sKt))
        tBK_cv = thr_copy_B.retile(tBK)
        tBS_cv = thr_copy_B.retile(tBS)
        tBU_cv = thr_copy_B_t.retile(tBU)
        tBKt_cv = thr_copy_B_state_t.retile(tBKt)
        tBsK = thr_copy_B.partition_S(sK)
        tBsS = thr_copy_B.partition_S(sS)
        tBsU = thr_copy_B_t.partition_S(sUt)
        tBsKt = thr_copy_B_state_t.partition_S(sKt)

        # Accumulators
        acc_scores = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((chunk, chunk)), Float32
        )
        acc_rhs = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((chunk, value_tile)), Float32
        )
        acc_o = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((chunk, value_tile)), Float32
        )
        acc_state = cute.make_rmem_tensor(
            thr_state_mma.partition_shape_C((value_tile, DK)), Float32
        )

        # Coordinate tensors (row, col) matching each accumulator layout.
        cCC = thr_mma.partition_C(cute.make_identity_tensor((chunk, chunk)))
        cCD = thr_mma.partition_C(cute.make_identity_tensor((chunk, value_tile)))
        cSS = thr_state_mma.partition_C(
            cute.make_identity_tensor((value_tile, DK))
        )

        acc_scores_mn = _mn_view(acc_scores)
        acc_rhs_mn = _mn_view(acc_rhs)
        acc_o_mn = _mn_view(acc_o)
        acc_state_mn = _mn_view(acc_state)
        cCC_mn = _mn_view(cCC)
        cCD_mn = _mn_view(cCD)
        cSS_mn = _mn_view(cSS)

        # --------------------------------------------------------------
        # Initial state: gmem f32 -> accumulator registers + bf16 SMEM copy
        # --------------------------------------------------------------
        s_off = vslice * value_tile * DK  # elements into the (DV, DK) plane
        gSin_full = mSin[seq_idx, head, None, None]  # (DV, DK) f32
        gSout_full = mSout[seq_idx, head, None, None]
        gSin = cute.make_tensor(
            cute.make_ptr(
                Float32,
                gSin_full.iterator.toint() + s_off * 4,
                _GMEM,
                assumed_align=16,
            ),
            cute.make_layout((value_tile, DK), stride=(DK, 1)),
        )
        gSout = cute.make_tensor(
            cute.make_ptr(
                Float32,
                gSout_full.iterator.toint() + s_off * 4,
                _GMEM,
                assumed_align=16,
            ),
            cute.make_layout((value_tile, DK), stride=(DK, 1)),
        )
        atom_c_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=64
        )
        copy_C_f32 = cute.make_tiled_copy_C(atom_c_f32, tiled_state_mma)
        thr_copy_C = copy_C_f32.get_slice(tidx)
        tCgSin = thr_copy_C.partition_S(gSin)
        tCgSout = thr_copy_C.partition_D(gSout)
        tCrState = thr_copy_C.retile(acc_state)
        tCsSf = thr_copy_C.partition_S(sSf)
        if cutlass.const_expr(self.async_state):
            # Overlap the initial FP32 state fetch with the first chunk's
            # Q/K/V issue and gate preparation. The shared tile uses the same
            # row-major layout as the MMA accumulator copy partition.
            thr_g2s_state = tiled_g2s_state.get_slice(tidx)
            tSgSin = thr_g2s_state.partition_S(gSin)
            tSsSf = thr_g2s_state.partition_D(sSf)
            cute.copy(tiled_g2s_state, tSgSin, tSsSf)
            cute.arch.cp_async_commit_group()
        else:
            cute.copy(copy_C_f32, tCgSin, tCrState)
            for i in cutlass.range_constexpr(cute.size(acc_state_mn.shape[0])):
                for j in cutlass.range_constexpr(cute.size(acc_state_mn.shape[1])):
                    sS[cSS_mn[i, j]] = dt(acc_state_mn[i, j])
            # Make sS visible before the first chunk's state GEMMs.
            cute.arch.sync_threads()

        thr_g2s = tiled_g2s.get_slice(tidx)
        thr_g2s_v = tiled_g2s_v.get_slice(tidx)
        cLoad = cute.make_identity_tensor((chunk, DK))
        tGcG = thr_g2s.partition_S(cLoad)
        cLoadV = cute.make_identity_tensor((chunk, value_tile))
        tGcGV = thr_g2s_v.partition_S(cLoadV)
        tGsK = thr_g2s.partition_D(sK)
        tGsQ = thr_g2s.partition_D(sQ)
        tGsV = thr_g2s_v.partition_D(sV)

        # Raw base addresses (elements) of the head slices; per-chunk tiles are
        # rebuilt with cute.make_ptr to keep the gmem address space + alignment.
        k_addr0 = mK.iterator.toint() + head_q * (DK * 2)
        q_addr0 = mQ.iterator.toint() + head_q * (DK * 2)
        v_addr0 = mV.iterator.toint() + (head * DV + vslice * value_tile) * 2
        o_addr0 = mO.iterator.toint() + (head * DV + vslice * value_tile) * 2
        thr_s2g = tiled_s2g.get_slice(tidx)
        tOsV = thr_s2g.partition_S(sV)
        # Both lean tile sizes can turn the scattered MMA C fragment into a
        # coalesced 16-byte output transaction through one stmatrix staging.
        if cutlass.const_expr(self.packed_output and chunk <= 32):
            atom_o_r2s = cute.make_copy_atom(
                warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False), dt
            )
            tiled_o_r2s = cute.make_tiled_copy_C(atom_o_r2s, tiled_mma)
            thr_o_r2s = tiled_o_r2s.get_slice(tidx)
            tCsV_out = thr_o_r2s.partition_D(sV)
            tCrO = tiled_o_r2s.retile(acc_o)
            tCrO_out = cute.make_rmem_tensor_like(tCrO, dt)
        kq_tile_layout = cute.make_layout((chunk, DK), stride=(Q_ROW, 1))
        v_tile_layout = cute.make_layout((chunk, value_tile), stride=(V_ROW, 1))
        if cutlass.const_expr(not self.local_head_params):
            global_dt_bias = Float32(mDtBias[head])
            global_neg_exp_alog = Float32(0.0) - cute.math.exp(
                Float32(mAlog[head]), fastmath=True
            )
        if cutlass.const_expr(self.fixed_one_chunk):
            num_chunks = 1
        else:
            num_chunks = (seqlen + (chunk - 1)) // chunk
        for ci in cutlass.range(num_chunks, at_least_once=True):
            chunk_base = ci * chunk
            len_c = cutlass.min(seqlen - chunk_base, Int32(chunk))
            num_blk = (len_c + (block - 1)) // block

            # ----------------------------------------------------------
            # Loads: K, Q, V chunk tiles via cp.async with row predication
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("load_issue")
            row0 = bos + chunk_base
            gK_c = cute.make_tensor(
                cute.make_ptr(
                    dt,
                    k_addr0 + row0 * (Q_ROW * 2),
                    _GMEM,
                    assumed_align=16,
                ),
                kq_tile_layout,
            )
            gQ_c = cute.make_tensor(
                cute.make_ptr(
                    dt,
                    q_addr0 + row0 * (Q_ROW * 2),
                    _GMEM,
                    assumed_align=16,
                ),
                kq_tile_layout,
            )
            gV_c = cute.make_tensor(
                cute.make_ptr(
                    dt,
                    v_addr0 + row0 * (V_ROW * 2),
                    _GMEM,
                    assumed_align=16,
                ),
                v_tile_layout,
            )
            tGgK = thr_g2s.partition_S(gK_c)
            tGgQ = thr_g2s.partition_S(gQ_c)
            tGgV = thr_g2s_v.partition_S(gV_c)
            if cutlass.const_expr(self.packed_output and chunk <= 32):
                gO_c = cute.make_tensor(
                    cute.make_ptr(
                        dt,
                        o_addr0 + row0 * (V_ROW * 2),
                        _GMEM,
                        assumed_align=16,
                    ),
                    v_tile_layout,
                )
                tOgO = thr_s2g.partition_D(gO_c)

            for m in cutlass.range_constexpr(cute.size(tGsK.shape[1])):
                tok = chunk_base + tGcG[(0, m, 0)][0]
                if cutlass.const_expr(self.predicated_loads):
                    valid = cute.make_rmem_tensor((1,), cutlass.Boolean)
                    valid[0] = tok < seqlen
                    cute.copy(
                        tiled_g2s,
                        tGgK[(None, m, None)],
                        tGsK[(None, m, None)],
                        pred=valid,
                    )
                    cute.copy(
                        tiled_g2s,
                        tGgQ[(None, m, None)],
                        tGsQ[(None, m, None)],
                        pred=valid,
                    )
                else:
                    if tok < seqlen:
                        cute.copy(
                            tiled_g2s,
                            tGgK[(None, m, None)],
                            tGsK[(None, m, None)],
                        )
                        cute.copy(
                            tiled_g2s,
                            tGgQ[(None, m, None)],
                            tGsQ[(None, m, None)],
                        )
                    else:
                        tGsK[(None, m, None)].fill(0)
                        # A padded Q row contributes only to a padded output row:
                        # score/output epilogues mask that row and no reduction
                        # crosses the MMA M dimension.  Its SMEM contents are
                        # therefore dead, unlike padded K/V rows that participate
                        # in contraction dimensions and must remain finite.
                        if cutlass.const_expr(self.clear_padded_q):
                            tGsQ[(None, m, None)].fill(0)
            for m in cutlass.range_constexpr(cute.size(tGsV.shape[1])):
                tok = chunk_base + tGcGV[(0, m, 0)][0]
                if cutlass.const_expr(self.predicated_loads):
                    valid = cute.make_rmem_tensor((1,), cutlass.Boolean)
                    valid[0] = tok < seqlen
                    cute.copy(
                        tiled_g2s_v,
                        tGgV[(None, m, None)],
                        tGsV[(None, m, None)],
                        pred=valid,
                    )
                else:
                    if tok < seqlen:
                        cute.copy(
                            tiled_g2s_v,
                            tGgV[(None, m, None)],
                            tGsV[(None, m, None)],
                        )
                    else:
                        tGsV[(None, m, None)].fill(0)
            cute.arch.cp_async_commit_group()
            cute.experimental.iket.range_pop()

            # ----------------------------------------------------------
            # Gates: separate warp groups compute decay and beta concurrently;
            # warp 0 then performs the prefix scan and exponential tables.
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("gates")
            if cutlass.const_expr(self.parallel_gates):
                if tidx < scan_size:
                    if tidx < active_rows:
                        tok = chunk_base + tidx
                        tok_safe = cutlass.min(tok, seqlen - 1)
                        row = bos + tok_safe
                        if cutlass.const_expr(self.local_head_params):
                            dt_bias = Float32(mDtBias[head])
                            neg_exp_alog = Float32(0.0) - cute.math.exp(
                                Float32(mAlog[head]), fastmath=True
                            )
                        else:
                            dt_bias = global_dt_bias
                            neg_exp_alog = global_neg_exp_alog
                        a_val = Float32(mA[row, head])
                        x = a_val + dt_bias
                        if cutlass.const_expr(self.poly_softplus):
                            ax = cutlass.max(x, Float32(0.0) - x)
                            tp = cutlass.min(ax, Float32(8.0)) * Float32(
                                0.25
                            ) - Float32(1.0)
                            if cutlass.const_expr(self.softplus_degree == 4):
                                corr = Float32(0.185445337)
                                corr = corr * tp + Float32(-0.293484563)
                                corr = corr * tp + Float32(0.143401101)
                                corr = corr * tp + Float32(-0.0486338200)
                                corr = corr * tp + Float32(0.0178948557)
                            else:
                                corr = Float32(-0.00852176081)
                                corr = corr * tp + Float32(-0.0664322004)
                                corr = corr * tp + Float32(0.200398788)
                                corr = corr * tp + Float32(-0.213032976)
                                corr = corr * tp + Float32(0.137233630)
                                corr = corr * tp + Float32(-0.0677231029)
                                corr = corr * tp + Float32(0.0181499273)
                            sp = cutlass.max(x, Float32(0.0)) + corr
                        else:
                            xc = cutlass.min(x, Float32(20.0))
                            exp_x = cute.math.exp(xc, fastmath=True)
                            if cutlass.const_expr(self.fast_softplus):
                                sp = cute.math.log(
                                    Float32(1.0) + exp_x, fastmath=True
                                )
                            else:
                                sp = cute.math.log1p(exp_x, fastmath=True)
                            sp = cutlass.max(sp, x)
                        sGr[tidx] = neg_exp_alog * sp
                    else:
                        sGr[tidx] = 0.0
                # Place beta immediately after the full warp-rounded decay
                # stripe, so the two transcendental chains occupy disjoint
                # warps for every supported token tile.
                beta_tidx = tidx - scan_size
                if 0 <= beta_tidx and beta_tidx < scan_size:
                    if beta_tidx < active_rows:
                        tok = chunk_base + beta_tidx
                        tok_safe = cutlass.min(tok, seqlen - 1)
                        row = bos + tok_safe
                        b_val = Float32(mB[row, head])
                        if cutlass.const_expr(self.fast_sigmoid):
                            sBeta[beta_tidx] = Float32(0.5) + Float32(
                                0.5
                            ) * cute.math.tanh(
                                Float32(0.5) * b_val, approx=True
                            )
                        else:
                            sBeta[beta_tidx] = Float32(1.0) / (
                                Float32(1.0)
                                + cute.math.exp(-b_val, fastmath=True)
                            )
                    else:
                        sBeta[beta_tidx] = 0.0
            else:
                if tidx < scan_size:
                    if tidx < active_rows:
                        tok = chunk_base + tidx
                        tok_safe = cutlass.min(tok, seqlen - 1)
                        row = bos + tok_safe
                        if cutlass.const_expr(self.local_head_params):
                            dt_bias = Float32(mDtBias[head])
                            neg_exp_alog = Float32(0.0) - cute.math.exp(
                                Float32(mAlog[head]), fastmath=True
                            )
                        else:
                            dt_bias = global_dt_bias
                            neg_exp_alog = global_neg_exp_alog
                        a_val = Float32(mA[row, head])
                        b_val = Float32(mB[row, head])
                        x = a_val + dt_bias
                        if cutlass.const_expr(self.poly_softplus):
                            ax = cutlass.max(x, Float32(0.0) - x)
                            tp = cutlass.min(ax, Float32(8.0)) * Float32(
                                0.25
                            ) - Float32(1.0)
                            if cutlass.const_expr(self.softplus_degree == 4):
                                corr = Float32(0.185445337)
                                corr = corr * tp + Float32(-0.293484563)
                                corr = corr * tp + Float32(0.143401101)
                                corr = corr * tp + Float32(-0.0486338200)
                                corr = corr * tp + Float32(0.0178948557)
                            else:
                                corr = Float32(-0.00852176081)
                                corr = corr * tp + Float32(-0.0664322004)
                                corr = corr * tp + Float32(0.200398788)
                                corr = corr * tp + Float32(-0.213032976)
                                corr = corr * tp + Float32(0.137233630)
                                corr = corr * tp + Float32(-0.0677231029)
                                corr = corr * tp + Float32(0.0181499273)
                            sp = cutlass.max(x, Float32(0.0)) + corr
                        else:
                            xc = cutlass.min(x, Float32(20.0))
                            exp_x = cute.math.exp(xc, fastmath=True)
                            if cutlass.const_expr(self.fast_softplus):
                                sp = cute.math.log(
                                    Float32(1.0) + exp_x, fastmath=True
                                )
                            else:
                                sp = cute.math.log1p(exp_x, fastmath=True)
                            sp = cutlass.max(sp, x)
                        sGr[tidx] = neg_exp_alog * sp
                        if cutlass.const_expr(self.fast_sigmoid):
                            sBeta[tidx] = Float32(0.5) + Float32(
                                0.5
                            ) * cute.math.tanh(
                                Float32(0.5) * b_val, approx=True
                            )
                        else:
                            sBeta[tidx] = Float32(1.0) / (
                                Float32(1.0) + cute.math.exp(-b_val, fastmath=True)
                            )
                    else:
                        sGr[tidx] = 0.0
                        sBeta[tidx] = 0.0
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("load_wait")
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.experimental.iket.range_pop()

            if cutlass.const_expr(self.async_state):
                if ci == 0:
                    cute.copy(copy_C_f32, tCsSf, tCrState)
                    for i in cutlass.range_constexpr(
                        cute.size(acc_state_mn.shape[0])
                    ):
                        for j in cutlass.range_constexpr(
                            cute.size(acc_state_mn.shape[1])
                        ):
                            sS[cSS_mn[i, j]] = dt(acc_state_mn[i, j])

            if warp_id == 0:
                cute.experimental.iket.range_push("scan")
                rG = cute.make_rmem_tensor((scan_stripes,), Float32)
                for half in cutlass.range_constexpr(scan_stripes):
                    rG[half] = sGr[half * 32 + lane]
                for d_log in cutlass.range_constexpr(scan_steps):
                    d = 1 << d_log
                    for half in cutlass.range_constexpr(scan_stripes):
                        n = cute.arch.shuffle_sync_up(
                            rG[half], d, mask=0xFFFFFFFF, mask_and_clamp=0
                        )
                        if lane >= d:
                            rG[half] = rG[half] + n
                if cutlass.const_expr(scan_stripes > 1):
                    stripe0_total = cute.arch.shuffle_sync(
                        rG[0], 31, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    rG[1] = rG[1] + stripe0_total
                for half in cutlass.range_constexpr(scan_stripes):
                    t_local = half * 32 + lane
                    g_cum = rG[half]
                    if cutlass.const_expr(chunk == 16):
                        if t_local < chunk:
                            sG[t_local] = g_cum
                            sLam[t_local] = cute.math.exp(g_cum, fastmath=True)
                    else:
                        sG[t_local] = g_cum
                        sLam[t_local] = cute.math.exp(g_cum, fastmath=True)
                cute.arch.sync_warp()
                g_last = sG[len_c - 1]
                for half in cutlass.range_constexpr(scan_stripes):
                    t_local = half * 32 + lane
                    if cutlass.const_expr(chunk == 16):
                        if t_local < chunk:
                            dd = cutlass.min(g_last - sG[t_local], Float32(0.0))
                            sLco[t_local] = cute.math.exp(dd, fastmath=True)
                    else:
                        dd = cutlass.min(g_last - sG[t_local], Float32(0.0))
                        sLco[t_local] = cute.math.exp(dd, fastmath=True)
                cute.experimental.iket.range_pop()

            # ----------------------------------------------------------
            # GEMM 1: KK^T -> -M (strictly lower)  and GEMM 2: QK^T -> W
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("g1_kk")
            acc_scores.fill(0.0)
            cute.copy(copy_A, tAsK[(None, None, 0)], tAK_cv[(None, None, 0)])
            cute.copy(copy_B, tBsK[(None, None, 0)], tBK_cv[(None, None, 0)])
            for kk in cutlass.range_constexpr(cute.size(tAK.shape[2])):
                k_next = (kk + 1) % cute.size(tAK.shape[2])
                cute.copy(
                    copy_A, tAsK[(None, None, k_next)], tAK_cv[(None, None, k_next)]
                )
                cute.copy(
                    copy_B, tBsK[(None, None, k_next)], tBK_cv[(None, None, k_next)]
                )
                cute.gemm(
                    tiled_mma,
                    acc_scores,
                    tAK[(None, None, kk)],
                    tBK[(None, None, kk)],
                    acc_scores,
                )
            cute.arch.sync_threads()  # scan tables (sG/sLam/sLco) now visible
            g_cols = cute.make_rmem_tensor(
                (cute.size(acc_scores_mn.shape[1]),), Float32
            )
            for j in cutlass.range_constexpr(cute.size(acc_scores_mn.shape[1])):
                g_cols[j] = sG[cCC_mn[0, j][1]]
            for i in cutlass.range_constexpr(cute.size(acc_scores_mn.shape[0])):
                row = cCC_mn[i, 0][0]
                g_row = sG[row]
                nbeta_row = Float32(0.0) - sBeta[row]
                for j in cutlass.range_constexpr(cute.size(acc_scores_mn.shape[1])):
                    coord = cCC_mn[i, j]
                    mval = Float32(0.0)
                    if row < active_rows and row > coord[1]:
                        ratio = cute.math.exp(g_row - g_cols[j], fastmath=True)
                        mval = nbeta_row * ratio * acc_scores_mn[i, j]
                    sM[coord] = dt(mval)
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("g2_qk")

            acc_scores.fill(0.0)
            cute.copy(copy_A, tAsQ[(None, None, 0)], tAQ_cv[(None, None, 0)])
            for kk in cutlass.range_constexpr(cute.size(tAQ.shape[2])):
                k_next = (kk + 1) % cute.size(tAQ.shape[2])
                cute.copy(
                    copy_A, tAsQ[(None, None, k_next)], tAQ_cv[(None, None, k_next)]
                )
                cute.gemm(
                    tiled_mma,
                    acc_scores,
                    tAQ[(None, None, kk)],
                    tBK[(None, None, kk)],
                    acc_scores,
                )
            for i in cutlass.range_constexpr(cute.size(acc_scores_mn.shape[0])):
                row = cCC_mn[i, 0][0]
                g_row_s = sG[row]
                for j in cutlass.range_constexpr(cute.size(acc_scores_mn.shape[1])):
                    coord = cCC_mn[i, j]
                    wval = Float32(0.0)
                    if row < active_rows and row >= coord[1]:
                        ratio = cute.math.exp(g_row_s - g_cols[j], fastmath=True)
                        wval = scale * ratio * acc_scores_mn[i, j]
                    sW[coord] = dt(wval)
            cute.experimental.iket.range_pop()

            # ----------------------------------------------------------
            # GEMM 3: rhs = K @ S^T ; GEMM 4: acc_o = Q @ S^T
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("g34_state")
            acc_rhs.fill(0.0)
            acc_o.fill(0.0)
            cute.copy(copy_B, tBsS[(None, None, 0)], tBS_cv[(None, None, 0)])
            for kk in cutlass.range_constexpr(cute.size(tBS.shape[2])):
                k_next = (kk + 1) % cute.size(tBS.shape[2])
                cute.copy(
                    copy_B, tBsS[(None, None, k_next)], tBS_cv[(None, None, k_next)]
                )
                cute.gemm(
                    tiled_mma,
                    acc_rhs,
                    tAK[(None, None, kk)],
                    tBS[(None, None, kk)],
                    acc_rhs,
                )
                cute.gemm(
                    tiled_mma,
                    acc_o,
                    tAQ[(None, None, kk)],
                    tBS[(None, None, kk)],
                    acc_o,
                )

            cute.experimental.iket.range_pop()
            # rhs = beta_i * (V - lam_i * KS); the barrier also publishes sM/sW
            cute.experimental.iket.range_push("rhs_epi")
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(cute.size(acc_rhs_mn.shape[0])):
                row = cCD_mn[i, 0][0]
                if row < active_rows:
                    beta_i = sBeta[row]
                    lam_i = sLam[row]
                    for j in cutlass.range_constexpr(
                        cute.size(acc_rhs_mn.shape[1])
                    ):
                        coord = cCD_mn[i, j]
                        v_val = Float32(sV[coord])
                        acc_rhs_mn[i, j] = beta_i * (
                            v_val - lam_i * acc_rhs_mn[i, j]
                        )
            cute.experimental.iket.range_pop()

            # ----------------------------------------------------------
            # Substitution: u = (I + M)^{-1} rhs, 16-row blocks
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("subst")
            for blk in cutlass.range_constexpr(num_blocks):
                if blk < num_blk:
                    if cutlass.const_expr(blk > 0):
                        # rhs += (-M)[:, blk-1 block] @ u[blk-1 block]
                        kb = blk - 1
                        cute.copy(
                            copy_A, tAsM[(None, None, kb)], tAM_cv[(None, None, kb)]
                        )
                        cute.copy(
                            copy_B_t,
                            tBsU[(None, None, kb)],
                            tBU_cv[(None, None, kb)],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_rhs,
                            tAM[(None, None, kb)],
                            tBU[(None, None, kb)],
                            acc_rhs,
                        )
                    # owning warps store this block's corrected rhs (bf16)
                    for i in cutlass.range_constexpr(cute.size(acc_rhs_mn.shape[0])):
                        row = cCD_mn[i, 0][0]
                        if row // block == blk:
                            for j in cutlass.range_constexpr(
                                cute.size(acc_rhs_mn.shape[1])
                            ):
                                coord = cCD_mn[i, j]
                                sU[coord] = dt(acc_rhs_mn[i, j])
                    cute.arch.sync_threads()

                    # Scalar forward substitution; lane c owns value column c.
                    if tidx < value_tile:
                        base = blk * block
                        if cutlass.const_expr(self.tail_subst and chunk == 32):
                            rows_left = len_c - base
                            if rows_left <= 4:
                                self._substitute_rows(
                                    sU, sUp, sM, sLco, tidx, base, 4
                                )
                            else:
                                r_vals = [Float32(0.0)] * 16
                                for ii in cutlass.range_constexpr(16):
                                    r_vals[ii] = Float32(sU[base + ii, tidx])
                                u_vals = [Float32(0.0)] * 16
                                for ii in cutlass.range_constexpr(16):
                                    acc = r_vals[ii]
                                    for jj in cutlass.range_constexpr(ii):
                                        acc = acc + Float32(
                                            sM[base + ii, base + jj]
                                        ) * u_vals[jj]
                                    u_vals[ii] = acc
                                for ii in cutlass.range_constexpr(16):
                                    sU[base + ii, tidx] = dt(u_vals[ii])
                                    sUp[base + ii, tidx] = dt(
                                        u_vals[ii] * sLco[base + ii]
                                    )
                        else:
                            r_vals = [Float32(0.0)] * subst_rows
                            for ii in cutlass.range_constexpr(subst_rows):
                                r_vals[ii] = Float32(sU[base + ii, tidx])
                            u_vals = [Float32(0.0)] * subst_rows
                            for ii in cutlass.range_constexpr(subst_rows):
                                acc = r_vals[ii]
                                for jj in cutlass.range_constexpr(ii):
                                    acc = acc + Float32(
                                        sM[base + ii, base + jj]
                                    ) * u_vals[jj]
                                u_vals[ii] = acc
                            for ii in cutlass.range_constexpr(subst_rows):
                                sU[base + ii, tidx] = dt(u_vals[ii])
                                sUp[base + ii, tidx] = dt(
                                    u_vals[ii] * sLco[base + ii]
                                )
                            if cutlass.const_expr(subst_rows < block):
                                for ii in cutlass.range_constexpr(block - subst_rows):
                                    sUp[base + subst_rows + ii, tidx] = dt(0.0)
                    cute.arch.sync_threads()

            cute.experimental.iket.range_pop()
            # ----------------------------------------------------------
            # Output: acc_o = scale * lam_i * QS + W @ u, then store
            # ----------------------------------------------------------
            cute.experimental.iket.range_push("g6_out")
            for i in cutlass.range_constexpr(cute.size(acc_o_mn.shape[0])):
                row = cCD_mn[i, 0][0]
                if row < active_rows:
                    row_scale = scale * sLam[row]
                    for j in cutlass.range_constexpr(
                        cute.size(acc_o_mn.shape[1])
                    ):
                        acc_o_mn[i, j] = acc_o_mn[i, j] * row_scale

            for kb in cutlass.range_constexpr(num_blocks):
                if kb < num_blk:
                    cute.copy(
                        copy_A, tAsW[(None, None, kb)], tAW_cv[(None, None, kb)]
                    )
                    cute.copy(
                        copy_B_t, tBsU[(None, None, kb)], tBU_cv[(None, None, kb)]
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_o,
                        tAW[(None, None, kb)],
                        tBU[(None, None, kb)],
                        acc_o,
                    )

            if cutlass.const_expr(self.packed_output and chunk <= 32):
                tCrO_out.store(tCrO.load().to(dt))
                cute.copy(tiled_o_r2s, tCrO_out, tCsV_out)
                cute.arch.sync_threads()
                for m in cutlass.range_constexpr(cute.size(tOsV.shape[1])):
                    tok = chunk_base + tGcGV[(0, m, 0)][0]
                    if tok < seqlen:
                        cute.copy(
                            tiled_s2g,
                            tOsV[(None, m, None)],
                            tOgO[(None, m, None)],
                        )
            else:
                for i in cutlass.range_constexpr(cute.size(acc_o_mn.shape[0])):
                    row = cCD_mn[i, 0][0]
                    tok = chunk_base + row
                    if tok < seqlen:
                        for j in cutlass.range_constexpr(
                            cute.size(acc_o_mn.shape[1])
                        ):
                            col = cCD_mn[i, j][1]
                            mO[bos + tok, head, vslice * value_tile + col] = dt(
                                acc_o_mn[i, j]
                            )

            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("g7_update")
            # ----------------------------------------------------------
            # State update: S = lam_chunk * S + U'^T @ K
            # ----------------------------------------------------------
            lam_chunk = sLam[len_c - 1]
            for i in cutlass.range_constexpr(cute.size(acc_state_mn.shape[0])):
                for j in cutlass.range_constexpr(cute.size(acc_state_mn.shape[1])):
                    acc_state_mn[i, j] = acc_state_mn[i, j] * lam_chunk

            for kb in cutlass.range_constexpr(num_blocks):
                if kb < num_blk:
                    cute.copy(
                        copy_A_state_t,
                        tAsUp[(None, None, kb)],
                        tAUp_cv[(None, None, kb)],
                    )
                    cute.copy(
                        copy_B_state_t,
                        tBsKt[(None, None, kb)],
                        tBKt_cv[(None, None, kb)],
                    )
                    cute.gemm(
                        tiled_state_mma,
                        acc_state,
                        tAUp[(None, None, kb)],
                        tBKt[(None, None, kb)],
                        acc_state,
                    )

            cute.experimental.iket.range_pop()
            # refresh bf16 state copy for the next chunk
            if chunk_base + chunk < seqlen:
                cute.arch.sync_threads()  # everyone done with old sS / smem bufs
                for i in cutlass.range_constexpr(cute.size(acc_state_mn.shape[0])):
                    for j in cutlass.range_constexpr(
                        cute.size(acc_state_mn.shape[1])
                    ):
                        coord = cSS_mn[i, j]
                        sS[coord] = dt(acc_state_mn[i, j])
                cute.arch.sync_threads()

        # --------------------------------------------------------------
        # Final state store
        # --------------------------------------------------------------
        cute.copy(copy_C_f32, tCrState, tCgSout)

    @cute.jit
    def _substitute_rows(
        self,
        sU: cute.Tensor,
        sUp: cute.Tensor,
        sM: cute.Tensor,
        sLco: cute.Tensor,
        tidx: Int32,
        base: cutlass.Constexpr,
        rows: cutlass.Constexpr,
    ):
        """Solve one compile-time row prefix and zero the unused state RHS."""
        dt = self.io_dtype
        r_vals = [Float32(0.0)] * rows
        for ii in cutlass.range_constexpr(rows):
            r_vals[ii] = Float32(sU[base + ii, tidx])
        u_vals = [Float32(0.0)] * rows
        for ii in cutlass.range_constexpr(rows):
            acc = r_vals[ii]
            for jj in cutlass.range_constexpr(ii):
                acc = acc + Float32(sM[base + ii, base + jj]) * u_vals[jj]
            u_vals[ii] = acc
        for ii in cutlass.range_constexpr(rows):
            sU[base + ii, tidx] = dt(u_vals[ii])
            sUp[base + ii, tidx] = dt(u_vals[ii] * sLco[base + ii])
        for ii in cutlass.range_constexpr(16 - rows):
            sUp[base + rows + ii, tidx] = dt(0.0)


def _mn_view(acc: cute.Tensor) -> cute.Tensor:
    """((2cols,2rows), M, N) fragment -> ((2rows, M), (2cols, N)) view."""
    ref = cute.make_layout(acc.layout.shape)
    mn = cute.make_layout(
        ((ref.shape[0][1], ref.shape[1]), (ref.shape[0][0], ref.shape[2])),
        stride=((ref.stride[0][1], ref.stride[1]), (ref.stride[0][0], ref.stride[2])),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))


# ----------------------------------------------------------------------
# Torch-facing wrapper with a one-off compile cache
# ----------------------------------------------------------------------

_SMALL_CACHE: dict[tuple, dict] = {}


def run_small(
    q,
    k,
    v,
    state,
    A_log,
    a,
    dt_bias,
    b,
    cu_seqlens,
    scale,
    *,
    chunk=64,
    value_split=2,
    subst_rows=16,
    parallel_gates=None,
    fast_softplus=None,
    poly_softplus=True,
    packed_output=True,
    local_head_params=None,
    fast_sigmoid=None,
    async_state=False,
    softplus_degree=6,
    clear_padded_q=False,
    predicated_loads=False,
    tail_subst=False,
):
    output = torch.empty_like(v)
    output_state = torch.empty_like(state)
    fixed_one_chunk = q.size(0) <= chunk
    num_seqs = cu_seqlens.size(0) - 1
    multi_sequence_chunk32 = chunk == 32 and num_seqs > 1
    if parallel_gates is None:
        parallel_gates = chunk == 16 or multi_sequence_chunk32
    if fast_softplus is None:
        fast_softplus = chunk == 16 or multi_sequence_chunk32
    if local_head_params is None:
        local_head_params = multi_sequence_chunk32
    if fast_sigmoid is None:
        # The 16-token tile has ample gate-warp slack and its original
        # exp/reciprocal chain is marginally faster.  Tanh shortens the
        # critical gate range for the wider chunk-32 tile.
        fast_sigmoid = chunk != 16
    cache = _SMALL_CACHE.setdefault(
        (
            chunk,
            value_split,
            subst_rows,
            fixed_one_chunk,
            parallel_gates,
            fast_softplus,
            poly_softplus,
            packed_output,
            local_head_params,
            fast_sigmoid,
            async_state,
            softplus_degree,
            clear_padded_q,
            predicated_loads,
            tail_subst,
        ),
        {},
    )
    if "compiled" not in cache:
        kern = GDNSmallKernel(
            chunk=chunk,
            value_split=value_split,
            subst_rows=subst_rows,
            fixed_one_chunk=fixed_one_chunk,
            parallel_gates=parallel_gates,
            fast_softplus=fast_softplus,
            poly_softplus=poly_softplus,
            packed_output=packed_output,
            local_head_params=local_head_params,
            fast_sigmoid=fast_sigmoid,
            async_state=async_state,
            softplus_degree=softplus_degree,
            clear_padded_q=clear_padded_q,
            predicated_loads=predicated_loads,
            tail_subst=tail_subst,
        )

        def dyn(t, align=16):
            tc = from_dlpack(t, assumed_align=align)
            tc.mark_compact_shape_dynamic(
                mode=0,
                stride_order=tuple(range(t.dim())),
                divisibility=1,
            )
            return tc

        q_c = dyn(q)
        k_c = dyn(k)
        v_c = dyn(v)
        a_c = dyn(a, align=4)
        b_c = dyn(b, align=4)
        alog_c = from_dlpack(A_log, assumed_align=4)
        dtb_c = from_dlpack(dt_bias, assumed_align=4)
        o_c = dyn(output)
        sin_c = dyn(state)
        sout_c = dyn(output_state)
        cu_c = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        cache["compiled"] = cute.compile(
            kern,
            q_c,
            k_c,
            v_c,
            a_c,
            b_c,
            alog_c,
            dtb_c,
            o_c,
            sin_c,
            sout_c,
            cu_c,
            Float32(scale),
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    cache["compiled"](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        output,
        state,
        output_state,
        cu_seqlens,
        Float32(scale),
        stream,
    )
    return output, output_state
