"""SM100/SM103 persistent grouped blockwise-scale FP8 GEMM kernels for the MoE.

Two kernels built from one parametrized class:

  GEMM1 (mode="g1"): C_perm = swiglu_quant( A_perm @ W13^T )
    A: (P_cap, 7168) fp8, per-(row, k-block) scales SFA_perm (56, P_cap) f32
    B: W13 (32, 4096, 7168) fp8 viewed with paired X1/X2 N-mode so each
       256-wide N tile computes matching gate/up columns.
    Epilogue: silu(x2) * x1, dynamic per-(row, 128-col) fp8 quantization.
    Out: C_perm (P_cap, 2048) fp8 + SFC (16, P_cap) f32.

  GEMM2 (mode="g2"): pairs_out = (C_perm @ W2^T) * pair_w
    A: C_perm, scales SFC. B: W2 (32, 7168, 2048). Out bf16 (P_cap, 7168).

Structure per CTA (384 threads):
  warps 0-7 : promotion + epilogue (per-k-tile TMEM->reg FFMA with SFA*SFB)
              warpgroup 0 owns output subtiles j=0 (acc cols 0:64,128:192),
              warpgroup 1 owns j=1 (acc cols 64:128,192:256)
  warp 8    : MMA (tcgen05, 128x256x32 instrs, per-128-K accumulator stages)
  warp 9    : TMA loads of A/B tiles
  warp 10   : cp.async loads of SFA (128 f32) and SFB (2 f32) per k-tile
  warp 11   : idle (present so setmaxnreg is warpgroup-uniform)

Scheduling is persistent + device-driven: tile counts come from the ctrl
buffer written by the routing kernel; every warp runs the same static
round-robin schedule computed from (tile_scan, pair_base, counts_final).
"""

from typing import Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils

from moe_dsl import (
    CTRL_TOTAL_MTILES,
    CTRL_PAIR_BASE,
    CTRL_TILE_SCAN,
    CTRL_COUNTS_FINAL,
    NUM_LOCAL,
)

import os

F8 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

IKET = os.environ.get("MOE_IKET", "") == "1"


class MoeGroupedGemm:
    def __init__(self, mode: str, m_tile: int = 128, num_ab_stage: int = 4):
        assert mode in ("g1", "g2")
        self.mode = mode
        self.m_tile = m_tile
        self.n_tile = 256
        self.k_tile = 128
        if mode == "g1":
            self.k_len = 7168
            self.n_loop = 16       # 4096 / 256
        else:
            self.k_len = 2048
            self.n_loop = 28       # 7168 / 256
        self.k_tile_cnt = self.k_len // self.k_tile
        self.acc_dtype = F32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler = (self.m_tile, self.n_tile, self.k_tile)

        self.num_ab_stage = num_ab_stage
        self.num_scale_stage = num_ab_stage + 2
        self.num_acc_stage = 2

        # warp layout
        self.promo_warps = 8
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.scale_warp_id = 10
        self.threads_per_cta = 384

        self.num_regs_promo = 216
        self.num_regs_other = 64

        self.promo_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * self.promo_warps
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (self.promo_warps + 1)
        )
        self.epi_subtile = (self.m_tile, 64)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,        # (P_cap, K) fp8
        b_raw: cute.Tensor,    # g1: (32, 4096, 7168) fp8 ; g2: (32, 7168, 2048)
        sfa: cute.Tensor,      # g1: hs_scale (56, T) f32 ; g2: SFC (16, P_cap) f32
        sfb_raw: cute.Tensor,  # g1: (32, 32, 56) f32 ; g2: (32, 56, 16) f32
        out: cute.Tensor,      # g1: (P_cap, 2048) fp8 ; g2: (P_cap, 7168) bf16
        sfc_or_pw: cute.Tensor,  # g1: SFC (16, P_cap) f32 ; g2: pair_w (P_cap,) f32
        ctrl: cute.Tensor,
        tok_out: cute.Tensor,    # (T, 7168) bf16 final output (g2 solo fast path)
        pair_dst: cute.Tensor,   # (P_cap,) i32 token idx if solo else -1
        grid_size: cutlass.Constexpr[int],
        stream: cuda_driver.CUstream,
    ):
        p_cap = a.shape[0]
        k_len = self.k_len

        # ---- logical B view (N, K, L) ----
        if cutlass.const_expr(self.mode == "g1"):
            # N mode (128, 2, 16): n = i + 128*h + 256*j -> W13 row i + 2048h + 128j
            mB = cute.make_tensor(
                b_raw.iterator,
                cute.make_layout(
                    ((128, 2, 16), k_len, NUM_LOCAL),
                    stride=((k_len, 2048 * k_len, 128 * k_len), 1, 4096 * k_len),
                ),
            )
        else:
            mB = cute.make_tensor(
                b_raw.iterator,
                cute.make_layout(
                    (7168, k_len, NUM_LOCAL),
                    stride=(k_len, 1, 7168 * k_len),
                ),
            )

        mA = cute.make_tensor(
            a.iterator,
            cute.make_layout((p_cap, k_len, 1), stride=(k_len, 1, 0)),
        )

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            F8,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, F8, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, F8, self.num_ab_stage
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        a_bytes = cute.size_in_bytes(F8, a_smem_layout)
        b_bytes = cute.size_in_bytes(F8, b_smem_layout)
        self.num_tma_load_bytes = a_bytes + b_bytes

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sSFA: cute.struct.Align[
                cute.struct.MemRange[F32, self.m_tile * self.num_scale_stage], 16
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[F32, 2 * self.num_scale_stage], 16
            ]
            sAmax: cute.struct.Align[cute.struct.MemRange[F32, 2 * self.m_tile], 16]
            sA: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(self.a_smem_layout_staged.outer)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(self.b_smem_layout_staged.outer)],
                1024,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            sfa,
            sfb_raw,
            out,
            sfc_or_pw,
            ctrl,
            tok_out,
            pair_dst,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA: cute.Tensor,      # (Kblk, P_cap)
        mSFB: cute.Tensor,      # raw scale tensor
        mOut: cute.Tensor,
        mSfcPw: cute.Tensor,
        ctrl: cute.Tensor,
        mTokOut: cute.Tensor,
        mPairDst: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        n_loop = cutlass.const_expr(self.n_loop)
        k_tile_cnt = cutlass.const_expr(self.k_tile_cnt)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        trivial_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=trivial_vmnk,
            defer_sync=True,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * self.promo_warps
            ),
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.promo_warps
            ),
            cta_layout_vmnk=trivial_vmnk,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(
            cute.make_layout((self.m_tile, self.num_scale_stage))
        )
        sSFB = storage.sSFB.get_tensor(cute.make_layout((2, self.num_scale_stage)))
        sAmax = storage.sAmax.get_tensor(cute.make_layout(2 * self.m_tile))

        # global tensors tiled for MMA
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        # ---- device-driven schedule state (same for every warp) ----
        # lane l caches tile_scan[l+1], pair_base[l], counts_final[l]
        total_mtiles = ctrl[CTRL_TOTAL_MTILES]
        total_tiles = total_mtiles * n_loop
        scan_next = Int32(0)
        base_l = Int32(0)
        cnt_l = Int32(0)
        if lane < NUM_LOCAL:
            scan_next = ctrl[CTRL_TILE_SCAN + 1 + lane]
            base_l = ctrl[CTRL_PAIR_BASE + lane]
            cnt_l = ctrl[CTRL_COUNTS_FINAL + lane]

        # =============== TMA warp ===============
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                mt_g = tile_id // n_loop
                nt = tile_id % n_loop
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                m_idx = base // self.m_tile + (mt_g - mt_prev)

                tAgA_slice = tAgA[(None, m_idx, None, 0)]
                tBgB_slice = tBgB[(None, nt, None, e)]

                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("tma_acq")
                    ab_pipeline.producer_acquire(ab_producer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_tile)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    ab_producer_state.advance()
                tile_id += gdim
            ab_pipeline.producer_tail(ab_producer_state)

        # =============== Scale warp ===============
        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            atom_sfa = cute.make_copy_atom(
                cpasync.CopyG2SOp(), F32, num_bits_per_copy=128
            )
            tiled_copy_sfa = cute.make_tiled_copy_tv(
                atom_sfa, cute.make_layout(32), cute.make_layout(4)
            )
            thr_copy_sfa = tiled_copy_sfa.get_slice(lane)
            atom_sfb = cute.make_copy_atom(
                cpasync.CopyG2SOp(), F32, num_bits_per_copy=32
            )

            scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_stage
            )
            p_cap = mSFA.layout.stride[0]
            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                mt_g = tile_id // n_loop
                nt = tile_id % n_loop
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                row0 = base + (mt_g - mt_prev) * self.m_tile

                # SFB row base offsets for this tile
                if cutlass.const_expr(self.mode == "g1"):
                    # blocks (nt, kb) and (16+nt, kb) of (32, 32, 56)
                    sfb_ptr = mSFB.iterator + e * (32 * 56) + nt * 56
                    sfb_stride = 16 * 56
                else:
                    # blocks (2nt, kb), (2nt+1, kb) of (32, 56, 16)
                    sfb_ptr = mSFB.iterator + e * (56 * 16) + (2 * nt) * 16
                    sfb_stride = 16

                # per-tile source tokens for gathered A scales (g1 only)
                toks = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                if cutlass.const_expr(self.mode == "g1"):
                    # in g1 the mPairDst argument carries pair_src (slot -> token)
                    for r in cutlass.range_constexpr(4):
                        toks[r] = mPairDst[row0 + lane * 4 + r]
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    scale_pipeline.producer_acquire(scale_producer_state)
                    st = scale_producer_state.index
                    if cutlass.const_expr(self.mode == "g1"):
                        # gather sfa[k_tile, tok] for the tile's 128 rows
                        for r in cutlass.range_constexpr(4):
                            tok = toks[r]
                            src_ptr = mSFA.iterator + k_tile * p_cap
                            if tok >= 0:
                                src_ptr = src_ptr + tok
                            gsfa1 = cute.make_tensor(src_ptr, cute.make_layout(1))
                            ssfa1 = cute.make_tensor(
                                sSFA.iterator + (st * self.m_tile + lane * 4 + r),
                                cute.make_layout(1),
                            )
                            cute.copy(atom_sfb, gsfa1, ssfa1)
                    else:
                        gsfa = cute.make_tensor(
                            (mSFA.iterator + k_tile * p_cap + row0).align(16),
                            cute.make_layout(self.m_tile),
                        )
                        tSg = thr_copy_sfa.partition_S(gsfa)
                        tSs = thr_copy_sfa.partition_D(
                            cute.make_tensor(
                                (sSFA.iterator + st * self.m_tile).align(16),
                                cute.make_layout(self.m_tile),
                            )
                        )
                        cute.copy(tiled_copy_sfa, tSg, tSs)
                    if lane < 2:
                        gsfb = cute.make_tensor(
                            sfb_ptr + (lane * sfb_stride + k_tile),
                            cute.make_layout(1),
                        )
                        ssfb = cute.make_tensor(
                            sSFB.iterator + (st * 2 + lane), cute.make_layout(1)
                        )
                        cute.copy(atom_sfb, gsfb, ssfb)
                    scale_pipeline.producer_commit(scale_producer_state)
                    scale_producer_state.advance()
                tile_id += gdim
            scale_pipeline.producer_tail(scale_producer_state)

        # =============== MMA warp ===============
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("mma_accw")
                    acc_pipeline.producer_acquire(acc_producer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("mma_abw")
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kb in cutlass.range(num_kblocks, unroll_full=True):
                        kcoord = (None, None, kb, ab_consumer_state.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kcoord],
                            tCrB[kcoord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                tile_id += gdim
            acc_pipeline.producer_tail(acc_producer_state)

        # =============== idle warp (register-uniformity companion) ======
        if warp_idx == self.promo_warps + 3:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # =============== Promotion / epilogue warps ===============
        if warp_idx < self.promo_warps:
            cute.arch.setmaxregister_increase(self.num_regs_promo)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            wg = warp_idx // 4  # 0 or 1: output subtile id
            epi_tidx = tidx % 128
            row = epi_tidx  # row within tile for Ld32x32b partitioning

            # (EPI_M_TILE, EPI_N_TILE, EPI_M, EPI_N, STAGE)
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], self.epi_subtile
            )
            tmem_load_atom = sm100_utils.get_tmem_load_op(
                (self.m_tile, self.n_tile, self.k_tile),
                utils.LayoutEnum.ROW_MAJOR,
                BF16,
                self.acc_dtype,
                self.epi_subtile,
                False,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

            # partition the output gmem tensor with the same tiled copy so the
            # per-thread value order matches the TMEM fragments by construction
            if cutlass.const_expr(self.mode == "g1"):
                out_tile_n = 128
            else:
                out_tile_n = 256
            gOut_tiled = cute.local_tile(
                cute.make_tensor(
                    mOut.iterator,
                    cute.make_layout(
                        (mOut.shape[0], out_tile_n * n_loop),
                        stride=(out_tile_n * n_loop, 1),
                    ),
                ),
                (self.m_tile, out_tile_n),
                (None, None),
            )
            gOut_epi = cute.flat_divide(gOut_tiled, self.epi_subtile)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN)
            tTR_gOut = thr_copy_t2r.partition_D(gOut_epi)

            frag_shape = tTR_gOut[(None, None, None, 0, 0, 0, 0)].shape
            tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            facc0 = cute.make_rmem_tensor(frag_shape, F32)
            facc1 = cute.make_rmem_tensor(frag_shape, F32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )

            if cutlass.const_expr(self.mode == "g1"):
                j0 = wg          # x1 half
                j1 = wg + 2      # x2 half (same output columns)
            else:
                j0 = 2 * wg      # contiguous 128-column half
                j1 = 2 * wg + 1

            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                mt_g = tile_id // n_loop
                nt = tile_id % n_loop
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                me = cute.arch.shuffle_sync(cnt_l, e)
                mt = mt_g - mt_prev
                row_g = base + mt * self.m_tile + row
                valid = row + mt * self.m_tile < me

                facc0.fill(0.0)
                facc1.fill(0.0)

                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("pr_scw")
                    scale_pipeline.consumer_wait(scale_consumer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    st = scale_consumer_state.index
                    sa = sSFA[(row, st)]
                    if cutlass.const_expr(self.mode == "g1"):
                        c0 = sa * sSFB[(0, st)]
                        c1 = sa * sSFB[(1, st)]
                    else:
                        c0 = sa * sSFB[(wg, st)]
                        c1 = c0

                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("pr_accw")
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("pr_work")
                    ai = acc_consumer_state.index
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j0, ai)],
                        tTR_rAcc,
                    )
                    if valid:
                        v = tTR_rAcc.load().to(F32)
                        f = facc0.load()
                        facc0.store(v * c0 + f)
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j1, ai)],
                        tTR_rAcc,
                    )
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    if valid:
                        v = tTR_rAcc.load().to(F32)
                        f = facc1.load()
                        facc1.store(v * c1 + f)

                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                # ---------------- epilogue ----------------
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_push("pr_epi")
                m_idx = base // self.m_tile + mt
                if cutlass.const_expr(self.mode == "g1"):
                    # swiglu: out = x1 * silu(x2) ; x1 = facc0, x2 = facc1
                    x1 = facc0.load()
                    x2 = facc1.load()
                    e2 = cute.math.exp2(x2 * -1.4426950408889634, fastmath=True)
                    sig = cute.math.rcp(e2 + 1.0, fastmath=True)
                    res = x1 * (x2 * sig)
                    facc0.store(res)
                    am = Float32(0.0)
                    for i in cutlass.range_constexpr(cute.size(facc0)):
                        am = cute.arch.fmax(am, cute.math.abs(facc0[i]))
                    # exchange amax across the two warpgroups
                    sAmax[wg * self.m_tile + row] = am
                    self.promo_sync_barrier.arrive_and_wait()
                    other = sAmax[(1 - wg) * self.m_tile + row]
                    am = cute.arch.fmax(am, other)
                    scale = am * (1.0 / 224.0)
                    inv = Float32(0.0)
                    if am > 0.0:
                        inv = 224.0 / am
                    q = cute.make_rmem_tensor(frag_shape, F8)
                    qv = facc0.load() * inv
                    q.store(qv.to(F8))
                    if valid:
                        cute.autovec_copy(
                            q, tTR_gOut[(None, None, None, 0, wg, m_idx, nt)]
                        )
                        if wg == 0:
                            mSfcPw[(nt, row_g)] = scale
                    self.promo_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                else:
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    if valid:
                        w = mSfcPw[row_g]
                        meta = mPairDst[row_g]
                        tok = meta & 1073741823
                        multi = meta >= 1073741824
                        frag_lay = tTR_gOut[
                            (None, None, None, 0, 0, 0, 0)
                        ].layout
                        col0 = nt * 256 + wg * 128
                        base0 = mTokOut.iterator + (tok * 7168 + col0)
                        base1 = base0 + 64
                        rC = cute.make_rmem_tensor(frag_shape, BF16)
                        rC.store((facc0.load() * w).to(BF16))
                        if multi:
                            for i in cutlass.range_constexpr(
                                cute.size(frag_shape) // 8
                            ):
                                v32 = cute.make_tensor(
                                    cute.recast_ptr(
                                        rC.iterator + 8 * i, dtype=cutlass.Int32
                                    ),
                                    cute.make_layout(4),
                                )
                                cute.arch.inline_ptx(
                                    "red.global.add.noftz.v4.bf16x2 [{$r0}],"
                                    " {{$r1}, {$r2}, {$r3}, {$r4}};",
                                    read_only_args=[
                                        (base0 + 8 * i).toint(),
                                        v32[0], v32[1], v32[2], v32[3],
                                    ],
                                )
                        else:
                            cute.autovec_copy(
                                rC, cute.make_tensor(base0.align(16), frag_lay)
                            )
                        rC.store((facc1.load() * w).to(BF16))
                        if multi:
                            for i in cutlass.range_constexpr(
                                cute.size(frag_shape) // 8
                            ):
                                v32 = cute.make_tensor(
                                    cute.recast_ptr(
                                        rC.iterator + 8 * i, dtype=cutlass.Int32
                                    ),
                                    cute.make_layout(4),
                                )
                                cute.arch.inline_ptx(
                                    "red.global.add.noftz.v4.bf16x2 [{$r0}],"
                                    " {{$r1}, {$r2}, {$r3}, {$r4}};",
                                    read_only_args=[
                                        (base1 + 8 * i).toint(),
                                        v32[0], v32[1], v32[2], v32[3],
                                    ],
                                )
                        else:
                            cute.autovec_copy(
                                rC, cute.make_tensor(base1.align(16), frag_lay)
                            )
                tile_id += gdim

            # tmem dealloc
            tmem.relinquish_alloc_permit()
            self.promo_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
