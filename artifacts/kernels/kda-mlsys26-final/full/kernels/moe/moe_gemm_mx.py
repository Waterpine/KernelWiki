"""SM100/SM103 persistent grouped MXFP8 block-scale GEMM kernels for the MoE.

Same two logical GEMMs as moe_gemm.py, but the per-128-block f32 scales are
decomposed as s = r * 2^e with r in (0.5, 1]: the residual r is folded into
the fp8 operand values (one extra rounding) and the power-of-two part rides
as UE8M0 scale factors applied *in hardware* by the tcgen05 block-scale MMA
(kind mxf8f6f4, sf_vec_size 32; the per-128 scale byte is replicated 4x).

This removes the per-k-tile TMEM promotion round-trip entirely: the MMA
accumulates the full K extent in TMEM and the epilogue reads the accumulator
once per tile.

TMEM budget at N=256 f32 acc: 2 overlapped acc buffers (phase-indexed, the
CUTLASS blockscaled example trick) + SFA (16 cols) + SFB (32 cols):
  buffer0 = cols [0, 256), buffer1 = cols [208, 464), SF = cols [464, 512).
The epilogue reads the subtile overlapping the *other* buffer first, then
releases the (single-stage) acc pipeline early so the MMA can start the next
tile while the rest of the accumulator is still being drained.

Structure per CTA (384 threads):
  warps 0-7 : epilogue (single TMEM->reg pass per tile + SwiGLU/quant/store)
  warp 8    : MMA (tcgen05 mxf8 block-scale, S2T scale-factor staging)
  warp 9    : TMA loads of A/B/SFA/SFB tiles
  warps10-11: idle (setmaxnreg uniformity)
"""

import os

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as bsl

from moe_dsl import (
    CTRL_TOTAL_MTILES,
    CTRL_PAIR_BASE,
    CTRL_TILE_SCAN,
    CTRL_COUNTS_FINAL,
    NUM_LOCAL,
)

F8 = cutlass.Float8E4M3FN
E8M0 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

IKET = os.environ.get("MOE_IKET", "") == "1"

SF_VEC = 32


class MoeGroupedGemmMX:
    def __init__(self, mode: str, m_tile: int = 128, num_ab_stage: int = 4):
        assert mode in ("g1", "g2")
        self.mode = mode
        self.m_tile = m_tile
        self.n_tile = 256
        self.k_tile = 128
        if mode == "g1":
            self.k_len = 7168
            self.n_loop = 16       # 4096 / 256 (paired X1/X2 view)
        else:
            self.k_len = 2048
            self.n_loop = 28       # 7168 / 256
        self.k_tile_cnt = self.k_len // self.k_tile
        self.kb_cnt = self.k_len // 128          # scale blocks along K
        self.acc_dtype = F32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler = (self.m_tile, self.n_tile, self.k_tile)

        self.num_ab_stage = num_ab_stage
        # single-stage acc pipeline with two phase-indexed overlapped buffers
        self.num_acc_stage = 1

        # TMEM columns
        self.sfa_tmem_cols = (self.m_tile // 32) * 4      # 16
        self.sfb_tmem_cols = (self.n_tile // 32) * 4      # 32
        self.sf_tmem_cols = self.sfa_tmem_cols + self.sfb_tmem_cols  # 48
        self.acc_tmem_cols = self.n_tile * 2 - self.sf_tmem_cols     # 464

        # warp layout
        self.epi_warps = 8
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.threads_per_cta = 384

        self.num_regs_epi = 216
        # NOT an exact 64K-reg fit (216*256 + 72*128 = 64512): an exact fit
        # (e.g. 80) stalls setmaxnreg.inc forever and deadlocks the kernel.
        self.num_regs_other = 72
        # L2 data prefetch distance in k-tiles (0 disables). Measured a 5-7%
        # REGRESSION on B300 (L2 pollution beats the latency win) — keep the
        # code path but default off.
        self.pf_dist = int(os.environ.get("MOE_PF", "0"))

        self.epi_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * self.epi_warps
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (self.epi_warps + 1)
        )
        self.epi_subtile = (self.m_tile, 64)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,        # (P_cap, K) fp8 (residual-folded)
        b_raw: cute.Tensor,    # g1: (32, 4096, 7168) fp8 ; g2: (32, 7168, 2048)
        sfa_raw: cute.Tensor,  # ue8m0 bytes, atom-packed: (P_cap/128 * KB * 512,)
        sfb_raw: cute.Tensor,  # ue8m0 bytes, atom-packed per expert
        out: cute.Tensor,      # g1: (P_cap, 2048) fp8 ; g2: (P_cap, 7168) bf16
        sfc_or_pw: cute.Tensor,  # g1: SFC ue8m0 bytes (P_cap/128*16*512,) ; g2: pair_w (P_cap,) f32
        ctrl: cute.Tensor,
        tok_out: cute.Tensor,    # (T, 7168) bf16 final output (g2 solo fast path)
        pair_dst: cute.Tensor,   # (P_cap,) i32 token idx if solo else -1
        grid_size: cutlass.Constexpr[int],
        stream: cuda_driver.CUstream,
    ):
        p_cap = a.shape[0]
        k_len = self.k_len
        kb = self.kb_cnt

        # ---- logical B view (N, K, L) over the tile-contiguous storage ----
        # w13f storage: (e, j16, kt56, h2, i128, k128) so each k-tile pull is
        # one sequential 32KB burst. Paired N = (i, h, j).
        if cutlass.const_expr(self.mode == "g1"):
            mB = cute.make_tensor(
                b_raw.iterator,
                cute.make_layout(
                    ((128, 2, 16), (128, 56), NUM_LOCAL),
                    stride=(
                        (128, 16384, 56 * 32768),
                        (1, 32768),
                        4096 * k_len,
                    ),
                ),
            )
            # SFB atoms stored atom-row-major (W13 row atoms), K atoms inner:
            # byte(e, r, kb, k1) = e*32*kb*512 + (r//128)*(kb*512) + kb*512*
            mSFB = cute.make_tensor(
                cute.recast_ptr(sfb_raw.iterator, dtype=E8M0),
                cute.make_layout(
                    (((32, 4), 2, 16), ((SF_VEC, 4), kb), NUM_LOCAL),
                    stride=(
                        ((16, 4), 16 * kb * 512, kb * 512),
                        ((0, 1), 512),
                        32 * kb * 512,
                    ),
                ),
            )
        else:
            # w2f storage: (e, nt28, kt16, r256, k128)
            mB = cute.make_tensor(
                b_raw.iterator,
                cute.make_layout(
                    ((256, 28), (128, 16), NUM_LOCAL),
                    stride=((128, 16 * 32768), (1, 32768), 7168 * k_len),
                ),
            )
            mSFB = cute.make_tensor(
                cute.recast_ptr(sfb_raw.iterator, dtype=E8M0),
                cute.make_layout(
                    (((32, 4), 56), ((SF_VEC, 4), kb), NUM_LOCAL),
                    stride=(((16, 4), kb * 512), ((0, 1), 512), 56 * kb * 512),
                ),
            )

        mA = cute.make_tensor(
            a.iterator,
            cute.make_layout((p_cap, k_len, 1), stride=(k_len, 1, 0)),
        )
        mSFA = cute.make_tensor(
            cute.recast_ptr(sfa_raw.iterator, dtype=E8M0),
            cute.make_layout(
                (((32, 4), p_cap // 128), ((SF_VEC, 4), kb), 1),
                stride=(((16, 4), kb * 512), ((0, 1), 512), 0),
            ),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            F8,
            F8,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            E8M0,
            SF_VEC,
            self.cta_group,
            self.mma_tiler[:2],
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, F8, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, F8, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = bsl.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = bsl.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )

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
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSFA,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSFB,
            sfb_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_bytes = cute.size_in_bytes(F8, a_smem_layout)
        b_bytes = cute.size_in_bytes(F8, b_smem_layout)
        sfa_bytes = cute.size_in_bytes(E8M0, cute.filter_zeros(sfa_smem_layout))
        sfb_bytes = cute.size_in_bytes(E8M0, cute.filter_zeros(sfb_smem_layout))
        self.num_tma_load_bytes = a_bytes + b_bytes + sfa_bytes + sfb_bytes

        sfa_smem_cosize = cute.cosize(cute.filter_zeros(self.sfa_smem_layout_staged))
        sfb_smem_cosize = cute.cosize(cute.filter_zeros(self.sfb_smem_layout_staged))

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sAmax: cute.struct.Align[cute.struct.MemRange[F32, 4 * self.m_tile], 16]
            sSFA: cute.struct.Align[cute.struct.MemRange[E8M0, sfa_smem_cosize], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[E8M0, sfb_smem_cosize], 128]
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
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            out,
            sfc_or_pw,
            ctrl,
            tok_out,
            pair_dst,
            a,
            b_raw,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
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
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mOut: cute.Tensor,
        mSfcPw: cute.Tensor,
        ctrl: cute.Tensor,
        mTokOut: cute.Tensor,
        mPairDst: cute.Tensor,
        mAraw: cute.Tensor,
        mBraw: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
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
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

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
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.epi_warps
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
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sAmax = storage.sAmax.get_tensor(cute.make_layout(4 * self.m_tile))

        # global tensors tiled for MMA
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)

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
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # two overlapped acc buffers selected by pipeline phase
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (self.n_tile - self.sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                ),
            ),
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        # ---- device-driven schedule state (same for every warp) ----
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
                tAgSFA_slice = tAgSFA[(None, m_idx, None, 0)]
                tBgSFB_slice = tBgSFB[(None, nt, None, e)]

                pa = mAraw.iterator + (m_idx * self.m_tile + lane) * self.k_len
                pb = mBraw.iterator
                if cutlass.const_expr(self.mode == "g1"):
                    pb = (
                        mBraw.iterator
                        + e * (4096 * 7168)
                        + nt * (56 * 32768)
                        + lane * 128
                    )
                else:
                    pb = (
                        mBraw.iterator
                        + e * (7168 * 2048)
                        + nt * (16 * 32768)
                        + lane * 128
                    )

                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if cutlass.const_expr(self.pf_dist > 0):
                        kpf = k_tile + self.pf_dist
                        if kpf < k_tile_cnt:
                            koff = kpf * 128
                            for c in cutlass.range_constexpr(4):
                                cute.arch.inline_ptx(
                                    "prefetch.global.L2 [{$r0}];",
                                    read_only_args=[
                                        (pa + (c * 32 * self.k_len) + koff).toint()
                                    ],
                                )
                            # B is tile-contiguous: 32KB burst per k-tile
                            for c in cutlass.range_constexpr(8):
                                cute.arch.inline_ptx(
                                    "prefetch.global.L2 [{$r0}];",
                                    read_only_args=[
                                        (
                                            pb + kpf * 32768 + c * 4096
                                        ).toint()
                                    ],
                                )
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("tma_acq")
                    ab_pipeline.producer_acquire(ab_producer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    si = ab_producer_state.index
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile)],
                        tAsA[(None, si)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_tile)],
                        tBsB[(None, si)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, k_tile)],
                        tAsSFA[(None, si)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, k_tile)],
                        tBsSFB[(None, si)],
                        tma_bar_ptr=tma_bar,
                    )
                    ab_producer_state.advance()
                tile_id += gdim
            ab_pipeline.producer_tail(ab_producer_state)

        # =============== MMA warp ===============
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + self.acc_tmem_cols, dtype=E8M0
            )
            tCtSFA_layout = bsl.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                SF_VEC,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + self.acc_tmem_cols + self.sfa_tmem_cols, dtype=E8M0
            )
            tCtSFB_layout = bsl.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                SF_VEC,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # S2T copy partitions
            tCsSFA_compact = cute.filter_zeros(sSFA)
            tCtSFA_compact = cute.filter_zeros(tCtSFA)
            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(self.cta_group), E8M0
            )
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
            thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfa, tCsSFA_s2t_
            )
            tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

            tCsSFB_compact = cute.filter_zeros(sSFB)
            tCtSFB_compact = cute.filter_zeros(tCtSFB)
            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
            thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
            tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfb, tCsSFB_s2t_
            )
            tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                tCtAcc = tCtAcc_base[
                    (None, None, None, acc_producer_state.phase ^ 1)
                ]
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_push("mma_accw")
                acc_pipeline.producer_acquire(acc_producer_state)
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_pop()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_push("mma_abw")
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    if cutlass.const_expr(IKET):
                        cute.experimental.iket.range_pop()
                    si = ab_consumer_state.index
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_s2t[(None, None, None, None, si)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_s2t[(None, None, None, None, si)],
                        tCtSFB_s2t,
                    )
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kb_i in cutlass.range(num_kblocks, unroll_full=True):
                        kcoord = (None, None, kb_i, si)
                        tiled_mma.set(
                            tcgen05.Field.SFA,
                            tCtSFA[(None, None, kb_i)].iterator,
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB,
                            tCtSFB[(None, None, kb_i)].iterator,
                        )
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

        # =============== idle warps (register-uniformity companions) =====
        if warp_idx == self.epi_warps + 2:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx == self.epi_warps + 3:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # =============== Epilogue warps ===============
        if warp_idx < self.epi_warps:
            cute.arch.setmaxregister_increase(self.num_regs_epi)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            wg = warp_idx // 4  # 0 or 1: output subtile id
            epi_tidx = tidx % 128
            row = epi_tidx

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
            facc0 = cute.make_rmem_tensor(frag_shape, F32)
            facc1 = cute.make_rmem_tensor(frag_shape, F32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(self.mode == "g1"):
                j0 = wg          # x1 half
                j1 = wg + 2      # x2 half (same output columns)
            else:
                j0 = 2 * wg      # contiguous 128-column half
                j1 = 2 * wg + 1

            par = Int32(0)
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

                phase = acc_consumer_state.phase
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_push("ep_accw")
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("ep_t2r")

                # phase 0: buffer at cols [0,256): read the high subtile first
                # (it overlaps buffer 1), release early, then read the low one.
                # phase 1: buffer at cols [208,464): read the low subtile first
                # (it overlaps buffer 0), release early, then read the high one.
                if phase == 0:
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j1, 0)],
                        facc1,
                    )
                else:
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j0, 1)],
                        facc0,
                    )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                if phase == 0:
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j0, 0)],
                        facc0,
                    )
                else:
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, j1, 1)],
                        facc1,
                    )

                if cutlass.const_expr(IKET):
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("ep_epi")

                # ---------------- epilogue ----------------
                m_idx = base // self.m_tile + mt
                if cutlass.const_expr(self.mode == "g1"):
                    # swiglu: out = x1 * silu(x2) ; x1 = facc0, x2 = facc1
                    for i in cutlass.range_constexpr(cute.size(facc1)):
                        x2i = facc1[i]
                        thi = cute.math.tanh(x2i * 0.5, approx=True)
                        facc1[i] = x2i * (thi * 0.5 + 0.5)
                    res = facc0.load() * facc1.load()
                    facc0.store(res)
                    am0 = Float32(0.0)
                    am1 = Float32(0.0)
                    for i in cutlass.range_constexpr(0, cute.size(facc0), 2):
                        am0 = cute.arch.fmax(am0, cute.math.abs(facc0[i]))
                        am1 = cute.arch.fmax(am1, cute.math.abs(facc0[i + 1]))
                    am = cute.arch.fmax(am0, am1)
                    # exchange amax across the two warpgroups (parity buffers)
                    sAmax[(par * 2 + wg) * self.m_tile + row] = am
                    self.epi_sync_barrier.arrive_and_wait()
                    other = sAmax[(par * 2 + 1 - wg) * self.m_tile + row]
                    am = cute.arch.fmax(am, other)
                    par = par ^ 1
                    # pow2 quant scale: 2^(ebyte-127) >= am/224
                    tb = cute.make_rmem_tensor(cute.make_layout(1), F32)
                    ib = cute.make_tensor(
                        cute.recast_ptr(tb.iterator, dtype=Int32),
                        cute.make_layout(1),
                    )
                    tb[0] = am * Float32(1.0 / 224.0)
                    bits = ib[0]
                    ebyte = (bits >> 23) & 255
                    if (bits & 8388607) != 0:
                        ebyte = ebyte + 1
                    if ebyte > 254:
                        ebyte = Int32(254)
                    inv = Float32(0.0)
                    if am > 0.0:
                        # inv = 2^(127-ebyte)  (exact pow2 built from bits)
                        ib[0] = (254 - ebyte) << 23
                        inv = tb[0]
                    else:
                        ebyte = Int32(127)
                    q = cute.make_rmem_tensor(frag_shape, F8)
                    qv = facc0.load() * inv
                    q.store(qv.to(F8))
                    if valid:
                        cute.autovec_copy(
                            q, tTR_gOut[(None, None, None, 0, wg, m_idx, nt)]
                        )
                        if wg == 0:
                            # SFC ue8m0 atom bytes: 4 replicated bytes (u32)
                            sfc_off = (
                                m_idx * (16 * 512)
                                + nt * 512
                                + (row % 32) * 16
                                + (row // 32) * 4
                            )
                            bb = ebyte & 255
                            word = bb | (bb << 8) | (bb << 16) | (bb << 24)
                            dst32 = cute.make_tensor(
                                cute.recast_ptr(
                                    mSfcPw.iterator + sfc_off, dtype=Int32
                                ),
                                cute.make_layout(1),
                            )
                            dst32[0] = word
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
            self.epi_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
