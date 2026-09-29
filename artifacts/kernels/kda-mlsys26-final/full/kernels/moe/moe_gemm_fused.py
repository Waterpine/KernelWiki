"""Fused grouped GEMM1+GEMM2 persistent kernel (SM100/SM103, CuTe-DSL).

One persistent kernel processes a unified per-expert tile queue:
for each expert e: [g1 tiles of e (mtiles_e x 16)] then [g2 tiles of e
(mtiles_e x 28)]. GEMM2 tiles gate on a per-expert readiness counter
(ctrl[CTRL_G1_DONE+e]) that GEMM1 tiles bump after their stores are fenced,
so the two weight streams overlap and there is no inter-kernel barrier.

Deadlock-free by construction: tiles are assigned round-robin in queue order,
every g2 tile's g1 dependencies precede it in the queue, and g1 tiles never
block, so the minimal unprocessed tile can always make progress.

Both phases share identical CTA tile shapes (M128 N256 K128), so the SMEM
buffers, MMA fragments and pipelines are common; only the TMA descriptors,
scale sources and epilogues differ per phase.
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

from moe_dsl import (
    CTRL_TOTAL_MTILES,
    CTRL_PAIR_BASE,
    CTRL_TILE_SCAN,
    CTRL_COUNTS_FINAL,
    CTRL_G1_DONE,
    NUM_LOCAL,
)

F8 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

IKET = os.environ.get("MOE_IKET", "") == "1"

NL1 = 16   # g1 n tiles per m tile (4096 / 256)
NL2 = 28   # g2 n tiles per m tile (7168 / 256)
NLT = NL1 + NL2  # 44
KC1 = 56   # g1 k tiles (7168 / 128)
KC2 = 16   # g2 k tiles (2048 / 128)


class MoeFusedGemm:
    def __init__(self, m_tile: int = 128, num_ab_stage: int = 4):
        self.m_tile = m_tile
        self.n_tile = 256
        self.k_tile = 128
        self.acc_dtype = F32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler = (self.m_tile, self.n_tile, self.k_tile)

        self.num_ab_stage = num_ab_stage
        self.num_scale_stage = num_ab_stage + 2
        self.num_acc_stage = 2

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
        a1: cute.Tensor,       # A_perm (P_cap, 7168) fp8
        b1: cute.Tensor,       # W13 (32, 4096, 7168) fp8
        sfa1: cute.Tensor,     # hs_scale (56, T) f32 (gathered via pair_src)
        sfb1: cute.Tensor,     # (32, 32, 56) f32
        a2: cute.Tensor,       # C_perm (P_cap, 2048) fp8 (also g1 output)
        b2: cute.Tensor,       # W2 (32, 7168, 2048) fp8
        sfa2: cute.Tensor,     # SFC (32, P_cap) f32 (also g1 scale output)
        sfb2: cute.Tensor,     # (32, 56, 16) f32
        tok_out: cute.Tensor,  # (T, 7168) bf16 final output
        pair_w: cute.Tensor,   # (P_cap,) f32
        pair_src: cute.Tensor, # (P_cap,) i32
        pair_dst: cute.Tensor, # (P_cap,) i32 (token | multi bit)
        ctrl: cute.Tensor,
        grid_size: cutlass.Constexpr[int],
        stream: cuda_driver.CUstream,
    ):
        p_cap = a1.shape[0]

        mB1 = cute.make_tensor(
            b1.iterator,
            cute.make_layout(
                ((128, 2, 16), 7168, NUM_LOCAL),
                stride=((7168, 2048 * 7168, 128 * 7168), 1, 4096 * 7168),
            ),
        )
        mB2 = cute.make_tensor(
            b2.iterator,
            cute.make_layout(
                (7168, 2048, NUM_LOCAL), stride=(2048, 1, 7168 * 2048)
            ),
        )
        mA1 = cute.make_tensor(
            a1.iterator, cute.make_layout((p_cap, 7168, 1), stride=(7168, 1, 0))
        )
        mA2 = cute.make_tensor(
            a2.iterator, cute.make_layout((p_cap, 2048, 1), stride=(2048, 1, 0))
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

        tma_atom_a1, tma_tensor_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), mA1, a_smem_layout,
            self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), mB1, b_smem_layout,
            self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        tma_atom_a2, tma_tensor_a2 = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), mA2, a_smem_layout,
            self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), mB2, b_smem_layout,
            self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
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
            tma_atom_a1, tma_tensor_a1,
            tma_atom_b1, tma_tensor_b1,
            tma_atom_a2, tma_tensor_a2,
            tma_atom_b2, tma_tensor_b2,
            sfa1, sfb1, sfa2, sfb2,
            a2,        # g1 output (C_perm)
            tok_out,
            pair_w, pair_src, pair_dst,
            ctrl,
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
        tma_atom_a1: cute.CopyAtom, mA1_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom, mB1_nkl: cute.Tensor,
        tma_atom_a2: cute.CopyAtom, mA2_mkl: cute.Tensor,
        tma_atom_b2: cute.CopyAtom, mB2_nkl: cute.Tensor,
        mSFA1: cute.Tensor,
        mSFB1: cute.Tensor,
        mSFA2: cute.Tensor,
        mSFB2: cute.Tensor,
        mC1: cute.Tensor,      # (P_cap, 2048) fp8: g1 output
        mTokOut: cute.Tensor,  # (T, 7168) bf16
        mPairW: cute.Tensor,
        mPairSrc: cute.Tensor,
        mPairDst: cute.Tensor,
        ctrl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a1)
            cpasync.prefetch_descriptor(tma_atom_b1)
            cpasync.prefetch_descriptor(tma_atom_a2)
            cpasync.prefetch_descriptor(tma_atom_b2)

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

        gA1 = cute.local_tile(
            mA1_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB1 = cute.local_tile(
            mB1_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gA2 = cute.local_tile(
            mA2_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB2 = cute.local_tile(
            mB2_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA1 = thr_mma.partition_A(gA1)
        tCgB1 = thr_mma.partition_B(gB1)
        tCgA2 = thr_mma.partition_A(gA2)
        tCgB2 = thr_mma.partition_B(gB2)

        tAsA1, tAgA1 = cpasync.tma_partition(
            tma_atom_a1, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA1, 0, 3),
        )
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_atom_b1, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB1, 0, 3),
        )
        tAsA2, tAgA2 = cpasync.tma_partition(
            tma_atom_a2, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA2, 0, 3),
        )
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_atom_b2, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB2, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.griddepcontrol_wait()

        # ---- schedule state ----
        total_mtiles = ctrl[CTRL_TOTAL_MTILES]
        g1_total = total_mtiles * NL1
        total_tiles = total_mtiles * NLT
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
                is_g1 = tile_id < g1_total
                pt = tile_id
                nl = Int32(NL1)
                if is_g1 == False:  # noqa: E712
                    pt = tile_id - g1_total
                    nl = Int32(NL2)
                mt_g = pt // nl
                nt = pt - mt_g * nl
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                mtiles_e = cute.arch.shuffle_sync(scan_next, e) - mt_prev
                mt = mt_g - mt_prev
                g1cnt = mtiles_e * NL1
                lt = Int32(0)
                if is_g1 == False:  # noqa: E712
                    lt = g1cnt
                m_idx = base // self.m_tile + mt

                if lt >= g1cnt:
                    # gate on all g1 tiles of this expert being complete
                    need = mtiles_e * NL1
                    if lane == 0:
                        done = cute.arch.atomic_add(
                            ctrl.iterator + (CTRL_G1_DONE + e), Int32(0),
                            sem="acquire", scope="gpu",
                        )
                        while done < need:
                            cute.arch.inline_ptx(
                                "nanosleep.u32 {$r0};", read_only_args=[Int32(256)]
                            )
                            done = cute.arch.atomic_add(
                                ctrl.iterator + (CTRL_G1_DONE + e), Int32(0),
                                sem="acquire", scope="gpu",
                            )
                    cute.arch.sync_warp()

                if is_g1:
                    tAgA_slice = tAgA1[(None, m_idx, None, 0)]
                    tBgB_slice = tBgB1[(None, nt, None, e)]
                    for k_tile in cutlass.range(KC1, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state)
                        tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(
                            tma_atom_a1, tAgA_slice[(None, k_tile)],
                            tAsA1[(None, ab_producer_state.index)],
                            tma_bar_ptr=tma_bar,
                        )
                        cute.copy(
                            tma_atom_b1, tBgB_slice[(None, k_tile)],
                            tBsB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=tma_bar,
                        )
                        ab_producer_state.advance()
                else:
                    tAgA_slice = tAgA2[(None, m_idx, None, 0)]
                    tBgB_slice = tBgB2[(None, nt, None, e)]
                    for k_tile in cutlass.range(KC2, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state)
                        tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(
                            tma_atom_a2, tAgA_slice[(None, k_tile)],
                            tAsA2[(None, ab_producer_state.index)],
                            tma_bar_ptr=tma_bar,
                        )
                        cute.copy(
                            tma_atom_b2, tBgB_slice[(None, k_tile)],
                            tBsB2[(None, ab_producer_state.index)],
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
            t_len = mSFA1.layout.stride[0]
            p_cap = mSFA2.layout.stride[0]
            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                is_g1 = tile_id < g1_total
                pt = tile_id
                nl = Int32(NL1)
                if is_g1 == False:  # noqa: E712
                    pt = tile_id - g1_total
                    nl = Int32(NL2)
                mt_g = pt // nl
                nt = pt - mt_g * nl
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                mtiles_e = cute.arch.shuffle_sync(scan_next, e) - mt_prev
                mt = mt_g - mt_prev
                g1cnt = mtiles_e * NL1
                lt = Int32(0)
                if is_g1 == False:  # noqa: E712
                    lt = g1cnt
                row0 = base + mt * self.m_tile

                if lt >= g1cnt:
                    need = mtiles_e * NL1
                    if lane == 0:
                        done = cute.arch.atomic_add(
                            ctrl.iterator + (CTRL_G1_DONE + e), Int32(0),
                            sem="acquire", scope="gpu",
                        )
                        while done < need:
                            cute.arch.inline_ptx(
                                "nanosleep.u32 {$r0};", read_only_args=[Int32(256)]
                            )
                            done = cute.arch.atomic_add(
                                ctrl.iterator + (CTRL_G1_DONE + e), Int32(0),
                                sem="acquire", scope="gpu",
                            )
                    cute.arch.sync_warp()

                if is_g1:
                    sfb_ptr = mSFB1.iterator + (e * (32 * 56) + nt * 56)
                    toks = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                    for r in cutlass.range_constexpr(4):
                        toks[r] = mPairSrc[row0 + lane * 4 + r]
                    for k_tile in cutlass.range(KC1, unroll=1):
                        scale_pipeline.producer_acquire(scale_producer_state)
                        st = scale_producer_state.index
                        for r in cutlass.range_constexpr(4):
                            tok = toks[r]
                            src_ptr = mSFA1.iterator + k_tile * t_len
                            if tok >= 0:
                                src_ptr = src_ptr + tok
                            gsfa1 = cute.make_tensor(src_ptr, cute.make_layout(1))
                            ssfa1 = cute.make_tensor(
                                sSFA.iterator + (st * self.m_tile + lane * 4 + r),
                                cute.make_layout(1),
                            )
                            cute.copy(atom_sfb, gsfa1, ssfa1)
                        if lane < 2:
                            gsfb = cute.make_tensor(
                                sfb_ptr + (lane * (16 * 56) + k_tile),
                                cute.make_layout(1),
                            )
                            ssfb = cute.make_tensor(
                                sSFB.iterator + (st * 2 + lane), cute.make_layout(1)
                            )
                            cute.copy(atom_sfb, gsfb, ssfb)
                        scale_pipeline.producer_commit(scale_producer_state)
                        scale_producer_state.advance()
                else:
                    sfb_ptr = mSFB2.iterator + (e * (56 * 16) + (2 * nt) * 16)
                    for k_tile in cutlass.range(KC2, unroll=1):
                        scale_pipeline.producer_acquire(scale_producer_state)
                        st = scale_producer_state.index
                        gsfa = cute.make_tensor(
                            (mSFA2.iterator + k_tile * p_cap + row0).align(16),
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
                                sfb_ptr + (lane * 16 + k_tile),
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
                kc = Int32(KC1)
                if tile_id >= g1_total:
                    kc = Int32(KC2)
                for k_tile in cutlass.range(kc, unroll=1):
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kb in cutlass.range(num_kblocks, unroll_full=True):
                        kcoord = (None, None, kb, ab_consumer_state.index)
                        cute.gemm(
                            tiled_mma, tCtAcc,
                            tCrA[kcoord], tCrB[kcoord], tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                tile_id += gdim
            acc_pipeline.producer_tail(acc_producer_state)

        # =============== idle warp ===============
        if warp_idx == self.promo_warps + 3:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # =============== Promotion / epilogue warps ===============
        if warp_idx < self.promo_warps:
            cute.arch.setmaxregister_increase(self.num_regs_promo)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            wg = warp_idx // 4
            epi_tidx = tidx % 128
            row = epi_tidx

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
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

            # g1 output partition (fp8 columns, 128 per tile)
            gC1_tiled = cute.local_tile(
                cute.make_tensor(
                    mC1.iterator,
                    cute.make_layout(
                        (mC1.shape[0], 2048), stride=(2048, 1)
                    ),
                ),
                (self.m_tile, 128),
                (None, None),
            )
            gC1_epi = cute.flat_divide(gC1_tiled, self.epi_subtile)
            tTR_gC1 = thr_copy_t2r.partition_D(gC1_epi)

            frag_shape = tTR_gC1[(None, None, None, 0, 0, 0, 0)].shape
            frag_lay = tTR_gC1[(None, None, None, 0, 0, 0, 0)].layout
            tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            facc0 = cute.make_rmem_tensor(frag_shape, F32)
            facc1 = cute.make_rmem_tensor(frag_shape, F32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )

            tile_id = Int32(bidx)
            while tile_id < total_tiles:
                is_g1 = tile_id < g1_total
                pt = tile_id
                nl = Int32(NL1)
                if is_g1 == False:  # noqa: E712
                    pt = tile_id - g1_total
                    nl = Int32(NL2)
                mt_g = pt // nl
                nt = pt - mt_g * nl
                ballot = cute.arch.vote_ballot_sync(scan_next <= mt_g)
                e = cute.arch.popc(ballot)
                mt_prev = cute.arch.shuffle_sync(scan_next, e - 1)
                if e == 0:
                    mt_prev = Int32(0)
                base = cute.arch.shuffle_sync(base_l, e)
                mtiles_e = cute.arch.shuffle_sync(scan_next, e) - mt_prev
                mt = mt_g - mt_prev
                g1cnt = mtiles_e * NL1
                lt = Int32(0)
                if is_g1 == False:  # noqa: E712
                    lt = g1cnt
                me = cute.arch.shuffle_sync(cnt_l, e)
                kc = Int32(KC1)
                if is_g1 == False:  # noqa: E712
                    kc = Int32(KC2)
                row_g = base + mt * self.m_tile + row
                valid = row + mt * self.m_tile < me

                facc0.fill(0.0)
                facc1.fill(0.0)

                j0 = Int32(2 * wg)
                j1 = Int32(2 * wg + 1)
                if is_g1:
                    j0 = Int32(wg)
                    j1 = Int32(wg + 2)

                for k_tile in cutlass.range(kc, unroll=1):
                    scale_pipeline.consumer_wait(scale_consumer_state)
                    st = scale_consumer_state.index
                    sa = sSFA[(row, st)]
                    c0 = sa * sSFB[(0, st)]
                    c1 = sa * sSFB[(1, st)]
                    if is_g1 == False:  # noqa: E712
                        c0 = sa * sSFB[(wg, st)]
                        c1 = c0

                    acc_pipeline.consumer_wait(acc_consumer_state)
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

                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                # ---------------- epilogue ----------------
                m_idx = base // self.m_tile + mt
                if is_g1:
                    x1 = facc0.load()
                    x2 = facc1.load()
                    e2 = cute.math.exp2(x2 * -1.4426950408889634, fastmath=True)
                    sig = cute.math.rcp(e2 + 1.0, fastmath=True)
                    res = x1 * (x2 * sig)
                    facc0.store(res)
                    am = Float32(0.0)
                    for i in cutlass.range_constexpr(cute.size(facc0)):
                        am = cute.arch.fmax(am, cute.math.abs(facc0[i]))
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
                            q, tTR_gC1[(None, None, None, 0, wg, m_idx, nt)]
                        )
                        if wg == 0:
                            mSFA2[(nt, row_g)] = scale
                    # make stores visible to the async proxy + other CTAs,
                    # then bump the per-expert readiness counter
                    cute.arch.fence_acq_rel_gpu()
                    self.promo_sync_barrier.arrive_and_wait()
                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            cute.arch.atomic_add(
                                ctrl.iterator + (CTRL_G1_DONE + e), Int32(1),
                                sem="release", scope="gpu",
                            )
                else:
                    if valid:
                        w = mPairW[row_g]
                        meta = mPairDst[row_g]
                        tok = meta & 1073741823
                        multi = meta >= 1073741824
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

            tmem.relinquish_alloc_permit()
            self.promo_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
