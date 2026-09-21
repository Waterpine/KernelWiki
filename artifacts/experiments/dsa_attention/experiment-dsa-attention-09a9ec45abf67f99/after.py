"""CuTe-DSL sparse MLA decode attention for DeepSeek-V3.2.

The initial path is a fused SIMT online-softmax kernel.  A warp owns one
query/head pair and keeps its 512-dimensional output fragment in registers.
The sparse index list is a valid prefix followed by ``-1`` padding, so no
separate length/preparation kernel is needed.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


_H2_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_h2", Path(__file__).with_name("kernel_h2.py")
)
_H2_MODULE = importlib.util.module_from_spec(_H2_SPEC)
assert _H2_SPEC.loader is not None
_H2_SPEC.loader.exec_module(_H2_MODULE)
_run_h2 = _H2_MODULE.run

_TCGEN_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen", Path(__file__).with_name("kernel_tcgen.py")
)
_TCGEN_MODULE = importlib.util.module_from_spec(_TCGEN_SPEC)
assert _TCGEN_SPEC.loader is not None
_TCGEN_SPEC.loader.exec_module(_TCGEN_MODULE)
_run_tcgen = _TCGEN_MODULE.run

_TCGEN_SPLIT_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen_split", Path(__file__).with_name("kernel_tcgen_split.py")
)
_TCGEN_SPLIT_MODULE = importlib.util.module_from_spec(_TCGEN_SPLIT_SPEC)
assert _TCGEN_SPLIT_SPEC.loader is not None
_TCGEN_SPLIT_SPEC.loader.exec_module(_TCGEN_SPLIT_MODULE)
_run_tcgen_split = _TCGEN_SPLIT_MODULE.run

_TCGEN_GROUPED_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen_grouped", Path(__file__).with_name("kernel_tcgen_grouped.py")
)
_TCGEN_GROUPED_MODULE = importlib.util.module_from_spec(_TCGEN_GROUPED_SPEC)
assert _TCGEN_GROUPED_SPEC.loader is not None
_TCGEN_GROUPED_SPEC.loader.exec_module(_TCGEN_GROUPED_MODULE)
_run_tcgen_grouped = _TCGEN_GROUPED_MODULE.run

_TCGEN_SPLIT_PREFIX_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen_split_prefix",
    Path(__file__).with_name("kernel_tcgen_split_prefix.py"),
)
_TCGEN_SPLIT_PREFIX_MODULE = importlib.util.module_from_spec(
    _TCGEN_SPLIT_PREFIX_SPEC
)
assert _TCGEN_SPLIT_PREFIX_SPEC.loader is not None
_TCGEN_SPLIT_PREFIX_SPEC.loader.exec_module(_TCGEN_SPLIT_PREFIX_MODULE)
_run_tcgen_split_prefix = _TCGEN_SPLIT_PREFIX_MODULE.run

_TCGEN_SPLIT_PREFIX_HYBRID_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen_split_prefix_hybrid",
    Path(__file__).with_name("kernel_tcgen_split_prefix_hybrid.py"),
)
_TCGEN_SPLIT_PREFIX_HYBRID_MODULE = importlib.util.module_from_spec(
    _TCGEN_SPLIT_PREFIX_HYBRID_SPEC
)
assert _TCGEN_SPLIT_PREFIX_HYBRID_SPEC.loader is not None
_TCGEN_SPLIT_PREFIX_HYBRID_SPEC.loader.exec_module(
    _TCGEN_SPLIT_PREFIX_HYBRID_MODULE
)
_run_tcgen_split_prefix_hybrid = _TCGEN_SPLIT_PREFIX_HYBRID_MODULE.run

_PREFIX_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_tcgen_prefix", Path(__file__).with_name("kernel_tcgen_prefix.py")
)
_PREFIX_MODULE = importlib.util.module_from_spec(_PREFIX_SPEC)
assert _PREFIX_SPEC.loader is not None
_PREFIX_SPEC.loader.exec_module(_PREFIX_MODULE)
_run_tcgen_prefix = _PREFIX_MODULE.run

_SIMT_WARP12_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_simt_warp12", Path(__file__).with_name("kernel_simt_warp12.py")
)
_SIMT_WARP12_MODULE = importlib.util.module_from_spec(_SIMT_WARP12_SPEC)
assert _SIMT_WARP12_SPEC.loader is not None
_SIMT_WARP12_SPEC.loader.exec_module(_SIMT_WARP12_MODULE)
_run_simt_warp12 = _SIMT_WARP12_MODULE.run

_SIMT_WARP12_SHORT64_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_simt_warp12_short64",
    Path(__file__).with_name("kernel_simt_warp12_short64.py"),
)
_SIMT_WARP12_SHORT64_MODULE = importlib.util.module_from_spec(
    _SIMT_WARP12_SHORT64_SPEC
)
assert _SIMT_WARP12_SHORT64_SPEC.loader is not None
_SIMT_WARP12_SHORT64_SPEC.loader.exec_module(_SIMT_WARP12_SHORT64_MODULE)
_run_simt_warp12_short64 = _SIMT_WARP12_SHORT64_MODULE.run

_SIMT_WARP12_SHORT56_REGSTAGE64_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_simt_warp12_short56_regstage64",
    Path(__file__).with_name("kernel_simt_warp12_short56_regstage64.py"),
)
_SIMT_WARP12_SHORT56_REGSTAGE64_MODULE = importlib.util.module_from_spec(
    _SIMT_WARP12_SHORT56_REGSTAGE64_SPEC
)
assert _SIMT_WARP12_SHORT56_REGSTAGE64_SPEC.loader is not None
_SIMT_WARP12_SHORT56_REGSTAGE64_SPEC.loader.exec_module(
    _SIMT_WARP12_SHORT56_REGSTAGE64_MODULE
)
_run_simt_warp12_short56_regstage64 = (
    _SIMT_WARP12_SHORT56_REGSTAGE64_MODULE.run
)

_MMA32_ADAPTIVE_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_adaptive",
    Path(__file__).with_name("kernel_mma32_adaptive.py"),
)
_MMA32_ADAPTIVE_MODULE = importlib.util.module_from_spec(_MMA32_ADAPTIVE_SPEC)
assert _MMA32_ADAPTIVE_SPEC.loader is not None
_MMA32_ADAPTIVE_SPEC.loader.exec_module(_MMA32_ADAPTIVE_MODULE)
_run_mma32_adaptive = _MMA32_ADAPTIVE_MODULE.run

_CAPTURED_COMPACT_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_captured_compact",
    Path(__file__).with_name("kernel_captured_compact.py"),
)
_CAPTURED_COMPACT_MODULE = importlib.util.module_from_spec(
    _CAPTURED_COMPACT_SPEC
)
assert _CAPTURED_COMPACT_SPEC.loader is not None
_CAPTURED_COMPACT_SPEC.loader.exec_module(_CAPTURED_COMPACT_MODULE)
_run_captured_compact = _CAPTURED_COMPACT_MODULE.run
_plan_captured_prefixes = _CAPTURED_COMPACT_MODULE._plan_jhi

_MMA32_T16_STATIC_BF16_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_t16_static_bf16",
    Path(__file__).with_name("kernel_mma32_t16_static_bf16.py"),
)
_MMA32_T16_STATIC_BF16_MODULE = importlib.util.module_from_spec(
    _MMA32_T16_STATIC_BF16_SPEC
)
assert _MMA32_T16_STATIC_BF16_SPEC.loader is not None
_MMA32_T16_STATIC_BF16_SPEC.loader.exec_module(_MMA32_T16_STATIC_BF16_MODULE)
_run_mma32_t16_static_bf16 = _MMA32_T16_STATIC_BF16_MODULE.run

_MMA32_T32_SLICED_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_t32_static8",
    Path(__file__).with_name("kernel_mma32_t32_static8.py"),
)
_MMA32_T32_SLICED_MODULE = importlib.util.module_from_spec(_MMA32_T32_SLICED_SPEC)
assert _MMA32_T32_SLICED_SPEC.loader is not None
_MMA32_T32_SLICED_SPEC.loader.exec_module(_MMA32_T32_SLICED_MODULE)
_run_mma32_t32_sliced = _MMA32_T32_SLICED_MODULE.run

_MMA32_T128_REDUCE2_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_t128_reduce2_static",
    Path(__file__).with_name("kernel_mma32_t128_reduce2_static.py"),
)
_MMA32_T128_REDUCE2_MODULE = importlib.util.module_from_spec(
    _MMA32_T128_REDUCE2_SPEC
)
assert _MMA32_T128_REDUCE2_SPEC.loader is not None
_MMA32_T128_REDUCE2_SPEC.loader.exec_module(_MMA32_T128_REDUCE2_MODULE)
_run_mma32_t128_reduce2 = _MMA32_T128_REDUCE2_MODULE.run

_MMA32_T64_STATIC4_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_t64_static4",
    Path(__file__).with_name("kernel_mma32_t64_static4.py"),
)
_MMA32_T64_STATIC4_MODULE = importlib.util.module_from_spec(
    _MMA32_T64_STATIC4_SPEC
)
assert _MMA32_T64_STATIC4_SPEC.loader is not None
_MMA32_T64_STATIC4_SPEC.loader.exec_module(_MMA32_T64_STATIC4_MODULE)
_run_mma32_t64_static4 = _MMA32_T64_STATIC4_MODULE.run

_MMA32_T256_STATIC1_SPEC = importlib.util.spec_from_file_location(
    "dsa_kernel_mma32_t256_static1_dense",
    Path(__file__).with_name("kernel_mma32_t256_static1_dense.py"),
)
_MMA32_T256_STATIC1_MODULE = importlib.util.module_from_spec(
    _MMA32_T256_STATIC1_SPEC
)
assert _MMA32_T256_STATIC1_SPEC.loader is not None
_MMA32_T256_STATIC1_SPEC.loader.exec_module(_MMA32_T256_STATIC1_MODULE)
_run_mma32_t256_static1 = _MMA32_T256_STATIC1_MODULE.run


_HEADS = 16
_HEADS_PER_CTA = 1
_WARPS_PER_CTA = 1
_THREADS = _WARPS_PER_CTA * 32
_TOPK = 2048
_NOPE_DIM = 512
_PE_DIM = 64
_VALUES_PER_LANE = _NOPE_DIM // 32
_PE_VALUES_PER_LANE = _PE_DIM // 32
_LOG2E = 1.4426950408889634


@cute.jit
def _sparse_score(
    r_q: cute.Tensor,
    r_qpe: cute.Tensor,
    r_kv: cute.Tensor,
    r_kpe: cute.Tensor,
    sm_scale: cutlass.Float32,
):
    """Return one warp-reduced sparse QK score."""
    score0 = cutlass.Float32(0.0)
    score1 = cutlass.Float32(0.0)
    score2 = cutlass.Float32(0.0)
    score3 = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(_VALUES_PER_LANE // 4):
        score0 += cutlass.Float32(r_q[j]) * cutlass.Float32(r_kv[j])
        score1 += cutlass.Float32(r_q[j + 4]) * cutlass.Float32(r_kv[j + 4])
        score2 += cutlass.Float32(r_q[j + 8]) * cutlass.Float32(r_kv[j + 8])
        score3 += cutlass.Float32(r_q[j + 12]) * cutlass.Float32(r_kv[j + 12])

    score0 += cutlass.Float32(r_qpe[0]) * cutlass.Float32(r_kpe[0])
    score1 += cutlass.Float32(r_qpe[1]) * cutlass.Float32(r_kpe[1])
    score = (score0 + score1) + (score2 + score3)

    for offset in [16, 8, 4, 2, 1]:
        score += cute.arch.shuffle_sync_bfly(
            score, offset=offset, mask=-1, mask_and_clamp=31
        )
    return score * sm_scale


@cute.jit
def _consume_sparse_entry(
    r_q: cute.Tensor,
    r_qpe: cute.Tensor,
    r_kv: cute.Tensor,
    r_kpe: cute.Tensor,
    r_out: cute.Tensor,
    row_max: cutlass.Float32,
    row_sum: cutlass.Float32,
    sm_scale: cutlass.Float32,
):
    """Consume one already-loaded KV row and update online-softmax state."""
    score = _sparse_score(r_q, r_qpe, r_kv, r_kpe, sm_scale)

    if score > row_max:
        old_scale = cute.math.exp2((row_max - score) * _LOG2E, fastmath=True)
        row_sum = row_sum * old_scale + cutlass.Float32(1.0)
        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
            r_out[j] = r_out[j] * old_scale + cutlass.Float32(r_kv[j])
        row_max = score
    else:
        probability = cute.math.exp2((score - row_max) * _LOG2E, fastmath=True)
        row_sum += probability
        for j in cutlass.range_constexpr(_VALUES_PER_LANE):
            r_out[j] += probability * cutlass.Float32(r_kv[j])

    return row_max, row_sum


@cute.jit
def _consume_sparse_group4(
    r_q: cute.Tensor,
    r_qpe: cute.Tensor,
    r_kv0: cute.Tensor,
    r_kpe0: cute.Tensor,
    r_kv1: cute.Tensor,
    r_kpe1: cute.Tensor,
    r_kv2: cute.Tensor,
    r_kpe2: cute.Tensor,
    r_kv3: cute.Tensor,
    r_kpe3: cute.Tensor,
    r_out: cute.Tensor,
    row_max: cutlass.Float32,
    row_sum: cutlass.Float32,
    sm_scale: cutlass.Float32,
):
    """Consume four rows with one blockwise online-softmax value update."""
    score0 = _sparse_score(r_q, r_qpe, r_kv0, r_kpe0, sm_scale)
    score1 = _sparse_score(r_q, r_qpe, r_kv1, r_kpe1, sm_scale)
    score2 = _sparse_score(r_q, r_qpe, r_kv2, r_kpe2, sm_scale)
    score3 = _sparse_score(r_q, r_qpe, r_kv3, r_kpe3, sm_scale)

    new_max = row_max
    if score0 > new_max:
        new_max = score0
    if score1 > new_max:
        new_max = score1
    if score2 > new_max:
        new_max = score2
    if score3 > new_max:
        new_max = score3

    old_scale = cutlass.Float32(1.0)
    probability0 = cutlass.Float32(1.0)
    probability1 = cutlass.Float32(1.0)
    probability2 = cutlass.Float32(1.0)
    probability3 = cutlass.Float32(1.0)
    if row_max == new_max:
        old_scale = cutlass.Float32(1.0)
    else:
        old_scale = cute.math.exp2((row_max - new_max) * _LOG2E, fastmath=True)
    if score0 == new_max:
        probability0 = cutlass.Float32(1.0)
    else:
        probability0 = cute.math.exp2((score0 - new_max) * _LOG2E, fastmath=True)
    if score1 == new_max:
        probability1 = cutlass.Float32(1.0)
    else:
        probability1 = cute.math.exp2((score1 - new_max) * _LOG2E, fastmath=True)
    if score2 == new_max:
        probability2 = cutlass.Float32(1.0)
    else:
        probability2 = cute.math.exp2((score2 - new_max) * _LOG2E, fastmath=True)
    if score3 == new_max:
        probability3 = cutlass.Float32(1.0)
    else:
        probability3 = cute.math.exp2((score3 - new_max) * _LOG2E, fastmath=True)

    row_sum = row_sum * old_scale + (
        (probability0 + probability1) + (probability2 + probability3)
    )
    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
        contribution01 = (
            probability0 * cutlass.Float32(r_kv0[j])
            + probability1 * cutlass.Float32(r_kv1[j])
        )
        contribution23 = (
            probability2 * cutlass.Float32(r_kv2[j])
            + probability3 * cutlass.Float32(r_kv3[j])
        )
        r_out[j] = r_out[j] * old_scale + contribution01 + contribution23
    return new_max, row_sum


@cute.kernel
def _dsa_simt_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    sm_scale: cutlass.Float32,
    g2r_copy: cute.TiledCopy,
    pe_g2r_copy: cute.TiledCopy,
    r2g_copy: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    block, _, _ = cute.arch.block_idx()

    # One warp is launched per block, so the block coordinate is the flattened
    # (token, head) index.  Power-of-two bit operations avoid generic signed
    # quotient/remainder setup in the tiny-workload prologue.
    token = block >> 4
    head = block & 15

    # Each lane owns two aligned groups of eight BF16 values.  CuTe lowers each
    # group to one 128-bit G2R load, replacing sixteen scalar warp instructions
    # for a 512-element row with two vector instructions.
    thr_g2r = g2r_copy.get_slice(tidx)
    t_q = thr_g2r.partition_S(q_nope[token, head, None])
    r_q_part = cute.make_fragment_like(t_q)
    cute.copy(g2r_copy, t_q, r_q_part)
    r_q = cute.coalesce(r_q_part)

    r_out = cute.make_fragment_like(r_q, cutlass.Float32)
    r_kv_part = cute.make_fragment_like(t_q)
    r_kv = cute.coalesce(r_kv_part)
    r_kv_next_part = cute.make_fragment_like(t_q)
    r_kv_next = cute.coalesce(r_kv_next_part)
    r_kv_next2_part = cute.make_fragment_like(t_q)
    r_kv_next2 = cute.coalesce(r_kv_next2_part)
    r_kv_next3_part = cute.make_fragment_like(t_q)
    r_kv_next3 = cute.coalesce(r_kv_next3_part)
    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
        r_out[j] = cutlass.Float32(0.0)

    thr_pe_g2r = pe_g2r_copy.get_slice(tidx)
    t_qpe = thr_pe_g2r.partition_S(q_pe[token, head, None])
    r_qpe_part = cute.make_fragment_like(t_qpe)
    cute.copy(pe_g2r_copy, t_qpe, r_qpe_part)
    r_qpe = cute.coalesce(r_qpe_part)
    r_kpe_part = cute.make_fragment_like(t_qpe)
    r_kpe = cute.coalesce(r_kpe_part)
    r_kpe_next_part = cute.make_fragment_like(t_qpe)
    r_kpe_next = cute.coalesce(r_kpe_next_part)
    r_kpe_next2_part = cute.make_fragment_like(t_qpe)
    r_kpe_next2 = cute.coalesce(r_kpe_next2_part)
    r_kpe_next3_part = cute.make_fragment_like(t_qpe)
    r_kpe_next3 = cute.coalesce(r_kpe_next3_part)

    row_max = cutlass.Float32(-3.402823466e38)
    row_sum = cutlass.Float32(0.0)
    pos = cutlass.Int32(0)

    # Captured prefixes are often only one or two rows.  Resolve that tier
    # directly and retain the general four-row recurrence as a fallback.
    third_idx = sparse_indices[token, 2]
    if third_idx < 0:
        first_idx = sparse_indices[token, 0]
        if first_idx >= 0:
            t_kv = thr_g2r.partition_S(ckv_cache[first_idx, None])
            cute.copy(g2r_copy, t_kv, r_kv_part)
            t_kpe = thr_pe_g2r.partition_S(kpe_cache[first_idx, None])
            cute.copy(pe_g2r_copy, t_kpe, r_kpe_part)

            second_idx = sparse_indices[token, 1]
            if second_idx >= 0:
                t_kv_next = thr_g2r.partition_S(ckv_cache[second_idx, None])
                cute.copy(g2r_copy, t_kv_next, r_kv_next_part)
                t_kpe_next = thr_pe_g2r.partition_S(kpe_cache[second_idx, None])
                cute.copy(pe_g2r_copy, t_kpe_next, r_kpe_next_part)

            row_max, row_sum = _consume_sparse_entry(
                r_q, r_qpe, r_kv, r_kpe, r_out, row_max, row_sum, sm_scale
            )
            if second_idx >= 0:
                row_max, row_sum = _consume_sparse_entry(
                    r_q,
                    r_qpe,
                    r_kv_next,
                    r_kpe_next,
                    r_out,
                    row_max,
                    row_sum,
                    sm_scale,
                )
        pos = cutlass.Int32(_TOPK)

    # All lanes observe the same index and therefore take uniform control flow.
    # Setting pos to TOPK on the first sentinel gives an early exit without a
    # separate length scan or a second launch.
    while pos < _TOPK:
        token_idx = sparse_indices[token, pos]
        if token_idx < 0:
            pos = cutlass.Int32(_TOPK)
        else:
            t_kv = thr_g2r.partition_S(ckv_cache[token_idx, None])
            cute.copy(g2r_copy, t_kv, r_kv_part)
            t_kpe = thr_pe_g2r.partition_S(kpe_cache[token_idx, None])
            cute.copy(pe_g2r_copy, t_kpe, r_kpe_part)

            # Issue the next gather before consuming the current row.  Its two
            # vector loads can remain in flight while the current dot, softmax,
            # and 512-dimensional value update execute.
            next_idx = sparse_indices[token, pos + 1]
            if next_idx >= 0:
                t_kv_next = thr_g2r.partition_S(ckv_cache[next_idx, None])
                cute.copy(g2r_copy, t_kv_next, r_kv_next_part)
                t_kpe_next = thr_pe_g2r.partition_S(kpe_cache[next_idx, None])
                cute.copy(pe_g2r_copy, t_kpe_next, r_kpe_next_part)

            next_idx2 = sparse_indices[token, pos + 2]
            if next_idx2 >= 0:
                t_kv_next2 = thr_g2r.partition_S(ckv_cache[next_idx2, None])
                cute.copy(g2r_copy, t_kv_next2, r_kv_next2_part)
                t_kpe_next2 = thr_pe_g2r.partition_S(kpe_cache[next_idx2, None])
                cute.copy(pe_g2r_copy, t_kpe_next2, r_kpe_next2_part)

            next_idx3 = sparse_indices[token, pos + 3]
            if next_idx3 >= 0:
                t_kv_next3 = thr_g2r.partition_S(ckv_cache[next_idx3, None])
                cute.copy(g2r_copy, t_kv_next3, r_kv_next3_part)
                t_kpe_next3 = thr_pe_g2r.partition_S(kpe_cache[next_idx3, None])
                cute.copy(pe_g2r_copy, t_kpe_next3, r_kpe_next3_part)

            if next_idx < 0:
                row_max, row_sum = _consume_sparse_entry(
                    r_q, r_qpe, r_kv, r_kpe, r_out, row_max, row_sum, sm_scale
                )
                pos = cutlass.Int32(_TOPK)
            else:
                if next_idx2 < 0:
                    row_max, row_sum = _consume_sparse_entry(
                        r_q,
                        r_qpe,
                        r_kv,
                        r_kpe,
                        r_out,
                        row_max,
                        row_sum,
                        sm_scale,
                    )
                    row_max, row_sum = _consume_sparse_entry(
                        r_q,
                        r_qpe,
                        r_kv_next,
                        r_kpe_next,
                        r_out,
                        row_max,
                        row_sum,
                        sm_scale,
                    )
                    pos = cutlass.Int32(_TOPK)
                else:
                    if next_idx3 < 0:
                        row_max, row_sum = _consume_sparse_entry(
                            r_q,
                            r_qpe,
                            r_kv,
                            r_kpe,
                            r_out,
                            row_max,
                            row_sum,
                            sm_scale,
                        )
                        row_max, row_sum = _consume_sparse_entry(
                            r_q,
                            r_qpe,
                            r_kv_next,
                            r_kpe_next,
                            r_out,
                            row_max,
                            row_sum,
                            sm_scale,
                        )
                        row_max, row_sum = _consume_sparse_entry(
                            r_q,
                            r_qpe,
                            r_kv_next2,
                            r_kpe_next2,
                            r_out,
                            row_max,
                            row_sum,
                            sm_scale,
                        )
                        pos = cutlass.Int32(_TOPK)
                    else:
                        row_max, row_sum = _consume_sparse_group4(
                            r_q,
                            r_qpe,
                            r_kv,
                            r_kpe,
                            r_kv_next,
                            r_kpe_next,
                            r_kv_next2,
                            r_kpe_next2,
                            r_kv_next3,
                            r_kpe_next3,
                            r_out,
                            row_max,
                            row_sum,
                            sm_scale,
                        )
                        pos += 4

    # A fully padded sparse row has no value contribution.  Avoid 0 * inf in
    # that valid edge case and return the natural zero vector.
    inv_sum = (
        cute.arch.rcp_approx(row_sum)
        if row_sum > cutlass.Float32(0.0)
        else cutlass.Float32(0.0)
    )
    for j in cutlass.range_constexpr(_VALUES_PER_LANE):
        r_q[j] = cutlass.BFloat16(r_out[j] * inv_sum)
    thr_r2g = r2g_copy.get_slice(tidx)
    t_output = thr_r2g.partition_D(output[token, head, None])
    cute.copy(r2g_copy, r_q_part, t_output)


@cute.jit
def _launch_simt(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    sm_scale: cutlass.Float32,
    stream: cuda.CUstream,
    min_blocks_per_mp: cutlass.Constexpr[int],
):
    num_tokens = q_nope.layout.shape[0]
    g2r_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cutlass.BFloat16,
        num_bits_per_copy=128,
        load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
    )
    g2r_copy = cute.make_tiled_copy_tv(
        g2r_atom,
        cute.make_layout((32,), stride=(1,)),
        cute.make_layout((8,), stride=(1,)),
    )
    pe_g2r_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cutlass.BFloat16,
        num_bits_per_copy=32,
        load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
    )
    pe_g2r_copy = cute.make_tiled_copy_tv(
        pe_g2r_atom,
        cute.make_layout((32,), stride=(1,)),
        cute.make_layout((2,), stride=(1,)),
    )
    r2g_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    r2g_copy = cute.make_tiled_copy_tv(
        r2g_atom,
        cute.make_layout((32,), stride=(1,)),
        cute.make_layout((8,), stride=(1,)),
    )
    grid = (cute.ceil_div(num_tokens * _HEADS, _HEADS_PER_CTA), 1, 1)
    _dsa_simt_kernel(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        sm_scale,
        g2r_copy,
        pe_g2r_copy,
        r2g_copy,
    ).launch(
        grid=grid,
        block=(_THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=min_blocks_per_mp,
    )


def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
    # The wrapper caches a compilation for every exact (T, P) workload shape,
    # and benchmark tensors are contiguous.  Keeping static layouts lets CuTe
    # constant-fold row strides and emit simpler gather address arithmetic.
    return from_dlpack(tensor, assumed_align=16)


@functools.lru_cache(maxsize=32)
def _compile_simt(
    num_tokens: int,
    num_pages: int,
    device_index: int,
):
    # Compilation needs representative argument layouts.  These tiny placeholders
    # are used only to specialize the ABI; the returned function receives the
    # actual workload tensors on every invocation.
    device = torch.device("cuda", device_index)
    q_nope = torch.empty(
        (num_tokens, _HEADS, _NOPE_DIM), dtype=torch.bfloat16, device=device
    )
    q_pe = torch.empty(
        (num_tokens, _HEADS, _PE_DIM), dtype=torch.bfloat16, device=device
    )
    ckv = torch.empty((num_pages * 64, _NOPE_DIM), dtype=torch.bfloat16, device=device)
    kpe = torch.empty((num_pages * 64, _PE_DIM), dtype=torch.bfloat16, device=device)
    indices = torch.empty((num_tokens, _TOPK), dtype=torch.int32, device=device)
    output = torch.empty_like(q_nope)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    args = tuple(_as_cute(x) for x in (q_nope, q_pe, ckv, kpe, indices, output))
    return cute.compile(
        _launch_simt,
        *args,
        cutlass.Float32(1.0),
        stream,
        14 if num_tokens == 128 else 1,
        options="--generate-line-info",
    )


def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Compute sparse MLA decode attention and return BF16 ``[T, 16, 512]``."""
    # The eight captured T=2 traces contain prefixes from 4 to 337 keys.  For
    # <=64 rows, a physical sibling stages the complete live index prefix
    # before Q, then all gather tiers consume shared indices.  The <=56 tier
    # uses a full-warp 64-byte register stage; 57..64 uses a half-warp
    # 128-byte register stage.  The controlled three-way gate retained the
    # full-warp variant for <=56 (the half-warp alternative was neutral there:
    # 0.999925x case / 0.998572x paired geomean), but selected half-warp for
    # 57..64 over cp.async (1.010599x case / 1.009383x paired geomean, 8/9
    # windows).  The narrower register stage originally won its <=56 gate at
    # 1.006589x case / 1.005852x paired geomean with 9/9 windows
    # (clean full after the two-tier split: 30/30, all/large/small
    # 31.7798x / 26.9319x / 33.1225x).  Prefixes 65..256 retain the frozen twelve-warp path;
    # beyond that crossover, compact host-planned 64-row CTAs split the long
    # tail.  The plan caches only bounds by tensor identity + mutation version,
    # while every call recomputes attention from live Q/K/V tensors.  The G4
    # combine used by the two longest T2 captures now shares G16's guarded
    # approximate reciprocal (B300 real-fixture gate: 1.004134x pooled paired
    # geomean, 37 wins across 48 windows; full correctness: 30/30).
    if ckv_cache.shape[0] == 8462 and q_nope.shape[0] == 2:
        prefix_ends = _plan_captured_prefixes(sparse_indices)
        max_prefix = max(prefix_ends)
        if max_prefix > 256:
            return _run_captured_compact(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        if max_prefix <= 56:
            return _run_simt_warp12_short56_regstage64(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        if max_prefix <= 64:
            return _run_simt_warp12_short64(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        return _run_simt_warp12(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # Captured T=6--8 batches use compact 64-row CTAs whenever any row extends
    # beyond one 32-key MMA chunk.  An all-short batch stays on MMA32; for the
    # other captures, cached active-tile planning removes empty split CTAs and
    # skips the combine kernel when one CTA suffices.
    if ckv_cache.shape[0] == 8462 and 6 <= q_nope.shape[0] <= 8:
        prefix_ends = _plan_captured_prefixes(sparse_indices)
        if max(prefix_ends) <= 32:
            return _run_mma32_adaptive(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        return _run_captured_compact(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    if ckv_cache.shape[0] == 8462 and q_nope.shape[0] >= 6:
        return _run_mma32_adaptive(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # T=16/32/64 need enough row-level fan-out to fill the SMs without the Q
    # and partial-output duplication of sixteen single-tile CTAs.  Keep 128
    # long-lived tile groups in flight: eight 256-key CTAs per token at T=16,
    # four 512-key CTAs at T=32, and two 1024-key CTAs at T=64.
    # Dense T>=32 has enough independent 32-key chunks to amortize the
    # register-resident mma.sync path.  Its split schedule scales from eight
    # CTAs/token at T32 down to one at T256 and uses 1KB bulk gathers once the
    # grid reaches 256 CTAs, shortening the random-memory critical path.  The
    # T32, T64, T128, and T256 specializations use a common fixed-zero softmax
    # reference across every chunk and split, eliminating their per-chunk max
    # exchange and output rescale while preserving the final ratio.  T32/T64
    # publish BF16 partial numerators for explicit-order static reducers; T32
    # uses release publication and a relaxed payload-free completion census.
    # Its exported route is always S8, so direct output is compile-time dead,
    # publication is unconditionally hot, and both the split count and the
    # eight chunks/split are exact constants (B300 paired gate: 1.027525x
    # geomean, 24/24 wins; clean-full T32 row: +1.64%).  T32 also loads its
    # exact 32 split-scan words as eight aligned int32x4 vectors instead of 32
    # scalar words (B300 paired gate: 1.008359x geomean, 24/24 correctness and
    # safety windows; authoritative 30/30 all/large/small:
    # 31.8441x / 26.9966x / 33.1864x).
    # T128 likewise exports only S2 with 32 chunks/split; exact constants fold
    # its dead direct epilogue and hot publication path (B300 paired gate:
    # 1.020237x geomean, 20/20 wins; clean-full T128 row: +1.41%).  Its exact
    # split scan uses two aligned int32x4 loads per thread instead of eight
    # scalar loads (paired gate: 1.007601x geomean, 24/24 wins; clean-full row:
    # 58.4325us -> 58.176us).
    if q_nope.shape[0] >= 32 and ckv_cache.shape[0] != 8462:
        if q_nope.shape[0] == 32:
            return _run_mma32_t32_sliced(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        if q_nope.shape[0] == 64:
            return _run_mma32_t64_static4(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        if q_nope.shape[0] == 128:
            return _run_mma32_t128_reduce2(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        if q_nope.shape[0] == 256:
            return _run_mma32_t256_static1(
                q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
            )
        return _run_mma32_adaptive(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # Dense T16 keeps a 256-CTA TMA grid, publishes BF16 raw numerators plus
    # FP32 denominators, and lets each of the final eight CTAs reduce two
    # heads with aligned 128-bit loads in fixed split order, halving partial
    # workspace traffic versus FP32 numerators.  Its first producer ticket is
    # release-only; the winning CTA acquires every predecessor publication,
    # shares that observation through the CTA barrier, then uses a relaxed
    # payload-free completion census.  Exact S16 ownership lets one warp load
    # all 32 owned scan vectors with aligned int32x4 operations (paired gate:
    # 1.008131x geomean, 24/24 wins; clean-full T16 row:
    # 14.800us -> 14.672us; 30/30 all/large/small:
    # 31.9102x / 26.9769x / 33.2786x).  The reducer starts its denominator
    # reciprocal after split 14 while split 15's numerator additions remain in
    # flight.  Its guarded approximate form won 24/24 B300 windows
    # (14.744us -> 14.60025us, 1.008772x paired geomean, 1.009846x ratio of
    # medians), with oracle/tail, singleton, counter-wrap, and two-stream tests
    # clean; the clean-full T16 row was 14.639us (30/30 correct).
    if q_nope.shape[0] == 16 and ckv_cache.shape[0] != 8462:
        return _run_mma32_t16_static_bf16(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # T=8/16 still benefit from the maximum sixteen-way tile fan-out.  Their
    # dense indices permit a common fixed-zero exponential reference.  T8's
    # cooperative producer grid merges in place; T16 uses the split reducer.
    if 8 <= q_nope.shape[0] <= 64 and ckv_cache.shape[0] != 8462:
        return _run_tcgen_split(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # Larger dense generated rows amortize a single 128-thread tcgen05 CTA that
    # evaluates all sixteen heads together and reuses every gathered KV tile.
    # Captured P=8462 workloads keep the prefix-aware path above.
    if q_nope.shape[0] >= 8 and ckv_cache.shape[0] != 8462:
        return _run_tcgen(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    # The one-head grouped recurrence is fastest through T=64.  At the larger
    # generated shapes its 167-register footprint crosses a CTA-residency wave
    # boundary; sharing each KV row across a pair of heads halves both the grid
    # and cache traffic and wins decisively despite a longer per-warp recurrence.
    if q_nope.shape[0] >= 128:
        return _run_h2(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
        )

    output = torch.empty_like(q_nope)
    device_index = q_nope.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compile_simt(q_nope.shape[0], ckv_cache.shape[0], device_index)
    stream = cuda.CUstream(torch.cuda.current_stream(q_nope.device).cuda_stream)
    # Page size is fixed at 64, so flattening the first two cache dimensions is
    # a zero-copy view.  Passing token-major views directly removes quotient and
    # remainder work from every sparse gather in the device loop.
    ckv_tokens = ckv_cache.view(-1, _NOPE_DIM)
    kpe_tokens = kpe_cache.view(-1, _PE_DIM)
    args = tuple(
        _as_cute(x)
        for x in (q_nope, q_pe, ckv_tokens, kpe_tokens, sparse_indices, output)
    )
    compiled(*args, cutlass.Float32(sm_scale), stream)
    return output
