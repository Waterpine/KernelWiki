"""DSA sparse MLA decode attention for NVIDIA B300 (SM103) in CuTe-DSL.

Computes, per query token t:
    out[t] = softmax( (q_nope[t] @ Kc^T + q_pe[t] @ Kp^T) * sm_scale ) @ Kc
over the valid (non -1) rows of sparse_indices[t] gathered from the paged
compressed-KV cache (ckv) and RoPE cache (kpe). Token-level indices address
flattened [num_pages*64, 512] / [num_pages*64, 64] caches directly.

Design:
  * One kernel launch per call. Grid = (T * NSPLIT) CTAs; each CTA handles a
    slice of the token's 2048 top-k slots ("flash-decoding" split-K).
  * Register MMA (mma.sync m16n8k16 x8 warps). QK^T is computed transposed
    (S^T = K @ Q^T) on a (4,1,2) atom layout: A = gathered K/V rows (each
    smem byte ldsm'd once) and B = Q (duplicated x4 instead of x8), with a
    pairwise smem reduce of the two k-half accumulators. PV stays on the
    (1,8,1) layout (warps split output cols).
  * Depth-2 TMA bulk (cp.async.bulk per row+cache) gather with mbarrier
    completion: stage barriers are armed with constant full-tile bytes at
    kernel entry and every row is gathered unconditionally (invalid slots
    gather cache row 0 - L2-hot, finite, exactly masked), so the gathers
    launch straight off per-warp uniform index loads with no scan/arm chain.
    Iteration it waits tile it, consumes it (QK/softmax/PV), then re-arms
    and issues tile it+2 into the freed stage. Q moves on its own mbarrier
    (barQ), issued by warps 6/7 right after the entry barrier.
  * Softmax without running max (scores bounded; upper clamp for NaN-safety).
    The general path lowers that guard through native fmin; the scalar micro
    body retains its algebraic clamp because native fmin measured slower there.
  * T=1/two-KV microkernel: one 16-CTA cluster assigns one head per CTA;
    warp 0 computes the two BF16 dot products, then all eight warps emit the
    512-value row after one shared-memory handoff. It preserves fp32 score and
    P.V accumulation plus the hi/lo BF16 P representation, while bypassing the
    general gather/MMA/merge choreography for this launch-overhead floor case.
  * Per-tile validity scan parallel to the gathers: empty splits skip the
    main loop and epilogue-store work entirely, empty tiles skip their MMA,
    and prefix-packed tiles let fully-dead warps skip QK.
  * Strided tile->split mapping (split s owns tiles {s + j*NSPLIT}): with
    prefix-packed indices, a token with <= NSPLIT*64 valid rows lands at most
    ONE nonempty tile per split, so medium tokens (129..1024 valid at ns16)
    consume in parallel instead of 2 serial tiles on the low splits. Exact
    for any -1 pattern (pure slot permutation; the solo probe covers the
    complement ranges). NTILES==1 configs (ns32) are mapping-invariant.
  * Split 0 publishes the DIRECT decision to its own smem right at the
    pre-loop probe (sAf/sAny final there); mainloop barriers order it for
    the epilogue, which drops the re-derive + extra CTA barrier.
  * Solo fast path: split 0 prefetches the other splits' index slots after
    issuing its gathers (exact, any -1 pattern); when no other split has work
    it writes the normalized output directly, skipping partials and the merge.
  * Otherwise per-split partials merged in-kernel, column-split across the
    last NM splits (NM = 8 within one wave, else 4; launch-id tagged flags).
  * Partials (gmem mPartO, DSMEM sSlot, pair-hop rows) are carried in fp32
    end to end; the accumulated output is rounded to bf16 exactly once, at
    the final normalized store. (bf16 partials cost 1-2 output ULPs on
    cancelling elements and fail the official every-element 1e-2 gate.)
  * Softmax P is split hi/lo into two bf16 planes (p = p_hi + p_lo, exact to
    ~2^-17) and PV runs two MMAs per k-step, so the P quantization noise
    (~2^-9 rel, the other half of the official-gate 2-ULP collisions vs the
    baseline's own noise) drops below the baseline's rounding floor.
  * Packed-prefix QK warp-skip: warps whose 16-row m-tile lies beyond the
    tile's valid count skip QK (their acc stays zero; dead rows are
    validity-masked and land zero in P via the masked store).
  * Padded smem strides plus a lane-contiguous QK k-half exchange remove the
    measured shared-memory bank conflicts in both ldsm and cross-warp traffic.
  * The underfilled ns32/T=2 specialization reads its duplicated prologue
    indices through the read-only path with an L2 residency hint; ns16 keeps
    ordinary loads, where the same hint measured slower.
"""

import math
import os
from typing import Any, cast

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int16, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute as _cute_ir, llvm
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import cpasync, warp as warp_ops
from cutlass.cute.runtime import from_dlpack
