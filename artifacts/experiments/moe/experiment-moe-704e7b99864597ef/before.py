"""FP8 block-scale DeepSeek-V3 MoE — CuTe-DSL kernels for B200/B300 (sm_10x).
(contest run: seed submission)

Pipeline per call (3 kernel launches from one compiled host function):
 1. routing: fused sigmoid + group-top-k + CSR build + activation permute
    with per-row requantization
 2. gemm1:   grouped GEMM (full-K TMEM accumulation) + SwiGLU -> bf16
             intermediate; cooperative tail quantizes rows to FP8
 3. gemm2:   grouped GEMM + weighted combine into bf16 output

Weights are requantized once per weight set on the host so every 128-column
block carries a single scale across the whole K range (enables uniform-scale
MMA accumulation). The transform is cached keyed on the tensors' identity
(data_ptr + version); outputs are always computed from live inputs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from moe_common import HIDDEN, INTER, KB1, KB2, META_SIZE
from moe_routing import RoutingKernel, WarmL2
from moe_gemms import MoeGemm1, MoeGemm2, M_TILE, N1_TILES, N2_TILES2, Z_ROWS
from moe_g12 import MoeGemm12
from moe_g1x2 import MoeGemm1X2


class _Launcher:
   def __init__(self):
       lg = os.environ.get("MOE_G12N_LATE_GATE", "0") == "1"
       lt = os.environ.get("MOE_G12N_LEAN_TAIL", "1") == "1"
       ltf = os.environ.get("MOE_G12F_LEAN_TAIL", "1") == "1"
       # pull-permute measured DRAM-zero-sum at large T (routing -83us,
       # G1 +76us @32768): G1 already runs at the DRAM ceiling, so moving
       # the permute bytes into it buys nothing. Kept for experiments.
       self.pull = os.environ.get("MOE_G1_PULL", "0") == "1"
       rc = os.environ.get("MOE_RED_COMB", "1") == "1"
       self.red_comb = rc
       kblk = int(os.environ.get("MOE_KBLK", "128"))
       self.rk = RoutingKernel(M_TILE, N1_TILES, N2_TILES2, Z_ROWS, nthr=256)
       # P0 ILP-2 (parked, default 1): two topk cascades per warp iteration
       # measured WORSE at T=11948 (P0 31.2->38.7us, P3 +6 from kernel-wide
       # register pressure at the 384-thread/168-reg ceiling; 40B spills
       # with prefetch kept, 8B without). P0's ~4.6us/token is MIO/shuffle
       # THROUGHPUT saturation across 12 warps, not per-chain latency —
       # more ILP only adds pressure. Logits L2 presweep (MOE_RK_LPF) also
       # neutral: the lbuf double-buffer already hides the loads.
       rk_ilp = int(os.environ.get("MOE_RK_ILP", "1"))
       self.rk_big = RoutingKernel(M_TILE, N1_TILES, N2_TILES2, Z_ROWS,
                                   nthr=384, p0_ilp=rk_ilp)
       # T < FUSE_LO: one-CTA routing (CTA barriers + smem-carried counts;
       # the multi-CTA rk costs a flat ~13.9us of serial phase latency)
       self.rk_tiny = RoutingKernel(M_TILE, N1_TILES, N2_TILES2, Z_ROWS,
                                    nthr=256, tiny=True)
       self.g12f = MoeGemm12(fuse_rt=True, lean_tail=ltf, red_comb=rc,
                             kblk=kblk)
       # dense-P3 fused variant for mid T ([512, 2048)): expert-major
       # permute with progressive perm_done release, so the first m-tiles'
       # GEMM1 A-gates open while the permute is still running (token-major
       # P2' opens them only near its end, which is why plain g12f lost to
       # the 2-kernel path at T=901)
       self.g12fd = MoeGemm12(fuse_rt=True, lean_tail=ltf, red_comb=rc,
                              kblk=kblk, dense_p3=True)
       # W2 L2-prefetch distance (queue items ahead) for the nofuse instance
       w2pf = int(os.environ.get("MOE_W2PF", "0"))
       self.g12n = MoeGemm12(fuse_rt=False, late_gate=lg, lean_tail=lt,
                             red_comb=rc, kblk=kblk, w2pf=w2pf)
       # GEMM1 K-split experiment (MOE_KSPLIT=1): measured NET NEGATIVE and
       # parked. Straggler-wave splitting (T=14/16) loses because the f32
       # partial round-trip through psum adds ~98MB of L2/DRAM traffic to a
       # DRAM-saturated phase (+13us), and G2 pops already overlap the G1
       # tail wave; deep-latency splitting (T=1) wins ~7us of stream but
       # gives ~5 back in owner combine latency (poll + gmem loads). Also:
       # any role branching around the t2r rounds serializes the LDTMs
       # (~2.5us x 4/item) — the unified straight-line-t2r form is the only
       # viable shape at the 168-reg ceiling.
       self.ksplit = os.environ.get("MOE_KSPLIT", "0") == "1"
       self.g12nk = (
           MoeGemm12(fuse_rt=False, late_gate=lg, lean_tail=lt,
                     red_comb=rc, kblk=kblk, ksplit=True)
           if self.ksplit else self.g12n
       )
       # M-pair instance for large T: GEMM1 items cover two adjacent m-tiles
       # of one expert sharing a single B stream (-33% TMA bytes per tile);
       # routing is launched with do_pair=2 so the g1 item prefix counts
       # ceil(m-tiles/2) pair slots per expert
       self.g12p = (
           MoeGemm12(fuse_rt=False, late_gate=lg, lean_tail=lt,
                     red_comb=rc, pair=True)
           if kblk == 128 else self.g12n
       )
       # static-segment fused variant (T <= 128): barrier-free prologue.
       # Correct but measured SLOWER than routing+g12n: its item table is
       # 4.5x larger than the active set, and the skip machinery's pops +
       # cursor polling contend so hard that P0's own atomics starve (IKET:
       # pop avg 6.2us, P0 32us at T=7). Kept env-gated for a possible
       # compact-id revival; default OFF.
       self.g12s = MoeGemm12(fuse_rt=True, red_comb=rc, static_seg=True)
       self.use_static = rc and os.environ.get("MOE_G12_STATIC", "0") == "1"
       self.g1_tail = MoeGemm1(fused=False, pull=self.pull)
       self.warm = WarmL2()
       self.use2sm = os.environ.get("MOE_G1_2SM", "0") == "1"
       self.g1x2 = MoeGemm1X2()
       self.g2 = MoeGemm2()

   @cute.jit
   def launch(
       self,
       logits: cute.Tensor,
       bias: cute.Tensor,
       hidden: cute.Tensor,
       hs_scale: cute.Tensor,
       w13_nkl: cute.Tensor,
       w13m: cute.Tensor,
       w2_nkl: cute.Tensor,
       w2m: cute.Tensor,
       w13t: cute.Tensor,
       w2t: cute.Tensor,
       out: cute.Tensor,
       topk_e: cute.Tensor,
       topk_w: cute.Tensor,
       counts: cute.Tensor,
       offsets: cute.Tensor,
       cursors: cute.Tensor,
       g1_off: cute.Tensor,
       g2_off: cute.Tensor,
       dense_off: cute.Tensor,
       mt_done: cute.Tensor,
       perm_done: cute.Tensor,
       row_token: cute.Tensor,
       row_weight: cute.Tensor,
       perm_amax: cute.Tensor,
       zc_tokens: cute.Tensor,
       mt_tokens: cute.Tensor,
       tok_cnt: cute.Tensor,
       topk_slot: cute.Tensor,
       abuf: cute.Tensor,
       cbuf_bf: cute.Tensor,
       cbuf: cute.Tensor,
       camax: cute.Tensor,
       cscale_row: cute.Tensor,
       sbuf: cute.Tensor,
       meta: cute.Tensor,
       gq: cute.Tensor,
       psum: cute.Tensor,
       pflag: cute.Tensor,
       local_off: cutlass.Int32,
       rsf: cutlass.Float32,
       nwarm: cutlass.Int32,
       pf_bytes: cutlass.Int32,
       stream,
       stream2,
   ):
       T_ = logits.shape[0]
       do_perm_big = cutlass.Int32(0) if self.pull else cutlass.Int32(1)
       do_pair_big = cutlass.Int32(1) if self.use2sm else cutlass.Int32(0)
       if T_ >= G12_CUT:
           if T_ >= 4096:
               self.rk_big.launch(
                   logits, bias, hidden, abuf, hs_scale, topk_e, topk_w,
                   counts, offsets, cursors, g1_off, g2_off, dense_off,
                   mt_done, perm_done, row_token, row_weight, perm_amax,
                   zc_tokens, mt_tokens, tok_cnt, topk_slot, meta, w13t,
                   pflag, local_off, rsf, do_perm_big, do_pair_big,
                   cutlass.Int32(0), pf_bytes, stream,
               )
           else:
               self.rk.launch(
                   logits, bias, hidden, abuf, hs_scale, topk_e, topk_w,
                   counts, offsets, cursors, g1_off, g2_off, dense_off,
                   mt_done, perm_done, row_token, row_weight, perm_amax,
                   zc_tokens, mt_tokens, tok_cnt, topk_slot, meta, w13t,
                   pflag, local_off, rsf, do_perm_big, do_pair_big,
                   cutlass.Int32(0), pf_bytes, stream,
               )
           if cutlass.const_expr(self.use2sm):
               self.g1x2.launch(
                   abuf, w13_nkl, w13m, cbuf_bf, cbuf, camax, cscale_row,
                   perm_amax, row_token, offsets, cursors, g1_off, dense_off,
                   zc_tokens, out, gq, meta, stream,
               )
           else:
               self.g1_tail.launch(
                   abuf, w13_nkl, w13m, cbuf_bf, cbuf, camax, cscale_row,
                   perm_amax, row_token, offsets, cursors, g1_off, dense_off,
                   mt_done, zc_tokens, out, hidden, hs_scale, perm_done,
                   meta, stream,
               )
           self.g2.launch(
               cbuf, w2_nkl, w2m, cscale_row, row_token, row_weight, offsets,
               cursors, g2_off, counts, out, sbuf, mt_tokens, topk_slot, meta,
               stream,
           )
       else:
           s_hi = cutlass.Int32(128 if self.use_static else 0)
           if T_ <= s_hi:
               self.g12s.launch(
                   logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                   w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax, cscale_row,
                   perm_amax, topk_e, topk_w, tok_cnt, row_token, row_weight,
                   offsets, cursors, dense_off, g1_off, g2_off, mt_done,
                   perm_done, zc_tokens, out, sbuf, mt_tokens, topk_slot,
                   counts, meta, psum, pflag, local_off, rsf, stream,
               )
           elif (T_ >= G12_FUSE_LO) & (T_ < G12_FUSE_HI):
               self.g12f.launch(
                   logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                   w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax, cscale_row, perm_amax,
                   topk_e, topk_w, tok_cnt, row_token, row_weight, offsets,
                   cursors, dense_off, g1_off, g2_off, mt_done, perm_done,
                   zc_tokens, out, sbuf, mt_tokens, topk_slot, counts, meta,
                   psum, pflag, local_off, rsf, stream,
               )
           elif (T_ >= G12_FUSE_HI) & (T_ < G12_FUSE_HI2):
               self.g12fd.launch(
                   logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                   w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax, cscale_row, perm_amax,
                   topk_e, topk_w, tok_cnt, row_token, row_weight, offsets,
                   cursors, dense_off, g1_off, g2_off, mt_done, perm_done,
                   zc_tokens, out, sbuf, mt_tokens, topk_slot, counts, meta,
                   psum, pflag, local_off, rsf, stream,
               )
           else:
               if T_ >= PAIR_CUT:
                   # M-pair path: routing counts pair slots (do_pair=2), the
                   # pair GEMM instance shares one B stream per m-tile pair
                   self.rk_big.launch(
                       logits, bias, hidden, abuf, hs_scale, topk_e, topk_w,
                       counts, offsets, cursors, g1_off, g2_off, dense_off,
                       mt_done, perm_done, row_token, row_weight, perm_amax,
                       zc_tokens, mt_tokens, tok_cnt, topk_slot, meta, w13t,
                       pflag, local_off, rsf, cutlass.Int32(1),
                       cutlass.Int32(2), cutlass.Int32(0), pf_bytes, stream,
                   )
                   self.g12p.launch(
                       logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                       w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax,
                       cscale_row, perm_amax, topk_e, topk_w, tok_cnt,
                       row_token, row_weight, offsets, cursors, dense_off,
                       g1_off, g2_off, mt_done, perm_done, zc_tokens, out,
                       sbuf, mt_tokens, topk_slot, counts, meta, psum, pflag,
                       local_off, rsf, stream,
                   )
               elif T_ < G12_FUSE_LO:
                   # tiny 2-kernel path: routing may K-split gemm1 items
                   # (ks_en=1, MOE_KSPLIT experiment); the paired ksplit
                   # instance decodes them
                   ks_en = cutlass.Int32(1 if self.ksplit else 0)
                   if T_ < RK_TINY_HI:
                       self.rk_tiny.launch(
                           logits, bias, hidden, abuf, hs_scale, topk_e,
                           topk_w, counts, offsets, cursors, g1_off, g2_off,
                           dense_off, mt_done, perm_done, row_token,
                           row_weight, perm_amax, zc_tokens, mt_tokens,
                           tok_cnt, topk_slot, meta, w13t, pflag, local_off,
                           rsf, cutlass.Int32(1), cutlass.Int32(0),
                           ks_en, pf_bytes, stream,
                       )
                   else:
                       self.rk.launch(
                           logits, bias, hidden, abuf, hs_scale, topk_e,
                           topk_w, counts, offsets, cursors, g1_off, g2_off,
                           dense_off, mt_done, perm_done, row_token,
                           row_weight, perm_amax, zc_tokens, mt_tokens,
                           tok_cnt, topk_slot, meta, w13t, pflag, local_off,
                           rsf, cutlass.Int32(1), cutlass.Int32(0),
                           ks_en, pf_bytes, stream,
                       )
                   self.g12nk.launch(
                       logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                       w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax, cscale_row, perm_amax,
                       topk_e, topk_w, tok_cnt, row_token, row_weight, offsets,
                       cursors, dense_off, g1_off, g2_off, mt_done, perm_done,
                       zc_tokens, out, sbuf, mt_tokens, topk_slot, counts, meta,
                       psum, pflag, local_off, rsf, stream,
                   )
               else:
                   if T_ >= 4096:
                       self.rk_big.launch(
                           logits, bias, hidden, abuf, hs_scale, topk_e,
                           topk_w, counts, offsets, cursors, g1_off, g2_off,
                           dense_off, mt_done, perm_done, row_token,
                           row_weight, perm_amax, zc_tokens, mt_tokens,
                           tok_cnt, topk_slot, meta, w13t, pflag, local_off,
                           rsf, cutlass.Int32(1), cutlass.Int32(0),
                           cutlass.Int32(0), pf_bytes, stream,
                       )
                   else:
                       self.rk.launch(
                           logits, bias, hidden, abuf, hs_scale, topk_e,
                           topk_w, counts, offsets, cursors, g1_off, g2_off,
                           dense_off, mt_done, perm_done, row_token,
                           row_weight, perm_amax, zc_tokens, mt_tokens,
                           tok_cnt, topk_slot, meta, w13t, pflag, local_off,
                           rsf, cutlass.Int32(1), cutlass.Int32(0),
                           cutlass.Int32(0), pf_bytes, stream,
                       )
                   self.g12n.launch(
                       logits, bias, hidden, hs_scale, abuf, w13_nkl, w13m,
                       w2_nkl, w2m, w13t, w2t, cbuf_bf, cbuf, camax, cscale_row, perm_amax,
                       topk_e, topk_w, tok_cnt, row_token, row_weight, offsets,
                       cursors, dense_off, g1_off, g2_off, mt_done, perm_done,
                       zc_tokens, out, sbuf, mt_tokens, topk_slot, counts, meta,
                       psum, pflag, local_off, rsf, stream,
                   )


# merged-kernel dispatch thresholds (baked at compile time):
# T in [FUSE_LO, FUSE_HI) runs the fully fused kernel (routing prologue);
# other T < CUT runs routing kernel + merged GEMM; T >= CUT runs the
# three-kernel large path (legacy: the merged kernel + epilogue reds now
# win at every measured size, so the default cut covers all workloads).
G12_CUT = int(os.environ.get("MOE_G12_CUT", "33000"))
G12_FUSE_LO = int(os.environ.get("MOE_G12_FUSE_LO", "24"))
G12_FUSE_HI = int(os.environ.get("MOE_G12_FUSE_HI", "512"))
# [FUSE_HI, FUSE_HI2) runs the dense-P3 fused variant (expert-major permute
# with progressive perm_done release). Session 7 measured parity-to--5us vs
# the 2-kernel path at T=901; the pair-split epilogue flipped it: the promo
# warps' P3 share is no longer contended by the 4-stall epi chain, and the
# fused variant now wins by ~4us (250.4 vs 254.4 @901). Default covers 901.
G12_FUSE_HI2 = int(os.environ.get("MOE_G12_FUSE_HI2", "1024"))
# T < RK_TINY_HI uses the one-CTA routing kernel: -4.6us @T=1, -3.7 @T=7,
# but a single CTA starves P0/P3 parallelism above ~10 tokens (+3us @T=15)
RK_TINY_HI = int(os.environ.get("MOE_RK_TINY_HI", "12"))
# T >= PAIR_CUT uses the M-pair GEMM1 instance (rk_big do_pair=2). Measured
# +50us at 11948: pair items consume BOTH TMEM acc stages, so the epilogue
# release latency (accw ~11us/item) is fully exposed instead of hidden by the
# 2-stage ping-pong, and S=3 staging costs TMA depth. Even with a perfect
# ~4us release the per-tile ceiling (~24us) barely beats the single-item
# 25.5us. Kept for experiments; default OFF.
PAIR_CUT = int(os.environ.get("MOE_PAIR_CUT", "99999"))

# MB of W13t swept into L2 by the side-stream warmer while routing runs
WARM_MB = int(os.environ.get("MOE_WARM_MB", "0"))
# MB of W13t prefetched into L2 from inside the routing kernel's entry
# (2-kernel path, T >= 512). Measured and REJECTED at 96/128/256MB: at 901
# g12n got +2..9us slower (prefetch desyncs the paced DRAM streams / L2
# pollution) and at 11948 the P3 permute's ~164MB of traffic evicts the
# prefetched head anyway. Kept env-gated for experiments; default OFF.
RK_PF_MB = int(os.environ.get("MOE_RK_PF_MB", "0"))

_LAUNCHER = _Launcher()
_COMPILED = None
_STREAM2 = None
_WARM_COMPILED = None
_WARM_EV = None
_WS = {}
_PREP = {}


_MB2 = 1 << 21  # 2 MiB
_KBLK = int(os.environ.get("MOE_KBLK", "128"))


def _get_ws(T, device):
   """Workspace buffers carved from one fresh slab with a fixed layout.

   Placement matters: when the allocator lands `perm_amax` (read by GEMM1
   promotions on the critical path) near the grid-barrier/work-queue lines,
   GEMM1 slows ~1.5x. A dedicated slab makes placement history-independent:
   big TMA-streamed buffers 2MB-aligned up front, then an isolated control
   page for `meta`, cool metadata far behind it, and the row-scale vectors
   (perm_amax et al.) in their own 2MB region at the end.
   """
   ws = _WS.get(T)
   if ws is None:
       P_max = T * 8 + 32 * 256 + 256
       # with epilogue reds the multi-token staging buffer is never touched
       # (keep a stub so the kernel signature is unchanged); the legacy
       # combine path still needs the full P_max rows.
       s_rows = 8 if (_LAUNCHER.red_comb and G12_CUT > 32768) else P_max
       # gemm1 K-split partial slab (MOE_KSPLIT experiment): two 128KB f32
       # tiles per logical g1 item; only the tiny 2-kernel path (T <
       # FUSE_LO) can split, others get a stub. pflag is polled/released
       # cross-CTA: own 64KB-aligned region.
       ks_rows = 512 if (T < G12_FUSE_LO and _LAUNCHER.ksplit) else 8
       layout = [
           ("abuf", (P_max, HIDDEN), torch.float8_e4m3fn, _MB2),
           ("cbuf_bf", (P_max, INTER), torch.bfloat16, _MB2),
           ("cbuf", (P_max, INTER), torch.float8_e4m3fn, _MB2),
           ("sbuf", (s_rows, HIDDEN), torch.bfloat16, _MB2),
           ("camax", (KB2, P_max), torch.float32, _MB2),
           ("meta", (META_SIZE,), torch.int32, _MB2),
           ("gq", (74 * 32,), torch.int32, 65536),
           ("topk_e", (T, 8), torch.int32, 65536),
           ("topk_w", (T, 8), torch.float32, 512),
           ("counts", (32,), torch.int32, 512),
           ("offsets", (33,), torch.int32, 512),
           ("cursors", (33,), torch.int32, 512),
           ("g1_off", (33,), torch.int32, 512),
           ("g2_off", (33,), torch.int32, 512),
           ("dense_off", (33,), torch.int32, 512),
           ("mt_done", (P_max // 128 + 8,), torch.int32, 512),
           ("perm_done", (P_max // 128 + 8,), torch.int32, 512),
           ("row_token", (P_max,), torch.int32, 512),
           ("zc_tokens", (T,), torch.int32, 512),
           ("mt_tokens", (T,), torch.int32, 512),
           ("tok_cnt", (T,), torch.int32, 512),
           ("topk_slot", (T, 8), torch.int32, 512),
           ("perm_amax", (P_max,), torch.float32, _MB2),
           ("row_weight", (P_max,), torch.float32, 512),
           ("cscale_row", (P_max,), torch.float32, 512),
           # K-split state appended so every prior offset is untouched
           # (the layout above is placement-tuned; inserting the psum slab
           # mid-layout shifted the control lines and cost +8..13us)
           ("psum", (ks_rows * 65536,), torch.float32, _MB2),
           ("pflag", (512,), torch.int32, 65536),
       ]
       esz = {torch.int32: 4, torch.float32: 4,
              torch.float8_e4m3fn: 1, torch.bfloat16: 2}
       offs = {}
       off = 0
       for name, shape, dt, al in layout:
           off = (off + al - 1) // al * al
           offs[name] = off
           n = esz[dt]
           for s in shape:
               n *= s
           off += n
       slab = torch.zeros(off + _MB2, dtype=torch.uint8, device=device)
       shift = (-slab.data_ptr()) % _MB2
       ws = {"_slab": slab}
       for name, shape, dt, al in layout:
           o = shift + offs[name]
           n = esz[dt]
           for s in shape:
               n *= s
           ws[name] = slab[o:o + n].view(dt).view(shape)
       _WS[T] = ws
   return ws


@torch.no_grad()
def _prep_weights(w13, w13s, w2, w2s):
   """Requantize weights to one scale per 128-column block (whole-K).

   Cached on tensor identity; recomputed whenever the tensors are replaced
   or modified in place (data_ptr/_version change)."""
   key = (
       w13.data_ptr(), w13._version, w13s.data_ptr(), w13s._version,
       w2.data_ptr(), w2._version, w2s.data_ptr(), w2s._version,
   )
   ent = _PREP.get("cur")
   if ent is not None and ent[0] == key:
       return ent[1]
   m13 = w13s.amax(dim=2).contiguous()  # (32, 32)
   r13 = w13s / m13.unsqueeze(2)  # (32, 32, 56)
   w13p = torch.empty_like(w13)
   for e in range(32):
       wf = w13[e].float().view(32, 128, 56, 128)
       wf.mul_(r13[e].view(32, 1, 56, 1))
       w13p[e] = wf.view(2 * INTER, HIDDEN).to(torch.float8_e4m3fn)
   # interleave gate/up 128-row blocks: item n reads ONE contiguous
   # (256, K) box instead of two 2048-rows-apart halves
   w13p = (
       w13p.view(32, 2, 16, 128, HIDDEN)
       .permute(0, 2, 1, 3, 4)
       .reshape(32, 2 * INTER, HIDDEN)
       .contiguous()
   )
   m13 = m13.view(32, 2, 16).permute(0, 2, 1).reshape(32, 32).contiguous()
   m2 = w2s.amax(dim=2).contiguous()  # (32, 56)
   r2 = w2s / m2.unsqueeze(2)  # (32, 56, 16)
   w2p = torch.empty_like(w2)
   for e in range(32):
       wf = w2[e].float().view(56, 128, 16, 128)
       wf.mul_(r2[e].view(56, 1, 16, 1))
       w2p[e] = wf.view(HIDDEN, INTER).to(torch.float8_e4m3fn)
   # tile-contiguous copies for the merged kernel: each (256n, KBLKk) TMA
   # box is one contiguous block ([e][nb][kb][256][KBLK]), so weight
   # streaming becomes sequential DRAM bursts instead of 128B strips
   # strided by the K pitch (148 interleaved striders thrash row buffers)
   kbt = _KBLK
   w13t = (
       w13p.view(32, 16, 256, HIDDEN // kbt, kbt).permute(0, 1, 3, 2, 4)
       .contiguous().view(32 * 16 * (HIDDEN // kbt), 256, kbt)
   )
   w2t = (
       w2p.view(32, 28, 256, INTER // kbt, kbt).permute(0, 1, 3, 2, 4)
       .contiguous().view(32 * 28 * (INTER // kbt), 256, kbt)
   )
   out = (w13p, m13, w2p, m2, w13t, w2t)
   _PREP["cur"] = (key, out)
   return out


def _cvt(x, dyn=False):
   t = from_dlpack(x, assumed_align=16)
   if dyn:
       t = t.mark_layout_dynamic(leading_dim=x.dim() - 1)
   return t


@torch.no_grad()
def run(
   routing_logits: torch.Tensor,
   routing_bias: torch.Tensor,
   hidden_states: torch.Tensor,
   hidden_states_scale: torch.Tensor,
   gemm1_weights: torch.Tensor,
   gemm1_weights_scale: torch.Tensor,
   gemm2_weights: torch.Tensor,
   gemm2_weights_scale: torch.Tensor,
   local_expert_offset: int,
   routed_scaling_factor: float,
):
   global _COMPILED
   device = hidden_states.device
   T = routing_logits.shape[0]
   ws = _get_ws(T, device)
   out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=device)

   if isinstance(local_expert_offset, torch.Tensor):
       local_expert_offset = int(local_expert_offset.item())
   if isinstance(routed_scaling_factor, torch.Tensor):
       routed_scaling_factor = float(routed_scaling_factor.item())

   w13p, w13m, w2p, w2m, w13t, w2t = _prep_weights(
       gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale
   )

   stream = cutlass_torch.current_stream()
   global _STREAM2
   if _STREAM2 is None:
       _STREAM2 = torch.cuda.Stream(device)
   s2 = cutlass_torch.cuda.CUstream(_STREAM2.cuda_stream)
   args = (
       _cvt(routing_logits.contiguous(), dyn=True),
       _cvt(routing_bias.contiguous()),
       _cvt(hidden_states, dyn=True),
       _cvt(hidden_states_scale, dyn=True),
       _cvt(w13p.permute(1, 2, 0)),
       _cvt(w13m),
       _cvt(w2p.permute(1, 2, 0)),
       _cvt(w2m),
       _cvt(w13t),
       _cvt(w2t),
       _cvt(out, dyn=True),
       _cvt(ws["topk_e"], dyn=True),
       _cvt(ws["topk_w"], dyn=True),
       _cvt(ws["counts"]),
       _cvt(ws["offsets"]),
       _cvt(ws["cursors"]),
       _cvt(ws["g1_off"]),
       _cvt(ws["g2_off"]),
       _cvt(ws["dense_off"]),
       _cvt(ws["mt_done"], dyn=True),
       _cvt(ws["perm_done"], dyn=True),
       _cvt(ws["row_token"], dyn=True),
       _cvt(ws["row_weight"], dyn=True),
       _cvt(ws["perm_amax"], dyn=True),
       _cvt(ws["zc_tokens"], dyn=True),
       _cvt(ws["mt_tokens"], dyn=True),
       _cvt(ws["tok_cnt"], dyn=True),
       _cvt(ws["topk_slot"], dyn=True),
       _cvt(ws["abuf"], dyn=True),
       _cvt(ws["cbuf_bf"], dyn=True),
       _cvt(ws["cbuf"], dyn=True),
       _cvt(ws["camax"], dyn=True),
       _cvt(ws["cscale_row"], dyn=True),
       _cvt(ws["sbuf"], dyn=True),
       _cvt(ws["meta"]),
       _cvt(ws["gq"]),
       _cvt(ws["psum"], dyn=True),
       _cvt(ws["pflag"]),
       cutlass.Int32(local_expert_offset),
       cutlass.Float32(routed_scaling_factor),
       cutlass.Int32(WARM_MB * ((1 << 20) // 4)),
       cutlass.Int32(RK_PF_MB * (1 << 20) if T >= 512 else 0),
       stream,
       s2,
   )
   if _COMPILED is None:
       _COMPILED = cute.compile(_LAUNCHER.launch, *args)
   _COMPILED(*args)
   if WARM_MB > 0 and T >= 4096:
       # fire the L2 warmer AFTER this call's kernels on the side stream:
       # it executes in the inter-call gap (weights are identical across
       # calls) and pre-warms W13's head for the next call's GEMM1 phase.
       # Launched before routing it would residency-block the 384-thread
       # routing grid (no SM has register room for both) and delay it by
       # its whole duration.
       global _WARM_COMPILED, _WARM_EV
       if _WARM_EV is None:
           _WARM_EV = torch.cuda.Event()
       torch.cuda.current_stream().record_event(_WARM_EV)
       _STREAM2.wait_event(_WARM_EV)
       wargs = (
           _cvt(w13t), _cvt(ws["meta"]),
           cutlass.Int32(WARM_MB * ((1 << 20) // 4)), s2,
       )
       if _WARM_COMPILED is None:
           _WARM_COMPILED = cute.compile(_LAUNCHER.warm.launch, *wargs)
       _WARM_COMPILED(*wargs)
   return out
