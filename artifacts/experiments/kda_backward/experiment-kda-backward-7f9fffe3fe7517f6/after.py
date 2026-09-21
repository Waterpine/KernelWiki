"""KDA backward: warp-specialized TMA, sliced H96 epilogues, and G2 PDL."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
import cuda.bindings.driver as cuda_drv

import kda_cute as KC
from kda_cute import _ptr, BT, KD
import kda_g1
import kda_g2
import kda_g2_h96


PRIORITIZE_H64_VARLEN = True
COMPACT_FINAL_LOADS = True
FINAL_LOAD_BATCH = 8
G2_HOIST_BETA = True


class _C:
   g1 = {}
   g1a = {}
   g2 = {}
   fin = {}
   kg1 = {}
   kg2 = {}
   s1 = None
   s_hi = None
   ev_a = None
   ev_b = None


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, grad_out):
   dev = q.device
   T = q.shape[1]
   H = q.shape[2]
   stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

   # Prep+intra go out as one compiled launch packet so intra's host-side
   # allocation and argument packing never surfaces as a device idle gap.
   # Prep itself emits the chunk tables and cleared accumulators (each block
   # walks cu_seqlens), so no metadata kernel opens the timed span.
   bufs = KC.prepare_intra(q, k, g, beta, A_log, dt_bias, cu_seqlens,
                           v.reshape(-1, KD), float(scale))
   if _C.s1 is None:
       _C.s1 = torch.cuda.Stream()
       _C.s_hi = torch.cuda.Stream(priority=-1)
       _C.ev_a = torch.cuda.Event()
       _C.ev_b = torch.cuda.Event()
   bufs["dh"] = torch.empty(bufs["nt_up"] * H * KD, KD, device=dev, dtype=torch.bfloat16)
   bufs["dv2"] = torch.empty(bufs["nt_up"] * BT * H, KD, device=dev, dtype=torch.bfloat16)
   prioritize_fwd = H == 64 and (
       bufs["n_seq"] == 1 or
       (PRIORITIZE_H64_VARLEN and bufs["n_seq"] == 6))
   _C.ev_a.record()
   if prioritize_fwd:
       with torch.cuda.stream(_C.s_hi):
           _C.s_hi.wait_event(_C.ev_a)
           KC.run_fwdscan(bufs, initial_state, H, dev)
       KC.run_revscan(bufs, grad_out.reshape(-1, KD), H, dev)
   else:
       with torch.cuda.stream(_C.s1):
           _C.s1.wait_event(_C.ev_a)
           KC.run_revscan(bufs, grad_out.reshape(-1, KD), H, dev)
           _C.ev_b.record(_C.s1)
       KC.run_fwdscan(bufs, initial_state, H, dev)

   nt_up = bufs["nt_up"]
   rows = nt_up * BT * H
   dq_pre = torch.empty(rows, KD, device=dev, dtype=torch.bfloat16)
   daqk = torch.empty(rows, BT, device=dev, dtype=torch.bfloat16)
   stream_g1a = (
       cuda_drv.CUstream(_C.s_hi.cuda_stream) if prioritize_fwd else stream
   )
   g1a_args = (
       _ptr(grad_out, cutlass.BFloat16), _ptr(bufs["vnew"], cutlass.BFloat16),
       _ptr(bufs["h"], cutlass.BFloat16), _ptr(bufs["gc"], cutlass.Float16),
       _ptr(bufs["tbl"], cutlass.Int32),
       _ptr(dq_pre, cutlass.BFloat16), _ptr(daqk, cutlass.BFloat16),
       Int32(nt_up), Float32(scale), Int32(T), stream_g1a)
   if prioritize_fwd:
       with torch.cuda.stream(_C.s_hi):
           if H not in _C.g1a:
               # G1A's warp-1 TMA producer overlaps warp-0 TMEM allocation.
               _C.g1a[H] = cute.compile(kda_g1.KGrad1A(H).launch, *g1a_args)
           _C.g1a[H](*g1a_args)
           _C.ev_b.record(_C.s_hi)
   else:
       if H not in _C.g1a:
           # G1A's warp-1 TMA producer overlaps warp-0 TMEM allocation.
           _C.g1a[H] = cute.compile(kda_g1.KGrad1A(H).launch, *g1a_args)
       _C.g1a[H](*g1a_args)
   torch.cuda.current_stream().wait_event(_C.ev_b)

   dk_pre = torch.empty(rows, KD, device=dev, dtype=torch.bfloat16)
   # The public gate gradient is BF16; round the local per-token handoff once
   # while retaining FP32 accumulation in the final reverse scan.
   dg_pre = torch.empty(rows, KD, device=dev, dtype=torch.bfloat16)
   db_pre = torch.empty(rows, device=dev, dtype=torch.float32)
   dv_out = torch.empty_like(v)
   dakk = torch.empty(rows, BT, device=dev, dtype=torch.bfloat16)

   g1_args = (
       _ptr(grad_out, cutlass.BFloat16), _ptr(v, cutlass.BFloat16),
       _ptr(bufs["vnew"], cutlass.BFloat16), _ptr(bufs["dv2"], cutlass.BFloat16),
       _ptr(bufs["h"], cutlass.BFloat16), _ptr(bufs["dh"], cutlass.BFloat16),
       _ptr(bufs["qn"], cutlass.BFloat16), _ptr(bufs["kn"], cutlass.BFloat16),
       _ptr(bufs["gc"], cutlass.Float16), _ptr(bufs["bs"], cutlass.Float16),
       _ptr(bufs["nmat"], cutlass.BFloat16), _ptr(bufs["tbl"], cutlass.Int32),
       _ptr(dq_pre, cutlass.BFloat16), _ptr(dk_pre, cutlass.BFloat16),
       _ptr(dg_pre, cutlass.BFloat16), _ptr(db_pre, cutlass.Float32),
       _ptr(dv_out, cutlass.BFloat16), _ptr(daqk, cutlass.BFloat16),
       _ptr(dakk, cutlass.BFloat16),
       Int32(nt_up), Float32(scale), Int32(T), stream)
   if H not in _C.g1:
       _C.kg1[H] = kda_g1.KGrad1(H)
       _C.g1[H] = cute.compile(_C.kg1[H].launch, *g1_args)
   _C.g1[H](*g1_args)

   dq = torch.empty_like(q)
   dk = torch.empty_like(k)
   dg = torch.empty_like(g)
   dbeta = torch.empty_like(beta)

   g2_args = (
       _ptr(daqk, cutlass.BFloat16), _ptr(dakk, cutlass.BFloat16),
       _ptr(bufs["qn"], cutlass.BFloat16), _ptr(bufs["kn"], cutlass.BFloat16),
       _ptr(bufs["gc"], cutlass.Float16), _ptr(bufs["bs"], cutlass.Float16),
       _ptr(bufs["rq"], cutlass.Float32), _ptr(bufs["rk"], cutlass.Float32),
       _ptr(bufs["tbl"], cutlass.Int32),
       _ptr(dq_pre, cutlass.BFloat16), _ptr(dk_pre, cutlass.BFloat16),
       _ptr(dg_pre, cutlass.BFloat16), _ptr(db_pre, cutlass.Float32),
       _ptr(dq, cutlass.BFloat16), _ptr(dk, cutlass.BFloat16),
       _ptr(dbeta, cutlass.BFloat16),
       Int32(H), Int32(nt_up), stream)
   g2_key = (H, G2_HOIST_BETA)
   if g2_key not in _C.g2:
       # The 256-column staged G2 reaches two-CTA residency.  Its balanced
       # warpgroup epilogue is now faster for both benchmark head counts.
       _C.kg2[g2_key] = kda_g2_h96.KGrad2()
       g2_compile_args = (
           *g2_args[:-1], H == 96, G2_HOIST_BETA, g2_args[-1])
       _C.g2[g2_key] = cute.compile(
           _C.kg2[g2_key].launch, *g2_compile_args)
   _C.g2[g2_key](*g2_args)

   fin_args = (
       _ptr(dq_pre, cutlass.BFloat16), _ptr(dk_pre, cutlass.BFloat16),
       _ptr(dg_pre, cutlass.BFloat16), _ptr(db_pre, cutlass.Float32),
       _ptr(bufs["gc"], cutlass.Float16), _ptr(k, cutlass.BFloat16),
       _ptr(g, cutlass.BFloat16), _ptr(beta, cutlass.BFloat16),
       _ptr(bufs["qn"], cutlass.BFloat16), _ptr(bufs["kn"], cutlass.BFloat16),
       _ptr(bufs["rq"], cutlass.Float32), _ptr(bufs["rk"], cutlass.Float32),
       _ptr(bufs["bs"], cutlass.Float16), _ptr(A_log, cutlass.Float32),
       _ptr(dt_bias, cutlass.Float32), _ptr(bufs["tbl"], cutlass.Int32),
       _ptr(dq, cutlass.BFloat16), _ptr(dk, cutlass.BFloat16),
       _ptr(dg, cutlass.BFloat16), _ptr(dbeta, cutlass.BFloat16),
       _ptr(bufs["dA_log"], cutlass.Float32), _ptr(bufs["ddt"], cutlass.Float32),
       Int32(H), Int32(bufs["nt_up"]), stream)
   fin_key = (H, COMPACT_FINAL_LOADS, FINAL_LOAD_BATCH)
   if fin_key not in _C.fin:
       fin_compile_args = (
           *fin_args[:-1], COMPACT_FINAL_LOADS, FINAL_LOAD_BATCH,
           H == 96, fin_args[-1])
       _C.fin[fin_key] = cute.compile(kda_g2.launch_final, *fin_compile_args)

   _C.fin[fin_key](*fin_args)

   return dq, dk, dv_out, dg, dbeta, bufs["dA_log"], bufs["ddt"]


if __name__ == "__main__":
   pass
