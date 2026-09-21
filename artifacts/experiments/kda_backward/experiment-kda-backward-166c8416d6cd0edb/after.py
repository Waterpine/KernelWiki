"""KDA (Kimi Delta Attention) training backward pass on Blackwell, CuTe-DSL.

Kernels per call (see kda_cute.py):
 P  elementwise prepare (chunk-parallel): l2norm, beta sigmoid, gate
    cumsum/Jacobian, gate factors, operand tiles ep/b1/aq2/ak2/kbg
 A  chunk matrices: Aqk/Akk, T_A = (I+Akk)^-1, w, u, dv_a, and the fused
    gated-operand dumps qg/kg/kge/akb from its resident tiles
 B  forward state scan  ->  h (per-chunk state), v_new
 C  backward state scan ->  dh, dv_new           (independent of B)
 Q  (H64) early dqi contraction overlapped with the scan tail
 D  matrix backward -> dv, raw fragment tiles dqt/dkI/dkT, hdh, db_v
 F  gradient assembly -> dq, dk, dg, dbeta (+ per-chunk parameter partials)
 E  reduce per-chunk dA_log/ddt_bias partials
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
   "CUTE_DSL_CACHE_DIR",
   str(Path(__file__).resolve().parent / ".cute_cache"))

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
from cuda.bindings import driver as cuda_driver  # noqa: E402

import kda_cute  # noqa: E402

_compiled = {}
_dummy_cu = {}
_scan_streams = {}
_prep_streams = {}
_workspaces = {}


def _get_workspace(device, H, T):
    """Persistent scratch tensors and their static dlpack views.

    Every intermediate is fully overwritten before it is read on each call,
    so reusing one workspace per (device, H, T) is safe and removes ~30
    allocations plus ~30 from_dlpack view constructions from the critical
    host path in front of Kernel P (the GPU idles behind that host work).
    """
    key = (device, H, T)
    ws = _workspaces.get(key)
    if ws is not None:
        return ws
    bf = torch.bfloat16
    f32 = torch.float32
    ntub_max = (T + 63) // 64 + kda_cute.MAXS - 1

    def big(dtype=bf):
        return torch.empty(T, H, 128, dtype=dtype, device=device)

    ws = {}
    t = ws["t"] = {
        "w": big(), "u": big(), "qg": big(), "kg": big(), "dva": big(),
        "ep": big(), "gate_jac": big(), "kge": big(), "b1g": big(),
        "aq2g": big(), "ak2g": big(), "akbg": big(), "kbgg": big(),
        "dqi": big(), "ia": big(), "dki": big(), "ic": big(), "ib": big(),
        "vn": big(), "dvn": big(),
        "TA": torch.empty(ntub_max * 64, H, 64, dtype=bf, device=device),
        "egl": torch.empty(ntub_max, H, 128, dtype=f32, device=device),
        "gref": torch.empty(ntub_max, H, 4, 128, dtype=f32, device=device),
        "rstd": torch.empty(H, T, 3, dtype=f32, device=device),
        "ht": torch.empty(ntub_max, H, 128, 128, dtype=bf, device=device),
        "dht": torch.empty(ntub_max, H, 128, 128, dtype=bf, device=device),
        "hdhv": torch.empty(ntub_max, H, 128, dtype=f32, device=device),
        "dbv": torch.empty(ntub_max, H, 64, dtype=f32, device=device),
        "red": torch.empty(2, ntub_max, H, 128, dtype=f32, device=device),
        "chunk_meta": torch.empty(ntub_max, 2, dtype=torch.int32,
                                  device=device),
        "seq_meta": torch.empty(kda_cute.MAXS, 5, dtype=torch.int32,
                                device=device),
        "full_meta": torch.empty(ntub_max, 2, dtype=torch.int32,
                                 device=device),
        "tail_meta": torch.empty(kda_cute.MAXS, 3, dtype=torch.int32,
                                 device=device),
    }
    v = ws["v"] = {}
    for n in ("w", "u", "qg", "kg", "dva", "ep", "gate_jac", "kge", "b1g",
              "aq2g", "ak2g", "akbg", "kbgg", "dqi", "ia", "dki", "ic",
              "ib", "vn", "dvn"):
        v[n + "_a"] = _av(t[n])
        v[n + "_b"] = _bv(t[n])
    v["TA_f"] = _fd(t["TA"].permute(0, 2, 1))
    v["egl_f"] = _fd(t["egl"])
    v["gref_v"] = _fd(t["gref"].permute(0, 2, 3, 1))
    v["rstd_v"] = _fd(t["rstd"].permute(1, 2, 0))
    v["ht_vh"] = _vh(t["ht"])
    v["ht_hv"] = _hv(t["ht"])
    v["dht_vh"] = _vh(t["dht"])
    v["dht_hv"] = _hv(t["dht"])
    v["hdhv_f"] = _fd(t["hdhv"])
    v["dbv_f"] = _fd(t["dbv"])
    v["red_f"] = _fd(t["red"])
    v["chunk_meta_v"] = _fd(t["chunk_meta"], 4)
    v["seq_meta_v"] = _fd(t["seq_meta"], 4)
    v["full_meta_v"] = _fd(t["full_meta"], 4)
    v["tail_meta_v"] = _fd(t["tail_meta"], 4)
    _workspaces[key] = ws
    return ws


def _get_scan_stream(device):
   entry = _scan_streams.get(device)
   if entry is None:
       # Higher stream priority lets the work distributor interleave this
       # grid's CTAs with the caller-stream grid instead of admitting them
       # only after the caller grid's queue drains.
       entry = (torch.cuda.Stream(device=device, priority=-1),
                torch.cuda.Event())
       _scan_streams[device] = entry
   return entry


def _get_prep_stream(device):
   entry = _prep_streams.get(device)
   if entry is None:
       entry = (torch.cuda.Stream(device=device), torch.cuda.Event())
       _prep_streams[device] = entry
   return entry


def _av(x):  # (T, D, H) view of a [T, H, D] tensor
   return from_dlpack(x.permute(0, 2, 1), assumed_align=16)


def _bv(x):  # (D, T, H) view
   return from_dlpack(x.permute(2, 0, 1), assumed_align=16)


def _hv(x):  # (K, V, NT*H) view of [NT, H, K, V]
   return from_dlpack(x.view(-1, 128, 128).permute(1, 2, 0),
                      assumed_align=16)


def _vh(x):  # (V, K, NT*H), convenient for scan-state TMA stores
   return from_dlpack(x.view(-1, 128, 128).permute(2, 1, 0),
                      assumed_align=16)


def _fd(x, align=16):
   return from_dlpack(x, assumed_align=align)


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens,
       grad_out):
   B, T, H, Dh = q.shape
   assert Dh == 128
   device = q.device
   bf = torch.bfloat16
   num_seqs = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
   ntub = (T + 63) // 64 + (num_seqs - 1)
   # All chunk-indexed tensors use this workload-independent bound so that
   # compiled entries (keyed on (H, T, scale)) see identical shapes on every
   # call.  from_dlpack layouts are static: TMA descriptors bake the gmem
   # box at compile time, and a smaller first-call ntub would silently
   # zero-fill chunks gc >= that bound on later varlen calls.
   ntub_max = (T + 63) // 64 + kda_cute.MAXS - 1

   if cu_seqlens is None:
       dk_ = _dummy_cu.get((device, T))
       if dk_ is None:
           dk_ = torch.tensor([0, T], dtype=torch.long, device=device)
           _dummy_cu[(device, T)] = dk_
       cu_t = dk_
   else:
       cu_t = cu_seqlens

   # intermediates
   w = torch.empty(T, H, 128, dtype=bf, device=device)
   u = torch.empty(T, H, 128, dtype=bf, device=device)
   qg = torch.empty(T, H, 128, dtype=bf, device=device)
   kg = torch.empty(T, H, 128, dtype=bf, device=device)
   dva = torch.empty(T, H, 128, dtype=bf, device=device)
   TA = torch.empty(ntub_max * 64, H, 64, dtype=bf, device=device)
   egl = torch.empty(ntub_max, H, 128, dtype=torch.float32, device=device)
   ep = torch.empty(T, H, 128, dtype=bf, device=device)
   # Reused half-chunk gate factors (exp(G16), exp(G48-G16), and the two
   # suffix factors); physical channel contiguity makes each slot coalesced.
   gref = torch.empty(ntub_max, H, 4, 128, dtype=torch.float32,
                      device=device)
   gref_v = _fd(gref.permute(0, 2, 3, 1))
   gate_jac = torch.empty(T, H, 128, dtype=bf, device=device)
   # Physical [H,T,3] keeps reciprocal q/k norms and sigmoid(beta) for
   # adjacent tokens contiguous.  Kernels see the logical [T,3,H] view.
   rstd = torch.empty(H, T, 3, dtype=torch.float32, device=device)
   rstd_v = _fd(rstd.permute(1, 2, 0))
   ht = torch.empty(ntub_max, H, 128, 128, dtype=bf, device=device)
   dht = torch.empty(ntub_max, H, 128, 128, dtype=bf, device=device)
   vn = torch.empty(T, H, 128, dtype=bf, device=device)
   dvn = torch.empty(T, H, 128, dtype=bf, device=device)
   # Head-independent packed mapping produced once, then consumed by every
   # chunk-parallel kernel instead of repeating the MAXS sequence search.
   chunk_meta = torch.empty(ntub_max, 2, dtype=torch.int32, device=device)
   seq_meta = torch.empty(kda_cute.MAXS, 4, dtype=torch.int32, device=device)
   if cu_seqlens is None:
       # These views are never compiled or launched for the fixed variant;
       # aliases avoid adding allocator traffic to that accepted fast path.
       full_chunk_meta = chunk_meta
       tail_chunk_meta = chunk_meta
   else:
       full_chunk_meta = torch.empty(ntub_max, 2, dtype=torch.int32,
                                     device=device)
       tail_chunk_meta = torch.empty(kda_cute.MAXS, 3, dtype=torch.int32,
                                     device=device)

   kge = torch.empty(T, H, 128, dtype=bf, device=device)
   b1g = torch.empty(T, H, 128, dtype=bf, device=device)
   aq2g = torch.empty(T, H, 128, dtype=bf, device=device)
   ak2g = torch.empty(T, H, 128, dtype=bf, device=device)
   akbg = torch.empty(T, H, 128, dtype=bf, device=device)
   kbgg = torch.empty(T, H, 128, dtype=bf, device=device)
   dqi = torch.empty(T, H, 128, dtype=bf, device=device)
   ia_ = torch.empty(T, H, 128, dtype=bf, device=device)
   dki = torch.empty(T, H, 128, dtype=bf, device=device)
   ic_ = torch.empty(T, H, 128, dtype=bf, device=device)
   ib_ = torch.empty(T, H, 128, dtype=bf, device=device)
   hdhv = torch.empty(ntub_max, H, 128, dtype=torch.float32, device=device)
   dbv = torch.empty(ntub_max, H, 64, dtype=torch.float32, device=device)

   dq = torch.empty(T, H, 128, dtype=bf, device=device)
   dk = torch.empty(T, H, 128, dtype=bf, device=device)
   dg = torch.empty(T, H, 128, dtype=bf, device=device)
   dv = torch.empty(T, H, 128, dtype=bf, device=device)
   db = torch.empty(T, H, dtype=bf, device=device)
   dA = torch.empty(H, dtype=torch.float32, device=device)
   dbias = torch.empty(H * 128, dtype=torch.float32, device=device)
   red = torch.empty(2, ntub_max, H, 128,
                     dtype=torch.float32, device=device)
   dbg = _dummy_cu.get((device, "dbg"))
   if dbg is None:
       dbg = torch.zeros(4, 1, 1, 1, dtype=torch.float32, device=device)
       _dummy_cu[(device, "dbg")] = dbg

   q0, k0, v0, g0, do0 = q[0], k[0], v[0], g[0], grad_out[0]
   beta0 = beta[0]

   chunk_meta_v = _fd(chunk_meta, 4)
   seq_meta_v = _fd(seq_meta, 4)
   full_chunk_meta_v = _fd(full_chunk_meta, 4)
   tail_chunk_meta_v = _fd(tail_chunk_meta, 4)
   argsM = (_fd(cu_t, 8), chunk_meta_v, seq_meta_v,
            full_chunk_meta_v, tail_chunk_meta_v)
   argsP = (_av(q0), _av(k0), _av(g0), _fd(beta0, 2),
            _av(ep), _av(gate_jac),
            _av(b1g), _av(aq2g), _av(ak2g),
            _av(kbgg), _fd(egl), gref_v, rstd_v, chunk_meta_v,
            _fd(A_log, 4), _fd(dt_bias, 4))
   argsA = (_av(aq2g), _av(ak2g), _av(b1g), _bv(kbgg), _bv(v0), _bv(do0),
            _bv(w), _bv(u),
            _fd(TA.permute(0, 2, 1)), _bv(dva),
            _av(qg), _av(kg), _av(kge), _av(akbg), gref_v,
            rstd_v, chunk_meta_v)
   argsA_full = argsA[:-1] + (full_chunk_meta_v,)
   argsA_tail = argsA[:-1] + (tail_chunk_meta_v,)
   argsB = (_av(w), _av(u), _bv(kg), _fd(egl), _fd(initial_state),
            _vh(ht), _bv(vn), seq_meta_v)
   argsC = (_av(kg), _bv(qg), _bv(w), _bv(do0), _av(dva), _fd(egl),
            _vh(dht), _bv(dvn), seq_meta_v)
   argsQ = (_av(do0), _hv(ht), _av(dqi), chunk_meta_v)
   argsD = (_av(kge), _bv(b1g), _bv(aq2g), _bv(akbg),
            _fd(beta0, 2), _av(v0), _av(do0),
            _av(vn), _av(dvn), _bv(dvn), _hv(ht), _hv(dht),
            _fd(TA.permute(0, 2, 1)),
            _av(dqi), _av(dki), _av(ia_), _av(ib_), _av(ic_),
            _av(dv), _fd(hdhv), _fd(dbv),
            gref_v, rstd_v,
            chunk_meta_v)
   argsD_full = argsD[:-1] + (full_chunk_meta_v,)
   argsD_tail = argsD[:-1] + (tail_chunk_meta_v,)
   # Reconstruct q-hat/k-hat in F from the prepared aq2=q-hat*ep and
   # b1=k-hat/ep operands.  D has just consumed both buffers, whereas the
   # original q/k tensors have been cold since P.
   argsF = (_av(aq2g), _av(b1g), _av(ep), _av(g0), _av(gate_jac),
            _av(dqi), _av(ia_), _av(dki), _av(ic_), _av(ib_),
            gref_v, _fd(egl), _fd(hdhv), _fd(dbv), rstd_v,
            _av(dq), _av(dk), _av(dg), _fd(db, 2), _fd(red),
            _fd(cu_t, 8), _fd(dt_bias, 4))
   argsF_full = argsF[:-2] + (full_chunk_meta_v, argsF[-1])
   argsF_tail = argsF[:-2] + (tail_chunk_meta_v, argsF[-1])
   argsE = (_fd(red), _fd(dA, 4), _fd(dbias, 4))

   early_dqi = H == 64
   fixed_seq = cu_seqlens is None
   # Split every packed-varlen queue into uniform full/tail launches so the
   # compiler can erase per-CTA full-chunk branching, including H=64.
   queue_a = not fixed_seq
   overlap_tail = not fixed_seq and num_seqs == 6
   uniform_seq = not fixed_seq and num_seqs == kda_cute.MAXS
   metadata_needed = not fixed_seq and not uniform_seq
   key = (H, T, float(scale), fixed_seq, uniform_seq)
   entry = _compiled.get(key)
   if entry is None:
       km = kda_cute.KernelM()
       kp = kda_cute.KernelP(H, T, float(scale), fixed_seq=fixed_seq,
                             uniform_seq=uniform_seq)
       ka = kda_cute.KernelA(H, T, float(scale), fixed_seq=fixed_seq,
                             uniform_seq=uniform_seq)
       ka_full = (kda_cute.KernelA(H, T, float(scale), work_kind=1)
                  if queue_a and not uniform_seq else
                  kda_cute.KernelA(H, T, float(scale), fixed_seq=True,
                                   uniform_seq=True, work_kind=1)
                  if uniform_seq else None)
       ka_tail = (kda_cute.KernelA(H, T, float(scale), work_kind=2)
                  if queue_a and not uniform_seq else None)
       kb = kda_cute.KernelB(H, T, fixed_seq=fixed_seq,
                             uniform_seq=uniform_seq)
       kc = kda_cute.KernelC(H, T, fixed_seq=fixed_seq,
                             uniform_seq=uniform_seq)
       kq = kda_cute.KernelQ(H, T,
                             fixed_seq=fixed_seq or uniform_seq,
                             uniform_seq=uniform_seq)
       kd = kda_cute.KernelD(H, T, float(scale),
                             external_dqi=early_dqi,
                             fixed_seq=fixed_seq)
       kd_full = (kda_cute.KernelD(H, T, float(scale),
                                   external_dqi=early_dqi,
                                   uniform_seq=uniform_seq, work_kind=1)
                  if not fixed_seq else None)
       kd_tail = (kda_cute.KernelD(H, T, float(scale),
                                   external_dqi=early_dqi, work_kind=2)
                  if not fixed_seq and not uniform_seq else None)
       kf = kda_cute.KernelF(H, T, float(scale), fixed_seq=fixed_seq)
       kf_full = (kda_cute.KernelF(H, T, float(scale),
                                   uniform_seq=uniform_seq, work_kind=1)
                  if not fixed_seq else None)
       kf_tail = (kda_cute.KernelF(H, T, float(scale), work_kind=2)
                  if not fixed_seq and not uniform_seq else None)
       ke = kda_cute.KernelE(H)
       cur0 = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
       fm = (cute.compile(km.host, *argsM, num_seqs, ntub, cur0)
             if metadata_needed else None)
       fp = cute.compile(kp.host, *argsP, num_seqs, ntub, cur0)
       fa = (cute.compile(ka.host, *argsA, num_seqs, ntub, cur0)
             if not queue_a else
             cute.compile(ka_full.host, *argsA_full,
                          num_seqs, ntub, cur0))
       fa_tail = (cute.compile(ka_tail.host, *argsA_tail,
                               num_seqs, ntub, cur0)
                  if queue_a and not uniform_seq else None)
       fb = cute.compile(kb.host, *argsB, num_seqs, cur0)
       fc = cute.compile(kc.host, *argsC, num_seqs, cur0)
       fq = (cute.compile(kq.host, *argsQ, num_seqs, ntub, cur0)
             if early_dqi else None)
       fd = (cute.compile(kd.host, *argsD, num_seqs, ntub, cur0)
             if fixed_seq else
             cute.compile(kd_full.host, *argsD_full,
                          num_seqs, ntub, cur0))
       fd_tail = (cute.compile(kd_tail.host, *argsD_tail,
                               num_seqs, ntub, cur0)
                  if not fixed_seq and not uniform_seq else None)
       ff = (cute.compile(kf.host, *argsF, num_seqs, ntub, cur0)
             if fixed_seq else
             cute.compile(kf_full.host, *argsF_full,
                          num_seqs, ntub, cur0))
       ff_tail = (cute.compile(kf_tail.host, *argsF_tail,
                               num_seqs, ntub, cur0)
                  if not fixed_seq and not uniform_seq else None)
       fe = cute.compile(ke.host, *argsE, ntub, cur0)
       entry = (fm, fp, fa, fa_tail, fb, fc, fq,
                fd, fd_tail, ff, ff_tail, fe)
       _compiled[key] = entry

   fm, fp, fa, fa_tail, fb, fc, fq, fd, fd_tail, ff, ff_tail, fe = entry
   cur_torch = torch.cuda.current_stream(device)
   cur = cuda_driver.CUstream(cur_torch.cuda_stream)
   if metadata_needed:
       fm(*argsM, num_seqs, ntub, cur)
   fp(*argsP, num_seqs, ntub, cur)

   if not queue_a:
       fa(*argsA, num_seqs, ntub, cur)
   else:
       fa(*argsA_full, num_seqs, ntub, cur)
       if not uniform_seq:
           fa_tail(*argsA_tail, num_seqs, ntub, cur)

   # B and C are independent given A's outputs; run them concurrently so
   # the sequential-scan latency of one hides under the other.  The work
   # distributor admits the caller-stream grid first (it sits directly
   # behind A; the side stream joins through an event), so the longer
   # chain goes on the caller stream: C alone for H96, B followed by the
   # dependent early-dqi contraction for H64.
   scan_stream, scan_done = _get_scan_stream(device)
   scan_driver = cuda_driver.CUstream(scan_stream.cuda_stream)
   scan_stream.wait_stream(cur_torch)
   if H == 96:
       fb(*argsB, num_seqs, scan_driver)
       fc(*argsC, num_seqs, cur)
   else:
       fc(*argsC, num_seqs, scan_driver)
       fb(*argsB, num_seqs, cur)
       fq(*argsQ, num_seqs, ntub, cur)
   scan_done.record(scan_stream)
   cur_torch.wait_event(scan_done)

   if fixed_seq:
       fd(*argsD, num_seqs, ntub, cur)
       ff(*argsF, num_seqs, ntub, cur)
   elif overlap_tail:
       # Full and partial-tail chunks are disjoint dependency chains.  Keep
       # each D->F pair ordered, while allowing the compact tail F launch to
       # use execution slots left by the tensor-heavy full D launch.
       tail_stream, tail_done = _get_prep_stream(device)
       tail_driver = cuda_driver.CUstream(tail_stream.cuda_stream)
       tail_stream.wait_stream(cur_torch)
       fd(*argsD_full, num_seqs, ntub, cur)
       fd_tail(*argsD_tail, num_seqs, ntub, tail_driver)
       ff(*argsF_full, num_seqs, ntub, cur)
       ff_tail(*argsF_tail, num_seqs, ntub, tail_driver)
       tail_done.record(tail_stream)
       cur_torch.wait_event(tail_done)
   else:
       fd(*argsD_full, num_seqs, ntub, cur)
       if not uniform_seq:
           fd_tail(*argsD_tail, num_seqs, ntub, cur)
       ff(*argsF_full, num_seqs, ntub, cur)
       if not uniform_seq:
           ff_tail(*argsF_tail, num_seqs, ntub, cur)
   reduce_nt = T // kda_cute.BT if uniform_seq else ntub
   fe(*argsE, reduce_nt, cur)

   return (dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0),
           dg.unsqueeze(0), db.unsqueeze(0), dA, dbias)
