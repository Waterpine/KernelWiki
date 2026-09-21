"""KDA (Kimi Delta Attention) training backward pass on Blackwell, CuTe-DSL.

Four kernels per call (see kda_cute.py):
  A  prepare (chunk-parallel):  l2norm, beta sigmoid, gate cumsum, Aqk/Akk,
     T_A = (I+Akk)^-1, w(-), u, qg(*scale), kg, dv_a, eglast
  B  forward state scan  ->  h (per-chunk state), v_new
  C  backward state scan ->  dh, dv_new           (independent of B)
  D  fused gradient assembly -> dq, dk, dv, dg, dbeta, dA_log, ddt_bias
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

import kda_cute  # noqa: E402

_compiled = {}
_dummy_cu = {}


def _av(x):  # (T, D, H) view of a [T, H, D] tensor
    return from_dlpack(x.permute(0, 2, 1), assumed_align=16)


def _bv(x):  # (D, T, H) view
    return from_dlpack(x.permute(2, 0, 1), assumed_align=16)


def _hv(x):  # (K, V, NT*H) view of [NT, H, K, V]
    return from_dlpack(x.view(-1, 128, 128).permute(1, 2, 0),
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

    if cu_seqlens is None:
        dk_ = _dummy_cu.get(device)
        if dk_ is None:
            dk_ = torch.zeros(2, dtype=torch.long, device=device)
            _dummy_cu[device] = dk_
        cu_t = dk_
    else:
        cu_t = cu_seqlens

    # intermediates
    w = torch.empty(T, H, 128, dtype=bf, device=device)
    u = torch.empty(T, H, 128, dtype=bf, device=device)
    qg = torch.empty(T, H, 128, dtype=bf, device=device)
    kg = torch.empty(T, H, 128, dtype=bf, device=device)
    dva = torch.empty(T, H, 128, dtype=bf, device=device)
    TA = torch.empty(ntub * 64, H, 64, dtype=bf, device=device)
    egl = torch.empty(ntub, H, 128, dtype=torch.float32, device=device)
    ht = torch.empty(ntub, H, 128, 128, dtype=bf, device=device)
    dht = torch.empty(ntub, H, 128, 128, dtype=bf, device=device)
    vn = torch.empty(T, H, 128, dtype=bf, device=device)
    dvn = torch.empty(T, H, 128, dtype=bf, device=device)

    dq = torch.empty(T, H, 128, dtype=bf, device=device)
    dk = torch.empty(T, H, 128, dtype=bf, device=device)
    dg = torch.empty(T, H, 128, dtype=bf, device=device)
    dv = torch.empty(T, H, 128, dtype=bf, device=device)
    db = torch.empty(T, H, dtype=bf, device=device)
    dA = torch.zeros(H, dtype=torch.float32, device=device)
    dbias = torch.zeros(H * 128, dtype=torch.float32, device=device)

    q0, k0, v0, g0, do0 = q[0], k[0], v[0], g[0], grad_out[0]
    beta0 = beta[0]

    argsA = (_av(q0), _av(k0), _av(g0), _fd(beta0, 2), _bv(v0), _bv(do0),
             _av(w), _av(u), _av(qg), _av(kg),
             _fd(TA.permute(0, 2, 1)), _av(dva), _fd(egl), _fd(cu_t, 8),
             _fd(A_log, 4), _fd(dt_bias, 4))
    argsB = (_av(w), _av(u), _bv(kg), _fd(egl), _fd(initial_state),
             _fd(ht), _av(vn), _fd(cu_t, 8))
    argsC = (_av(kg), _bv(qg), _bv(w), _bv(do0), _av(dva), _fd(egl),
             _fd(dht), _av(dvn), _fd(cu_t, 8))
    argsD = (_av(q0), _av(k0), _av(g0), _fd(beta0, 2), _av(v0), _av(do0),
             _av(vn), _av(dvn), _bv(dvn), _hv(ht), _hv(dht),
             _fd(TA.permute(0, 2, 1)),
             _av(dq), _av(dk), _av(dg), _av(dv), _fd(db, 2), _fd(dA, 4),
             _fd(dbias, 4), _fd(cu_t, 8), _fd(A_log, 4), _fd(dt_bias, 4))

    key = (H, num_seqs, T, float(scale))
    entry = _compiled.get(key)
    if entry is None:
        ka = kda_cute.KernelA(H, num_seqs, T, float(scale))
        kb = kda_cute.KernelB(H, num_seqs, T)
        kc = kda_cute.KernelC(H, num_seqs, T)
        kd = kda_cute.KernelD(H, num_seqs, T, float(scale))
        fa = cute.compile(ka.host, *argsA, ntub)
        fb = cute.compile(kb.host, *argsB)
        fc = cute.compile(kc.host, *argsC)
        fd = cute.compile(kd.host, *argsD, ntub)
        entry = (fa, fb, fc, fd)
        _compiled[key] = entry

    fa, fb, fc, fd = entry
    fa(*argsA)
    fb(*argsB)
    fc(*argsC)
    fd(*argsD)

    return (dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0),
            dg.unsqueeze(0), db.unsqueeze(0), dA, dbias)
