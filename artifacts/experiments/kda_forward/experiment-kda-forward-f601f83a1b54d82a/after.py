"""KDA forward candidate: pkda (CuTe-DSL m64/m128 champion) adapter."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pkda  # noqa: E402


@torch.no_grad()
def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    out = torch.empty_like(v)
    pkda.fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        A_log,
        dt_bias.reshape(q.shape[2], 128),
        -5.0,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        write_final=False,
        task_specialize=True,
        tma_beta=True,
        tma_state=True,
        fp16_gate=True,
        cache_dtb=True,
        cache_rf=4,
        solve_ldsm=True,
        decay_96=True,
        decay_128=True,
        v_fold=True,
        early_v=True,
        ptree_no_handoff=True,
        joint_norm=3,
        gram_tc=True,
        inp_pipe=True,
        regs=True,
        kd_fold=True,
        walk2=3,
    )
    return out
