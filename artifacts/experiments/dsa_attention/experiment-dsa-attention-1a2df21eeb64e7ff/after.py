"""Shape-only dispatcher for the B300 DSA sparse MLA decode kernels.

Each process imports exactly one implementation body for its static token
count.  The cached callable therefore depends only on shape; all implementation
bodies rebuild the views of current-call inputs and outputs on every call. This
also keeps fresh CuTe materialization within the conductor's per-workload bound;
only the T=1, T=2, or combined long body needed by that shape is imported.
The long body additionally keys T=6 compilation on ``num_pages`` so its
flattened-cache bound can be folded without retaining any input-derived data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_RUNNERS = {}


def _load_sibling_run(stem):
    path = Path(__file__).resolve().with_name(f"{stem}.py")
    module_name = f"_dsa_shape_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DSA shape implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens = int(q_nope.shape[0])
    if num_tokens == 1:
        stem = "kernel_t1"
    elif num_tokens == 2:
        stem = "kernel_t2"
    else:
        stem = "kernel_long"

    selected = _RUNNERS.get(stem)
    if selected is None:
        selected = _load_sibling_run(stem)
        _RUNNERS[stem] = selected
    return selected(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
