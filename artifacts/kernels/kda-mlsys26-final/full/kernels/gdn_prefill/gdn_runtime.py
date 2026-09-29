"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Net Chunked Prefill - Blackwell SM100 Adapter
==========================================================

Bridges FlashInfer's PyTorch-based ``chunk_gated_delta_rule()`` API to the
CuTe DSL chunked GDN kernel for SM100 (Blackwell).

Follows the same compile-once-cache-and-replay pattern used by the decode
kernels in ``gdn_decode_pretranspose.py``.

State layout: ``[N, H, V, K]``.
"""

import functools
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from gdn_core import GatedDeltaNetChunkedKernel


@functools.cache
def _get_num_sm(device_index: int) -> int:
    """Cache the one device property needed by the persistent scheduler."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    value_split: int,
    adaptive_b32_sequence_order: bool,
    zigzag_sequence_order: bool,
    sequence_order_rotation: int,
    warp_local_inverse: bool,
    is_persistent: bool,
    fixed_num_chunks: int,
    single_sequence: bool,
    fast_sigmoid: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(
            f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16"
        )


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported state dtype {torch_dtype}, expected float32 or bfloat16"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_checkpoints: Optional[torch.Tensor] = None,
    fast_sigmoid: Optional[bool] = None,
) -> None:
    """Execute the Blackwell chunked GDN prefill kernel.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        A_log: ``(HO,)`` float32, log decay coefficient
        a: ``(total_tokens, HO)`` bfloat16, decay input
        dt_bias: ``(HO,)`` float32, per-head decay bias
        b: ``(total_tokens, HO)`` bfloat16, update-gate input
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int64
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: store intermediate state every N tokens (0 = disabled)
        cu_checkpoints: ``(num_seqs + 1,)`` int32, cumulative checkpoint counts
        output_checkpoints: ``(total_checkpoints, HO, DK, DK)`` float32/bfloat16, or None
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    B = cu_seqlens.size(0) - 1
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sm = _get_num_sm(device_index)
    # One CTA owns a recurrent (sequence, value-head) stream.  With at most
    # 13 sequences, the native eight heads leave B300's 148 SMs underfilled.
    # Split the independent 128-row value/state dimension into two 64-row
    # virtual heads; beyond this crossover duplicated Q/K work outweighs the
    # extra parallelism.
    value_split = 2 if not is_GQA and B <= 13 else 1
    adaptive_b32_sequence_order = B == 32
    zigzag_sequence_order = 32 <= B <= 48 and B not in (34, 35)
    # These captured schedules benefit from assigning each inverse warp the
    # two 8x8 blocks consumed by its following 16x16 correction. This removes
    # the intervening CTA barrier; other B values retain the lower-divergence
    # two-warp inversion path.
    warp_local_inverse = B in (34, 35, 39, 56, 57)
    if B == 32:
        sequence_order_rotation = B - 2
    elif B == 34:
        sequence_order_rotation = 7
    elif B == 35:
        sequence_order_rotation = 2
    elif B in (37, 48):
        sequence_order_rotation = 3
    elif B == 43:
        sequence_order_rotation = 14
    elif B == 57:
        sequence_order_rotation = 7
    elif B == 56:
        sequence_order_rotation = 11
    else:
        sequence_order_rotation = 0
    # The sole sequence spans the full dynamic token dimension, so its CuTe
    # TMA atoms are already bounded and can be used without mutable maps.
    single_sequence = B == 1
    # Remove persistent-scheduler work whenever every recurrent virtual-head
    # stream fits in one resident wave.  Counting the split value heads keeps
    # this rule transferable across batch sizes and GPU SM counts.
    is_persistent = B * HV * value_split > num_sm
    # Coarse token-count buckets specialize loop control while keeping every
    # tensor extent dynamic; they are shared by all shapes in each size class.
    total_tokens = q.size(0)
    # The hardware tanh identity saves gate-warp instructions for short
    # prefills, while long-running streams favor the original exp/reciprocal
    # sequence.  Keep this a broad latency/throughput size class.
    if fast_sigmoid is None:
        # The tanh identity is decisive for the underfilled single-sequence
        # direct grid and remains slightly favorable once the persistent
        # scheduler is active.  Direct N=2..4 grids have enough independent
        # gate warps that exp/reciprocal overlaps better with the main loop.
        fast_sigmoid = total_tokens < 4096 and (single_sequence or B > 4)
    if total_tokens <= 64:
        fixed_num_chunks = 1
    elif single_sequence and total_tokens <= 128:
        fixed_num_chunks = 2
    elif single_sequence and total_tokens <= 192:
        fixed_num_chunks = 3
    else:
        fixed_num_chunks = 0
    _output_state = output_state if store_final_state else None

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        value_split,
        adaptive_b32_sequence_order,
        zigzag_sequence_order,
        sequence_order_rotation,
        warp_local_inverse,
        is_persistent,
        fixed_num_chunks,
        single_sequence,
        fast_sigmoid,
    )

    if "compiled" not in cache:
        # --- First call: compile the kernel ---
        max_active_clusters = num_sm

        value_tile = 128 // value_split
        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(value_tile, 64, 128),
            mma_tiler_qkv=(value_tile, 64, 64),
            mma_tiler_kv=(value_tile, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            enable_checkpoints=enable_checkpoints,
            # Direct N<=4 grids contain at most 64 virtual-head streams, so
            # persistence cannot redistribute a second tile and only adds
            # scheduler divmod/advance control.
            is_persistent=is_persistent,
            value_split=value_split,
            # The full-token Q/K/V descriptors are safe because beta=0
            # neutralizes every padded tail row; O remains sequence-bounded.
            embedded_qkv=True,
            adaptive_b32_sequence_order=adaptive_b32_sequence_order,
            zigzag_sequence_order=zigzag_sequence_order,
            sequence_order_rotation=sequence_order_rotation,
            warp_local_inverse=warp_local_inverse,
            fixed_num_chunks=fixed_num_chunks,
            single_sequence=single_sequence,
            fast_sigmoid=fast_sigmoid,
        )

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        A_log_cute = from_dlpack(A_log, assumed_align=16)
        a_cute = from_dlpack(a, assumed_align=16)
        a_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16)
        b_cute = from_dlpack(b, assumed_align=16)
        b_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()

        # CUDA allocation alignment plus the 512-byte contiguous K-row stride
        # make 32-byte vector accesses valid for both state tensors.
        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(_initial_state, assumed_align=32)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(_output_state, assumed_align=32)
            s_out_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_checkpoints_cute = None
        cu_checkpoints_cute = None
        if enable_checkpoints:
            s_checkpoints_cute = from_dlpack(output_checkpoints, assumed_align=16)
            s_checkpoints_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
            cu_checkpoints_cute = from_dlpack(
                cu_checkpoints, assumed_align=4
            ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, not single_sequence
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            A_log_cute,
            a_cute,
            dt_bias_cute,
            b_cute,
            o_cute,
            cu_seqlens_cute,
            s_in_cute,
            s_out_cute,
            s_checkpoints_cute,
            cu_checkpoints_cute,
            checkpoint_every_n_tokens,
            scale,
            workspace_cute,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, B, HQ, HV, not single_sequence
    )
    ws_key = f"workspace_{q.device.index}"
    if ws_key not in cache or cache[ws_key].size(0) < workspace_size:
        cache[ws_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[ws_key]

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        A_log,
        a,
        dt_bias,
        b,
        output,
        cu_seqlens,
        _initial_state,
        _output_state,
        output_checkpoints,
        cu_checkpoints,
        checkpoint_every_n_tokens,
        scale,
        workspace,
        stream,
    )
