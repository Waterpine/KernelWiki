"""FP8 block-scale DeepSeek-V3 MoE implemented entirely with CuTe-DSL.

The implementation uses an exact fused no-aux-loss router, packs the local
routes into expert-contiguous rows, executes two persistent Blackwell
blockwise grouped GEMMs, and fuses SwiGLU with the intervening FP8
requantization.  PyTorch is used only for owning workspace allocations and
for passing pointers/current-stream handles to the CuTe kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


# The benchmark imports this file directly with importlib.  Make the sibling
# package discoverable independent of the benchmark driver's working directory.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from solution.gemm import grouped_blockwise_gemm
from solution.ops import combine, pack_hidden, swiglu_quantize
from solution.routing import RoutingWorkspace, run_routing


_HIDDEN = 7168
_INTERMEDIATE = 2048
_GEMM1_OUT = 2 * _INTERMEDIATE
_LOCAL_EXPERTS = 32


@dataclass(slots=True)
class _ComputeWorkspace:
    """Reusable tensors for the two-GEMM local-expert pipeline."""

    seq_len: int
    capacity: int
    packed_hidden: torch.Tensor
    packed_hidden_scale: torch.Tensor
    gemm1_out: torch.Tensor
    intermediate: torch.Tensor
    intermediate_scale: torch.Tensor
    gemm2_out: torch.Tensor
    output: torch.Tensor


_COMPUTE_WORKSPACES: dict[tuple[int, int], _ComputeWorkspace] = {}


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _allocate_compute_workspace(
    seq_len: int,
    capacity: int,
    device: torch.device,
) -> _ComputeWorkspace:
    # Capacity is observed after exact routing.  It is allocation metadata only:
    # every buffer region consumed by this invocation is overwritten below.
    return _ComputeWorkspace(
        seq_len=seq_len,
        capacity=capacity,
        packed_hidden=torch.empty(
            (capacity, _HIDDEN), dtype=torch.float8_e4m3fn, device=device
        ),
        packed_hidden_scale=torch.empty(
            (capacity, _HIDDEN // 128), dtype=torch.float32, device=device
        ),
        gemm1_out=torch.empty(
            (capacity, _GEMM1_OUT), dtype=torch.bfloat16, device=device
        ),
        intermediate=torch.empty(
            (capacity, _INTERMEDIATE), dtype=torch.float8_e4m3fn, device=device
        ),
        intermediate_scale=torch.empty(
            (capacity, _INTERMEDIATE // 128), dtype=torch.float32, device=device
        ),
        gemm2_out=torch.empty(
            (capacity, _HIDDEN), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty(
            (seq_len, _HIDDEN), dtype=torch.bfloat16, device=device
        ),
    )


def _compute_workspace(
    hidden_states: torch.Tensor,
    needed_rows: int,
) -> _ComputeWorkspace:
    seq_len = int(hidden_states.shape[0])
    key = (_device_index(hidden_states), seq_len)
    workspace = _COMPUTE_WORKSPACES.get(key)
    if workspace is None or workspace.capacity < needed_rows:
        workspace = _allocate_compute_workspace(
            seq_len, needed_rows, hidden_states.device
        )
        _COMPUTE_WORKSPACES[key] = workspace
    return workspace


def _validate_inputs(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> int:
    """Fail early on layouts that would otherwise become raw-pointer hazards."""

    seq_len = int(hidden_states.shape[0])
    expected = (
        (routing_logits, torch.float32, (seq_len, 256), "routing_logits"),
        (routing_bias, torch.bfloat16, (256,), "routing_bias"),
        (hidden_states, torch.float8_e4m3fn, (seq_len, _HIDDEN), "hidden_states"),
        (
            hidden_states_scale,
            torch.float32,
            (_HIDDEN // 128, seq_len),
            "hidden_states_scale",
        ),
        (
            gemm1_weights,
            torch.float8_e4m3fn,
            (_LOCAL_EXPERTS, _GEMM1_OUT, _HIDDEN),
            "gemm1_weights",
        ),
        (
            gemm1_weights_scale,
            torch.float32,
            (_LOCAL_EXPERTS, _GEMM1_OUT // 128, _HIDDEN // 128),
            "gemm1_weights_scale",
        ),
        (
            gemm2_weights,
            torch.float8_e4m3fn,
            (_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE),
            "gemm2_weights",
        ),
        (
            gemm2_weights_scale,
            torch.float32,
            (_LOCAL_EXPERTS, _HIDDEN // 128, _INTERMEDIATE // 128),
            "gemm2_weights_scale",
        ),
    )
    device = hidden_states.device
    if device.type != "cuda":
        raise ValueError("all MoE tensors must be CUDA tensors")
    for tensor, dtype, shape, name in expected:
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return seq_len


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
) -> torch.Tensor:
    """Compute the local-rank contribution to the DeepSeek-V3 MoE layer."""

    seq_len = _validate_inputs(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )

    routing: RoutingWorkspace = run_routing(
        routing_logits,
        routing_bias,
        int(local_expert_offset),
        float(routed_scaling_factor),
    )

    # The grouped-GEMM scheduler needs a host launch bound.  Reading this one
    # scalar also avoids doing the worst-case 8*T work when only roughly T
    # routes belong to this rank.  All preceding routing kernels are on the
    # same stream, so the value is complete when item() returns.
    packed_rows = int(routing.total_padded.item())
    workspace = _compute_workspace(hidden_states, packed_rows)

    if packed_rows:
        pack_hidden(
            hidden_states,
            hidden_states_scale,
            routing.row_to_expanded,
            workspace.packed_hidden,
            workspace.packed_hidden_scale,
            packed_rows,
            seq_len,
        )

        grouped_blockwise_gemm(
            workspace.packed_hidden,
            gemm1_weights,
            workspace.gemm1_out,
            workspace.packed_hidden_scale,
            gemm1_weights_scale,
            routing.gidx,
            packed_rows,
            _GEMM1_OUT,
            _HIDDEN,
            _LOCAL_EXPERTS,
        )

        swiglu_quantize(
            workspace.gemm1_out,
            workspace.intermediate,
            workspace.intermediate_scale,
            packed_rows,
        )

        grouped_blockwise_gemm(
            workspace.intermediate,
            gemm2_weights,
            workspace.gemm2_out,
            workspace.intermediate_scale,
            gemm2_weights_scale,
            routing.gidx,
            packed_rows,
            _HIDDEN,
            _INTERMEDIATE,
            _LOCAL_EXPERTS,
        )

    combine(
        workspace.gemm2_out,
        routing.route_slots,
        routing.route_weights,
        workspace.output,
        packed_rows,
        seq_len,
    )
    return workspace.output


__all__ = ["run"]
