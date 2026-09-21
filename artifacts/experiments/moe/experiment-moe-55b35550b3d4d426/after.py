"""FP8 block-scale DeepSeek-V3 MoE implemented entirely with CuTe-DSL.

The implementation uses an exact fused no-aux-loss router, packs the local
routes into expert-contiguous rows, executes two persistent Blackwell
blockwise grouped GEMMs, and fuses SwiGLU with the intervening FP8
requantization.  PyTorch is used only for owning workspace allocations and
for passing pointers/current-stream handles to the CuTe kernels.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys

import cuda.bindings.driver as cuda
import torch


# The benchmark imports this file directly with importlib.  Make the sibling
# package discoverable independent of the benchmark driver's working directory.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from solution.gemm import GemmTactic, grouped_blockwise_gemm
from solution.ops import (
    combine,
    pack_hidden,
    permute_and_pack_hidden,
    permute_and_pack_hidden_device_m,
    swiglu_quantize,
)
from solution.routing import (
    RoutingWorkspace,
    _run_routing_trusted as run_routing,
)


_HIDDEN = 7168
_INTERMEDIATE = 2048
_GEMM1_OUT = 2 * _INTERMEDIATE
_LOCAL_EXPERTS = 32

# The longer-K gate/up projection benefits from N-fast traversal and A reuse;
# the shorter-K down projection retains M-fast traversal.  Keeping the two
# tactics explicit also lets later tuning vary them independently.
_GEMM1_TACTIC = GemmTactic(raster_along_m=False)
_GEMM2_TACTIC = GemmTactic(raster_along_m=True, ab_stage_delta=-1)
# Once packed M is large enough to fill the machine without cluster-N
# cooperation, N-fast traversal with independent CTAs improves the down
# projection.  Keep this separate from the balanced PAD128 and PAD64 tactics.
_GEMM2_VERY_LARGE_TACTIC = GemmTactic(
    cluster_shape_mn=(1, 1),
    raster_along_m=False,
    ab_stage_delta=-1,
)
_GEMM2_VERY_LARGE_MIN_PACKED_ROWS = 32768
_GEMM1_M64_TACTIC = GemmTactic(
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=(1, 2),
    raster_along_m=False,
)
_GEMM2_M64_TACTIC = GemmTactic(
    mma_tiler_mn=(64, 128),
    cluster_shape_mn=(1, 2),
    raster_along_m=True,
)
# PAD64 substantially reduces expert padding in the moderate-size regime. At
# larger row counts its smaller tcgen05 tile loses throughput, so keep those
# projections on PAD128.
_M64_MAX_TOKENS = 901


def _check_cuda(result, operation: str):
    """Unwrap a cuda-python driver result with useful failure context."""

    if isinstance(result, tuple):
        status, *values = result
    else:
        status, values = result, []
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with {status}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


@dataclass(slots=True)
class _MappedMState:
    """Device-writable host scalar and pre-pack event for one workspace."""

    host_ptr: int | None
    device_ptr: cuda.CUdeviceptr
    host_value: ctypes.c_int32
    event: cuda.CUevent | None
    device_index: int
    stream: cuda.CUstream | None = None
    pointer_may_be_in_flight: bool = False
    event_recorded: bool = False

    @classmethod
    def allocate(cls, device_index: int) -> _MappedMState:
        # Driver allocations and events are context-sensitive.  Querying the
        # current stream materializes/selects PyTorch's primary context before
        # the driver maps the host allocation into its unified address space.
        with torch.cuda.device(device_index):
            torch.cuda.current_stream(device_index)
            host_ptr = _check_cuda(
                cuda.cuMemHostAlloc(
                    ctypes.sizeof(ctypes.c_int32),
                    cuda.CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc(mapped total_padded)",
            )
            try:
                device_ptr = _check_cuda(
                    cuda.cuMemHostGetDevicePointer(host_ptr, 0),
                    "cuMemHostGetDevicePointer(total_padded)",
                )
                flags = cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING
                event = _check_cuda(
                    cuda.cuEventCreate(flags),
                    "cuEventCreate(total_padded)",
                )
            except Exception:
                cuda.cuMemFreeHost(host_ptr)
                raise
        try:
            host_value = ctypes.c_int32.from_address(int(host_ptr))
            host_value.value = -1
            return cls(
                int(host_ptr),
                device_ptr,
                host_value,
                event,
                device_index,
            )
        except Exception:
            with torch.cuda.device(device_index):
                cuda.cuEventDestroy(event)
                cuda.cuMemFreeHost(host_ptr)
            raise

    def begin_pointer_use(self, stream: cuda.CUstream) -> None:
        """Bind the stream before a routing launch can write the host pointer."""

        if self.pointer_may_be_in_flight:
            raise RuntimeError("previous mapped total_padded use is still active")
        self.stream = stream
        # Set this before passing ``device_ptr`` to CuTe.  Even if compilation,
        # launch, or event recording raises after partially enqueueing work,
        # cleanup now knows it must synchronize before freeing/reusing memory.
        self.pointer_may_be_in_flight = True
        self.event_recorded = False

    def record(self) -> None:
        event = self.event
        if event is None:
            raise RuntimeError("total_padded event was already destroyed")
        stream = self.stream
        if stream is None or not self.pointer_may_be_in_flight:
            raise RuntimeError("mapped total_padded pointer use was not started")
        _check_cuda(
            cuda.cuEventRecord(event, stream),
            "cuEventRecord(total_padded)",
        )
        self.event_recorded = True

    def wait(self) -> int:
        event = self.event
        if event is None:
            raise RuntimeError("total_padded event was already destroyed")
        if not self.event_recorded:
            raise RuntimeError("total_padded event was not recorded")
        _check_cuda(
            cuda.cuEventSynchronize(event),
            "cuEventSynchronize(total_padded)",
        )
        self.pointer_may_be_in_flight = False
        self.event_recorded = False
        return int(self.host_value.value)

    def synchronize_pointer_use(self) -> None:
        """Make an exposed mapped pointer safe to reuse or free after failure."""

        if not self.pointer_may_be_in_flight:
            return
        if self.event_recorded and self.event is not None:
            _check_cuda(
                cuda.cuEventSynchronize(self.event),
                "cuEventSynchronize(total_padded failure)",
            )
        elif self.stream is not None:
            _check_cuda(
                cuda.cuStreamSynchronize(self.stream),
                "cuStreamSynchronize(total_padded failure)",
            )
        else:
            _check_cuda(
                cuda.cuCtxSynchronize(),
                "cuCtxSynchronize(total_padded failure)",
            )
        self.pointer_may_be_in_flight = False
        self.event_recorded = False

    def close(self) -> None:
        event = self.event
        host_ptr = self.host_ptr
        if event is None and host_ptr is None:
            return
        with torch.cuda.device(self.device_index):
            self.synchronize_pointer_use()
            if event is not None:
                _check_cuda(
                    cuda.cuEventDestroy(event),
                    "cuEventDestroy(total_padded)",
                )
                self.event = None
            if host_ptr is not None:
                _check_cuda(
                    cuda.cuMemFreeHost(host_ptr),
                    "cuMemFreeHost(total_padded)",
                )
                self.host_ptr = None
                self.device_ptr = cuda.CUdeviceptr(0)
                self.host_value = ctypes.c_int32()

    def __del__(self) -> None:
        # Interpreter shutdown may tear down CUDA before module globals.  The
        # cache owns this state for the process lifetime, so cleanup here is
        # necessarily best-effort.
        event = getattr(self, "event", None)
        host_ptr = getattr(self, "host_ptr", None)
        if event is not None or host_ptr is not None:
            try:
                with torch.cuda.device(self.device_index):
                    if getattr(self, "pointer_may_be_in_flight", False):
                        if (
                            getattr(self, "event_recorded", False)
                            and event is not None
                        ):
                            cuda.cuEventSynchronize(event)
                        elif getattr(self, "stream", None) is not None:
                            cuda.cuStreamSynchronize(self.stream)
                        else:
                            cuda.cuCtxSynchronize()
                    if event is not None:
                        cuda.cuEventDestroy(event)
                    if host_ptr is not None:
                        cuda.cuMemFreeHost(host_ptr)
            except Exception:
                pass


@dataclass(slots=True)
class _OutputClearState:
    """Reusable auxiliary stream for hiding a large output memset."""

    stream: cuda.CUstream | None
    ready_event: cuda.CUevent | None
    done_event: cuda.CUevent | None
    device_index: int
    pending: bool = False

    @classmethod
    def allocate(cls, device_index: int) -> _OutputClearState:
        with torch.cuda.device(device_index):
            torch.cuda.current_stream(device_index)
            stream = _check_cuda(
                cuda.cuStreamCreate(
                    cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
                ),
                "cuStreamCreate(output clear)",
            )
            ready_event = None
            done_event = None
            try:
                flags = cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING
                ready_event = _check_cuda(
                    cuda.cuEventCreate(flags),
                    "cuEventCreate(output clear ready)",
                )
                done_event = _check_cuda(
                    cuda.cuEventCreate(flags),
                    "cuEventCreate(output clear done)",
                )
            except Exception:
                if ready_event is not None:
                    cuda.cuEventDestroy(ready_event)
                cuda.cuStreamDestroy(stream)
                raise
        return cls(stream, ready_event, done_event, device_index)

    def begin(self, output: torch.Tensor, main_stream: cuda.CUstream) -> None:
        stream = self.stream
        ready_event = self.ready_event
        done_event = self.done_event
        if stream is None or ready_event is None or done_event is None:
            raise RuntimeError("output clear state was already destroyed")
        _check_cuda(
            cuda.cuEventRecord(ready_event, main_stream),
            "cuEventRecord(output clear ready)",
        )
        _check_cuda(
            cuda.cuStreamWaitEvent(stream, ready_event, 0),
            "cuStreamWaitEvent(output clear ready)",
        )
        _check_cuda(
            cuda.cuMemsetD32Async(
                output.data_ptr(), 0, output.numel() // 2, stream
            ),
            "cuMemsetD32Async(aux output clear)",
        )
        _check_cuda(
            cuda.cuEventRecord(done_event, stream),
            "cuEventRecord(output clear done)",
        )
        self.pending = True

    def join(self, main_stream: cuda.CUstream) -> None:
        if not self.pending:
            return
        done_event = self.done_event
        if done_event is None:
            raise RuntimeError("output clear state was already destroyed")
        _check_cuda(
            cuda.cuStreamWaitEvent(main_stream, done_event, 0),
            "cuStreamWaitEvent(output clear done)",
        )
        self.pending = False

    def close(self) -> None:
        stream = self.stream
        ready_event = self.ready_event
        done_event = self.done_event
        if stream is None and ready_event is None and done_event is None:
            return
        with torch.cuda.device(self.device_index):
            if stream is not None:
                cuda.cuStreamSynchronize(stream)
            if ready_event is not None:
                cuda.cuEventDestroy(ready_event)
                self.ready_event = None
            if done_event is not None:
                cuda.cuEventDestroy(done_event)
                self.done_event = None
            if stream is not None:
                cuda.cuStreamDestroy(stream)
                self.stream = None
        self.pending = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
    mapped_m: _MappedMState | None = None
    output_clear: _OutputClearState | None = None


_COMPUTE_WORKSPACES: dict[tuple[int, int], _ComputeWorkspace] = {}


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _compute_workspace_key(tensor: torch.Tensor) -> tuple[int, int]:
    return (_device_index(tensor), int(tensor.shape[0]))


def _allocate_compute_workspace(
    seq_len: int,
    capacity: int,
    device: torch.device,
    mapped_m: _MappedMState | None = None,
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
        mapped_m=mapped_m,
    )


def _compute_workspace(
    hidden_states: torch.Tensor,
    needed_rows: int,
    mapped_m: _MappedMState | None = None,
) -> _ComputeWorkspace:
    seq_len = int(hidden_states.shape[0])
    key = _compute_workspace_key(hidden_states)
    workspace = _COMPUTE_WORKSPACES.get(key)
    if workspace is None:
        workspace = _allocate_compute_workspace(
            seq_len, needed_rows, hidden_states.device, mapped_m
        )
        _COMPUTE_WORKSPACES[key] = workspace
    elif workspace.capacity < needed_rows:
        # Preserve the mapped scalar/event across a capacity grow.  The caller
        # synchronizes the bounded speculative pack before replacing buffers.
        workspace = _allocate_compute_workspace(
            seq_len,
            needed_rows,
            hidden_states.device,
            workspace.mapped_m or mapped_m,
        )
        _COMPUTE_WORKSPACES[key] = workspace
    elif workspace.mapped_m is None and mapped_m is not None:
        workspace.mapped_m = mapped_m
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

    # The entry point has a fixed benchmark/operator contract.  Rechecking
    # eight tensors' metadata on every invocation delays the first CUDA launch
    # after the caller's timing event; trust that contract and read only the
    # runtime axis needed for size dispatch and launch geometry.
    seq_len = int(hidden_states.shape[0])

    use_m64 = seq_len <= _M64_MAX_TOKENS
    pad_m = 64 if use_m64 else 128
    gemm1_tactic = _GEMM1_M64_TACTIC if use_m64 else _GEMM1_TACTIC
    gemm2_tactic = _GEMM2_M64_TACTIC if use_m64 else _GEMM2_TACTIC

    routing: RoutingWorkspace = run_routing(
        routing_logits,
        routing_bias,
        int(local_expert_offset),
        float(routed_scaling_factor),
        pad_m=pad_m,
        live_gidx_only=not use_m64,
    )

    if use_m64:
        # Keep the latency-oriented PAD64 path unchanged.  Its exact host M
        # sizes both the allocation and the one-CTA-per-row pack launch.
        packed_rows = int(routing.total_padded.item())
        workspace = _compute_workspace(hidden_states, packed_rows)
        if (
        not use_m64
        and packed_rows >= _GEMM2_VERY_LARGE_MIN_PACKED_ROWS
    ):
        gemm2_tactic = _GEMM2_VERY_LARGE_TACTIC

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
    else:
        # PAD128 steady state overlaps the scalar synchronization with a
        # fixed-grid pack.  The first call for a (device, T) key still waits
        # synchronously so its compute buffers have exact observed capacity.
        key = _compute_workspace_key(hidden_states)
        workspace = _COMPUTE_WORKSPACES.get(key)
        stream = cuda.CUstream(
            torch.cuda.current_stream(hidden_states.device).cuda_stream
        )

        if workspace is None:
            async_m = _AsyncMState.allocate(key[0])
            try:
                async_m.enqueue(routing.total_padded, stream)
                packed_rows = async_m.wait()
                workspace = _compute_workspace(
                    hidden_states, packed_rows, async_m
                )
            except Exception:
                async_m.close()
                raise

            if packed_rows:
                permute_and_pack_hidden(
                    hidden_states,
                    hidden_states_scale,
                    routing.counts,
                    routing.offsets,
                    routing.expert_routes,
                    routing.route_slots,
                    routing.gidx,
                    workspace.packed_hidden,
                    workspace.packed_hidden_scale,
                    packed_rows,
                    seq_len,
                    stream,
                )
        else:
            async_m = workspace.async_m
            if async_m is None:
                # Defensive upgrade for a cache populated without the PAD128
                # async state; dispatch policy normally makes this impossible.
                async_m = _AsyncMState.allocate(key[0])
                workspace.async_m = async_m

            async_m.enqueue(routing.total_padded, stream)
            permute_and_pack_hidden_device_m(
                hidden_states,
                hidden_states_scale,
                routing.counts,
                routing.offsets,
                routing.total_padded,
                routing.expert_routes,
                routing.route_slots,
                routing.gidx,
                workspace.packed_hidden,
                workspace.packed_hidden_scale,
                seq_len,
                stream,
            )
            packed_rows = async_m.wait()

            if packed_rows > workspace.capacity:
                # The speculative kernel clamps itself to the old capacity,
                # so it cannot go OOB.  Finish it before retiring those
                # buffers, grow exactly, then replay the complete pack.
                _check_cuda(
                    cuda.cuStreamSynchronize(stream),
                    "cuStreamSynchronize(capacity grow)",
                )
                workspace = _compute_workspace(
                    hidden_states, packed_rows, async_m
                )
                permute_and_pack_hidden_device_m(
                    hidden_states,
                    hidden_states_scale,
                    routing.counts,
                    routing.offsets,
                    routing.total_padded,
                    routing.expert_routes,
                    routing.route_slots,
                    routing.gidx,
                    workspace.packed_hidden,
                    workspace.packed_hidden_scale,
                    seq_len,
                    stream,
                )

    if packed_rows:

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
            tactic=gemm1_tactic,
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
            tactic=gemm2_tactic,
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
