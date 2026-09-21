"""Fused gate/beta preprocessing for the GDN prefill kernel."""

import functools

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import EnableTVMFFI
from cutlass.cute.runtime import from_dlpack


class GateBetaPreprocess:
    """Compute the log-space decay gate and sigmoid update gate in one launch."""

    threads_per_cta = 256
    num_heads = 8

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        gate_beta: cute.Tensor,
        num_elements: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = cutlass.Int32(bidx) * self.threads_per_cta + cutlass.Int32(tidx)

        if idx < num_elements:
            head = idx % self.num_heads
            x = cutlass.Float32(a[idx]) + dt_bias[head]

            # Match torch.nn.functional.softplus's default beta=1, threshold=20.
            softplus = x
            if x <= cutlass.Float32(20.0):
                softplus = cute.math.log1p(cute.math.exp(x))

            gate = -cute.math.exp(a_log[head]) * softplus

            b_val = cutlass.Float32(b[idx])
            beta = cutlass.Float32(0.0)
            if b_val >= cutlass.Float32(0.0):
                z = cute.math.exp(-b_val)
                beta = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + z)
            else:
                z = cute.math.exp(b_val)
                beta = z / (cutlass.Float32(1.0) + z)

            gate_beta[idx] = gate
            gate_beta[num_elements + idx] = beta

    @cute.jit
    def __call__(
        self,
        a_iter: cute.Pointer,
        b_iter: cute.Pointer,
        a_log_iter: cute.Pointer,
        dt_bias_iter: cute.Pointer,
        gate_beta_iter: cute.Pointer,
        num_elements: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        input_layout = cute.make_layout((num_elements,))
        param_layout = cute.make_layout((self.num_heads,))
        output_layout = cute.make_layout((num_elements * 2,))

        a = cute.make_tensor(a_iter, input_layout)
        b = cute.make_tensor(b_iter, input_layout)
        a_log = cute.make_tensor(a_log_iter, param_layout)
        dt_bias = cute.make_tensor(dt_bias_iter, param_layout)
        gate_beta = cute.make_tensor(gate_beta_iter, output_layout)

        self.kernel(a, b, a_log, dt_bias, gate_beta, num_elements).launch(
            grid=[cute.ceil_div(num_elements, self.threads_per_cta), 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )


@functools.cache
def _get_compiled_gate_beta(a_dtype: torch.dtype, b_dtype: torch.dtype):
    """Return a mutable per-dtype cache, matching the core kernel cache style."""
    return {}


def preprocess_gate_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape or a.dim() != 2 or a.shape[1] != 8:
        raise ValueError("expected contiguous a and b tensors with shape [T, 8]")
    if a_log.shape != (8,) or dt_bias.shape != (8,):
        raise ValueError("expected A_log and dt_bias tensors with shape [8]")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("expected contiguous a and b tensors")

    gate_beta = torch.empty(
        (2, a.shape[0], a.shape[1]), dtype=torch.float32, device=a.device
    )
    cache = _get_compiled_gate_beta(a.dtype, b.dtype)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    num_elements = a.numel()

    if "compiled" not in cache:
        a_tensor = from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
        b_tensor = from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
        a_log_tensor = from_dlpack(a_log, assumed_align=16, enable_tvm_ffi=True)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16, enable_tvm_ffi=True)
        gate_beta_tensor = from_dlpack(
            gate_beta, assumed_align=16, enable_tvm_ffi=True
        )
        cache["compiled"] = cute.compile[EnableTVMFFI](
            GateBetaPreprocess(),
            a_tensor.iterator,
            b_tensor.iterator,
            a_log_tensor.iterator,
            dt_bias_tensor.iterator,
            gate_beta_tensor.iterator,
            cutlass.Int32(num_elements),
            stream=current_stream,
        )

    cache["compiled"](
        a.data_ptr(),
        b.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        gate_beta.data_ptr(),
        num_elements,
        stream=current_stream,
    )
    return gate_beta[0], gate_beta[1]
