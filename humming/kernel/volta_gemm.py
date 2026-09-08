import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import torch

from humming.jit.runtime import KernelRuntime


@dataclasses.dataclass(kw_only=True)
class VoltaHummingGemmKernel(KernelRuntime):
    """Packed low-bit, group-scaled WMMA kernel for SM70."""

    name: ClassVar[str] = "volta_humming_gemm"
    kernel_symbol: ClassVar[str] = "volta_humming_gemm"
    weight_bits: int
    group_size: int
    scale_dtype: torch.dtype

    def init_kernel(self):
        if self.sm_version != 70:
            raise RuntimeError("VoltaHummingGemmKernel is only valid on SM70.")
        if self.weight_bits not in (2, 3, 4):
            raise ValueError("SM70 GSQ kernel supports uint2, uint3, and uint4 weights only.")
        if self.group_size not in (64, 128):
            raise ValueError("SM70 GSQ kernel supports group sizes 64 and 128 only.")
        if self.scale_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("SM70 GSQ kernel requires float16 or bfloat16 group scales.")
        is_bf16 = self.scale_dtype == torch.bfloat16
        # Include the implementation revision in the generated translation
        # unit. Header-only kernels otherwise risk reusing a stale JIT cubin
        # when only the included .cuh changes.
        self.code = '#define HUMMING_VOLTA_GEMM_REVISION 3\n#include <humming/kernel/volta_gemm.cuh>\n'
        self.kernel_expr = (
            f"{self.kernel_symbol}<{self.weight_bits}, {self.group_size}, "
            f"{'true' if is_bf16 else 'false'}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        )
        self.prepare()

    def _launch(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, outputs=None):
        self.check_context()
        if inputs.dtype != torch.float16 or weights.dtype != torch.int32:
            raise ValueError("SM70 GSQ kernel requires FP16 inputs and packed int32 weights.")
        if inputs.ndim != 2 or weights.ndim != 2 or scales.ndim != 2:
            raise ValueError("SM70 GSQ kernel expects rank-2 dense tensors.")
        shape_m, shape_k = inputs.shape
        shape_n = weights.shape[0]
        if shape_k % self.group_size or shape_k % 16 or shape_n % 16:
            raise ValueError("SM70 GSQ kernel requires N/K multiples of 16 and K aligned to its group size.")
        if outputs is None:
            outputs = torch.empty((shape_m, shape_n), dtype=torch.float16, device=inputs.device)
        config = cbd.CUlaunchConfig()
        # Four warps cooperate on four adjacent N tiles and share one A tile.
        # This is especially important for decode, where shape_m is usually 1.
        config.gridDimX = (shape_n + 63) // 64
        config.gridDimY = (shape_m + 15) // 16
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(inputs.device).cuda_stream
        values = (inputs.data_ptr(), weights.data_ptr(), scales.data_ptr(), outputs.data_ptr(), shape_m, shape_n, shape_k)
        cbd.cuLaunchKernelEx(config, self.func, (values, self.arg_types), 0)
        return outputs

    def __call__(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, outputs=None):
        # A Torch custom op is opaque to Dynamo but remains captureable by a
        # CUDA graph.  Keep the direct eager launch for the normal path so it
        # has no dispatcher overhead.
        if torch.compiler.is_compiling():
            result = torch.ops.humming.volta_humming_gemm(
                inputs, weights, scales, self.weight_bits, self.group_size
            )
            if outputs is not None:
                outputs.copy_(result)
                return outputs
            return result
        return self._launch(inputs, weights, scales, outputs)


@dataclasses.dataclass(kw_only=True)
class VoltaHummingGemvKernel(VoltaHummingGemmKernel):
    """Single-token decode kernel that avoids padded 16-row WMMA work."""

    name: ClassVar[str] = "volta_humming_gemv"
    kernel_symbol: ClassVar[str] = "volta_humming_gemv"

    def _launch(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, outputs=None):
        self.check_context()
        if inputs.dtype != torch.float16 or weights.dtype != torch.int32:
            raise ValueError("SM70 GSQ kernel requires FP16 inputs and packed int32 weights.")
        if inputs.ndim != 2 or weights.ndim != 2 or scales.ndim != 2 or inputs.shape[0] != 1:
            raise ValueError("SM70 GSQ GEMV requires one FP16 activation row and rank-2 packed tensors.")
        _, shape_k = inputs.shape
        shape_n = weights.shape[0]
        if shape_k % self.group_size or shape_k % 32 or shape_n % 16:
            raise ValueError("SM70 GSQ GEMV requires K aligned to 32/group and N aligned to 16.")
        if outputs is None:
            outputs = torch.empty((1, shape_n), dtype=torch.float16, device=inputs.device)
        config = cbd.CUlaunchConfig()
        config.gridDimX = (shape_n + 31) // 32
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 256
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(inputs.device).cuda_stream
        values = (inputs.data_ptr(), weights.data_ptr(), scales.data_ptr(), outputs.data_ptr(), 1, shape_n, shape_k)
        cbd.cuLaunchKernelEx(config, self.func, (values, self.arg_types), 0)
        return outputs

    def __call__(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, outputs=None):
        if torch.compiler.is_compiling():
            result = torch.ops.humming.volta_humming_gemv(
                inputs, weights, scales, self.weight_bits, self.group_size
            )
            if outputs is not None:
                outputs.copy_(result)
                return outputs
            return result
        return self._launch(inputs, weights, scales, outputs)


@torch.library.custom_op("humming::volta_humming_gemm", mutates_args=())
def _volta_humming_gemm_op(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_bits: int,
    group_size: int,
) -> torch.Tensor:
    """Opaque Dynamo boundary for the preloaded Volta CUDA-driver kernel."""
    kernel = VoltaHummingGemmKernel(
        weight_bits=weight_bits,
        group_size=group_size,
        scale_dtype=scales.dtype,
    )
    return kernel._launch(inputs, weights, scales)


@_volta_humming_gemm_op.register_fake
def _volta_humming_gemm_fake(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_bits: int,
    group_size: int,
) -> torch.Tensor:
    del scales, weight_bits, group_size
    return torch.empty((inputs.shape[0], weights.shape[0]), dtype=inputs.dtype, device=inputs.device)


@torch.library.custom_op("humming::volta_humming_gemv", mutates_args=())
def _volta_humming_gemv_op(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_bits: int,
    group_size: int,
) -> torch.Tensor:
    kernel = VoltaHummingGemvKernel(
        weight_bits=weight_bits,
        group_size=group_size,
        scale_dtype=scales.dtype,
    )
    return kernel._launch(inputs, weights, scales)


@_volta_humming_gemv_op.register_fake
def _volta_humming_gemv_fake(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_bits: int,
    group_size: int,
) -> torch.Tensor:
    del scales, weight_bits, group_size
    return torch.empty((inputs.shape[0], weights.shape[0]), dtype=inputs.dtype, device=inputs.device)
