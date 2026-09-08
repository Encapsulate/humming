import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import torch

from humming.jit.runtime import KernelRuntime


@dataclasses.dataclass(kw_only=True)
class VoltaHummingGemmKernel(KernelRuntime):
    """Packed low-bit, group-scaled WMMA reference kernel for SM70."""

    name: ClassVar[str] = "volta_humming_gemm"
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
        self.code = '#include <humming/kernel/volta_gemm.cuh>\n'
        self.kernel_expr = (
            f"volta_humming_gemm<{self.weight_bits}, {self.group_size}, "
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

    def __call__(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, outputs=None):
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
        config.gridDimX = (shape_n + 15) // 16
        config.gridDimY = (shape_m + 15) // 16
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(inputs.device).cuda_stream
        values = (inputs.data_ptr(), weights.data_ptr(), scales.data_ptr(), outputs.data_ptr(), shape_m, shape_n, shape_k)
        cbd.cuLaunchKernelEx(config, self.func, (values, self.arg_types), 0)
        return outputs
