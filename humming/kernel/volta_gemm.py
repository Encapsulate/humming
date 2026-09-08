import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import torch

from humming.jit.runtime import KernelRuntime


@dataclasses.dataclass(kw_only=True)
class VoltaHummingGemmKernel(KernelRuntime):
    """Packed uint2/uint3, group-128 WMMA reference kernel for SM70."""

    name: ClassVar[str] = "volta_humming_gemm"
    weight_bits: int
    scale_dtype: torch.dtype

    def init_kernel(self):
        if self.sm_version != 70:
            raise RuntimeError("VoltaHummingGemmKernel is only valid on SM70.")
        if self.weight_bits not in (2, 3):
            raise ValueError("SM70 GSQ kernel supports uint2 and uint3 weights only.")
        if self.scale_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("SM70 GSQ kernel requires float16 or bfloat16 group scales.")
        is_bf16 = self.scale_dtype == torch.bfloat16
        self.code = '#include <humming/kernel/volta_gemm.cuh>\n'
        self.kernel_expr = f"volta_humming_gemm<{self.weight_bits}, {'true' if is_bf16 else 'false'}>"
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
        if shape_k % 128 or shape_k % 16 or shape_n % 16:
            raise ValueError("SM70 GSQ kernel requires N/K multiples of 16 and K multiple of 128.")
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
