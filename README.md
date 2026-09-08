
# Humming

Humming is a high-performance, lightweight, and highly flexible JIT (Just-In-Time) compiled GEMM kernel library specifically designed for quantized inference.

## Volta / V100 support in this fork

This fork adds a usable SM70 baseline for Tesla V100 systems.  Humming's
quantization and packing kernels still run as JIT CUDA kernels; dense layers
with FP16 activations and quantized weights are dequantized once by
`transform()` and execute through PyTorch/cuBLAS.  This preserves Humming's
weight formats while avoiding native instructions unavailable on Volta.

The SM70 path currently supports dense FP16-activation layers only.  MoE,
FP8/INT8 activations, and the native Humming GEMM remain SM75+ features.  A
future native Volta backend must use `m8n8k4` fragments because Volta does not
implement the SM75 `ldmatrix` or newer `m16n8k16` instructions.

For Volta, use a PyTorch CUDA runtime whose FP16 cuBLAS path supports SM70.
The fallback was verified with PyTorch 2.5/CUDA 12.1.  Install Humming's normal
Python dependencies into that environment, then run your program from the
repository checkout:

```bash
python -m pip install -e .
python your_program.py
```

See [SM70 notes](docs/sm70.md) for the constraints and rationale.

## Key Features

- **High Flexibility**
    - Supports inference for any weight type under 8-bit across **FP16 / BF16 / FP8 / FP4 / INT8 / INT4** activations (provided the activation's dynamic range covers the weight type).
    - Supports various quantization strategies.
    - Supports various scale types (BF16, FP16, E4M3, E5M2, and UE8M0).
    - Supports both **Dense GEMM** and **MoE GEMM**.
- **High Compatibility**: supports all NVIDIA GPUs from **SM75+** (Turing architecture) and beyond.
- **High Performance**
    * Delivers State-of-the-Art (SOTA) throughput and efficiency across a wide range of computational scenarios.
- **Ultra-Lightweight**
    * Minimal dependencies: Requires only **PyTorch** and **NVCC**.
    * Compact footprint: The package size is only **100+KB**.


## Support Matrix

| Activation Type | Supported Devices | Supported Weight Types |
| :--- | :--- | :--- |
| **FP16** (e5m10) | SM70 fallback; SM75+ native | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8, kExp ≤ 5) |
| **BF16** (e8m7) | SM80+ | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8) |
| **FP8** (e4m3) | SM89+ | • Symmetric INT1-5<br>• INT1-4 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 4, kMan ≤ 3) |
| **FP8** (e5m2) | SM89+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 5, kMan ≤ 2) |
| **FP4** (e2m1) | SM120+ | • Symmetric INT1-3<br>• INT1-2 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 2, kMan ≤ 1) |
| **INT8** | SM75+ | • Symmetric INT1-8<br>• INT1-7 with dynamic zero point |
| **INT4** | SM80+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point |


## Getting Started


### Installation

```bash
git clone <your-fork-url> humming-sm70
cd humming-sm70
```

For the V100 baseline, use the existing runtime command above rather than
installing a duplicate CUDA or PyTorch stack.


### Usage Example


```python
import torch
from humming.layer import HummingLayer

layer = HummingLayer(
    shape_n=8192,
    shape_k=8192,
    weight_config={"dtype": "int6"},
    torch_dtype=torch.float16,
).cuda()

weight = torch.randn((8192, 8192), dtype=torch.float16, device="cuda:0")
inputs = torch.randn((128, 8192), dtype=torch.float16, device="cuda:0")

# Load unquantized weight and quantize to layer quantization format
layer.load_from_unquantized(weight)
# Transform weight to humming format and prepare default kernels
layer.transform()

# Run quantized GEMM (tuning_config is optional, auto-selected by default)
output = layer(inputs)

print("Quantized GEMM Output:")
print(output)
print("\nReference Output:")
print(inputs.matmul(weight.T))
```


## Acknowledgement

This project is highly inspired by

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin/) and [vLLM](https://github.com/vllm-project/vllm) Marlin Kernel
- [lmdeploy](https://github.com/InternLM/lmdeploy/) GEMM kernel
- [CUTLASS](https://github.com/nvidia/cutlass)
