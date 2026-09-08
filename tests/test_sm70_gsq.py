import pytest
import torch

from humming import dtypes
from humming.layer import HummingLayer
from humming.utils.weight import dequantize_weight


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="requires an SM70 GPU",
)


@pytest.mark.parametrize(
    ("weight_dtype", "group_size"),
    [(dtypes.uint2, 128), (dtypes.uint3, 128), (dtypes.uint4, 64)],
)
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
def test_sm70_packed_gsq_native_layer(weight_dtype, group_size, scale_dtype):
    """Keep GSQ packed and execute it via the Volta WMMA path."""
    torch.manual_seed(70 + weight_dtype.num_bits)
    # N is packable but deliberately not a 256 multiple, exercising output
    # padding and trimming just as the vLLM adapter does.
    shape_m, shape_n, shape_k = 5, 480, 384
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        torch_dtype=torch.float16,
        weight_config={
            "dtype": str(weight_dtype),
            "group_size": group_size,
            "scale_dtype": str(dtypes.DataType.from_torch_dtype(scale_dtype)),
        },
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=True,
    ).cuda()
    layer.load_from_unquantized(
        torch.randn(shape_n, shape_k, device="cuda", dtype=torch.float16)
    )
    layer.bias.copy_(torch.randn_like(layer.bias))

    inputs = torch.randn(shape_m, shape_k, device="cuda", dtype=torch.float16)
    weight_ref = dequantize_weight(
        layer.weight, layer.weight_scale, None, None, weight_dtype, packed=True
    ).to(torch.float16)
    expected = inputs @ weight_ref.T + layer.bias

    layer.transform()
    assert layer.volta_native
    assert not hasattr(layer, "volta_weight")

    outputs = layer(inputs)
    supplied = torch.empty_like(outputs)
    returned = layer(inputs, outputs=supplied)
    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
    assert returned.data_ptr() == supplied.data_ptr()
    torch.testing.assert_close(supplied, expected, rtol=0, atol=0)
