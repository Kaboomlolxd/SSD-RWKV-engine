from __future__ import annotations

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_u8_grouped_to_tensor,
    encode_scale_u8,
    encode_scale_u8_grouped,
    packed_length_scale_u8_grouped,
)
from rwkv_ssd.tools.quant_quality import compare_tensors


def _entry(length: int, shape: list[int]) -> TensorEntry:
    return TensorEntry(
        name="blocks.0.att.key.weight",
        layer_id=0,
        dtype="bfloat16",
        shape=shape,
        offset=0,
        length=length,
        alignment=1,
        residency="streamed",
        dequant="scale_u8_grouped",
    )


def test_grouped_u8_roundtrip_and_size() -> None:
    t = torch.randn(32, 32, dtype=torch.bfloat16)
    blob = encode_scale_u8_grouped(t, group_size=128)
    assert len(blob) == packed_length_scale_u8_grouped(t.numel(), group_size=128)
    out = decode_scale_u8_grouped_to_tensor(blob, _entry(len(blob), [32, 32]), torch.device("cpu"))
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_grouped_u8_reduces_outlier_scale_error() -> None:
    torch.manual_seed(11)
    t = torch.randn(4096, dtype=torch.float32) * 0.01
    t[0] = 100.0
    t = t.reshape(64, 64).to(torch.bfloat16)
    grouped = encode_scale_u8_grouped(t, group_size=128)
    grouped_out = decode_scale_u8_grouped_to_tensor(
        grouped, _entry(len(grouped), [64, 64]), torch.device("cpu")
    )
    global_blob = encode_scale_u8(t)
    global_entry = _entry(len(global_blob), [64, 64])
    global_entry = TensorEntry(**{**global_entry.__dict__, "dequant": "scale_u8"})
    from rwkv_ssd.runtime.dequant import decode_weight_to_tensor

    global_out = decode_weight_to_tensor(global_blob, global_entry, torch.device("cpu"))
    assert compare_tensors(t, grouped_out)["rmse"] < compare_tensors(t, global_out)["rmse"]
