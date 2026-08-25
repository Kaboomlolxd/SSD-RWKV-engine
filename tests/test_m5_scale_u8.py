"""M5 scale_u8 codec roundtrip."""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.dequant import decode_weight_blob
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import encode_scale_u8
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes


def test_scale_u8_roundtrip_bfloat16() -> None:
    t = torch.randn(32, 32, dtype=torch.bfloat16)
    entry = TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="bfloat16",
        shape=[32, 32],
        offset=0,
        length=8 + 32 * 32,
        alignment=4096,
        residency="streamed",
        dequant="scale_u8",
    )
    raw = encode_scale_u8(t)
    decoded = tensor_from_bytes(decode_weight_blob(raw, entry), entry, torch.device("cpu"))
    assert decoded.shape == t.shape
    assert decoded.dtype == t.dtype
    err = (decoded.float() - t.float()).abs().max().item()
    assert err < 0.05
