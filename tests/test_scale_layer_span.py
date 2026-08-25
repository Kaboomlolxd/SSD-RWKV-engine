"""Batched scale_u8 / scale_u4 layer decode from one read span."""

from __future__ import annotations

import struct

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_layer_from_span,
    decode_scale_to_tensor,
    encode_scale_u4,
    encode_scale_u8,
)


def _entry(name: str, offset: int, numel: int, *, dequant: str) -> TensorEntry:
    length = 8 + numel if dequant == "scale_u8" else 8 + (numel + 1) // 2
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype="bfloat16",
        shape=[numel],
        offset=offset,
        length=length,
        alignment=4096,
        residency="streamed",
        dequant=dequant,
    )


def test_scale_u8_layer_span_matches_per_tensor() -> None:
    t0 = torch.linspace(-1.0, 1.0, 64, dtype=torch.bfloat16)
    t1 = torch.linspace(0.5, 2.0, 32, dtype=torch.bfloat16)
    b0 = encode_scale_u8(t0)
    b1 = encode_scale_u8(t1)
    base = 4096
    off1 = base + len(b0)
    raw = b0 + b1
    entries = [
        _entry("blocks.0.a", base, 64, dequant="scale_u8"),
        _entry("blocks.0.b", off1, 32, dequant="scale_u8"),
    ]
    batched = decode_scale_layer_from_span(raw, entries, base, torch.device("cpu"), bits=8)
    e0 = decode_scale_to_tensor(b0, entries[0], torch.device("cpu"), bits=8)
    e1 = decode_scale_to_tensor(b1, entries[1], torch.device("cpu"), bits=8)
    assert torch.allclose(batched["blocks.0.a"].float(), e0.float(), atol=0.05)
    assert torch.allclose(batched["blocks.0.b"].float(), e1.float(), atol=0.05)


def test_scale_u4_layer_span_matches_per_tensor() -> None:
    t0 = torch.randn(48, dtype=torch.bfloat16)
    b0 = encode_scale_u4(t0)
    base = 0
    entries = [_entry("blocks.0.w", base, 48, dequant="scale_u4")]
    raw = b0
    batched = decode_scale_layer_from_span(raw, entries, base, torch.device("cpu"), bits=4)
    single = decode_scale_to_tensor(b0, entries[0], torch.device("cpu"), bits=4)
    assert torch.allclose(batched["blocks.0.w"].float(), single.float(), atol=0.08)
