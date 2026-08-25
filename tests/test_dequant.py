"""Dequant codec entry point."""

from __future__ import annotations

import pytest

from rwkv_ssd.runtime.dequant import decode_weight_blob
from rwkv_ssd.runtime.manifest import TensorEntry


def test_none_passthrough() -> None:
    entry = TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="float32",
        shape=[4, 4],
        offset=0,
        length=64,
        alignment=4096,
        residency="streamed",
        dequant="none",
    )
    data = b"x" * 64
    assert decode_weight_blob(data, entry) == data


def test_unknown_codec_raises() -> None:
    entry = TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="float32",
        shape=[4, 4],
        offset=0,
        length=64,
        alignment=4096,
        residency="streamed",
        dequant="q4_k",
    )
    with pytest.raises(ValueError, match="unsupported dequant"):
        decode_weight_blob(b"x" * 64, entry)
