"""Contiguous layer span detection for single-read layer loads."""

from __future__ import annotations

from rwkv_ssd.runtime.layer_io import entries_contiguous_span, entries_layer_read_span
from rwkv_ssd.runtime.manifest import TensorEntry


def _entry(name: str, offset: int, length: int, layer_id: int = 0) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=layer_id,
        dtype="float32",
        shape=[4, 4],
        offset=offset,
        length=length,
        alignment=4096,
        residency="streamed",
        dequant="none",
    )


def test_contiguous_span_detected() -> None:
    entries = [
        _entry("blocks.0.a", 4096, 100),
        _entry("blocks.0.b", 4096 + 100, 200),
    ]
    assert entries_contiguous_span(entries) == (4096, 300)


def test_gap_returns_none() -> None:
    entries = [
        _entry("blocks.0.a", 0, 100),
        _entry("blocks.0.b", 200, 100),
    ]
    assert entries_contiguous_span(entries) is None
    assert entries_layer_read_span(entries) == (0, 300)
