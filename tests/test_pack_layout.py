"""P2.b pack layout ordering."""

from __future__ import annotations

from rwkv_ssd.runtime.pack_layout import (
    DEFAULT_SECTOR_BYTES,
    pad_between_layers,
    sort_tensor_names,
)


def _lid(name: str) -> int:
    if "blocks.1." in name:
        return 1
    if "blocks.0." in name:
        return 0
    return -1


def test_layer_grouped_sort() -> None:
    names = ["blocks.1.a", "embed.weight", "blocks.0.b", "blocks.0.a"]
    ordered = sort_tensor_names(names, "layer_grouped", _lid)
    assert ordered.index("blocks.0.a") < ordered.index("blocks.0.b")
    assert ordered.index("blocks.0.b") < ordered.index("blocks.1.a")


def test_sector_pad_advances_offset() -> None:
    off = pad_between_layers(1000, 0, 1, DEFAULT_SECTOR_BYTES)
    assert off == DEFAULT_SECTOR_BYTES
    same = pad_between_layers(1000, 0, 0, DEFAULT_SECTOR_BYTES)
    assert same == 1000
