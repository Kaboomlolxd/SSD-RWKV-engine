"""trinity_layer: one zlib decompress per layer."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.trinity_codec import (
    LayerZlibCache,
    decompress_trinity_layer_blob,
    decode_trinity_layer_to_tensor,
    encode_trinity_layer_bundle,
    encode_trinity_lut2,
)
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.residency import apply_residency_policy
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack


def test_residency_policy_keeps_trinity_layer_inner_spans() -> None:
    entry = TensorEntry(
        name="emb.weight",
        layer_id=-1,
        dtype="bfloat16",
        shape=[4, 4],
        offset=0,
        length=100,
        alignment=4096,
        residency="streamed",
        dequant="trinity_layer",
        inner_offset=8,
        inner_length=40,
    )
    out = apply_residency_policy([entry], "partial", 2)[0]
    assert out.inner_offset == 8
    assert out.inner_length == 40


def test_trinity_layer_raw_bundle_roundtrip() -> None:
    t = torch.randn(16, 16, dtype=torch.bfloat16)
    lut = encode_trinity_lut2(t)
    disk, spans = encode_trinity_layer_bundle([lut], compress=False)
    assert disk[:4] == b"TCL\x02"
    io, ilen = spans[0]
    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=[16, 16],
        offset=0,
        length=len(disk),
        alignment=4096,
        residency="streamed",
        dequant="trinity_layer",
        inner_offset=io,
        inner_length=ilen,
    )
    cache = LayerZlibCache()
    out = decode_trinity_layer_to_tensor(disk, entry, cache, torch.device("cpu"))
    assert out.shape == t.shape
    disk_z, _ = encode_trinity_layer_bundle([lut], compress=True)
    out_z = decode_trinity_layer_to_tensor(
        disk_z, entry, LayerZlibCache(), torch.device("cpu")
    )
    assert torch.equal(out, out_z)


def test_trinity_layer_bundle_roundtrip() -> None:
    t = torch.randn(16, 16, dtype=torch.bfloat16)
    lut = encode_trinity_lut2(t)
    disk, spans = encode_trinity_layer_bundle([lut])
    layer = decompress_trinity_layer_blob(disk)
    assert len(spans) == 1
    io, ilen = spans[0]
    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=[16, 16],
        offset=0,
        length=len(disk),
        alignment=4096,
        residency="streamed",
        dequant="trinity_layer",
        inner_offset=io,
        inner_length=ilen,
    )
    cache = LayerZlibCache()
    out = decode_trinity_layer_to_tensor(disk, entry, cache, torch.device("cpu"))
    assert out.shape == t.shape


def test_trinity_layer_pack_streaming_golden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RWKV_ALLOW_TRINITY_LAYER", "1")
    pack = create_synthetic_pack(
        tmp_path / "pack",
        pack_codec="trinity",
        pack_layout="layer_grouped",
        quiet=True,
    )
    manifest = __import__(
        "rwkv_ssd.runtime.manifest", fromlist=["Manifest"]
    ).Manifest.load(pack)
    assert manifest.meta.get("pack_codec") == "trinity_layer"
    from tests.helpers import greedy_token_ids

    resident = greedy_token_ids(pack, "tl", mode="resident", max_tokens=8)
    streaming = greedy_token_ids(pack, "tl", mode="streaming", max_tokens=8)
    assert resident == streaming
