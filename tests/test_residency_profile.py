"""Partial residency profile wiring."""

import json
from pathlib import Path

from rwkv_ssd.runtime.residency import apply_residency_policy
from rwkv_ssd.runtime.manifest import TensorEntry


def _entry(name: str, layer_id: int) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=layer_id,
        dtype="float32",
        shape=[4, 4],
        offset=0,
        length=64,
        alignment=4096,
        residency="streamed",
    )


def test_residency_profile_json(tmp_path: Path) -> None:
    profile = {
        "resident_layer_ids": [0, 1],
        "always_resident_tensors": ["embed", "head"],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    entries = [
        _entry("embed.weight", 0),
        _entry("blocks.1.weight", 1),
        _entry("blocks.2.weight", 2),
        _entry("head.weight", 9999),
    ]
    from rwkv_ssd.runtime.residency import load_residency_profile

    out = apply_residency_policy(
        entries, "partial", num_layers=4, profile=load_residency_profile(path)
    )
    by_name = {e.name: e.residency for e in out}
    assert by_name["embed.weight"] == "resident"
    assert by_name["blocks.1.weight"] == "resident"
    assert by_name["blocks.2.weight"] == "streamed"
    assert by_name["head.weight"] == "resident"
