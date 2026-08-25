"""Tests for pack_composition (B3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_bench import pack_full_stats


def test_pack_composition_written_in_meta_json(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack
    from rwkv_ssd.tools.pack_runtime import _compute_pack_composition

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    comp = _compute_pack_composition(pack)
    assert comp["weights_bin_mb"] > 0
    assert comp["total_mb"] > 0

    meta_path = pack / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "pack_composition" in meta
    assert meta["pack_composition"]["weights_bin_mb"] == comp["weights_bin_mb"]


def test_manifest_pack_composition_accessor(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    manifest = Manifest.load(pack)
    comp = manifest.pack_composition()
    assert comp["weights_bin_mb"] > 0
    assert "total_mb" in comp


def test_manifest_pack_composition_empty_when_missing(tmp_path: Path) -> None:
    """Packs built before B3 have no pack_composition block in manifest.json."""
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    manifest_path = pack / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "meta" in m and "pack_composition" in m["meta"]:
        m["meta"].pop("pack_composition")
    manifest_path.write_text(json.dumps(m), encoding="utf-8")

    manifest = Manifest.load(pack)
    assert manifest.pack_composition() == {}
