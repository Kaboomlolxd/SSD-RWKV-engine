"""Tests for pack_full_stats and storage_ratio_adjusted honesty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_ssd.runtime.pack_bench import pack_full_stats, pack_read_stats


def test_pack_full_stats_reports_all_components(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    stats = pack_full_stats(pack)

    assert stats["weights_bin_mb"] > 0
    assert "manifest_json_mb" in stats
    assert "meta_json_mb" in stats
    assert "shadow_bin_mb" in stats
    assert "decode_cache_mb" in stats
    assert "state_cache_mb" in stats
    assert "sidecar_mb" in stats
    assert "total_mb" in stats

    assert (pack / "manifest.json").is_file()
    assert (pack / "meta.json").is_file()
    assert (pack / "manifest.json").stat().st_size > 0
    assert (pack / "meta.json").stat().st_size > 0

    components = (
        stats["weights_bin_mb"]
        + stats["shadow_bin_mb"]
        + stats["manifest_json_mb"]
        + stats["meta_json_mb"]
        + stats["decode_cache_mb"]
        + stats["state_cache_mb"]
        + stats["sidecar_mb"]
    )
    assert abs(stats["total_mb"] - components) < 0.02, (
        f"total_mb={stats['total_mb']} but sum of components={components}"
    )


def test_pack_full_stats_counts_shadow_sidecar(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    weights_mb_before = pack_full_stats(pack)["weights_bin_mb"]

    shadow_path = pack / "shadow.bin"
    shadow_path.write_bytes(b"\x00" * (200 * 1024 * 1024))
    try:
        stats = pack_full_stats(pack)
        assert stats["shadow_bin_mb"] >= 200 * 1024 * 1024 / (1024 * 1024) - 0.5
        assert stats["total_mb"] > weights_mb_before
    finally:
        shadow_path.unlink(missing_ok=True)


def test_pack_full_stats_counts_decode_cache(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    decode_dir = pack / ".decode_cache"
    decode_dir.mkdir()
    (decode_dir / "layer_0.bin").write_bytes(b"\x00" * (4 * 1024 * 1024))
    try:
        stats = pack_full_stats(pack)
        assert stats["decode_cache_mb"] >= 4
        assert stats["sidecar_mb"] == 0.0
    finally:
        for f in decode_dir.iterdir():
            f.unlink()
        decode_dir.rmdir()


def test_pack_full_stats_legacy_field_kept(tmp_path: Path) -> None:
    """``weights_mb`` field kept for back-compat (matches pack_read_stats)."""
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    full = pack_full_stats(pack)
    legacy = pack_read_stats(pack)
    assert full["weights_mb"] == legacy["weights_mb"]


def test_pack_full_stats_empty_dir_handled(tmp_path: Path) -> None:
    """A dir that is not a pack: returns zeros rather than crashing."""
    fake = tmp_path / "not-a-pack"
    fake.mkdir()
    (fake / "manifest.json").write_text("{}")
    stats = pack_full_stats(fake)
    assert stats["weights_bin_mb"] == 0.0
    assert stats["total_mb"] == 0.0
