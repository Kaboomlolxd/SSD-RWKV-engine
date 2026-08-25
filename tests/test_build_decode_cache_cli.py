"""build_decode_cache --print-summary / --verify-only (DNV-11 distribution helpers)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_print_summary_empty(tmp_path: Path) -> None:
    """No .decode_cache/ → JSON prints layer_count=0 with stable schema."""
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "meta.json").write_text(
        '{"weights_sha256": "abc", "n_layer": 4}', encoding="utf-8"
    )
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "rwkv_ssd.tools.build_decode_cache",
            "--pack",
            str(pack),
            "--print-summary",
        ],
        text=True,
    )
    summary = json.loads(out)
    assert summary["layer_count"] == 0
    assert summary["total_bytes"] == 0
    assert summary["total_mb"] == 0.0
    assert summary["weights_key"] == "abc"
    assert "pack" in summary


def test_print_summary_with_cache(synthetic_pack: Path) -> None:
    """After build, --print-summary reports nonzero layer_count and total_bytes."""
    cache_dir = synthetic_pack / ".decode_cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "layer_0_aaaa.bin").write_bytes(b"\x00" * 1024)
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "rwkv_ssd.tools.build_decode_cache",
            "--pack",
            str(synthetic_pack),
            "--print-summary",
        ],
        text=True,
    )
    summary = json.loads(out)
    assert summary["layer_count"] == 1
    assert summary["total_bytes"] == 1024
    assert summary["total_mb"] == round(1024 / 1e6, 2)


def test_print_summary_prefers_sharded_manifest_key(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "weights.shard.0.bin").write_bytes(b"0")
    (pack / "weights.shard.1.bin").write_bytes(b"1")
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "weights_file": "weights.shard.0.bin",
                "weights_files": ["weights.shard.0.bin", "weights.shard.1.bin"],
                "tensors": [
                    {
                        "name": "a",
                        "dtype": "float32",
                        "shape": [1],
                        "offset": 0,
                        "length": 1,
                        "alignment": 1,
                        "residency": "streamed",
                        "shard_file": "weights.shard.0.bin",
                    }
                ],
                "meta": {
                    "weights_sha256_by_file": {"weights.shard.0.bin": "a", "weights.shard.1.bin": "b"}
                },
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "rwkv_ssd.tools.build_decode_cache",
            "--pack",
            str(pack),
            "--print-summary",
        ],
        text=True,
    )
    summary = json.loads(out)
    assert summary["weights_key"].startswith("sharded:")
