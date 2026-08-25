"""CPU storage diagnostics across runtime-pack layouts."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.tools.shard_pack import shard_pack
from rwkv_ssd.tools.storage_diagnostic import run_storage_diagnostic


def _assert_schema(result: dict) -> None:
    assert result["schema_version"] == 1
    assert result["file_size_bytes"] > 0
    assert result["tensor_count"] > 0
    assert result["logical_tensor_bytes"] > 0
    assert result["sequential_tensor_read"]["median_ms"] >= 0
    assert result["sequential_tensor_read"]["median_mbps"] >= 0
    assert result["per_layer"]
    assert "likely_warm_cache" in result["cache_behavior"]
    assert result["hardware_validation"]["physical_multi_ssd_validated"] is False


def test_storage_diagnostic_single_file_pack(synthetic_pack: Path) -> None:
    result = run_storage_diagnostic(
        synthetic_pack,
        backend="pread",
        repeats=1,
        max_layers=2,
    )
    _assert_schema(result)
    assert result["sharded"] is False
    assert result["shard_concurrency"]["supported"] is False


def test_storage_diagnostic_layer_affinity_pack(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    output = tmp_path / "layer-sharded"
    shard_pack(synthetic_pack, output, n_shards=2, strategy="layer")
    result = run_storage_diagnostic(
        output,
        repeats=1,
        shard_workers=2,
        max_layers=2,
    )
    _assert_schema(result)
    assert result["sharded"] is True
    assert result["shard_concurrency"]["supported"] is True
    assert result["shard_concurrency"]["shard_count"] == 2


def test_storage_diagnostic_manifest_v2_striped_pack(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    output = tmp_path / "striped"
    shard_pack(
        synthetic_pack,
        output,
        n_shards=2,
        strategy="stripe",
        stripe_bytes=64,
    )
    result = run_storage_diagnostic(
        output,
        repeats=1,
        shard_workers=2,
        max_layers=2,
    )
    _assert_schema(result)
    assert result["manifest_version"] == 2
    assert result["layout"] == "striped_round_robin"
    assert result["sharded"] is True
