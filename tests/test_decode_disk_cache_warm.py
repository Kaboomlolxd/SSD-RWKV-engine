"""Tests for decode disk cache warm at load."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.decode_disk_cache import DecodeDiskCache, warm_disk_cache_layers
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.metrics import MetricsCollector


def _make_provider(tmp_path: Path, entries: list[TensorEntry]) -> ManifestWeightProvider:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"\x00" * 8192)
    from rwkv_ssd.runtime.weight_store import open_weight_store

    store = open_weight_store(weights, backend="mmap")
    metrics = MetricsCollector()
    return ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=entries,
        device=torch.device("cpu"),
        metrics=metrics,
        stream_layer_cache=False,
        decode_disk_cache="1",
        pack_dir=tmp_path,
        manifest_meta={"weights_sha256": "test"},
    )


def test_warm_disk_cache_skips_resident_layer(tmp_path) -> None:
    entries = [
        TensorEntry(
            name="blocks.3.att.receptance.weight",
            layer_id=3,
            dtype="bf16",
            shape=[4, 4],
            offset=0,
            length=64,
            alignment=4096,
            residency="streamed",
            dequant="none",
        ),
    ]
    provider = _make_provider(tmp_path, entries)
    provider.set_model_z({"blocks.3.att.receptance.weight": torch.zeros(4, 4)})
    disk = provider._disk_cache
    assert disk is not None
    by_layer = {3: entries}
    n = warm_disk_cache_layers(disk, provider, by_layer, [3])
    assert n == 0
    provider.close()


def test_warm_disk_cache_materializes_fused_lut_entries(
    synthetic_pack_trinity_lut2: Path, monkeypatch
) -> None:
    """Disk cache warm must bf16-decode fused att mats (not blob-only fused path)."""
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
    from rwkv_ssd.runtime.weight_store import open_weight_store
    from rwkv_ssd.runtime.metrics import MetricsCollector

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_ids = sorted(by_layer.keys())
    layer_id = next(i for i in layer_ids if i >= 0)
    entries = by_layer[layer_id]
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
        decode_disk_cache="1",
        pack_dir=synthetic_pack_trinity_lut2,
        manifest_meta=manifest.meta,
    )
    fused = provider.load_layer_tensors(entries)
    partial = provider.load_layer_tensors_materialized(entries)
    assert len(partial) >= len(fused)
    assert all(e.name in partial for e in entries if e.residency == "streamed")
    provider.close()


def test_promote_auto_bounded_on_large_model(monkeypatch) -> None:
    from rwkv_ssd.runtime.rwkv7_weights import promote_full_z_enabled

    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    assert promote_full_z_enabled(12) is False
    assert promote_full_z_enabled(2) is True
    monkeypatch.setenv("RWKV_PROMOTE_FULL_Z", "1")
    assert promote_full_z_enabled(12) is True
