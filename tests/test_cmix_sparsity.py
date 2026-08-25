"""Opt-in CPU CMix activation telemetry tests."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.weight_store import open_weight_store


def test_cmix_sparsity_reports_samples_fractions_and_tile_occupancy(
    synthetic_pack: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RWKV_CMIX_SPARSITY", "1")
    monkeypatch.setenv("RWKV_CMIX_TILE_STATS", "1")
    monkeypatch.setenv("RWKV_CMIX_TILE_SIZE", "2")
    manifest = Manifest.load(synthetic_pack)
    store = open_weight_store(manifest.weights_path, backend="pread")
    provider = ManifestWeightProvider(
        "streaming",
        store,
        manifest.tensors,
        torch.device("cpu"),
        MetricsCollector(),
    )
    try:
        provider.record_cmix_sparsity(torch.tensor([0.0, 0.0, 1.0, -2.0, 0.0]))
        stats = provider.cache_stats()
        assert stats["cmix_samples"] == 1
        assert stats["cmix_zero_elements"] == 3
        assert stats["cmix_active_elements"] == 2
        assert stats["cmix_total_elements"] == 5
        assert stats["cmix_tile_size"] == 2
        assert stats["cmix_tile_samples"] == 3
        assert stats["cmix_tile_active_fraction"] == 1 / 3
        assert stats["cmix_tile_occupancy"] == {"0.0": 2, "1.0": 1}
    finally:
        provider.close()
        store.close()


def test_cmix_sparsity_is_disabled_by_default(
    synthetic_pack: Path, monkeypatch
) -> None:
    monkeypatch.delenv("RWKV_CMIX_SPARSITY", raising=False)
    monkeypatch.delenv("RWKV_CMIX_TILE_STATS", raising=False)
    manifest = Manifest.load(synthetic_pack)
    store = open_weight_store(manifest.weights_path, backend="pread")
    provider = ManifestWeightProvider(
        "streaming",
        store,
        manifest.tensors,
        torch.device("cpu"),
        MetricsCollector(),
    )
    try:
        provider.record_cmix_sparsity(torch.tensor([0.0, 1.0]))
        stats = provider.cache_stats()
        assert stats["cmix_samples"] == 0
        assert stats["cmix_tile_samples"] == 0
    finally:
        provider.close()
        store.close()
