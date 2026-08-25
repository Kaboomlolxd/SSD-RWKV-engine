from __future__ import annotations

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.page_residency import (
    ExtentResidencyTracker,
    ObservedResidencyWeightStore,
)
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.weight_store import open_weight_store


def _entry() -> TensorEntry:
    return TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="u8",
        shape=[8],
        offset=4,
        length=8,
        alignment=1,
        residency="streamed",
    )


def test_extent_tracker_recognizes_covering_read_and_invalidation() -> None:
    tracker = ExtentResidencyTracker(ttl_s=30)
    assert not tracker.probably_resident(10, 5)
    tracker.observe(0, 32)
    assert tracker.probably_resident(10, 5)
    tracker.invalidate(8, 8)
    assert not tracker.probably_resident(10, 5)
    stats = tracker.stats()
    assert stats.observations == 1
    assert stats.probable_hits == 1
    assert stats.probable_misses == 2
    assert stats.invalidations == 1


def test_observed_store_tracks_real_span_reads(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(range(32)))
    inner = open_weight_store(path, backend="pread")
    store = ObservedResidencyWeightStore(inner)
    entry = _entry()
    try:
        assert not store.probably_resident([entry])
        assert store.read_bytes_span(0, 16) == bytes(range(16))
        assert store.probably_resident([entry])
        store.advise_release([entry])
        assert not store.probably_resident([entry])
    finally:
        store.close()


def test_provider_skips_os_prefetch_hint_for_probably_hot_extent(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(range(32)))
    store = ObservedResidencyWeightStore(open_weight_store(path, backend="pread"))
    entry = _entry()
    metrics = MetricsCollector()
    provider = ManifestWeightProvider(
        "streaming",
        store,
        [entry],
        torch.device("cpu"),
        metrics,
        prefetch=False,
    )
    try:
        store.read_bytes(entry)
        provider.hint_prefetch_layer([entry])
        assert metrics.page_cache_prefetch_skips == 1
    finally:
        provider.close()
        store.close()


def test_factory_enables_observed_residency_with_env(monkeypatch, tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"0" * 32)
    monkeypatch.setenv("RWKV_PAGE_RESIDENCY", "1")
    store = open_weight_store(path, backend="pread")
    try:
        assert isinstance(store, ObservedResidencyWeightStore)
    finally:
        store.close()
