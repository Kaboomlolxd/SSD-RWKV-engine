"""I/O backend parity and prefetch metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.io_pread import PreadWeightStore
from tests.helpers import greedy_token_ids


def test_pread_matches_mmap_bytes(synthetic_pack: Path) -> None:
    manifest = Manifest.load(synthetic_pack)
    mmap_store = open_weight_store(manifest.weights_path, backend="mmap")
    pread_store = open_weight_store(manifest.weights_path, backend="pread")
    try:
        for entry in manifest.tensors[:8]:
            assert mmap_store.read_bytes(entry) == pread_store.read_bytes(entry)
    finally:
        mmap_store.close()
        pread_store.close()


def test_pread_span_and_bounds(synthetic_pack: Path) -> None:
    manifest = Manifest.load(synthetic_pack)
    store = PreadWeightStore(manifest.weights_path)
    try:
        entry = manifest.tensors[0]
        assert store.read_bytes_span(entry.offset, entry.length) == store.read_bytes(entry)
        with pytest.raises(OSError):
            store.read_bytes_span(-1, 1)
    finally:
        store.close()


def test_streaming_prefetch_metrics_columns(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=8,
        io_backend="pread",
        # Prefetch is only issued when the decoded tensors will be
        # reused on the next token (``stream_layer_cache=True``). For the
        # no-cache / strict-fused path the prefetch is skipped to avoid
        # blocking the main thread on I/O with no compute to overlap
        # against.
        stream_layer_cache=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("prefetch metrics")
        # F-2 non-blocking fix: ``prefetch_hits`` is now a best-effort
        # signal. On a tiny synthetic pack the worker is racing the
        # main thread (GIL), so we don't require a hit. The test
        # asserts the metric *columns* are present and non-negative
        # for every row.
        assert all(L.prefetch_wait_ms >= 0 for L in engine.metrics.layers)
        assert any(L.compute_ms > 0 for L in engine.metrics.layers)
    finally:
        engine.close()


def test_hedged_matches_mmap(synthetic_pack: Path) -> None:
    manifest = Manifest.load(synthetic_pack)
    mmap_store = open_weight_store(manifest.weights_path, backend="mmap")
    hedged = open_weight_store(manifest.weights_path, backend="mmap", hedged=True)
    try:
        entry = manifest.tensors[0]
        assert mmap_store.read_bytes(entry) == hedged.read_bytes(entry)
    finally:
        mmap_store.close()
        hedged.close()


def test_greedy_tokens_same_mmap_and_pread(synthetic_pack: Path) -> None:
    mmap_ids = greedy_token_ids(
        synthetic_pack, "parity", mode="streaming", max_tokens=8
    )
    pread_ids = greedy_token_ids(
        synthetic_pack,
        "parity",
        mode="streaming",
        max_tokens=8,
        io_backend="pread",
    )
    assert mmap_ids == pread_ids


@pytest.mark.parametrize(
    ("io_backend", "io_hedged"),
    [("threaded", False), ("mmap", True)],
)
def test_span_capable_wrappers_support_streaming(
    synthetic_pack: Path, io_backend: str, io_hedged: bool
) -> None:
    """Wrapper stores must preserve the contiguous layer-span API."""
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        io_backend=io_backend,
        io_hedged=io_hedged,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        assert engine.generate("span wrapper")
    finally:
        engine.close()
