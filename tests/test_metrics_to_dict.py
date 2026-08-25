"""Tests for MetricsCollector.to_dict() (A4)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector


def test_to_dict_serializable_empty() -> None:
    m = MetricsCollector()
    d = m.to_dict()
    json.dumps(d)
    assert d["tokens_generated"] == 0
    assert d["total_wall_s"] == 0.0
    assert d["tok_s"] == 0.0
    assert d["layers"] == []


def test_to_dict_serializable_with_layers() -> None:
    m = MetricsCollector()
    m.tokens_generated = 10
    m.total_wall_s = 0.5
    m.prefill_wall_s = 0.2
    m.state_cache_hit = True
    m.prefetch_overlaps = 4
    m.weight_cache_bytes = 1024
    m.provider_cache_bytes = 2048
    m.z_bytes = 4096
    m.mtp_gate_open = 1
    m.cache_write_submits = 2
    m.cache_write_sync_ms = 0.5

    L = LayerTiming(layer_id=0)
    L.read_ms = 1.5
    L.staging_ms = 0.2
    L.h2d_ms = 0.0
    L.decode_device_ms = 0.15
    L.d2h_ms = 0.04
    L.compute_ms = 5.0
    L.prefetch_wait_ms = 0.1
    L.prefetch_hits = 1
    L.chunk_reads = 2
    L.layer_cache_hits = 1
    L.ngram_hits = 0
    L.shadow_hits = 0
    L.disk_cache_hits = 0
    m.layers.append(L)

    d = m.to_dict()
    blob = json.dumps(d)
    assert "tokens_generated" in blob
    assert d["tok_s"] == 20.0
    assert d["state_cache_hit"] is True
    assert d["prefetch_overlaps"] == 4
    assert d["mtp_gate_open"] == 1
    assert d["cache_write_submits"] == 2
    assert len(d["layers"]) == 1
    assert d["layers"][0]["layer_id"] == 0
    assert d["layers"][0]["read_ms"] == 1.5
    assert d["layers"][0]["decode_device_ms"] == 0.15
    assert d["layers"][0]["d2h_ms"] == 0.04
    assert d["layers"][0]["compute_ms"] == 5.0


def test_to_dict_in_engine(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        eng.generate("hello")
        d = eng.metrics.to_dict()
        assert d["tokens_generated"] == 4
        assert d["tok_s"] > 0
        assert d["layers"], "expected per-layer metrics"
        assert "cache_stats" not in d
    finally:
        eng.close()


def test_to_dict_includes_cache_stats_when_state_cache(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        state_cache=True,
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        eng.generate("hi")
        d = eng.metrics.to_dict()
        assert "cache_stats" not in d
    finally:
        eng.close()
