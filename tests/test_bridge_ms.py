"""Tests for bridge_ms derived field (B2)."""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_layer_io_stats_bridge_computed() -> None:
    from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector

    m = MetricsCollector()
    m.total_wall_s = 0.5
    m.tokens_generated = 10

    L = LayerTiming(layer_id=0)
    L.read_ms = 1.0
    L.staging_ms = 0.5
    L.h2d_ms = 0.0
    L.compute_ms = 2.0
    m.layers.append(L)

    from bench.bench_throughput import _layer_io_stats

    io = _layer_io_stats(m, backend="synthetic", tok_s=20.0, wall_s=0.5, max_tokens=10)
    assert "bridge_ms_per_token" in io
    assert "bridge_pct" in io
    assert io["bridge_ms_per_token"] is not None
    assert io["bridge_pct"] is not None

    ms_per_tok = 500.0 / 10.0
    classified = 1.0 + 0.5 + 0.0 + 2.0
    expected_bridge = max(0.0, ms_per_tok - classified / 10)
    assert abs(io["bridge_ms_per_token"] - round(expected_bridge, 2)) < 1e-3


def test_layer_io_stats_bridge_zero_when_no_layers() -> None:
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from bench.bench_throughput import _layer_io_stats

    m = MetricsCollector()
    io = _layer_io_stats(m, backend="synthetic", tok_s=0.0, wall_s=0.0, max_tokens=0)
    assert io["bridge_ms_per_token"] is None
    assert io["bridge_pct"] is None


def test_layer_io_stats_bridge_negative_clamped_to_zero() -> None:
    from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector
    from bench.bench_throughput import _layer_io_stats

    m = MetricsCollector()
    m.total_wall_s = 0.001
    m.tokens_generated = 10
    L = LayerTiming(layer_id=0)
    L.read_ms = 5.0
    L.staging_ms = 5.0
    L.compute_ms = 5.0
    m.layers.append(L)
    io = _layer_io_stats(m, backend="synthetic", tok_s=0.0, wall_s=0.001, max_tokens=10)
    assert io["bridge_ms_per_token"] == 0.0
    assert io["bridge_pct"] == 0.0
