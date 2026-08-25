"""MetricsCollector summary includes cache_writes and prefill."""

from __future__ import annotations

from rwkv_ssd.runtime.metrics import MetricsCollector


def test_summary_empty_layers() -> None:
    """No layers: short-circuit, doesn't include numeric fields."""
    m = MetricsCollector()
    assert m.summary() == "no layer timings recorded"


def test_summary_includes_cache_writes_and_prefill() -> None:
    m = MetricsCollector()
    m.tokens_generated = 8
    m.total_wall_s = 2.0
    m.prefill_wall_s = 0.456
    m.cache_write_submits = 11
    m.cache_write_sync_ms = 0.9
    L = m.start_layer(0)
    L.read_ms = 2.0
    L.compute_ms = 7.0
    out = m.summary()
    assert "cache_writes=11" in out
    assert "cache_sync_ms=0.9" in out
    assert "prefill=0.456s" in out
    assert "tok/s=4.00" in out


def test_summary_omits_prefill_when_zero() -> None:
    """Don't show prefill=0.000s when unset."""
    m = MetricsCollector()
    m.tokens_generated = 1
    m.total_wall_s = 1.0
    L = m.start_layer(0)
    L.compute_ms = 1.0
    out = m.summary()
    assert "prefill=" not in out
