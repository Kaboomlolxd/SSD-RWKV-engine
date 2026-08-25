"""P0.6: bench metrics accumulation fix.

The bench was previously calling ``eng.metrics.layers.clear()`` only after
warmup, then running multiple samples. ``start_layer`` always appends a
new row, so two samples on the same engine produced 2x the layer rows
and 2x the per-token ``compute_ms`` in the summary. This test pins the
behavior so a future refactor doesn't regress it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector


def test_clear_between_samples_keeps_metrics_isolated(tmp_path: Path) -> None:
    """The bench pattern is::

        for _ in range(samples):
            eng.metrics.layers.clear()
            eng.generate(...)

    so the per-sample metrics are isolated even when the same engine is
    reused. Verify the clear + append pattern is what we expect.
    """
    m = MetricsCollector()

    # First "sample": add 5 layer rows with 10ms compute each.
    m.layers.clear()
    for _ in range(5):
        L = m.start_layer(0)
        L.compute_ms = 10.0
    assert len(m.layers) == 5
    assert sum(L.compute_ms for L in m.layers) == 50.0

    # Second "sample": clear, add 5 fresh rows. Without the clear this
    # would have 10 rows (the bug); with clear it has 5.
    m.layers.clear()
    for _ in range(5):
        L = m.start_layer(0)
        L.compute_ms = 12.0
    assert len(m.layers) == 5
    assert sum(L.compute_ms for L in m.layers) == 60.0


def test_start_layer_always_appends_does_not_replace() -> None:
    """The ``start_layer`` design choice that made the bench bug possible:
    every call appends, no call replaces. Document it as a unit test
    so any future change to a "replace" semantics gets a heads-up.
    """
    m = MetricsCollector()
    m.start_layer(0).compute_ms = 1.0
    m.start_layer(0).compute_ms = 2.0
    m.start_layer(1).compute_ms = 3.0
    assert len(m.layers) == 3
    assert [L.compute_ms for L in m.layers] == [1.0, 2.0, 3.0]


def test_run_one_generate_signature() -> None:
    """Lock the bench helper signature: ``snapshot_layers`` stays the only
    way to slice pre-warmup rows, and the call must be cheap to run in
    a tight sample loop (no extra provider re-initialization)."""
    from bench.bench_io_ceiling import _run_one_generate
    import inspect

    sig = inspect.signature(_run_one_generate)
    assert list(sig.parameters) == ["eng", "ctx", "max_tokens", "snapshot_layers"]
    # All kwargs after the first two.
    for name in ("max_tokens", "snapshot_layers"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
