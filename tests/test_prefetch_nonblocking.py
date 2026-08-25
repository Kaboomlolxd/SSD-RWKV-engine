"""Non-blocking prefetch behavior (F-2 follow-up).

The previous ``begin_layer`` always blocked the main thread on the
in-flight prefetch future (``fut.result()``), which on a slow SSD cost
~80-150 ms per layer — 17% of the wall time on the F2 path. The fix is
to make the drain best-effort: if the worker is already done, use the
data; if not, abandon and fall through to the synchronous read.

These tests verify:

* ``begin_layer`` returns immediately even when the prefetch is still
  in flight (no main-thread block).
* ``prefetch_wait_ms`` is 0 on the abandoned path (no false attribution
  of the worker's time to the main thread).
* ``prefetch_entries`` does not block on the previous prefetch when
  submitting a new one.
* The drain still works on the fast path (worker done before
  ``begin_layer``).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.weight_provider import (
    ManifestWeightProvider,
    _PrefetchJobResult,
    _LayerSpanPrefetch,
)
from rwkv_ssd.runtime.weight_store import open_weight_store


def _make_provider(pack: Path, **kwargs):
    import torch

    manifest = Manifest.load(pack)
    device = torch.device("cpu")
    store = open_weight_store(manifest.weights_path)
    metrics = MetricsCollector()
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=device,
        metrics=metrics,
        stream_layer_cache=True,
        **kwargs,
    )
    return provider, store, manifest


def test_begin_layer_does_not_block_on_pending_prefetch(
    synthetic_pack: Path,
) -> None:
    """Submit a slow prefetch; ``begin_layer`` must return in < 50 ms."""
    provider, store, manifest = _make_provider(synthetic_pack, prefetch=True)
    try:
        layer_entries = [
            e
            for e in manifest.tensors
            if e.layer_id == 0 and e.name.startswith("blocks.")
        ]
        if not layer_entries:
            pytest.skip("no block tensors on layer 0")

        blocker = threading.Event()

        def slow_job():
            blocker.wait(timeout=5.0)
            return _PrefetchJobResult(raw_by_layer={}, tensors={})

        provider._executor = ThreadPoolExecutor(max_workers=1)
        fut = provider._executor.submit(slow_job)
        provider._prefetch_future = fut
        assert not fut.done()

        t0 = time.perf_counter()
        timing = provider.begin_layer(0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, (
            f"begin_layer blocked for {elapsed_ms:.1f} ms on a pending "
            f"prefetch — should be non-blocking"
        )
        assert timing.prefetch_wait_ms == 0.0

        blocker.set()
    finally:
        if provider._executor:
            provider._executor.shutdown(wait=True)
        provider.close()
        store.close()


def test_begin_layer_drains_when_prefetch_done(
    synthetic_pack: Path,
) -> None:
    """If the worker finishes before ``begin_layer``, the data is used."""
    provider, store, manifest = _make_provider(synthetic_pack, prefetch=True)
    try:
        layer_entries = [
            e
            for e in manifest.tensors
            if e.layer_id == 0 and e.name.startswith("blocks.")
        ]
        if not layer_entries:
            pytest.skip("no block tensors on layer 0")

        spans = _LayerSpanPrefetch(
            weights=(b"\x00" * 16, 0),
        )
        result = _PrefetchJobResult(
            raw_by_layer={0: spans}, tensors={}
        )
        fut: Future = Future()
        fut.set_result(result)
        provider._prefetch_future = fut
        assert fut.done()

        timing = provider.begin_layer(0)
        assert 0 in provider._prefetch_raw
        assert timing.prefetch_wait_ms >= 0.0
    finally:
        provider.close()
        store.close()


def test_prefetch_entries_does_not_block_on_pending(
    synthetic_pack: Path,
) -> None:
    """Submitting a new prefetch must not block on the previous one."""
    provider, store, manifest = _make_provider(synthetic_pack, prefetch=True)
    try:
        blocker = threading.Event()

        def slow_job():
            blocker.wait(timeout=5.0)
            return _PrefetchJobResult(raw_by_layer={}, tensors={})

        provider._executor = ThreadPoolExecutor(max_workers=2)
        slow_fut = provider._executor.submit(slow_job)
        provider._prefetch_future = slow_fut
        assert not slow_fut.done()

        t0 = time.perf_counter()
        provider.prefetch_entries(
            [
                e
                for e in manifest.tensors
                if e.layer_id == 1 and e.name.startswith("blocks.")
            ]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, (
            f"prefetch_entries blocked for {elapsed_ms:.1f} ms on "
            f"a pending previous prefetch — should be non-blocking"
        )

        # Keep the single-worker prefetch queue bounded. Replacing a running
        # future would leave stale reads queued behind it on every layer.
        assert provider._prefetch_future is slow_fut
        blocker.set()
    finally:
        if provider._executor:
            provider._executor.shutdown(wait=True)
        provider.close()
        store.close()


def test_streaming_end_to_end_non_blocking_prefetch(
    synthetic_pack: Path,
) -> None:
    """End-to-end smoke: prefetch is non-blocking on a tiny synthetic pack.

    With the F-2 non-blocking fix, ``begin_layer`` returns immediately
    whether or not the prefetch is done. The wait time is bounded by
    the I/O scheduling latency only (no ``fut.result()`` block). The
    test asserts:

    * The peak ``prefetch_wait_ms`` stays near zero (no main-thread
      block on the worker).
    * Every layer row has ``prefetch_wait_ms >= 0`` (no negative waits
      from the abandoned-future path).
    * Forward progress is made (layer rows are emitted).
    """
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=8,
        stream_layer_cache=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("non-blocking prefetch")
        max_wait = max(
            (L.prefetch_wait_ms for L in engine.metrics.layers),
            default=0.0,
        )
        # Best-effort: even when the worker IS done, the wait is a
        # few microseconds (dict copy). 50 ms is the design budget for
        # the abandoned path (no block).
        assert max_wait < 50.0, (
            f"prefetch_wait_ms peak {max_wait:.1f} ms — should be near-zero "
            f"on a tiny synthetic pack"
        )
        assert all(L.prefetch_wait_ms >= 0 for L in engine.metrics.layers)
        assert len(engine.metrics.layers) > 0
    finally:
        engine.close()
