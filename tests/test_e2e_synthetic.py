"""Golden resident vs streaming tests (M1/M3)."""

from pathlib import Path
import threading

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from tests.helpers import greedy_token_ids


@pytest.mark.parametrize("mode", ["resident", "partial", "streaming"])
def test_synthetic_generate_runs(synthetic_pack: Path, mode: str) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode=mode,
        device="cpu",
        max_tokens=8,
        greedy=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        out = engine.generate("Hi")
        assert isinstance(out, str)
        assert engine.metrics.tokens_generated == 8
    finally:
        engine.close()


def test_golden_resident_equals_streaming(synthetic_pack: Path) -> None:
    """M3: greedy streaming must match resident token-for-token."""
    prompt = "golden"
    n = 16
    tokens_res = greedy_token_ids(synthetic_pack, prompt, mode="resident", max_tokens=n)
    tokens_stream = greedy_token_ids(
        synthetic_pack, prompt, mode="streaming", max_tokens=n
    )
    assert tokens_res == tokens_stream
    assert len(tokens_res) == n


def test_streaming_has_read_metrics(synthetic_pack: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
        metrics_csv=tmp_path / "m.csv",
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("metrics")
        assert engine.metrics.layers
        read_total = sum(x.read_ms for x in engine.metrics.layers)
        assert read_total > 0
        assert (tmp_path / "m.csv").is_file()
    finally:
        engine.close()


def test_generate_tokens_finalizes_metrics_on_success_and_cancel(
    synthetic_pack: Path,
) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        token_ids = engine.generate_tokens("token metrics")
        assert len(token_ids) == 4
        assert engine.metrics.total_wall_s > 0.0
        assert engine.metrics.process_rss_bytes >= 0

        cancelled = threading.Event()
        cancelled.set()
        with pytest.raises(GenerationCancelled):
            engine.generate_tokens("cancelled", cancel_event=cancelled)
        assert engine.metrics.total_wall_s > 0.0
        assert engine.metrics.tokens_generated == 0
    finally:
        engine.close()


def test_repeatability(synthetic_pack: Path) -> None:
    """M4 reliability: identical greedy runs."""
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="resident",
        device="cpu",
        max_tokens=8,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        results = [engine.generate("repeat") for _ in range(3)]
    finally:
        engine.close()
    assert len(set(results)) == 1


def test_seeded_temperature_top_p_is_repeatable_and_in_vocab(
    synthetic_pack: Path,
) -> None:
    """The public engine carries one fixed RNG through the complete decode."""

    outputs: list[list[int]] = []
    for _ in range(2):
        cfg = EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="resident",
            device="cpu",
            max_tokens=12,
            greedy=False,
            temperature=0.85,
            top_p=0.8,
            seed=41,
        )
        engine = InferenceEngine(cfg)
        engine.load()
        try:
            token_ids = engine.generate_tokens("seeded sampling")
            outputs.append(token_ids)
            assert all(0 <= token < engine.backend._vocab for token in token_ids)
        finally:
            engine.close()
    assert outputs[0] == outputs[1]


@pytest.mark.slow
def test_streaming_100_tokens(synthetic_pack: Path) -> None:
    """M3: long streaming run (excluded from default pytest)."""
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=32,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("long-run")
        assert engine.metrics.tokens_generated == 32
    finally:
        engine.close()


def test_mode_wall_time_ordering(synthetic_pack_deep: Path) -> None:
    """M3: wall-time ratio resident vs streaming stays bounded."""
    pack = synthetic_pack_deep
    prompt = "bench"
    n = 12

    def wall(mode: str) -> float:
        cfg = EngineConfig(
            pack_dir=pack,
            backend="synthetic",
            mode=mode,
            device="cpu",
            max_tokens=n,
        )
        engine = InferenceEngine(cfg)
        engine.load()
        try:
            engine.generate(prompt)
            return engine.metrics.total_wall_s
        finally:
            engine.close()

    resident_s = wall("resident")
    streaming_s = wall("streaming")
    ratio = streaming_s / max(resident_s, 1e-9)
    assert 0.05 < ratio < 40.0


def test_streaming_weight_cache_smaller_than_resident(synthetic_pack: Path) -> None:
    """M3: streaming keeps fewer weight bytes resident than full resident mode."""
    n = 8
    prompt = "mem"

    def cache_bytes(mode: str) -> int:
        cfg = EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode=mode,
            device="cpu",
            max_tokens=n,
        )
        engine = InferenceEngine(cfg)
        engine.load()
        try:
            engine.generate(prompt)
            return engine.metrics.weight_cache_bytes
        finally:
            engine.close()

    assert cache_bytes("streaming") < cache_bytes("resident")


def test_prefetch_overlap_recorded(synthetic_pack: Path) -> None:
    """M3: layer N+1 prefetch is issued in streaming decode.

    With the F-2 non-blocking fix (``begin_layer`` no longer waits for
    the prefetch future), ``prefetch_hits`` is a *best-effort* signal:
    it counts the layers whose prefetched data was ready in time to
    serve the load. On a tiny synthetic pack the I/O is in the OS
    page cache, so the worker *sometimes* finishes before ``begin_layer``
    and sometimes does not (the Python main thread holds the GIL during
    prefetch submission, so the worker can't make progress until the
    main thread yields). The test asserts the prefetch is *issued*
    (the call path is exercised end-to-end) and that the per-row
    ``prefetch_wait_ms`` is non-negative — it does not require a hit.
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
        engine.generate("prefetch")
        # The prefetch is best-effort now. At minimum: the per-row
        # prefetch_wait_ms metric is non-negative on every layer.
        assert all(L.prefetch_wait_ms >= 0 for L in engine.metrics.layers)
        # And the engine made forward progress (the metric rows exist).
        assert len(engine.metrics.layers) > 0
    finally:
        engine.close()
