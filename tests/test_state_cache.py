"""M2.5 prefix / state cache tests."""

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState
from tests.helpers import greedy_token_ids


def _read_ms(metrics: MetricsCollector) -> float:
    return sum(x.read_ms for x in metrics.layers)


def test_state_cache_hit_skips_system_prefill_reads(synthetic_pack: Path) -> None:
    system = "SYSTEM:" * 12
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
        state_cache=True,
        system_prefix=system,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate(" user-one")
        cold_reads = _read_ms(engine.metrics)

        engine.metrics = MetricsCollector()
        engine.generate(" user-two")
        warm_reads = _read_ms(engine.metrics)

        assert engine.metrics.state_cache_hit
        assert warm_reads < cold_reads
        assert engine.prefix_cache is not None
        assert engine.prefix_cache.stats.hits >= 1
    finally:
        engine.close()


def test_state_cache_same_tokens_as_no_cache(synthetic_pack: Path) -> None:
    system = "SYS:"
    prompt = system + " ask"
    n = 8

    plain = greedy_token_ids(synthetic_pack, prompt, mode="streaming", max_tokens=n)

    cfg_cache = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=n,
        state_cache=True,
        system_prefix=system,
    )
    with InferenceEngine(cfg_cache) as engine:
        engine.generate(" ask")
        engine.metrics = MetricsCollector()
        engine.generate(" ask")
        assert engine.metrics.state_cache_hit

    cached_tokens = greedy_token_ids(
        synthetic_pack, prompt, mode="streaming", max_tokens=n
    )
    assert plain == cached_tokens


def test_prefix_cache_stores_rwkv7_state() -> None:
    cache = PrefixStateCache(max_entries=4)
    tensors = [torch.zeros(2, 3), torch.ones(1, 4)]
    cache.put("sys", RecurrentState(last_token_id=7, rwkv7_state=tensors))
    hit = cache.get("sys")
    assert hit is not None
    assert hit.rwkv7_state is not None
    assert hit.last_token_id == 7
    assert all(torch.equal(a, b) for a, b in zip(hit.rwkv7_state, tensors))
    tensors[0].fill_(99)
    again = cache.get("sys")
    assert again is not None and again.rwkv7_state is not None
    assert again.rwkv7_state[0][0, 0].item() == 0.0
