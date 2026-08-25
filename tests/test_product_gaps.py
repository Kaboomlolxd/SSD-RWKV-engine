"""Tests for remaining product-gap features."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState
from rwkv_ssd.runtime.transcript_cache import sha256_text, split_transcript


def test_split_transcript() -> None:
    rendered = "user: hello\nassistant: hi\nuser: follow up"
    key, stable, suffix = split_transcript(rendered)
    assert stable == "user: hello\nassistant: hi\n"
    assert suffix == "user: follow up"
    assert key == sha256_text(stable)


def test_transcript_prefix_cache_hit(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        state_cache=True,
        prefix_cache_mode="transcript",
    )
    with InferenceEngine(cfg) as engine:
        p1 = "user: hello\nassistant:"
        engine.generate(p1)
        assert engine.prefix_cache is not None
        engine.generate("user: hello\nassistant:\nuser: again\nassistant:")
        assert engine.prefix_cache.stats.hits >= 1


def test_generate_followup_without_full_replay(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
    )
    with InferenceEngine(cfg) as engine:
        engine.generate("user: hello\nassistant:")
        out = engine.generate_followup("\nuser: again\nassistant:", max_tokens=2)
        assert isinstance(out, str)
        assert engine.backend.get_recurrent_state() is not None


def test_prefix_cache_contains_without_stats(synthetic_pack: Path) -> None:
    cache = PrefixStateCache(max_entries=4, disk_dir=synthetic_pack)
    key = "abc"
    cache.put(
        key,
        RecurrentState(h=torch.zeros(4), last_token_id=1),
    )
    assert cache.contains(key)
    before = cache.stats.hits
    assert cache.contains(key)
    assert cache.stats.hits == before


def test_auto_cache_budget_sets_bytes(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        cache_budget_auto=True,
    )
    with InferenceEngine(cfg) as engine:
        assert engine.config.cache_budget_gb is not None
        assert engine.config.max_provider_cache_bytes > 0
