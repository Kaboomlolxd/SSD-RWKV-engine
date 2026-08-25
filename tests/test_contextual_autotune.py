from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.contextual_autotune import (
    ContextualAutotuner,
    KnobProfile,
    TuningContext,
    TuningObservation,
    machine_fingerprint,
)


def _profiles() -> list[KnobProfile]:
    return [
        KnobProfile("strict", cache_format="none", prefetch_enabled=False, estimated_ram_bytes=100),
        KnobProfile("packed", cache_format="packed", prefetch_enabled=True, estimated_ram_bytes=200),
        KnobProfile("prepared", cache_format="prepared", prepared_cache_bytes=512, estimated_ram_bytes=500),
    ]


def test_contextual_tuner_explores_then_selects_best_profile() -> None:
    tuner = ContextualAutotuner(_profiles(), hysteresis=0.05)
    context = TuningContext(prompt_tokens=20, expected_decode_tokens=32)
    observed = {"strict": 3.0, "packed": 6.0, "prepared": 5.0}
    for expected_name in ("strict", "packed", "prepared"):
        decision = tuner.choose(context)
        assert decision.exploratory
        assert decision.new_profile == expected_name
        tuner.record(context, expected_name, TuningObservation(tok_s=observed[expected_name]))
    decision = tuner.choose(context)
    assert not decision.exploratory
    assert decision.new_profile == "packed"


def test_context_buckets_learn_independently() -> None:
    tuner = ContextualAutotuner(_profiles())
    short = TuningContext(prompt_tokens=8, expected_decode_tokens=8)
    long = TuningContext(prompt_tokens=1000, expected_decode_tokens=200)
    assert tuner.choose(short).new_profile == "strict"
    assert tuner.choose(long).new_profile == "strict"
    assert tuner.snapshot()["contexts"] == 2


def test_ram_budget_filters_profiles() -> None:
    tuner = ContextualAutotuner(_profiles())
    context = TuningContext(
        prompt_tokens=1, expected_decode_tokens=1, available_ram_bytes=150
    )
    assert tuner.choose(context).new_profile == "strict"


def test_hysteresis_keeps_near_tied_current_profile() -> None:
    tuner = ContextualAutotuner(
        _profiles()[:2], max_explorations_per_context=2, hysteresis=0.1
    )
    context = TuningContext(prompt_tokens=1, expected_decode_tokens=20)
    first = tuner.choose(context)
    tuner.record(context, first.new_profile, TuningObservation(tok_s=10.0))
    second = tuner.choose(context)
    tuner.record(context, second.new_profile, TuningObservation(tok_s=10.5))
    assert tuner.choose(context).new_profile == "packed"


def test_profile_applies_multiple_engine_knobs() -> None:
    profile = KnobProfile(
        "test",
        cache_format="packed",
        prefetch_enabled=False,
        io_chunk_bytes=4096,
        io_chunk_policy="uniform",
        packed_cache_bytes=123,
    )
    tuner = ContextualAutotuner([profile], max_explorations_per_context=0)
    context = TuningContext(prompt_tokens=1, expected_decode_tokens=1)
    decision = tuner.choose(context)
    config = EngineConfig(pack_dir=Path("pack"))
    tuner.apply(config, decision)
    assert config.cache_format == "packed"
    assert config.prefetch_enabled is False
    assert config.io_chunk_bytes == 4096
    assert config.packed_cache_bytes == 123


def test_contextual_profile_persists_only_for_matching_machine(tmp_path) -> None:
    context = TuningContext(prompt_tokens=20, expected_decode_tokens=32)
    tuner = ContextualAutotuner(_profiles())
    decision = tuner.choose(context)
    tuner.record(context, decision.new_profile, TuningObservation(tok_s=4.0))
    path = tuner.save(tmp_path / "machine-profile.json", machine_fingerprint="machine-a")

    restored = ContextualAutotuner(_profiles())
    assert restored.load(path, machine_fingerprint="machine-a") is True
    assert restored.snapshot()["contexts"] == 1
    rejected = ContextualAutotuner(_profiles())
    assert rejected.load(path, machine_fingerprint="machine-b") is False
    assert rejected.snapshot()["contexts"] == 0


def test_machine_fingerprint_changes_with_backend_or_pack() -> None:
    one = machine_fingerprint(backend="synthetic", pack_identity="a")
    two = machine_fingerprint(backend="rwkvcpp", pack_identity="a")
    three = machine_fingerprint(backend="synthetic", pack_identity="b")
    assert len(one) == 64
    assert len({one, two, three}) == 3
