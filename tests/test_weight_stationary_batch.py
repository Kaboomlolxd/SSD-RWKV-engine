from __future__ import annotations

from rwkv_ssd.backends.synthetic import SyntheticBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine

from tests.helpers import greedy_token_ids


def test_weight_stationary_batch_matches_independent_generation(synthetic_pack) -> None:
    prompts = ["a", "longer", "xyz"]
    max_tokens = 4
    expected = [
        greedy_token_ids(
            synthetic_pack,
            prompt,
            max_tokens=max_tokens,
            cache_format="none",
        )
        for prompt in prompts
    ]
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=max_tokens,
        cache_format="none",
        prefetch_enabled=False,
    )
    with InferenceEngine(cfg) as engine:
        assert isinstance(engine.backend, SyntheticBackend)
        provider = engine._get_or_create_pack_provider()
        engine.metrics.reset_for_generate()
        actual = engine.backend.generate_greedy_batch(
            prompts, provider, max_tokens, engine.metrics
        )
        assert actual == expected


def test_weight_stationary_batch_amortizes_layer_loads(synthetic_pack) -> None:
    prompts = ["ab", "abcdef", "xyz"]
    max_tokens = 3
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=max_tokens,
        cache_format="none",
        prefetch_enabled=False,
    )
    with InferenceEngine(cfg) as engine:
        provider = engine._get_or_create_pack_provider()
        engine.metrics.reset_for_generate()
        engine.backend.generate_greedy_batch(prompts, provider, max_tokens, engine.metrics)
        n_layer = engine.backend.num_layers
        expected_sweeps = max(len(prompt.encode("utf-8")) for prompt in prompts) + max_tokens
        independent_sweeps = sum(len(prompt.encode("utf-8")) for prompt in prompts) + len(prompts) * max_tokens
        assert engine.metrics.batch_size == len(prompts)
        assert engine.metrics.weight_sweeps == expected_sweeps
        assert engine.metrics.weight_layer_loads == expected_sweeps * n_layer
        assert engine.metrics.weight_layer_loads < independent_sweeps * n_layer
        assert engine.metrics.tokens_generated == len(prompts) * max_tokens


def test_engine_generate_batch_returns_one_result_per_prompt(synthetic_pack) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        cache_format="none",
        prefetch_enabled=False,
    )
    with InferenceEngine(cfg) as engine:
        outputs = engine.generate_batch(["one", "two"])
        assert len(outputs) == 2
        metrics = engine.metrics.to_dict()
        assert metrics["batch_size"] == 2
        assert metrics["tokens_generated"] == 4
        assert metrics["weight_sweeps"] == 5


def test_engine_generate_batch_empty_input_is_noop(synthetic_pack) -> None:
    cfg = EngineConfig(pack_dir=synthetic_pack, backend="synthetic")
    with InferenceEngine(cfg) as engine:
        assert engine.generate_batch([]) == []
