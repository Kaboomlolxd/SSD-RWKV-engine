from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector
from rwkv_ssd.runtime.promotion_planner import PromotionCandidate, plan_promotions


def _planning_engine(synthetic_pack):
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="streaming",
            max_tokens=8,
            session_promotion=True,
            session_expected_tokens=64,
            session_promotion_bytes=10_000_000,
        )
    )
    backend = ChatRWKVBackend()
    backend._model = SimpleNamespace(
        z={"emb.weight": torch.zeros(4, 4, dtype=torch.float32)}
    )
    engine.backend = backend
    engine.manifest = Manifest.load(synthetic_pack)
    return engine


def test_engine_observation_builds_profitable_next_request_plan(synthetic_pack) -> None:
    engine = _planning_engine(synthetic_pack)
    engine.metrics.tokens_generated = 8
    for layer_id in sorted(engine.manifest.by_layer()):
        if layer_id < 0:
            continue
        for _ in range(8):
            engine.metrics.layers.append(
                LayerTiming(layer_id=layer_id, read_ms=2.0, staging_ms=1.0)
            )
    engine._observe_session_promotion()
    plan = engine._session_promotion_last_plan
    assert plan is not None
    assert plan.expected_remaining_tokens == 56
    assert plan.resident_bytes <= engine.config.session_promotion_bytes
    assert plan.selected
    assert engine._session_promotion_pending is plan


def test_engine_observation_short_session_avoids_promotion(synthetic_pack) -> None:
    engine = _planning_engine(synthetic_pack)
    engine.config.session_expected_tokens = 8
    engine.metrics.tokens_generated = 8
    engine.metrics.layers.append(LayerTiming(layer_id=0, read_ms=10.0))
    engine._observe_session_promotion()
    assert engine._session_promotion_pending is None


def test_engine_applies_plan_and_records_actual_byte_delta(
    synthetic_pack, monkeypatch
) -> None:
    engine = _planning_engine(synthetic_pack)
    provider = SimpleNamespace(
        _z_retention=SimpleNamespace(pinned_layer_ids=set()),
        cached_weight_bytes=lambda: 0,
        evict_streamed_layer=lambda layer_id, force=False: None,
    )
    monkeypatch.setattr(
        engine,
        "_get_or_create_streaming_provider",
        lambda model_z=None: provider,
    )

    import rwkv_ssd.runtime.rwkv7_weights as weights

    def fake_warm(z, _provider, _by_layer, layer_ids, metrics):
        for layer_id in layer_ids:
            z[f"blocks.{layer_id}.att.receptance.weight"] = torch.zeros(
                8, 8, dtype=torch.float32
            )
            metrics.start_layer(layer_id).staging_ms = 1.0
        return len(layer_ids)

    monkeypatch.setattr(weights, "warm_stream_cache_layers_into_z", fake_warm)
    plan = plan_promotions(
        [PromotionCandidate(0, 256, 1.0, 2.0)],
        expected_remaining_tokens=16,
        ram_cap_bytes=1024,
    )
    engine.config.session_promotion_bytes = 1024
    engine._session_promotion_pending = plan
    engine._apply_session_promotion_boundary()
    assert provider._z_retention.pinned_layer_ids == {0}
    assert engine.metrics.session_promoted_layers == [0]
    assert engine.metrics.session_promotion_bytes == 256
    assert engine.metrics.session_promotion_ms >= 0
    assert engine._session_observation_start == 1


def test_engine_rolls_back_when_actual_dense_bytes_exceed_cap(
    synthetic_pack, monkeypatch
) -> None:
    engine = _planning_engine(synthetic_pack)
    provider = SimpleNamespace(
        _z_retention=SimpleNamespace(pinned_layer_ids=set()),
        cached_weight_bytes=lambda: 0,
        evict_streamed_layer=lambda layer_id, force=False: None,
    )
    monkeypatch.setattr(
        engine,
        "_get_or_create_streaming_provider",
        lambda model_z=None: provider,
    )
    import rwkv_ssd.runtime.rwkv7_weights as weights

    def oversized(z, _provider, _by_layer, layer_ids, metrics):
        z["blocks.0.att.receptance.weight"] = torch.zeros(1024, dtype=torch.float32)
        return 1

    monkeypatch.setattr(weights, "warm_stream_cache_layers_into_z", oversized)
    engine.config.session_promotion_bytes = 128
    engine._session_promotion_pending = plan_promotions(
        [PromotionCandidate(0, 64, 1.0, 2.0)],
        expected_remaining_tokens=16,
        ram_cap_bytes=128,
    )
    engine._apply_session_promotion_boundary()
    assert provider._z_retention.pinned_layer_ids == set()
    assert "blocks.0.att.receptance.weight" not in engine.backend._model.z
    assert engine.metrics.session_promoted_layers == []


def test_session_and_adaptive_controllers_cannot_stack(synthetic_pack) -> None:
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="streaming",
            adaptive_residency=True,
            session_promotion=True,
            session_promotion_bytes=1024,
        )
    )
    with pytest.raises(ValueError, match="alternative controllers"):
        engine.load()


def test_session_promotion_metrics_are_structured() -> None:
    metrics = MetricsCollector(
        session_promoted_layers=[1, 3],
        session_promotion_bytes=4096,
        session_promotion_ms=2.5,
        session_promotion_estimated_net_ms=12.0,
        session_expected_remaining_tokens=40,
    )
    payload = metrics.to_dict()
    assert payload["session_promoted_layers"] == [1, 3]
    assert payload["session_promotion_bytes"] == 4096
    metrics.reset_for_generate()
    assert metrics.session_promoted_layers == []
