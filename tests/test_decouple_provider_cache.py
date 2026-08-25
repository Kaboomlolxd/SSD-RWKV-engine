"""Decoupled z eviction vs provider decoded-cache retention."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.stream_cache_policy import resolve_max_provider_cache_layers
from tests.helpers import greedy_token_ids


def test_resolve_provider_cache_cap_decoupled() -> None:
    cap = resolve_max_provider_cache_layers(
        0,
        12,
        max_layers_in_z=2,
        stream_layer_cache=True,
        decouple_provider_cache=True,
        warm_z=False,
    )
    assert cap == 12
    coupled = resolve_max_provider_cache_layers(
        0,
        12,
        max_layers_in_z=2,
        stream_layer_cache=True,
        decouple_provider_cache=False,
        warm_z=False,
    )
    assert coupled == 2


def test_decoupled_cache_survives_z_eviction(synthetic_pack_deep: Path) -> None:
    """After z LRU evicts a layer, provider tensors remain when decoupled."""
    from rwkv_ssd.runtime.engine import InferenceEngine

    cfg = EngineConfig(
        pack_dir=synthetic_pack_deep,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
        stream_layer_cache=True,
        max_layers_in_z=1,
        decouple_provider_cache=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("decouple")
        assert engine.metrics.provider_cache_bytes > 0
        assert sum(L.layer_cache_hits for L in engine.metrics.layers) > 0
    finally:
        engine.close()


def test_decoupled_matches_greedy(synthetic_pack_deep: Path) -> None:
    base = greedy_token_ids(
        synthetic_pack_deep, "parity", mode="streaming", max_tokens=8
    )
    decoupled = greedy_token_ids(
        synthetic_pack_deep,
        "parity",
        mode="streaming",
        max_tokens=8,
        stream_layer_cache=True,
        max_layers_in_z=1,
    )
    assert base == decoupled


def test_low_ram_preset_uses_partial_ssd_tier(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.throughput_defaults import apply_low_ram_defaults

    cfg = EngineConfig(pack_dir=tmp_path, mode="resident", low_ram=True)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "w.bin",
        tensors=[],
        meta={"n_layer": 4},
    )
    apply_low_ram_defaults(cfg, manifest)
    assert cfg.mode == "partial"
    assert cfg.stream_layer_cache is False
    assert cfg.max_layers_in_z == 0
    assert cfg.residency_profile is not None
