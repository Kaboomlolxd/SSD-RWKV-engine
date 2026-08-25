"""Stream layer RAM cache (throughput vs strict streaming)."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.weight_store import open_weight_store
from tests.helpers import greedy_token_ids


def test_layer_cache_matches_greedy_without_cache(synthetic_pack: Path) -> None:
    base = greedy_token_ids(synthetic_pack, "cache parity", mode="streaming", max_tokens=8)
    cached = greedy_token_ids(
        synthetic_pack,
        "cache parity",
        mode="streaming",
        max_tokens=8,
        stream_layer_cache=True,
    )
    assert base == cached


def test_bounded_cache_matches_greedy_without_cache(synthetic_pack: Path) -> None:
    base = greedy_token_ids(synthetic_pack, "bounded", mode="streaming", max_tokens=8)
    bounded = greedy_token_ids(
        synthetic_pack,
        "bounded",
        mode="streaming",
        max_tokens=8,
        stream_layer_cache=True,
        max_layers_in_z=1,
    )
    assert base == bounded


def test_explicit_disk_cache_format_matches_legacy_strict(synthetic_pack: Path) -> None:
    base = greedy_token_ids(
        synthetic_pack, "disk format", mode="streaming", max_tokens=8
    )
    explicit = greedy_token_ids(
        synthetic_pack,
        "disk format",
        mode="streaming",
        max_tokens=8,
        cache_format="none",
    )
    assert base == explicit


def test_layer_cache_records_hits(synthetic_pack: Path) -> None:
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
        engine.generate("cache hits")
        assert sum(L.layer_cache_hits for L in engine.metrics.layers) > 0
    finally:
        engine.close()


def test_bounded_provider_cache_smaller_than_full(synthetic_pack_deep: Path) -> None:
    """Bounded cache caps provider tensor RAM, not only model.z."""
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine

    pack = synthetic_pack_deep

    def provider_bytes(*, max_layers: int, provider_cap: int) -> int:
        cfg = EngineConfig(
            pack_dir=pack,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            max_tokens=8,
            stream_layer_cache=True,
            max_layers_in_z=max_layers,
            max_provider_cache_layers=provider_cap,
            decouple_provider_cache=True,
        )
        engine = InferenceEngine(cfg)
        engine.load()
        try:
            engine.generate("cap")
            return engine.metrics.provider_cache_bytes
        finally:
            engine.close()

    bounded = provider_bytes(max_layers=1, provider_cap=1)
    full = provider_bytes(max_layers=1, provider_cap=99)
    assert bounded < full


def test_fused_gate_tensors_participate_in_z_lru() -> None:
    from rwkv_ssd.runtime.z_layer_retention import ZLayerRetention, block_layer_ids_in_z

    z: dict[str, torch.Tensor] = {
        "blocks.0.att.x_r": torch.zeros(8),
        "blocks.1.att.x_r": torch.zeros(8),
        "blocks.2.att.x_r": torch.zeros(8),
    }
    retention = ZLayerRetention(max_layers_in_z=1, pinned_layer_ids={0})
    retention.touch(z, 1)
    retention.touch(z, 2)
    assert block_layer_ids_in_z(z) == [0, 2]
    assert retention._lru == [2]


def test_layer_weights_in_z_helper() -> None:
    from rwkv_ssd.runtime.rwkv7_weights import layer_weights_in_z

    z = {"blocks.3.att.receptance.weight": torch.zeros(4, 4)}
    assert layer_weights_in_z(z, 3)
    assert not layer_weights_in_z(z, 4)


def test_packed_cache_eviction_clears_tmix_aliases(
    synthetic_pack: Path,
) -> None:
    manifest = Manifest.load(synthetic_pack)
    store = open_weight_store(manifest.weights_path, backend="pread")
    provider = ManifestWeightProvider(
        "streaming",
        store,
        manifest.tensors,
        torch.device("cpu"),
        MetricsCollector(),
        max_packed_cache_bytes=10,
    )
    first = TensorEntry(
        "blocks.0.att.key.weight", 0, "u8", [1, 1], 0, 8, 1, "streamed"
    )
    second = TensorEntry(
        "blocks.1.att.key.weight", 1, "u8", [1, 1], 0, 8, 1, "streamed"
    )
    try:
        blob0 = b"0" * 8
        provider._register_fused_lut_blob(first, blob0)
        provider._fused_tmix_blobs["blocks.0.att."] = (
            (blob0, blob0, blob0, blob0),
            1,
            1,
        )
        assert provider.packed_weight_bytes() == 8
        provider._register_fused_lut_blob(second, b"1" * 8)
        assert "blocks.0.att.key.weight" not in provider._fused_lut_blobs
        assert "blocks.0.att." not in provider._fused_tmix_blobs
        assert provider.cache_stats()["packed_cache_evictions"] == 1
        assert provider.cache_stats()["packed_cache_bytes"] <= 10
    finally:
        provider.close()
        store.close()


def test_prepared_byte_cap_survives_multiple_layers_and_reuse(
    synthetic_pack: Path,
) -> None:
    manifest = Manifest.load(synthetic_pack)
    store = open_weight_store(manifest.weights_path, backend="pread")
    provider = ManifestWeightProvider(
        "streaming",
        store,
        manifest.tensors,
        torch.device("cpu"),
        MetricsCollector(),
        stream_layer_cache=True,
        max_provider_cache_bytes=48,
        max_provider_cache_layers=0,
    )
    try:
        for layer_id in range(3):
            tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            prepared = provider.prepare_layer_for_z(
                layer_id,
                {f"blocks.{layer_id}.ffn.test.weight": tensor},
            )
            before_reuse = provider.prepared_weight_bytes()
            reused = provider.prepare_layer_for_z(
                layer_id,
                {f"blocks.{layer_id}.ffn.test.weight": tensor.clone()},
            )
            assert reused is prepared
            assert provider.prepared_weight_bytes() == before_reuse
            provider._retain_provider_layer(layer_id)
            assert provider.prepared_weight_bytes() <= 48
        assert len(provider._provider_lru) <= 1
        assert provider.cache_stats()["provider_cache_evictions"] == 2
    finally:
        provider.close()
        store.close()


def test_selected_cache_format_is_available_at_load_and_in_json(
    synthetic_pack: Path,
) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=2,
        cache_format="prepared",
        prepared_cache_bytes=1024 * 1024,
    )
    with InferenceEngine(cfg) as engine:
        assert engine.metrics.cache_format == "prepared"
        assert engine.metrics.to_dict()["cache_format"] == "prepared"
        engine.generate("cache format metrics")
        payload = engine.metrics.to_dict()
        assert payload["cache_format"] == "prepared"
        assert payload["prepared_cache_bytes"] <= cfg.prepared_cache_bytes
