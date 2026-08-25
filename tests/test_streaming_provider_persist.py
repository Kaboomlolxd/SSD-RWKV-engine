"""Persistent streaming provider + promote frees provider RAM."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_store import open_weight_store


def test_promote_frees_provider_cache(synthetic_pack_deep: Path) -> None:
    manifest = Manifest.load(synthetic_pack_deep)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
        cfg = EngineConfig(
            pack_dir=synthetic_pack_deep,
            mode="streaming",
            device="cpu",
            stream_layer_cache=True,
            max_layers_in_z=1,
            decouple_provider_cache=True,
        )
        provider = create_weight_provider(
            cfg,
            store,
            manifest.tensors,
            torch.device("cpu"),
            metrics,
        )
        by_layer = manifest.by_layer()
        layer_ids = sorted(by_layer.keys())
        for layer_id in layer_ids:
            entries = by_layer[layer_id]
            tensors = provider.load_layer_tensors(entries)
            provider.prepare_layer_for_z(layer_id, tensors)
        assert provider.cached_weight_bytes() > 0
        provider.release_all_streamed_layers()
        block_bytes = sum(
            provider._layer_cached_bytes(lid) for lid in layer_ids if lid >= 0
        )
        assert block_bytes == 0
    finally:
        store.close()


@pytest.mark.chatrwkv
def test_engine_persists_chatrwkv_provider(tmp_path: Path) -> None:
    import os

    for var in (
        "RWKV_LUT_GEMM_FUSED",
        "RWKV_STRICT_FUSED_RETAIN",
        "RWKV_STRICT_FUSED_RETAIN_SHADOW",
        "RWKV_PREFER_FUSED_LUT",
        "RWKV_PROMOTE_FULL_Z",
        "RWKV_DECODE_SHADOW",
    ):
        os.environ.pop(var, None)
    ckpt = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    pack = Path("test_model/trinity_eval/trinity_lut2_0.1b")
    if not ckpt.is_file() or not pack.is_dir():
        return
    from rwkv_ssd.runtime.pack_profiles import resolve_trinity_pack

    pack = resolve_trinity_pack(pack)
    if not (pack / "quality_certificate.json").is_file():
        pytest.skip("active real lossy pack has no quality certificate")
    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt),
        backend="chatrwkv",
        mode="streaming",
        strategy="cpu bf16",
        device="cpu",
        max_tokens=4,
        greedy=True,
        skeleton_load=True,
    )
    with InferenceEngine(cfg) as engine:
        engine.generate("persist")
        prov1 = engine._streaming_provider
        assert prov1 is not None
        engine.generate("persist")
        assert engine._streaming_provider is prov1
