"""Provider ``model.z`` awareness skips redundant I/O."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_store import open_weight_store


def test_prefetch_skips_layers_resident_in_z(synthetic_pack_deep: Path) -> None:
    manifest = Manifest.load(synthetic_pack_deep)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
        cfg = EngineConfig(
            pack_dir=synthetic_pack_deep,
            mode="streaming",
            device="cpu",
            prefetch_enabled=True,
            stream_layer_cache=True,
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
        lid0 = layer_ids[0]
        tensors = provider.load_layer_tensors(by_layer[lid0])
        prepared = provider.prepare_layer_for_z(lid0, tensors)
        z = {name: t.clone() for name, t in prepared.items()}
        z.setdefault(f"blocks.{lid0}.att.receptance.weight", torch.zeros(8))
        provider.set_model_z(z)
        assert provider._layer_resident_in_z(lid0)
        assert not provider._layer_resident_in_z(layer_ids[1])
    finally:
        store.close()
