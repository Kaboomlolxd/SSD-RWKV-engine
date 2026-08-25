"""Provider LRU eviction by decoded byte budget."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def test_provider_byte_cap_evicts_old_layers(synthetic_pack_deep: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack_deep,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=6,
        stream_layer_cache=True,
        max_layers_in_z=1,
        decouple_provider_cache=True,
        max_provider_cache_layers=0,
        max_provider_cache_bytes=256 * 1024,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("byte cap")
        assert engine.metrics.provider_cache_bytes <= 256 * 1024 + 65536
    finally:
        engine.close()


def test_ram_budget_sets_provider_bytes(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.manifest import TensorEntry, Manifest
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    per_layer = 50 * 1024 * 1024
    tensors = [
        TensorEntry(
            "emb.weight",
            -1,
            "bf16",
            [512, 256],
            0,
            512 * 256 * 2,
            4096,
            "resident",
        ),
    ]
    for layer_id in range(20):
        tensors.append(
            TensorEntry(
                f"blocks.{layer_id}.att.key.weight",
                layer_id,
                "bf16",
                [256, 256],
                0,
                per_layer,
                4096,
                "streamed",
            )
        )
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=Path("w.bin"),
        tensors=tensors,
        meta={"n_layer": 20},
    )
    cfg = EngineConfig(pack_dir=tmp_path, ram_budget_gb=1.0)
    plan = apply_ram_budget_to_config(cfg, manifest, n_layer=20)
    assert plan.max_provider_cache_bytes > 0
    assert cfg.max_provider_cache_bytes == plan.max_provider_cache_bytes
    assert cfg.max_provider_cache_layers == 0
