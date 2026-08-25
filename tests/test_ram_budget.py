"""RAM budget planner for partial + bounded caches."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.ram_budget import compute_ram_budget_plan, global_decoded_bytes
from rwkv_ssd.runtime.throughput_defaults import apply_low_ram_defaults


def test_tiny_budget_pins_few_layers(synthetic_pack_deep: Path) -> None:
    from rwkv_ssd.runtime.manifest import Manifest

    manifest = Manifest.load(synthetic_pack_deep)
    plan = compute_ram_budget_plan(manifest, 0.15, n_layer=12)
    assert plan.max_provider_cache_bytes > 0
    assert len(plan.resident_layer_ids) >= 1
    assert plan.estimated_peak_bytes <= int(0.15 * 1e9 * 1.1)


def test_10gb_budget_200b_class_plan() -> None:
    """Illustrative 200B-class shape: large globals + ~2GB/layer."""
    n_embd = 8192
    vocab = 65536
    emb_numel = vocab * n_embd
    head_numel = n_embd * vocab
    per_layer = 2_000_000_000  # ~2 GB decoded per block (illustrative)

    from rwkv_ssd.runtime.manifest import TensorEntry, Manifest

    tensors = [
        TensorEntry(
            "emb.weight", -1, "bf16", [vocab, n_embd], 0, emb_numel * 2, 4096, "resident"
        ),
        TensorEntry(
            "head.weight", -1, "bf16", [vocab, n_embd], 0, head_numel * 2, 4096, "resident"
        ),
        TensorEntry(
            "ln_out.weight", -1, "bf16", [n_embd], 0, n_embd * 2, 4096, "resident"
        ),
        TensorEntry(
            "ln_out.bias", -1, "bf16", [n_embd], 0, n_embd * 2, 4096, "resident"
        ),
        TensorEntry(
            "blocks.0.ln0.weight", 0, "bf16", [n_embd], 0, n_embd * 2, 4096, "resident"
        ),
        TensorEntry(
            "blocks.0.ln0.bias", 0, "bf16", [n_embd], 0, n_embd * 2, 4096, "resident"
        ),
    ]
    for layer_id in range(100):
        tensors.append(
            TensorEntry(
                f"blocks.{layer_id}.att.key.weight",
                layer_id,
                "bf16",
                [n_embd, n_embd],
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
        meta={"n_layer": 100, "n_embd": n_embd},
    )
    plan = compute_ram_budget_plan(manifest, 10.0, n_layer=100)
    assert plan.estimated_peak_gb <= 10.5
    assert 0 in plan.resident_layer_ids
    assert plan.max_provider_cache_bytes > 0
    assert global_decoded_bytes(manifest.tensors) > 2e9


def test_low_ram_small_pack_uses_partial_ssd_tier(tmp_path: Path) -> None:
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", low_ram=True)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=Path("w.bin"),
        tensors=[],
        meta={"n_layer": 4},
    )
    apply_low_ram_defaults(cfg, manifest)
    assert cfg.mode == "partial"
    assert cfg.stream_layer_cache is False
    assert cfg.max_layers_in_z == 0


def test_ram_budget_config_partial_mode(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    cfg = EngineConfig(pack_dir=tmp_path, ram_budget_gb=10.0)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=Path("w.bin"),
        tensors=[
            __import__("rwkv_ssd.runtime.manifest", fromlist=["TensorEntry"]).TensorEntry(
                "emb.weight",
                -1,
                "bf16",
                [1024, 256],
                0,
                1024 * 256 * 2,
                4096,
                "resident",
            ),
        ],
        meta={"n_layer": 80},
    )
    apply_ram_budget_to_config(cfg, manifest, n_layer=80)
    assert cfg.mode == "partial"
    assert cfg.residency_profile_inline is not None
    assert cfg.max_provider_cache_bytes > 0
    assert cfg.max_provider_cache_layers == 0
