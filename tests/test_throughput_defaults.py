"""Streaming I/O defaults from thesis P2.a."""

from __future__ import annotations

import os
from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.throughput_defaults import (
    apply_auto_residency_policy,
    apply_cache_format_defaults,
    apply_streaming_defaults,
)


def test_auto_residency_selects_packed_for_quant_without_budget(tmp_path) -> None:
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                "blocks.0.att.key.weight",
                0,
                "bf16",
                [64, 64],
                0,
                1024,
                4096,
                "streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 1},
    )
    cfg = EngineConfig(
        pack_dir=tmp_path, mode="streaming", residency_policy="auto"
    )
    assert apply_auto_residency_policy(cfg, manifest) == "packed"


def test_auto_residency_uses_dense_when_cache_budget_fits(tmp_path) -> None:
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[TensorEntry("blocks.0.weight", 0, "bf16", [4, 4], 0, 32, 1, "streamed")],
        meta={"n_layer": 1},
    )
    cfg = EngineConfig(
        pack_dir=tmp_path,
        mode="streaming",
        residency_policy="auto",
        cache_budget_gb=0.001,
    )
    assert apply_auto_residency_policy(cfg, manifest) == "dense"


def test_streaming_enables_layer_size_chunks(tmp_path) -> None:
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_streaming_defaults(cfg)
    assert cfg.io_chunk_policy == "layer_size"
    assert cfg.mmap_sequential is True


def test_explicit_cache_format_overrides_auto_provider_cache(tmp_path) -> None:
    cfg = EngineConfig(
        pack_dir=tmp_path,
        mode="streaming",
        cache_format="none",
        stream_layer_cache=True,
        max_layers_in_z=2,
        max_provider_cache_layers=8,
        max_provider_cache_bytes=1234,
    )
    apply_cache_format_defaults(cfg)
    assert cfg.stream_layer_cache is False
    assert cfg.warm_z is False
    assert cfg.max_layers_in_z == 0
    assert cfg.max_provider_cache_layers == 0
    assert cfg.max_provider_cache_bytes == 0


def test_partial_applies_hot7_profile(tmp_path) -> None:
    from pathlib import Path

    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.throughput_defaults import apply_partial_defaults

    cfg = EngineConfig(pack_dir=tmp_path, mode="partial")
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=Path("weights.bin"),
        tensors=[],
        meta={"n_layer": 12},
    )
    apply_partial_defaults(cfg, manifest)
    assert cfg.residency_profile is not None
    name = cfg.residency_profile.name
    assert "hot7" in name or "hot4" in name


def test_partial_fused_defaults_hot3(tmp_path, monkeypatch) -> None:
    from rwkv_ssd.runtime.throughput_defaults import apply_partial_fused_defaults

    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    monkeypatch.delenv("RWKV_LUT_GEMM_FUSED", raising=False)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 12},
    )
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_partial_fused_defaults(cfg, manifest)
    assert cfg.mode == "partial"
    assert cfg.residency_profile is not None
    assert "hot3" in cfg.residency_profile.name
    assert cfg.stream_layer_cache is True
    assert cfg.max_layers_in_z == 2
    # The maintained SSD tier retains a bounded decoded working set. Full
    # provider retention is an explicit F5/cache-budget choice, not the
    # silent default for a streaming profile.
    assert cfg.max_provider_cache_layers == 2
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "0"
    assert os.environ.get("RWKV_LUT_GEMM_FUSED") == "1"


def test_bounded_fused_provider_cache_is_bounded_by_default(tmp_path, monkeypatch) -> None:
    """F2 retains a small decoded working set unless F5 is explicit."""
    from rwkv_ssd.runtime.throughput_defaults import apply_bounded_fused_defaults
    from rwkv_ssd.runtime.stream_cache_policy import (
        resolve_max_provider_cache_layers,
    )
    from rwkv_ssd.runtime.layer_keys import layer_ids

    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name=f"blocks.{i}.att.key.weight",
                layer_id=i,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
            for i in range(12)
        ],
        meta={"n_layer": 12},
    )
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_bounded_fused_defaults(cfg, manifest)
    assert cfg.stream_layer_cache is True
    assert cfg.decouple_provider_cache is True
    assert cfg.max_provider_cache_layers == 2
    n_block = len(layer_ids(manifest.tensors))
    resolved = resolve_max_provider_cache_layers(
        configured=cfg.max_provider_cache_layers,
        n_block_layers=n_block,
        max_layers_in_z=cfg.max_layers_in_z,
        stream_layer_cache=cfg.stream_layer_cache,
        decouple_provider_cache=cfg.decouple_provider_cache,
        warm_z=cfg.warm_z,
    )
    assert resolved == 2


def test_partial_ssd_tier_provider_cache_left_to_resolver(
    tmp_path, monkeypatch
) -> None:
    """Same as above for ``apply_partial_ssd_tier_defaults`` (F3 path) —
    must not hardcode the cache size; the resolver decides."""
    from rwkv_ssd.runtime.throughput_defaults import apply_partial_ssd_tier_defaults

    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 12},
    )
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_partial_ssd_tier_defaults(cfg, manifest)
    assert cfg.stream_layer_cache is False
    # F3 is the strict-fused (no-cache) path; the cache doesn't apply.
    assert cfg.max_provider_cache_layers == 0


def test_partial_ssd_tier_defaults_hot3(tmp_path, monkeypatch) -> None:
    from rwkv_ssd.runtime.throughput_defaults import apply_partial_ssd_tier_defaults

    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 12},
    )
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_partial_ssd_tier_defaults(cfg, manifest)
    assert cfg.mode == "partial"
    assert cfg.stream_layer_cache is False
    assert cfg.max_layers_in_z == 0
    assert cfg.residency_profile is not None
    assert "hot3" in cfg.residency_profile.name
    assert os.environ.get("RWKV_LUT_GEMM_FUSED") == "1"
    assert os.environ.get("RWKV_PREFER_FUSED_LUT") == "1"


def test_resident_unchanged(tmp_path) -> None:
    cfg = EngineConfig(pack_dir=tmp_path, mode="resident", io_chunk_policy="uniform")
    apply_streaming_defaults(cfg)
    assert cfg.io_chunk_policy == "uniform"
    assert cfg.mmap_sequential is False


def test_shadow_pack_defaults_decode_shadow_on(tmp_path, monkeypatch) -> None:
    shadow = tmp_path / "shadow.bin"
    shadow.write_bytes(b"\x00" * 8)
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.time_decay",
                layer_id=0,
                dtype="bf16",
                shape=[1],
                offset=0,
                length=2,
                alignment=4096,
                residency="streamed",
                fast_offset=0,
                fast_length=2,
            )
        ],
        meta={"shadow_file": "shadow.bin"},
    )
    monkeypatch.delenv("RWKV_DECODE_SHADOW", raising=False)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_streaming_defaults(cfg, manifest)
    assert os.environ.get("RWKV_DECODE_SHADOW") == "1"


def test_trinity_streaming_auto_stream_layer_cache(tmp_path) -> None:
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 12},
    )
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_streaming_defaults(cfg, manifest)
    assert cfg.stream_layer_cache is True


def test_stream_layer_cache_env_off(tmp_path, monkeypatch) -> None:
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 12},
    )
    monkeypatch.setenv("RWKV_STREAM_LAYER_CACHE", "0")
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_streaming_defaults(cfg, manifest)
    assert cfg.stream_layer_cache is False


def test_auto_residency_maps_all_ram_frontier_tiers(tmp_path, monkeypatch) -> None:
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_tier, select_ram_budget_tier

    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                "blocks.0.ffn.key.weight",
                0,
                "bf16",
                [32, 32],
                0,
                512,
                1,
                "streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": 1},
    )
    for budget_gb, expected in (
        (0.15, "packed"),
        (0.21, "prepared"),
        (0.27, "prepared"),
        (0.28, "prepared"),  # F4's distinct floor
        (0.39, "dense"),
    ):
        cfg = EngineConfig(
            pack_dir=tmp_path,
            mode="streaming",
            residency_policy="auto",
            ram_budget_gb=budget_gb,
        )
        tier = select_ram_budget_tier(budget_gb)
        apply_ram_budget_tier(cfg, manifest, tier=tier)
        assert apply_auto_residency_policy(cfg, manifest) == expected


def test_auto_residency_explicit_format_wins_and_static_is_legacy(tmp_path) -> None:
    manifest = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[TensorEntry("blocks.0.weight", 0, "bf16", [4, 4], 0, 32, 1, "streamed")],
        meta={"n_layer": 1},
    )
    explicit = EngineConfig(
        pack_dir=tmp_path,
        mode="streaming",
        residency_policy="auto",
        cache_format="dense",
    )
    explicit._ram_budget_tier_applied = "F1"
    assert apply_auto_residency_policy(explicit, manifest) == "dense"
    assert explicit.cache_format == "dense"

    legacy = EngineConfig(pack_dir=tmp_path, mode="streaming")
    legacy._ram_budget_tier_applied = "F5"
    assert apply_auto_residency_policy(legacy, manifest) == "auto"
    assert legacy.cache_format == "auto"
