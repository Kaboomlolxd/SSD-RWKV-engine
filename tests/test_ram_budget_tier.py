"""RAM budget auto-profile — F1/F2/F3/F5 tier selector."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.ram_budget import (
    apply_ram_budget_tier,
    select_ram_budget_tier,
)


def _empty_manifest(tmp_path: Path) -> Manifest:
    return Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "w.bin",
        tensors=[],
        meta={"n_layer": 12},
    )


def test_select_tier_fits_floor() -> None:
    """Tier floors: F5 ~382MB, F3 ~262MB, F2 ~203MB, F1 ~201MB."""
    assert select_ram_budget_tier(0.5) == "F5"
    assert select_ram_budget_tier(0.4) == "F5"
    assert select_ram_budget_tier(0.27) == "F3"
    assert select_ram_budget_tier(0.21) == "F2"
    assert select_ram_budget_tier(0.15) == "F1"
    assert select_ram_budget_tier(0.1) == "F1"


def test_select_tier_rejects_zero() -> None:
    with pytest.raises(ValueError):
        select_ram_budget_tier(0.0)


def test_apply_tier_f1_sets_no_promote(tmp_path: Path) -> None:
    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_ram_budget_tier(cfg, _empty_manifest(tmp_path), tier="F1")
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "0"


def test_apply_tier_f5_promotes(tmp_path: Path) -> None:
    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_ram_budget_tier(cfg, _empty_manifest(tmp_path), tier="F5")
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "1"


def test_apply_tier_unknown_raises(tmp_path: Path) -> None:
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    with pytest.raises(ValueError):
        apply_ram_budget_tier(cfg, _empty_manifest(tmp_path), tier="F9")


def test_ram_budget_picks_f5_for_500mb(tmp_path: Path) -> None:
    """End-to-end: ram_budget_gb=0.5 selects F5, not just F3."""
    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", ram_budget_gb=0.5)
    manifest = _empty_manifest(tmp_path)
    tier = select_ram_budget_tier(0.5)
    apply_ram_budget_tier(cfg, manifest, tier=tier)
    assert tier == "F5"
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "1"


def test_ram_budget_f5_keeps_streaming_mode(tmp_path: Path) -> None:
    """Tier + plan compose: F5's streaming mode survives plan application."""
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", ram_budget_gb=0.5)
    manifest = _empty_manifest(tmp_path)
    tier = select_ram_budget_tier(0.5)
    apply_ram_budget_tier(cfg, manifest, tier=tier)
    assert cfg.mode == "streaming"
    plan = apply_ram_budget_to_config(cfg, manifest, n_layer=12)
    assert cfg.mode == "streaming", "plan must not override tier-set mode for F5"
    assert plan.max_layers_in_z >= 1


def test_apply_tier_f5_sets_warm_z(tmp_path: Path) -> None:
    """F-4: ``apply_ram_budget_tier`` for F5 must set ``warm_z=True`` so the
    engine actually promotes all blocks into ``z`` instead of running as F2
    with cap=1 (the bug that left Fb at 1.9 tok/s instead of ~9 on 0.1B)."""
    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", ram_budget_gb=0.5)
    assert cfg.warm_z is False
    apply_ram_budget_tier(cfg, _empty_manifest(tmp_path), tier="F5")
    assert cfg.warm_z is True
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "1"


def test_planner_does_not_override_tier_applied(tmp_path: Path) -> None:
    """F-4: when a tier preset has been applied, ``apply_ram_budget_to_config``
    must not clobber ``warm_z``/``max_layers_in_z`` (the Fb regression)."""
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(
        pack_dir=tmp_path, mode="streaming", ram_budget_gb=0.5, max_layers_in_z=0
    )
    manifest = _empty_manifest(tmp_path)
    apply_ram_budget_tier(cfg, manifest, tier="F5")
    assert cfg.warm_z is True
    pre_z = cfg.max_layers_in_z
    apply_ram_budget_to_config(cfg, manifest, n_layer=12)
    assert cfg.warm_z is True, "planner must not clear warm_z set by tier preset"
    assert cfg.max_layers_in_z == pre_z, (
        "planner must not override max_layers_in_z when tier preset set it"
    )


def test_low_ram_with_budget_applies_tier_first(tmp_path: Path) -> None:
    """low_ram + ram_budget_gb must select F5 and keep warm_z (Fb path)."""
    from rwkv_ssd.runtime.throughput_defaults import apply_low_ram_defaults

    os.environ.pop("RWKV_PROMOTE_FULL_Z", None)
    cfg = EngineConfig(
        pack_dir=tmp_path, mode="streaming", low_ram=True, ram_budget_gb=0.5
    )
    apply_low_ram_defaults(cfg, _empty_manifest(tmp_path))
    assert getattr(cfg, "_ram_budget_tier_applied", None) == "F5"
    assert cfg.warm_z is True
    assert os.environ.get("RWKV_PROMOTE_FULL_Z") == "1"


def test_planner_no_throttle_on_small_models(tmp_path: Path) -> None:
    """F-3: on n_layer < 16, ``apply_ram_budget_to_config`` must NOT set a
    ``max_provider_cache_bytes`` cap (would throttle F1-F3 tok/s well below
    what docs claim)."""
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", ram_budget_gb=0.5)
    manifest = _empty_manifest(tmp_path)
    apply_ram_budget_to_config(cfg, manifest, n_layer=12)
    assert cfg.max_provider_cache_bytes == 0, (
        f"n_layer=12 must skip byte cap, got {cfg.max_provider_cache_bytes}"
    )


def test_planner_throttles_on_large_models(tmp_path: Path) -> None:
    """F-3 inverse: on n_layer ≥ 16 (7B+), ``apply_ram_budget_to_config`` does
    set a provider byte cap from the plan — that's the path calibrated for
    big models where 0.5 GB must not auto-promote."""
    from rwkv_ssd.runtime.ram_budget import apply_ram_budget_to_config

    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", ram_budget_gb=10.0)
    manifest = _empty_manifest(tmp_path)
    apply_ram_budget_to_config(cfg, manifest, n_layer=32)
    assert cfg.max_provider_cache_bytes > 0, (
        f"n_layer=32 must set byte cap, got {cfg.max_provider_cache_bytes}"
    )
