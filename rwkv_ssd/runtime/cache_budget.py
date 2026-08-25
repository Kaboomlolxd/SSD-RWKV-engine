"""Auto provider-cache budget helpers."""

from __future__ import annotations

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest


def resolve_auto_cache_budget_gb(cfg: EngineConfig, manifest: Manifest) -> float:
    """Pick a provider-cache budget from RAM tier or pack footprint."""
    if cfg.ram_budget_gb and cfg.ram_budget_gb > 0:
        return max(0.05, float(cfg.ram_budget_gb) * 0.25)
    weights_path = manifest.weights_path
    pack_gb = weights_path.stat().st_size / 1e9 if weights_path.is_file() else 0.0
    if pack_gb >= 1.0:
        return max(0.25, pack_gb * 0.15)
    return 0.25
