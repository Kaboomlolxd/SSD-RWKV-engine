"""Plan partial residency + cache caps to stay under a decoded-RAM budget."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.rwkv7_skeleton import is_global_z_key, layer_id_from_z_key

logger = logging.getLogger(__name__)

_BYTES_PER_ELEM = 2  # bf16 decode target for block weights
_STATE_RESERVE_BYTES = 512 * 1024 * 1024  # recurrent state + runtime headroom
_MAX_PINNED_LAYERS = 4

_TIER_MB = {
    "F1": 201.0,
    "F2": 203.0,
    "F3": 262.0,
    "F4": 280.0,
    "F5": 382.0,
}


def select_ram_budget_tier(budget_gb: float) -> str:
    """Pick the highest tier that fits ``budget_gb`` (RAM frontier presets F1-F5).

    The table is calibrated to the 0.1B numbers in ``docs/PRESETS.md``; on larger
    packs the same call returns a tier whose floor still leaves room for globals.
    Returns the tier name (``"F1"``..``"F5"``).
    """
    if budget_gb <= 0:
        raise ValueError("budget_gb must be positive")
    budget_mb = budget_gb * 1024.0
    for tier in ("F5", "F4", "F3", "F2", "F1"):
        if budget_mb >= _TIER_MB[tier]:
            return tier
    return "F1"


def apply_ram_budget_tier(
    config: Any,
    manifest: Manifest,
    *,
    tier: str,
) -> str:
    """Apply the named tier preset to ``config`` (no residency override).

    Mirrors the preset functions in ``throughput_defaults`` for the four
    RAM-frontier tiers; returns the tier name applied. Idempotent for ``F1``.

    Marks ``config._ram_budget_tier_applied = <tier>`` so a follow-up
    :func:`apply_ram_budget_to_config` does not override fields the tier
    preset has already set (F4/F5 mode/``warm_z``/``max_layers_in_z``).
    """
    from rwkv_ssd.runtime.throughput_defaults import (
        apply_bounded_fused_defaults,
        apply_partial_hot4_ssd_defaults,
        apply_partial_ssd_tier_defaults,
        apply_promote_max_defaults,
        apply_ssd_tier_fused_defaults,
    )

    if tier == "F1":
        apply_ssd_tier_fused_defaults(config, manifest)
    elif tier == "F2":
        apply_bounded_fused_defaults(config, manifest)
    elif tier == "F3":
        apply_partial_ssd_tier_defaults(config, manifest)
    elif tier == "F4":
        apply_partial_hot4_ssd_defaults(config, manifest)
    elif tier == "F5":
        apply_promote_max_defaults(config, manifest)
        # F5 needs warm_z=True so the LRU cap is bypassed and the engine can
        # hold every block in z. The F5 preset only sets max_layers_in_z when
        # warm_z is already True — without this it falls into the cold cap=1
        # path and Fb runs as a hybrid F1+F3 (the bug that left Fb at
        # 1.9 tok/s instead of ~9 on 0.1B). Set explicitly here.
        config.warm_z = True
    else:
        raise ValueError(f"unknown RAM budget tier: {tier!r}")
    config._ram_budget_tier_applied = tier
    return tier


@dataclass(frozen=True)
class RamBudgetPlan:
    budget_bytes: int
    global_bytes: int
    per_layer_bytes: int
    resident_layer_ids: list[int]
    max_layers_in_z: int
    max_provider_cache_layers: int
    max_provider_cache_bytes: int
    estimated_peak_bytes: int

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / 1e9

    @property
    def estimated_peak_gb(self) -> float:
        return self.estimated_peak_bytes / 1e9


def _is_skeleton_tensor(entry: TensorEntry) -> bool:
    if is_global_z_key(entry.name):
        return True
    if entry.name in ("blocks.0.ln0.weight", "blocks.0.ln0.bias"):
        return True
    return False


def decoded_tensor_bytes(entry: TensorEntry) -> int:
    return entry.numel * _BYTES_PER_ELEM


def layer_decoded_bytes(entries: list[TensorEntry], layer_id: int) -> int:
    total = 0
    prefix = f"blocks.{layer_id}."
    for entry in entries:
        if entry.layer_id == layer_id or entry.name.startswith(prefix):
            total += decoded_tensor_bytes(entry)
    return total


def global_decoded_bytes(entries: list[TensorEntry]) -> int:
    total = 0
    for entry in entries:
        if _is_skeleton_tensor(entry):
            total += decoded_tensor_bytes(entry)
    return total


def block_layer_ids(entries: list[TensorEntry]) -> list[int]:
    found: set[int] = set()
    for entry in entries:
        if entry.layer_id >= 0:
            found.add(entry.layer_id)
            continue
        lid = layer_id_from_z_key(entry.name)
        if lid is not None:
            found.add(lid)
    return sorted(found)


def compute_ram_budget_plan(
    manifest: Manifest,
    budget_gb: float,
    *,
    n_layer: int | None = None,
) -> RamBudgetPlan:
    """
    Fit globals + pinned early layers + bounded caches under ``budget_gb``.

    Middle/late layers stream from SSD; cross-token reuse uses ``.decode_cache/``.
    """
    entries = manifest.tensors
    budget_bytes = max(0, int(float(budget_gb) * 1e9))
    state_reserve = min(
        _STATE_RESERVE_BYTES,
        max(64 * 1024 * 1024, int(budget_bytes * 0.1)),
    )
    global_b = global_decoded_bytes(entries)
    layers = block_layer_ids(entries)
    if n_layer is not None and n_layer > 0:
        layers = [lid for lid in layers if 0 <= lid < n_layer]
    if not layers:
        n = int(manifest.meta.get("n_layer", 0))
        layers = list(range(max(0, n)))

    layer_sizes = {lid: layer_decoded_bytes(entries, lid) for lid in layers}
    sized = [layer_sizes[lid] for lid in layers if layer_sizes[lid] > 0]
    per_layer = int(sum(sized) / len(sized)) if sized else 32 * 1024 * 1024

    used = global_b + state_reserve
    pinned: list[int] = []

    for lid in layers:
        need = layer_sizes.get(lid, per_layer)
        if len(pinned) >= _MAX_PINNED_LAYERS:
            break
        if used + need > int(budget_bytes * 0.92):
            break
        pinned.append(lid)
        used += need

    last = layers[-1] if layers else None
    if last is not None and last not in pinned:
        need = layer_sizes.get(last, per_layer)
        if used + need <= int(budget_bytes * 0.95):
            pinned.append(last)
            used += need

    remaining = max(0, budget_bytes - used)
    provider_bytes = int(remaining * 0.88)
    provider_bytes = max(per_layer, provider_bytes)
    max_z = min(2, max(1, provider_bytes // max(2 * per_layer, 1)))
    provider_cap_layers = max(1, provider_bytes // max(per_layer, 1))

    # Peak: globals + pinned + provider byte window + z retention
    peak = global_b + sum(layer_sizes.get(lid, per_layer) for lid in pinned)
    peak += provider_bytes
    peak += max_z * per_layer
    peak += state_reserve

    return RamBudgetPlan(
        budget_bytes=budget_bytes,
        global_bytes=global_b,
        per_layer_bytes=per_layer,
        resident_layer_ids=sorted(set(pinned)),
        max_layers_in_z=max_z,
        max_provider_cache_layers=int(provider_cap_layers),
        max_provider_cache_bytes=int(provider_bytes),
        estimated_peak_bytes=int(peak),
    )


def residency_profile_from_plan(plan: RamBudgetPlan) -> dict[str, Any]:
    return {
        "description": f"ram budget ~{plan.budget_gb:.1f} GB (auto)",
        "resident_layer_ids": list(plan.resident_layer_ids),
        "always_resident_tensors": ["embed", "head", "ln_out"],
    }


def apply_ram_budget_to_config(
    config: Any,
    manifest: Manifest,
    *,
    n_layer: int | None = None,
) -> RamBudgetPlan:
    """Mutate ``config`` for partial + bounded caches under ``config.ram_budget_gb``.

    Sets the residency profile + provider byte cap from the plan but **does not
    override** ``config.mode``/``stream_layer_cache``/``warm_z``/
    ``max_layers_in_z`` if a tier preset has already set them
    (``config._ram_budget_tier_applied``). Call ``apply_ram_budget_tier`` first.

    On **small models** (``n_layer < 16``) leaves the provider byte cap uncapped
    — the 0.1B whole skeleton is already 201 MB so any per-layer byte cap
    throttles tok/s well below the docs claim. F4/F5 paths already skip the
    cap via the tier-applied marker.
    """
    budget = float(getattr(config, "ram_budget_gb", 0) or 0)
    if budget <= 0:
        raise ValueError("ram_budget_gb must be positive")
    plan = compute_ram_budget_plan(manifest, budget, n_layer=n_layer)
    tier_applied = getattr(config, "_ram_budget_tier_applied", None)
    small_model = n_layer is not None and 0 < n_layer < 16
    if not getattr(config, "mode", None) or config.mode == "resident":
        config.mode = "partial"
    if getattr(config, "warm_z", None) is None and not tier_applied:
        config.warm_z = False
    if not getattr(config, "decouple_provider_cache", False):
        config.decouple_provider_cache = True
    if not tier_applied:
        config.max_layers_in_z = plan.max_layers_in_z
        config.max_provider_cache_layers = 0
        if not small_model:
            config.max_provider_cache_bytes = plan.max_provider_cache_bytes
    if getattr(config, "decode_disk_cache", None) is None:
        config.decode_disk_cache = "auto"
    if not tier_applied:
        # F1-F3 (and any non-tier path) get a planner-built residency profile
        # so the per-layer load path knows which layers to pin. F4/F5 tier
        # presets either set their own residency profile (F3 via deploy/*.json)
        # or rely on warm-z promote (F5) — the planner's profile is wrong for
        # both, so skip it when a tier preset is in effect.
        config.residency_profile_inline = residency_profile_from_plan(plan)
    logger.info(
        "ram budget %.1f GB: tier=%s pin layers %s, max_z=%d, provider=%.2f GB, "
        "est. peak %.2f GB (globals %.2f GB, ~%.1f MB/layer)",
        plan.budget_gb,
        tier_applied or "none",
        plan.resident_layer_ids,
        plan.max_layers_in_z,
        plan.max_provider_cache_bytes / 1e9,
        plan.estimated_peak_gb,
        plan.global_bytes / 1e6,
        plan.per_layer_bytes / 1e6,
    )
    return plan
