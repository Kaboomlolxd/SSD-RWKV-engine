"""Bounded block-layer retention in ChatRWKV ``model.z`` during streaming."""

from __future__ import annotations

import os
from collections.abc import Callable

from rwkv_ssd.runtime.rwkv7_weights import layer_weights_in_z


def block_layer_ids_in_z(z: dict[str, torch.Tensor]) -> list[int]:
    """Sorted block layer ids that currently have weights in ``z``.

    The packed-block step mirrors ``att.ln_x.weight`` / ``att.ln_x.bias``
    into ``z`` (one weight + one bias per layer ≈ 3 KB) so the
    fused matmul can look them up by key. Those mirrors are *not*
    full-block tensors — they're a tiny per-layer scratch area
    overwritten on the next layer's mirror. The cap
    (``max_layers_in_z``) is about *full-block* tensors, so the
    ln_x mirrors are excluded from the count here. The mirrors
    themselves are cheap (~3 KB each) and never need to be
    evicted; the LRU eviction handles them by dropping the whole
    layer (and the next layer's mirror overwrites the stale ln_x
    anyway).
    """
    found: set[int] = set()
    for key in z:
        if not key.startswith("blocks."):
            continue
        # Skip the ln_x mirror — it's not a full-block tensor.
        if "att.ln_x.weight" in key or "att.ln_x.bias" in key:
            continue
        try:
            found.add(int(key.split("blocks.")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(found)


def _layer_has_block_tensors(z: dict[str, torch.Tensor], layer_id: int) -> bool:
    """True when ``layer_id`` has block tensors in ``z`` (gates-only counts for fused).

    The packed path mirrors ``att.ln_x.*`` into ``z`` as a tiny scratch pad;
    those alone must not count as a full block for the LRU (same rule as
    ``block_layer_ids_in_z``).
    """
    if layer_weights_in_z(z, layer_id):
        return True
    prefix = f"blocks.{layer_id}."
    for key in z:
        if not key.startswith(prefix):
            continue
        if "att.ln_x.weight" in key or "att.ln_x.bias" in key:
            continue
        return True
    return False


def _resolve_accuracy_pin_active() -> bool:
    """``RWKV_PIN_ACCURACY_LAYERS=auto|1|0`` — pin accuracy-sensitive layers."""
    raw = os.environ.get("RWKV_PIN_ACCURACY_LAYERS", "auto").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return True


def snapshot_pinned_layers(
    z: dict[str, torch.Tensor] | None,
    *,
    n_block_layers: int = 0,
    max_layers_in_z: int = 0,
    extra_layer_ids: set[int] | None = None,
) -> set[int]:
    """Layers to pin in provider cache / z retention.

    Sources merged:
    - Layers already resident in ``z`` at provider creation (skeleton / profile hot).
    - ``extra_layer_ids`` from caller (e.g. env-driven).
    - When ``RWKV_PIN_ACCURACY_LAYERS=auto`` (default) and a bounded-z
      streaming mode is active (max_z < n_block), the special "accuracy"
      block layers (the first and last block) are added — these correspond
      to the layers that bracket sensitive ops (final norm before head).
    """
    pinned: set[int] = set()
    if z is not None:
        pinned |= set(block_layer_ids_in_z(z))
    if extra_layer_ids:
        pinned |= set(extra_layer_ids)
    if (
        n_block_layers > 0
        and max_layers_in_z >= 0
        and max_layers_in_z < n_block_layers
        and _resolve_accuracy_pin_active()
    ):
        # Auto-pin first/last only when the z window has room for pins plus
        # at least one streamed LRU slot. With max_z<=2 (F2 bounded), pins
        # alone consumed the whole budget → 4 layers in z instead of 2
        # (BASELINE_BUGS F2 z-cap). Explicit RWKV_PIN_ACCURACY_LAYERS=1
        # still forces pins regardless of cap.
        explicit = os.environ.get("RWKV_PIN_ACCURACY_LAYERS", "auto").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        if explicit or max_layers_in_z >= 3:
            pinned.add(0)
            pinned.add(n_block_layers - 1)
    return pinned


def evict_block_layer_from_z(z: dict[str, torch.Tensor], layer_id: int) -> None:
    prefix = f"blocks.{layer_id}."
    for key in list(z.keys()):
        if key.startswith(prefix):
            del z[key]


class ZLayerRetention:
    """
    LRU cap on block tensors in ``z`` (globals are never counted).

    ``max_layers_in_z`` applies only to layers loaded during streaming decode,
    not to ``pinned_layer_ids`` from the skeleton.
    """

    def __init__(
        self,
        *,
        max_layers_in_z: int = 1,
        pinned_layer_ids: set[int] | None = None,
        warm_z: bool = False,
        on_evict: Callable[[int], None] | None = None,
    ) -> None:
        self.max_layers_in_z = max(0, int(max_layers_in_z))
        self.pinned_layer_ids = set(pinned_layer_ids or ())
        self.warm_z = warm_z
        self._on_evict = on_evict
        self._lru: list[int] = []

    def touch(self, z: dict[str, torch.Tensor], layer_id: int) -> None:
        """Mark ``layer_id`` as recently used; evict oldest streamed layers over cap."""
        if self.warm_z or not _layer_has_block_tensors(z, layer_id):
            return
        if layer_id in self.pinned_layer_ids:
            return
        if layer_id in self._lru:
            self._lru.remove(layer_id)
        self._lru.append(layer_id)
        while len(self._lru) > self.max_layers_in_z:
            old = self._lru.pop(0)
            if old in self.pinned_layer_ids:
                continue
            evict_block_layer_from_z(z, old)
            if self._on_evict is not None:
                self._on_evict(old)

    def should_keep_after_step(
        self,
        z: dict[str, torch.Tensor],
        layer_id: int,
        *,
        stream_layer_cache: bool,
    ) -> bool:
        if self.warm_z and stream_layer_cache:
            return True
        if stream_layer_cache:
            self.touch(z, layer_id)
            return True
        return False
