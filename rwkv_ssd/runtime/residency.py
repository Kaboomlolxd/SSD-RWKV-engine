"""Which tensors stay resident vs streamed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rwkv_ssd.runtime.manifest import TensorEntry


def load_residency_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_residency_policy(
    entries: list[TensorEntry],
    mode: str,
    num_layers: int,
    *,
    profile: dict[str, Any] | None = None,
) -> list[TensorEntry]:
    """
    mode:
      resident — all tensors resident (reference / debug)
      partial — embed, head, norms resident; middle layers streamed
      streaming — only small helpers resident; rest streamed

    Optional profile JSON (see deploy/partial_profile.example.json):
      resident_layer_ids: [0, 1, 30, 31]
      always_resident_tensors: ["embed", "head", ...]
    """
    if mode == "resident":
        return [_with_residency(e, "resident") for e in entries]

    if mode == "partial" and profile:
        resident_layers = set(profile.get("resident_layer_ids", []))
        always = tuple(profile.get("always_resident_tensors", ()))
        return [
            _with_residency(
                e,
                _classify_profile(e.name, e.layer_id, num_layers, resident_layers, always),
            )
            for e in entries
        ]

    return [
        _with_residency(e, _classify_default(e.name, e.layer_id, mode, num_layers))
        for e in entries
    ]


def _with_residency(entry: TensorEntry, residency: str) -> TensorEntry:
    return TensorEntry(
        name=entry.name,
        layer_id=entry.layer_id,
        dtype=entry.dtype,
        shape=entry.shape,
        offset=entry.offset,
        length=entry.length,
        alignment=entry.alignment,
        residency=residency,
        dequant=entry.dequant,
        inner_offset=entry.inner_offset,
        inner_length=entry.inner_length,
        fast_offset=entry.fast_offset,
        fast_length=entry.fast_length,
        fast_shard_file=entry.fast_shard_file,
        fast_stripes=entry.fast_stripes,
        shard_file=entry.shard_file,
        stripes=entry.stripes,
    )


def _classify_profile(
    name: str,
    layer_id: int,
    num_layers: int,
    resident_layers: set[int],
    always: tuple[str, ...],
) -> str:
    lower = name.lower()
    if _is_global_resident_name(name):
        return "resident"
    if layer_id in resident_layers or layer_id <= 0 or layer_id >= num_layers - 1:
        return "resident"
    if any(token in lower for token in always):
        return "resident"
    return "streamed"


def _is_global_resident_name(name: str) -> bool:
    """Tensors that stay in RAM for streaming / partial middle layers."""
    lower = name.lower()
    if lower == "emb.weight":
        return True
    if lower in {
        "head.weight",
        "output.weight",
    } or lower.endswith(".head.weight"):
        return True
    if lower.startswith("ln_out."):
        return True
    if lower in ("blocks.0.ln0.weight", "blocks.0.ln0.bias"):
        return True
    return False


def _classify_default(name: str, layer_id: int, mode: str, num_layers: int) -> str:
    if _is_global_resident_name(name):
        return "resident"
    if mode == "partial":
        if num_layers > 2 and (layer_id <= 0 or layer_id >= num_layers - 1):
            return "resident"
        if layer_id <= 0:
            return "resident"
        return "streamed"
    return "streamed"
