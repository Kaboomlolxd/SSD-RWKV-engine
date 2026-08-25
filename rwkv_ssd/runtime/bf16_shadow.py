"""Write bf16 shadow.bin alongside Trinity packs (trade disk for decode speed)."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import ALIGNMENT
from rwkv_ssd.runtime.pack_layout import align_offset, pad_between_layers

SHADOW_FILENAME = "shadow.bin"


def tensor_to_raw_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _tensor_numel(meta: dict) -> int:
    n = 1
    for d in meta.get("shape", ()):
        n *= int(d)
    return n


def write_bf16_shadow(
    state: dict[str, torch.Tensor],
    names: list[str],
    output_dir: Path,
    tensors_meta: list[dict],
    *,
    layer_id_fn,
    sector_bytes: int = 0,
    shadow_min_numel: int = 0,
    quiet: bool = False,
) -> Path:
    """
    Append ``shadow.bin`` with layer_grouped raw bf16 and set ``fast_offset`` /
    ``fast_length`` on each manifest tensor dict (mutates ``tensors_meta`` in place).

    When ``shadow_min_numel > 0``, only tensors with ``numel >= shadow_min_numel``
    are written; smaller tensors keep ``fast_offset=-1`` and decode via LUT at runtime.
    """
    shadow_path = output_dir / SHADOW_FILENAME
    meta_by_name = {m["name"]: m for m in tensors_meta}
    offset = 0
    prev_layer_id: int | None = None
    n_shadowed = 0
    n_skipped = 0

    with shadow_path.open("wb") as out:
        for name in names:
            if name not in meta_by_name:
                continue
            meta = meta_by_name[name]
            numel = _tensor_numel(meta)
            if shadow_min_numel > 0 and numel < shadow_min_numel:
                meta["fast_offset"] = -1
                meta["fast_length"] = 0
                n_skipped += 1
                continue
            layer_id = layer_id_fn(name)
            if prev_layer_id is not None and layer_id != prev_layer_id:
                offset = pad_between_layers(
                    offset, prev_layer_id, layer_id, sector_bytes
                )
                offset = align_offset(offset, ALIGNMENT)
                if offset > out.tell():
                    out.write(b"\x00" * (offset - out.tell()))
            t = state[name].contiguous()
            raw = tensor_to_raw_bytes(t)
            start = out.tell()
            out.write(raw)
            meta["fast_offset"] = start
            meta["fast_length"] = len(raw)
            offset = start + len(raw)
            prev_layer_id = layer_id
            n_shadowed += 1

    if not quiet:
        extra = ""
        if shadow_min_numel > 0:
            extra = f" ({n_shadowed} shadowed, {n_skipped} LUT-only, min_numel={shadow_min_numel})"
        print(
            f"Wrote bf16 shadow -> {shadow_path} "
            f"({shadow_path.stat().st_size / (1024**2):.2f} MiB){extra}"
        )
    return shadow_path
