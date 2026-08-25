"""Pack layout policies (P2.b channel / layer grouping)."""

from __future__ import annotations

# Default sector pad between layer groups (256 KiB — thesis NAND page stripe hint).
DEFAULT_SECTOR_BYTES = 256 * 1024


def sort_tensor_names(names: list[str], layout: str, layer_id_fn) -> list[str]:
    """Order tensor names for packing."""
    key = layout.strip().lower()
    if key in ("", "default", "alphabetical"):
        return sorted(names)
    if key in ("layer_grouped", "layer", "channel", "channel_aligned"):
        return sorted(names, key=lambda n: (layer_id_fn(n), n))
    raise ValueError(
        f"unknown pack_layout {layout!r} (use: default, layer_grouped)"
    )


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        return offset
    return (offset + alignment - 1) // alignment * alignment


def pad_between_layers(
    current_offset: int,
    prev_layer_id: int,
    next_layer_id: int,
    sector_bytes: int,
) -> int:
    """Pad ``weights.bin`` offset when advancing to a new layer id."""
    if sector_bytes <= 0 or prev_layer_id == next_layer_id:
        return current_offset
    return align_offset(current_offset, sector_bytes)
