"""Pre-decoded bf16 shadow reads (trade SSD for LUT/zlib decode)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rwkv_ssd.runtime.weight_store_base import WeightStore

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.tensor_loader import dtype_from_entry


def entry_has_shadow(entry: TensorEntry) -> bool:
    return entry.fast_offset >= 0 and entry.fast_length > 0


def layer_has_shadow(entries: list[TensorEntry]) -> bool:
    return bool(entries) and all(entry_has_shadow(e) for e in entries)


def layer_any_shadow(entries: list[TensorEntry]) -> bool:
    return any(entry_has_shadow(e) for e in entries)


def split_shadow_lut_entries(
    entries: list[TensorEntry],
) -> tuple[list[TensorEntry], list[TensorEntry]]:
    """Partition layer entries into shadow-fast vs LUT-decode subsets."""
    shadow: list[TensorEntry] = []
    lut: list[TensorEntry] = []
    for entry in entries:
        (shadow if entry_has_shadow(entry) else lut).append(entry)
    return shadow, lut


def entries_shadow_contiguous_span(entries: list[TensorEntry]) -> tuple[int, int] | None:
    """Return ``(base_fast_offset, total_length)`` when shadow blobs are back-to-back."""
    if not layer_has_shadow(entries):
        return None
    ordered = sorted(entries, key=lambda e: (e.fast_offset, e.name))
    base = ordered[0].fast_offset
    end = ordered[0].fast_offset + ordered[0].fast_length
    for entry in ordered[1:]:
        if entry.fast_offset != end:
            return None
        end += entry.fast_length
    return base, end - base


def entries_shadow_layer_read_span(entries: list[TensorEntry]) -> tuple[int, int] | None:
    """
    Return ``(base_fast_offset, total_length)`` covering shadow blobs in a layer.

    Works with selective shadow (partial layers): only entries with
    ``fast_offset >= 0`` are included. Alignment padding between blobs is included
    so callers can issue one read and slice per entry via ``fast_offset``.
    """
    shadowed = [e for e in entries if entry_has_shadow(e)]
    if not shadowed:
        return None
    base = min(e.fast_offset for e in shadowed)
    end = max(e.fast_offset + e.fast_length for e in shadowed)
    return base, end - base


def _shadow_slab_buffer(
    raw: bytes | memoryview | bytearray, base: int, entries: list[TensorEntry]
) -> bytearray:
    """Writable copy when ``torch.frombuffer`` cannot use ``raw`` directly."""
    shadowed = [e for e in entries if entry_has_shadow(e)]
    end = max(e.fast_offset + e.fast_length for e in shadowed)
    need = end - base
    if isinstance(raw, bytearray) and len(raw) == need:
        return raw
    region = memoryview(raw)
    if len(region) != need:
        region = region[:need]
    return bytearray(region)


def _shadow_flat_from_raw(
    raw: bytes | memoryview | bytearray,
    base: int,
    entries: list[TensorEntry],
    dt: torch.dtype,
) -> torch.Tensor:
    """bf16/fp16 flat view over a shadow span; copies only when the buffer is not usable."""
    shadowed = [e for e in entries if entry_has_shadow(e)]
    end = max(e.fast_offset + e.fast_length for e in shadowed)
    byte_len = end - base
    elem_size = torch.tensor([], dtype=dt).element_size()
    n_elem = byte_len // elem_size
    zero_copy = os.environ.get("RWKV_SHADOW_ZERO_COPY", "auto").strip().lower()
    allow_zc = zero_copy not in ("0", "false", "off", "no")
    if allow_zc and isinstance(raw, (memoryview, bytes, bytearray)):
        try:
            flat = torch.frombuffer(raw, dtype=dt, count=n_elem)
            if flat.numel() == n_elem:
                return flat
        except (TypeError, RuntimeError, ValueError):
            pass
    buf = _shadow_slab_buffer(raw, base, entries)
    return torch.frombuffer(buf, dtype=dt)


def decode_shadow_layer_from_span(
    raw: bytes | memoryview | bytearray,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load bf16 tensors from a shadow layer span (one copy, then views)."""
    shadowed = [e for e in entries if entry_has_shadow(e)]
    if not shadowed:
        return {}
    ordered = sorted(shadowed, key=lambda e: e.fast_offset)
    dt = dtype_from_entry(ordered[0])
    elem_size = torch.tensor([], dtype=dt).element_size()
    flat = _shadow_flat_from_raw(raw, base, entries, dt)
    if device.type != "cpu":
        flat = flat.to(device=device)
    out: dict[str, torch.Tensor] = {}
    readonly_src = isinstance(raw, memoryview) and raw.readonly
    for entry in ordered:
        elem_off = (entry.fast_offset - base) // elem_size
        n = entry.numel
        t = flat[elem_off : elem_off + n].reshape(entry.shape)
        if readonly_src:
            t = t.clone()
        out[entry.name] = t
    return out


def read_shadow_layer(
    store: WeightStore,
    entries: list[TensorEntry],
) -> tuple[bytes | memoryview | bytearray | None, int]:
    """
    Read one shadow layer span for ``entries``.

    Returns ``(raw, base_fast_offset)`` or ``(None, 0)`` when shadow is unavailable.
    """
    span = entries_shadow_layer_read_span(entries)
    if span is None:
        return None, 0
    base, total = span
    if any(entry.fast_stripes or entry.fast_shard_file for entry in entries):
        return read_shadow_layer_striped(store, entries, base=base, total=total)
    read_fn = getattr(store, "read_bytearray_span", None)
    if read_fn is not None:
        return read_fn(base, total), base
    read_span = getattr(store, "read_bytes_span", None)
    if read_span is None:
        return None, 0
    return read_span(base, total), base


def read_shadow_layer_striped(
    store: WeightStore,
    entries: list[TensorEntry],
    *,
    base: int | None = None,
    total: int | None = None,
) -> tuple[bytearray, int]:
    """Gather a layer from a striped BF16 shadow sidecar.

    ``fast_offset`` remains the logical/global address used by the decoder,
    while each ``fast_stripes`` item names a physical file extent.  Cloning
    those extents into ordinary :class:`TensorEntry` objects lets the normal
    CPU coalesced gather do the actual reads.  This keeps the shadow path
    independent from the weight-shard layout and works on Windows via pread.
    """
    shadowed = [entry for entry in entries if entry_has_shadow(entry)]
    if not shadowed:
        return bytearray(), 0
    if base is None or total is None:
        span = entries_shadow_layer_read_span(entries)
        if span is None:
            return bytearray(), 0
        base, total = span
    read_many = getattr(store, "read_entries_coalesced", None)
    if not callable(read_many):
        raise RuntimeError("striped BF16 shadow requires a coalescing weight store")

    clones: list[TensorEntry] = []
    placements: list[tuple[str, int, int, int]] = []
    for entry in shadowed:
        if entry.fast_stripes:
            for index, stripe in enumerate(
                sorted(entry.fast_stripes, key=lambda item: int(item["logical_offset"]))
            ):
                length = int(stripe["length"])
                name = f"{entry.name}.__shadow_{index}"
                clones.append(
                    TensorEntry(
                        name=name,
                        layer_id=entry.layer_id,
                        dtype=entry.dtype,
                        shape=[length],
                        offset=int(stripe["offset"]),
                        length=length,
                        alignment=entry.alignment,
                        residency="streamed",
                        shard_file=str(stripe["shard_file"]),
                    )
                )
                placements.append(
                    (name, int(entry.fast_offset), int(stripe["logical_offset"]), length)
                )
        else:
            if not entry.fast_shard_file:
                raise RuntimeError(
                    f"shadow entry {entry.name!r} has no physical shard file"
                )
            name = f"{entry.name}.__shadow_0"
            length = int(entry.fast_length)
            clones.append(
                TensorEntry(
                    name=name,
                    layer_id=entry.layer_id,
                    dtype=entry.dtype,
                    shape=[length],
                    offset=int(entry.fast_offset),
                    length=length,
                    alignment=entry.alignment,
                    residency="streamed",
                    shard_file=entry.fast_shard_file,
                )
            )
            placements.append((name, int(entry.fast_offset), 0, length))

    gathered = read_many(clones)
    out = bytearray(int(total))
    for name, logical_base, logical_offset, length in placements:
        payload = gathered.get(name)
        if payload is None or len(payload) != length:
            raise OSError(f"short striped shadow read for {name!r}")
        start = logical_base - int(base) + logical_offset
        end = start + length
        if start < 0 or end > len(out):
            raise ValueError(f"striped shadow placement for {name!r} exceeds layer span")
        out[start:end] = payload
    return out, int(base)
