"""Layer-scoped I/O helpers shared by prefetch and mmap advise."""

from __future__ import annotations

from rwkv_ssd.runtime.manifest import TensorEntry


def merge_layer_entries(
    by_layer: dict[int, list[TensorEntry]], layer_ids: list[int]
) -> list[TensorEntry]:
    """Concatenate tensor entries for ``layer_ids`` in order."""
    out: list[TensorEntry] = []
    for layer_id in layer_ids:
        out.extend(by_layer.get(layer_id, []))
    return out


def entries_contiguous_span(entries: list[TensorEntry]) -> tuple[int, int] | None:
    """
    Return ``(base_offset, total_length)`` when tensor blobs are packed back-to-back.

    Matches thesis Ch.10 sequential layer layout / ``layer_grouped`` packs.
    """
    if not entries:
        return None
    ordered = sorted(entries, key=lambda e: (e.offset, e.name))
    base = ordered[0].offset
    end = ordered[0].offset + ordered[0].length
    for entry in ordered[1:]:
        if entry.offset != end:
            return None
        end += entry.length
    return base, end - base


def entries_layer_read_span(entries: list[TensorEntry]) -> tuple[int, int] | None:
    """
    Return ``(base_offset, total_length)`` covering every tensor blob in a layer.

    Unlike ``entries_contiguous_span``, alignment padding between blobs is included
    so callers can issue one ``read_bytes_span`` and slice per entry.
    """
    if not entries:
        return None
    base = min(e.offset for e in entries)
    end = max(e.offset + e.length for e in entries)
    return base, end - base
