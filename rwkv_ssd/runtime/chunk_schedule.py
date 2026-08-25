"""Heterogeneous chunk schedules (P2.c) — uniform-K vs layer-size adaptive."""

from __future__ import annotations

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.io_chunked import DEFAULT_CHUNK_BYTES

LARGE_LAYER_BYTES = 256 * 1024
LARGE_CHUNK_BYTES = 128 * 1024
SMALL_CHUNK_BYTES = 64 * 1024


def chunk_bytes_for_entry(
    entry: TensorEntry,
    policy: str,
    default_bytes: int = DEFAULT_CHUNK_BYTES,
) -> int:
    key = policy.strip().lower()
    if key in ("", "off", "none", "uniform"):
        return default_bytes if default_bytes > 0 else DEFAULT_CHUNK_BYTES
    if key in ("layer_size", "adaptive", "heterogeneous"):
        if entry.length >= LARGE_LAYER_BYTES:
            return LARGE_CHUNK_BYTES
        return SMALL_CHUNK_BYTES
    raise ValueError(
        f"unknown chunk policy {policy!r} (use: uniform, layer_size)"
    )
