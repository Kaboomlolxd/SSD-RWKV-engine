"""Chunked tensor reads for within-layer micro-pipelining (P2.a)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rwkv_ssd.runtime.manifest import TensorEntry

# Default K from thesis micro-pipeline benches (8–16 chunks per tensor).
DEFAULT_CHUNK_BYTES = 64 * 1024


@dataclass
class ChunkReadStats:
    chunks: int = 0
    read_ms: float = 0.0


def read_tensor_chunked(
    read_range,
    entry: TensorEntry,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[bytes, ChunkReadStats]:
    """
    Read one tensor in ``chunk_bytes`` slices via ``read_range(entry, offset, length, dest)``.

    ``read_range`` signature: ``(entry, byte_offset, length, dest: memoryview) -> None``.
    """
    if entry.length <= 0:
        return b"", ChunkReadStats()
    if chunk_bytes <= 0 or entry.length <= chunk_bytes:
        buf = bytearray(entry.length)
        t0 = time.perf_counter()
        read_range(entry, 0, entry.length, memoryview(buf))
        return bytes(buf), ChunkReadStats(
            chunks=1, read_ms=(time.perf_counter() - t0) * 1000.0
        )

    buf = bytearray(entry.length)
    view = memoryview(buf)
    stats = ChunkReadStats()
    pos = 0
    while pos < entry.length:
        n = min(chunk_bytes, entry.length - pos)
        t0 = time.perf_counter()
        read_range(entry, pos, n, view[pos : pos + n])
        stats.read_ms += (time.perf_counter() - t0) * 1000.0
        stats.chunks += 1
        pos += n
    return bytes(buf), stats
