"""Explicit offset reads from weights.bin (P0 #4 — predictable profiling path)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from rwkv_ssd.runtime.io_read import pread_at
from rwkv_ssd.runtime.io_mmap_advise import layer_file_span
from rwkv_ssd.runtime.io_posix_fadvise import fadvise_dontneed, fadvise_willneed
from rwkv_ssd.runtime.manifest import TensorEntry


class PreadWeightStore:
    """Read tensor slices via ``pread`` (or seek+read on Windows)."""

    def __init__(self, weights_path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        self._fd = os.open(weights_path, flags)
        self._size = os.fstat(self._fd).st_size
        # Windows has no os.pread; its lseek+read fallback shares the file
        # descriptor cursor and must be serialized per shard. Different shard
        # stores still run concurrently in ShardedWeightStore.
        self._fallback_lock = threading.Lock()

    def _read_span_into(self, offset: int, view: memoryview) -> None:
        def read_all() -> None:
            pos = 0
            while pos < len(view):
                n = pread_at(self._fd, view[pos:], offset + pos)
                if n <= 0:
                    raise OSError(f"pread short read at {offset + pos}: got {n}")
                pos += n

        if hasattr(os, "pread"):
            read_all()
        else:
            with self._fallback_lock:
                read_all()

    def close(self) -> None:
        os.close(self._fd)

    def __enter__(self) -> PreadWeightStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        if byte_offset < 0 or length < 0 or byte_offset + length > entry.length:
            raise ValueError(f"read past end of tensor {entry.name}")
        if len(dest) < length:
            raise ValueError(
                f"dest too small for {entry.name}: need {length}, have {len(dest)}"
            )
        self._read_span_into(entry.offset + byte_offset, dest[:length])

    def read_bytes(self, entry: TensorEntry) -> bytes:
        self._validate_span(entry.offset, entry.length, entry.name)
        buf = bytearray(entry.length)
        view = memoryview(buf)
        self._read_span_into(entry.offset, view)
        return bytes(buf)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        self._validate_span(offset, length)
        buf = bytearray(length)
        view = memoryview(buf)
        self._read_span_into(offset, view)
        return bytes(buf)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        self._validate_span(offset, length)
        buf = bytearray(length)
        view = memoryview(buf)
        self._read_span_into(offset, view)
        return buf

    def _validate_span(
        self, offset: int, length: int, name: str = "backing file"
    ) -> None:
        if offset < 0 or length < 0 or offset + length > self._size:
            raise OSError(
                f"read past backing file for {name}: [{offset}, {offset + length})"
            )

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        span = layer_file_span(entries)
        if span is None:
            return False
        off, length = span
        return fadvise_willneed(self._fd, off, length)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        span = layer_file_span(entries)
        if span is None:
            return False
        off, length = span
        return fadvise_dontneed(self._fd, off, length)
