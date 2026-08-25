"""Cold SSD reads: reopen + pread per span (minimize OS page-cache reuse)."""

from __future__ import annotations

import os
from pathlib import Path

from rwkv_ssd.runtime.io_read import pread_at
from rwkv_ssd.runtime.manifest import TensorEntry


class ColdPreadWeightStore:
    """
    Open the weights file for each span read so the OS cannot reuse a warm mmap.

    On Windows this is the closest portable approximation to cold SSD without
    admin cache eviction. Pair with ``mmap_dontneed`` on release when using mmap
  for the primary store in benchmarks.
    """

    def __init__(self, weights_path: Path) -> None:
        self._path = Path(weights_path)
        self._flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            self._flags |= os.O_BINARY

    def close(self) -> None:
        pass

    def __enter__(self) -> ColdPreadWeightStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _read_span_cold(self, offset: int, length: int) -> bytes:
        fd = os.open(self._path, self._flags)
        try:
            buf = bytearray(length)
            view = memoryview(buf)
            pos = 0
            while pos < length:
                n = pread_at(fd, view[pos:], offset + pos)
                if n <= 0:
                    raise OSError(f"cold pread short read at {offset + pos}: got {n}")
                pos += n
            return bytes(buf)
        finally:
            os.close(fd)

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._read_span_cold(entry.offset, entry.length)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self._read_span_cold(offset, length)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        return bytearray(self._read_span_cold(offset, length))

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return False
