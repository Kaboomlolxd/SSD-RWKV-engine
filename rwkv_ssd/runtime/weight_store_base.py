"""Weight store protocol (avoids import cycles with threaded wrapper)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rwkv_ssd.runtime.manifest import TensorEntry

IO_BACKENDS = frozenset({"mmap", "pread", "threaded", "cold", "cold_pread", "cold-read"})


class WeightStore(ABC):
    # Read-only mmap stores can safely lend views to the native backend for
    # the lifetime of the store.  Read buffers from pread/cold stores must not
    # be retained across layer boundaries.
    stable_memoryviews: bool = False

    @abstractmethod
    def read_bytes(self, entry: TensorEntry) -> bytes:
        ...

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        if byte_offset < 0 or length < 0 or byte_offset + length > entry.length:
            raise ValueError(f"read past end of tensor {entry.name}")
        if len(dest) < length:
            raise ValueError(
                f"dest too small for {entry.name}: need {length}, have {len(dest)}"
            )
        data = self.read_bytes(entry)
        dest[:length] = data[byte_offset : byte_offset + length]

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        """Read a contiguous byte range from the backing store.

        Default implementation reads entry-by-entry; subclasses with mmap or
        pread should override for zero-copy or single-syscall reads.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement read_bytes_span; "
            "layer-span streaming may be slow"
        )

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        """Read a contiguous byte range as a mutable bytearray."""
        return bytearray(self.read_bytes_span(offset, length))

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        """Read a contiguous byte range as a zero-copy memoryview when possible."""
        return memoryview(self.read_bytes_span(offset, length))

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        """Hint the OS to prefetch pages for upcoming layer reads (Linux only)."""
        return False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        """Hint the OS to release pages after consumption (Linux only)."""
        return False

    def probably_resident(self, entries: list[TensorEntry]) -> bool:
        """Best-effort indication that all entry bytes are in the OS page cache."""
        return False

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> WeightStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
