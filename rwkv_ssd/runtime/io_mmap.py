"""Memory-mapped reads from weights.bin."""

from __future__ import annotations

import logging
import mmap
import os
from pathlib import Path

from rwkv_ssd.runtime.io_mmap_advise import advise_dontneed, advise_willneed
from rwkv_ssd.runtime.io_read import read_mmap_range
from rwkv_ssd.runtime.manifest import TensorEntry

logger = logging.getLogger(__name__)

# Linux: hint sequential access (P0 #4 madvise). No-op on Windows.
_MADV_SEQUENTIAL = getattr(mmap, "MADV_SEQUENTIAL", None)
# Linux: hint huge pages (transparent huge pages / THP). For 7B+ mmap
# regions the TLB pressure is real (3.5M page-table entries with 4 KB
# pages vs 7K with 2 MB huge pages). The kernel will only honor the
# hint if ``/sys/kernel/mm/transparent_hugepage/enabled`` is
# ``madvise`` or ``always``; the madvise call itself is cheap and
# safe to issue on every open. No-op on Windows / macOS.
_MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", None)


def _try_madvise(mm, advice: int | None, label: str) -> None:
    """Best-effort ``madvise`` — silently skip if the platform / kernel
    doesn't support the hint or the syscall is denied."""
    if advice is None:
        return
    try:
        mm.madvise(advice)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("madvise %s skipped: %s", label, exc)


class MmapWeightStore:
    def __init__(
        self,
        weights_path: Path,
        *,
        sequential_advise: bool = False,
        huge_page_advise: bool = True,
    ) -> None:
        self._path = weights_path
        self._file = open(weights_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if sequential_advise and _MADV_SEQUENTIAL is not None:
            _try_madvise(self._mmap, _MADV_SEQUENTIAL, "MADV_SEQUENTIAL")
        if huge_page_advise:
            # Default ON — for 0.1B (49 MB) the hint is a no-op
            # (THP ignores regions < 2 MB); for 7B+ (14 GB) it
            # reduces TLB pressure by 100-1000x. Set
            # ``RWKV_DISABLE_HUGE_PAGE_HINT=1`` to disable.
            if os.environ.get("RWKV_DISABLE_HUGE_PAGE_HINT", "") != "1":
                _try_madvise(self._mmap, _MADV_HUGEPAGE, "MADV_HUGEPAGE")

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> MmapWeightStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        read_mmap_range(self._mmap, entry, dest, byte_offset, length)

    def read_into(self, entry: TensorEntry, dest: memoryview) -> None:
        if len(dest) < entry.length:
            raise ValueError(
                f"buffer too small for {entry.name}: need {entry.length}, have {len(dest)}"
            )
        self.read_range(entry, 0, entry.length, dest[: entry.length])

    def read_bytes(self, entry: TensorEntry) -> bytes:
        start = entry.offset
        end = start + entry.length
        if start < 0 or entry.length < 0 or end > len(self._mmap):
            raise OSError(
                f"read past backing file for {entry.name}: [{start}, {end})"
            )
        return bytes(self._mmap[start:end])

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        """One memcpy for a contiguous layer span in ``weights.bin``."""
        self._validate_span(offset, length)
        return bytes(self._mmap[offset : offset + length])

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        """Zero-copy view into the mmap (decode may copy once into a writable slab)."""
        self._validate_span(offset, length)
        return memoryview(self._mmap)[offset : offset + length]

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        """One copy into a writable buffer suitable for ``torch.frombuffer``."""
        self._validate_span(offset, length)
        return bytearray(self._mmap[offset : offset + length])

    def _validate_span(self, offset: int, length: int) -> None:
        if offset < 0 or length < 0 or offset + length > len(self._mmap):
            raise OSError(
                f"read past backing file: [{offset}, {offset + length})"
            )

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return advise_willneed(self._mmap, entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return advise_dontneed(self._mmap, entries)
