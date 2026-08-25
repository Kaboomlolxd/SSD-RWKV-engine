"""mmap madvise helpers for layer-weight read-ahead (P0 #4 / P2.a)."""

from __future__ import annotations

import logging
import mmap

from rwkv_ssd.runtime.manifest import TensorEntry

logger = logging.getLogger(__name__)

MADV_WILLNEED = getattr(mmap, "MADV_WILLNEED", None)
MADV_DONTNEED = getattr(mmap, "MADV_DONTNEED", None)
MADV_SEQUENTIAL = getattr(mmap, "MADV_SEQUENTIAL", None)


def layer_file_span(entries: list[TensorEntry]) -> tuple[int, int] | None:
    """Byte range in weights.bin covering all ``entries`` (inclusive span)."""
    if not entries:
        return None
    start = min(e.offset for e in entries)
    end = max(e.offset + e.length for e in entries)
    if end <= start:
        return None
    return start, end - start


def madvise_range(mm: mmap.mmap, offset: int, length: int, option: int) -> bool:
    """Apply ``madvise`` to ``[offset, offset+length)``; return True if applied."""
    if length <= 0 or not hasattr(mm, "madvise"):
        return False
    try:
        mm.madvise(option, offset, length)
        return True
    except (OSError, AttributeError, ValueError, OverflowError) as exc:
        logger.debug("madvise(%s) skipped at %d+%d: %s", option, offset, length, exc)
        return False


def advise_willneed(mm: mmap.mmap, entries: list[TensorEntry]) -> bool:
    span = layer_file_span(entries)
    if span is None or MADV_WILLNEED is None:
        return False
    offset, length = span
    return madvise_range(mm, offset, length, MADV_WILLNEED)


def advise_dontneed(mm: mmap.mmap, entries: list[TensorEntry]) -> bool:
    span = layer_file_span(entries)
    if span is None or MADV_DONTNEED is None:
        return False
    offset, length = span
    return madvise_range(mm, offset, length, MADV_DONTNEED)
