"""Shared byte-range read helpers for weight stores."""

from __future__ import annotations

import mmap
import os

from rwkv_ssd.runtime.manifest import TensorEntry


def pread_at(fd: int, buf: memoryview, file_offset: int) -> int:
    """Read ``len(buf)`` bytes at ``file_offset`` (``os.pread`` or seek+read fallback)."""
    if hasattr(os, "pread"):
        # Python's os.pread signature is (fd, byte_count, offset); unlike
        # os.readv/preadv it does not accept a destination buffer.
        data = os.pread(fd, len(buf), file_offset)
        buf[: len(data)] = data
        return len(data)
    os.lseek(fd, file_offset, os.SEEK_SET)
    data = os.read(fd, len(buf))
    if not data:
        return 0
    buf[: len(data)] = data
    return len(data)


def read_mmap_range(
    mm: mmap.mmap, entry: TensorEntry, dest: memoryview, byte_offset: int, length: int
) -> None:
    """Copy ``length`` bytes from ``entry`` at ``byte_offset`` into ``dest``."""
    if byte_offset < 0 or length < 0 or byte_offset + length > entry.length:
        raise ValueError(f"read past end of tensor {entry.name}")
    if len(dest) < length:
        raise ValueError(f"dest too small for {entry.name}: need {length}, have {len(dest)}")
    start = entry.offset + byte_offset
    end = start + length
    if start < 0 or end > len(mm):
        raise OSError(
            f"read past backing file for {entry.name}: [{start}, {end})"
        )
    # Indexing an mmap is thread-safe for independent slices.  The old
    # seek()+readinto() implementation shared the mmap cursor, so concurrent
    # threaded/chunked reads could race and copy bytes from the wrong offset.
    chunk = mm[start:end]
    if len(chunk) != length:
        raise OSError(
            f"short mmap read for {entry.name}: wanted {length}, got {len(chunk)}"
        )
    dest[:length] = chunk


def read_entry_range(
    *,
    mmap_obj: mmap.mmap | None = None,
    fd: int | None = None,
    entry: TensorEntry,
    dest: memoryview,
    byte_offset: int,
    length: int,
) -> None:
    """Read a slice of one packed tensor into ``dest``."""
    if byte_offset < 0 or length < 0 or byte_offset + length > entry.length:
        raise ValueError(f"read past end of tensor {entry.name}")
    if len(dest) < length:
        raise ValueError(f"dest too small for {entry.name}: need {length}, have {len(dest)}")
    if mmap_obj is not None:
        read_mmap_range(mmap_obj, entry, dest, byte_offset, length)
        return
    if fd is None:
        raise ValueError("read_entry_range requires mmap_obj or fd")
    view = dest[:length]
    file_off = entry.offset + byte_offset
    offset = 0
    while offset < length:
        n = pread_at(fd, view[offset:], file_off + offset)
        if n <= 0:
            raise OSError(
                f"short read for {entry.name} at {file_off + offset}: got {n}"
            )
        offset += n
