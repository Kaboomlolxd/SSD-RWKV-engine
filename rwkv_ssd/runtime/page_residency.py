"""Portable observed page-residency estimates for prefetch decisions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


@dataclass(frozen=True)
class ResidencyStats:
    observations: int
    probable_hits: int
    probable_misses: int
    invalidations: int


class ExtentResidencyTracker:
    """Estimate OS-cache residency from recent completed reads.

    The estimate is deliberately conservative and portable.  Native page
    probes can later implement the same ``probably_resident`` contract.
    """

    def __init__(self, *, ttl_s: float = 30.0, max_extents: int = 4096) -> None:
        if ttl_s <= 0 or max_extents <= 0:
            raise ValueError("ttl_s and max_extents must be positive")
        self.ttl_s = float(ttl_s)
        self.max_extents = int(max_extents)
        self._extents: list[tuple[int, int, float]] = []
        self._lock = threading.Lock()
        self._observations = 0
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def observe(self, offset: int, length: int) -> None:
        if offset < 0 or length <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._observations += 1
            self._extents.append((int(offset), int(offset + length), now))
            cutoff = now - self.ttl_s
            self._extents = [item for item in self._extents if item[2] >= cutoff]
            if len(self._extents) > self.max_extents:
                self._extents = self._extents[-self.max_extents :]

    def probably_resident(self, offset: int, length: int) -> bool:
        now = time.monotonic()
        end = int(offset + length)
        with self._lock:
            cutoff = now - self.ttl_s
            hit = any(start <= offset and extent_end >= end and seen >= cutoff for start, extent_end, seen in self._extents)
            if hit:
                self._hits += 1
            else:
                self._misses += 1
            return hit

    def invalidate(self, offset: int, length: int) -> None:
        end = int(offset + length)
        with self._lock:
            self._invalidations += 1
            self._extents = [
                item
                for item in self._extents
                if item[1] <= offset or item[0] >= end
            ]

    def stats(self) -> ResidencyStats:
        with self._lock:
            return ResidencyStats(
                self._observations,
                self._hits,
                self._misses,
                self._invalidations,
            )


class ObservedResidencyWeightStore(WeightStore):
    """Transparent store wrapper that records recently read byte extents."""

    def __init__(self, inner: WeightStore, *, ttl_s: float = 30.0) -> None:
        self.inner = inner
        self.tracker = ExtentResidencyTracker(ttl_s=ttl_s)

    def read_bytes(self, entry: TensorEntry) -> bytes:
        payload = self.inner.read_bytes(entry)
        if not entry.stripes:
            self.tracker.observe(entry.offset, entry.length)
        return payload

    def read_range(self, entry, byte_offset, length, dest) -> None:
        self.inner.read_range(entry, byte_offset, length, dest)
        if not entry.stripes:
            self.tracker.observe(entry.offset + byte_offset, length)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        payload = self.inner.read_bytes_span(offset, length)
        self.tracker.observe(offset, length)
        return payload

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        payload = self.inner.read_bytearray_span(offset, length)
        self.tracker.observe(offset, length)
        return payload

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        payload = self.inner.read_memoryview_span(offset, length)
        self.tracker.observe(offset, length)
        return payload

    def probably_resident(self, entries: list[TensorEntry]) -> bool:
        ordinary = [entry for entry in entries if not entry.stripes]
        return bool(ordinary) and len(ordinary) == len(entries) and all(
            self.tracker.probably_resident(entry.offset, entry.length)
            for entry in ordinary
        )

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        if self.probably_resident(entries):
            return True
        return self.inner.advise_prefetch(entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        for entry in entries:
            if not entry.stripes:
                self.tracker.invalidate(entry.offset, entry.length)
        return self.inner.advise_release(entries)

    def close(self) -> None:
        self.inner.close()


__all__ = [
    "ExtentResidencyTracker",
    "ObservedResidencyWeightStore",
    "ResidencyStats",
]
