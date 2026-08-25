"""Prefill n-gram weight reuse (P2.e) — skip disk for repeated layer payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class NgramWeightCache:
    """
    Cache keyed by (layer_id, content hash of last successful read).

    When the same layer bytes appear in a prefill-heavy workload, skip re-read.
    """

    max_entries: int = 128
    hits: int = 0
    misses: int = 0
    _entries: dict[tuple[int, str], bytes] = field(default_factory=dict)
    _order: list[tuple[int, str]] = field(default_factory=list)

    @staticmethod
    def content_key(layer_id: int, raw: bytes) -> tuple[int, str]:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return (layer_id, digest)

    def get(self, layer_id: int, raw: bytes) -> bytes | None:
        key = self.content_key(layer_id, raw)
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        return self._entries[key]

    def put(self, layer_id: int, raw: bytes) -> None:
        key = self.content_key(layer_id, raw)
        if key in self._entries:
            return
        if len(self._entries) >= self.max_entries:
            evict = self._order.pop(0)
            del self._entries[evict]
        self._entries[key] = raw
        self._order.append(key)

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()
