"""Prefix recurrent-state cache (M2.5 prototype).

Caches hidden state after prefill of a fixed system prefix so follow-up
requests skip re-reading streamed layer weights for those tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


def _clone_external_state(state: object | None) -> object | None:
    """Clone backend-owned state without coupling this cache to NumPy/ggml."""
    if state is None:
        return None
    copier = getattr(state, "copy", None)
    if callable(copier):
        try:
            return copier()
        except TypeError:
            pass
    return copy.deepcopy(state)


def clone_rwkv7_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    """Deep-copy ChatRWKV RWKV-7 recurrent state (``model.generate_zero_state()``)."""
    return [t.clone() for t in state]


@dataclass
class RecurrentState:
    """Backend-owned recurrent state for RWKV-compatible backends."""

    last_token_id: int = 0
    h: torch.Tensor | None = None
    rwkv7_state: list[torch.Tensor] | None = None
    external_state: object | None = None

    def clone(self) -> "RecurrentState":
        return RecurrentState(
            last_token_id=int(self.last_token_id),
            h=self.h.clone() if self.h is not None else None,
            rwkv7_state=(
                clone_rwkv7_state(self.rwkv7_state)
                if self.rwkv7_state is not None
                else None
            ),
            external_state=_clone_external_state(self.external_state),
        )


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    prefill_ms_saved: float = 0.0


class PrefixStateCache:
    """Small in-memory LRU keyed by prefix text; optional SSD backing."""

    def __init__(
        self,
        max_entries: int = 16,
        *,
        disk_dir: Path | None = None,
    ) -> None:
        self._max = max(1, max_entries)
        self._entries: dict[str, RecurrentState] = {}
        self._order: list[str] = []
        self.stats = CacheStats()
        self._disk = None
        self._disk_dir = disk_dir
        if disk_dir is not None:
            from rwkv_ssd.runtime.state_disk_cache import StateDiskCache

            self._disk = StateDiskCache(disk_dir)

    def contains(self, prefix_key: str) -> bool:
        if prefix_key in self._entries:
            return True
        if self._disk is not None:
            from rwkv_ssd.runtime.state_disk_cache import _external_path, _path_for

            return _path_for(self._disk_dir, prefix_key).is_file() or _external_path(
                self._disk_dir, prefix_key
            ).is_file()
        return False

    def get(self, prefix_key: str) -> RecurrentState | None:
        state = self._entries.get(prefix_key)
        if state is None and self._disk is not None:
            loaded = self._disk.load(prefix_key)
            if loaded is not None:
                self.put(prefix_key, loaded)
                state = self._entries.get(prefix_key)
        if state is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        if prefix_key in self._order:
            self._order.remove(prefix_key)
        self._order.append(prefix_key)
        return RecurrentState(
            last_token_id=state.last_token_id,
            h=state.h.clone() if state.h is not None else None,
            rwkv7_state=(
                clone_rwkv7_state(state.rwkv7_state)
                if state.rwkv7_state is not None
                else None
            ),
            external_state=_clone_external_state(state.external_state),
        )

    def put(self, prefix_key: str, state: RecurrentState) -> None:
        if prefix_key in self._entries:
            self._order.remove(prefix_key)
        elif len(self._entries) >= self._max:
            evict = self._order.pop(0)
            del self._entries[evict]
        stored = RecurrentState(
            last_token_id=state.last_token_id,
            h=state.h.clone() if state.h is not None else None,
            rwkv7_state=(
                clone_rwkv7_state(state.rwkv7_state)
                if state.rwkv7_state is not None
                else None
            ),
            external_state=_clone_external_state(state.external_state),
        )
        self._entries[prefix_key] = stored
        self._order.append(prefix_key)
        if (
            self._disk is not None
            and (
                stored.rwkv7_state is not None
                or stored.external_state is not None
            )
        ):
            try:
                self._disk.store(prefix_key, stored)
            except (OSError, ValueError, RuntimeError, OverflowError):
                # Disk persistence is an optimization; an unsupported state
                # dtype or transient filesystem error must not invalidate the
                # in-memory cache or fail generation.
                pass

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()
        if self._disk is not None:
            self._disk.clear()
