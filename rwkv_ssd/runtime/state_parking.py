"""Byte-bounded RAM/SSD parking for portable recurrent session states."""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rwkv_ssd.runtime.snapshot import SnapshotMeta, load_snapshot, save_snapshot
from rwkv_ssd.runtime.sequence_state import clone_sequence_state
from rwkv_ssd.runtime.state_cache import RecurrentState, clone_rwkv7_state


class StateParkingCompatibilityError(ValueError):
    """A parked state belongs to a different backend/model/tokenizer."""


def state_nbytes(state: RecurrentState) -> int:
    total = 0
    if state.h is not None:
        total += state.h.numel() * state.h.element_size()
    if state.rwkv7_state is not None:
        total += sum(t.numel() * t.element_size() for t in state.rwkv7_state)
    if state.external_state is not None:
        total += int(np.asarray(state.external_state).nbytes)
    if state.sequence_state is not None:
        total += state.sequence_state.nbytes()
    return total


def clone_recurrent_state(state: RecurrentState) -> RecurrentState:
    external = state.external_state
    if external is not None:
        copier = getattr(external, "copy", None)
        external = copier() if callable(copier) else copy.deepcopy(external)
    return RecurrentState(
        last_token_id=int(state.last_token_id),
        h=state.h.clone() if state.h is not None else None,
        rwkv7_state=(
            clone_rwkv7_state(state.rwkv7_state)
            if state.rwkv7_state is not None
            else None
        ),
        external_state=external,
        sequence_state=clone_sequence_state(state.sequence_state),
    )


@dataclass
class StateParkingStats:
    ram_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    ram_evictions: int = 0
    disk_load_failures: int = 0
    bytes_written: int = 0
    bytes_restored: int = 0


class HierarchicalStateStore:
    """Keep hot session states in RAM and exact portable copies on SSD.

    The RAM tier is an LRU governed by serialized-state bytes, not entry count.
    Every successful ``put`` first writes an atomic snapshot, so RAM eviction
    cannot lose a session. Returned states are deep copies.
    """

    def __init__(self, root: str | Path, *, max_ram_bytes: int) -> None:
        if max_ram_bytes < 0:
            raise ValueError("max_ram_bytes must be non-negative")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_ram_bytes = int(max_ram_bytes)
        self._ram: OrderedDict[str, RecurrentState] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._ram_metadata: dict[str, tuple[str, str | None, str | None, int | None]] = {}
        self._ram_bytes = 0
        self.stats = StateParkingStats()

    @staticmethod
    def _digest(key: str) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        digest = self._digest(key)
        return self.root / digest[:2] / f"{digest}.snapshot"

    @property
    def ram_bytes(self) -> int:
        return self._ram_bytes

    @property
    def ram_keys(self) -> tuple[str, ...]:
        return tuple(self._ram.keys())

    def contains(self, key: str) -> bool:
        return key in self._ram or self._path(key).is_file()

    def _cache_ram(self, key: str, state: RecurrentState) -> None:
        size = state_nbytes(state)
        if key in self._ram:
            self._ram_bytes -= self._sizes.pop(key)
            del self._ram[key]
            self._ram_metadata.pop(key, None)
        if self.max_ram_bytes <= 0 or size > self.max_ram_bytes:
            return
        self._ram[key] = clone_recurrent_state(state)
        self._sizes[key] = size
        self._ram_bytes += size
        while self._ram and self._ram_bytes > self.max_ram_bytes:
            old_key, _old_state = self._ram.popitem(last=False)
            self._ram_bytes -= self._sizes.pop(old_key)
            self._ram_metadata.pop(old_key, None)
            self.stats.ram_evictions += 1

    def put(
        self,
        key: str,
        state: RecurrentState,
        *,
        backend: str = "portable",
        model_family: str = "rwkv7",
        model_fingerprint: str | None = None,
        tokenizer_fingerprint: str | None = None,
        serialization_version: int | None = None,
    ) -> Path:
        extras = {"parking_key_sha256": self._digest(key)}
        if model_fingerprint is not None:
            extras["model_fingerprint"] = str(model_fingerprint)
        if tokenizer_fingerprint is not None:
            extras["tokenizer_fingerprint"] = str(tokenizer_fingerprint)
        if serialization_version is not None:
            extras["state_serialization_version"] = int(serialization_version)
        target = self._path(key)
        meta = SnapshotMeta(
            backend=backend,
            mode="state_parking",
            model_family=model_family,
            last_token_id=int(state.last_token_id),
            extras=extras,
        )
        save_snapshot(target, state, meta)
        self.stats.bytes_written += target.stat().st_size
        self._cache_ram(key, state)
        self._ram_metadata[key] = (
            str(backend),
            str(model_fingerprint) if model_fingerprint is not None else None,
            str(tokenizer_fingerprint) if tokenizer_fingerprint is not None else None,
            int(serialization_version) if serialization_version is not None else None,
        )
        return target

    def get(
        self,
        key: str,
        *,
        backend: str | None = None,
        model_fingerprint: str | None = None,
        tokenizer_fingerprint: str | None = None,
        serialization_version: int | None = None,
    ) -> RecurrentState | None:
        state = self._ram.get(key)
        if state is not None:
            backend_meta, model_meta, tokenizer_meta, version_meta = self._ram_metadata.get(
                key, ("portable", None, None, None)
            )
            if backend is not None and backend_meta.strip().lower() != str(backend).strip().lower():
                raise StateParkingCompatibilityError(
                    f"state backend mismatch: parked={backend_meta!r} expected={backend!r}"
                )
            self._validate_fingerprint(model_meta, model_fingerprint, "model/pack")
            self._validate_fingerprint(tokenizer_meta, tokenizer_fingerprint, "tokenizer")
            if serialization_version is not None and version_meta != int(serialization_version):
                raise StateParkingCompatibilityError("state serialization version mismatch")
            self._ram.move_to_end(key)
            self.stats.ram_hits += 1
            return clone_recurrent_state(state)
        path = self._path(key)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            restored, meta = load_snapshot(path)
            if meta.extras.get("parking_key_sha256") != self._digest(key):
                raise ValueError("state parking key identity mismatch")
            if backend is not None and str(meta.backend).strip().lower() != str(backend).strip().lower():
                raise StateParkingCompatibilityError(
                    f"state backend mismatch: parked={meta.backend!r} expected={backend!r}"
                )
            self._validate_fingerprint(
                meta.extras.get("model_fingerprint"),
                model_fingerprint,
                "model/pack",
            )
            self._validate_fingerprint(
                meta.extras.get("tokenizer_fingerprint"),
                tokenizer_fingerprint,
                "tokenizer",
            )
            if serialization_version is not None:
                actual_version = meta.extras.get("state_serialization_version")
                if actual_version is None or int(actual_version) != int(serialization_version):
                    raise StateParkingCompatibilityError(
                        "state serialization version mismatch"
                    )
        except StateParkingCompatibilityError:
            # Compatibility failures are actionable and must not be silently
            # converted into a cache miss by callers such as HTTP session
            # restore.  The snapshot remains on disk for diagnosis/cleanup.
            raise
        except (OSError, ValueError, RuntimeError, OverflowError):
            self.stats.disk_load_failures += 1
            return None
        self.stats.disk_hits += 1
        self.stats.bytes_restored += path.stat().st_size
        self._cache_ram(key, restored)
        self._ram_metadata[key] = (
            str(meta.backend),
            str(meta.extras.get("model_fingerprint"))
            if meta.extras.get("model_fingerprint") is not None
            else None,
            str(meta.extras.get("tokenizer_fingerprint"))
            if meta.extras.get("tokenizer_fingerprint") is not None
            else None,
            int(meta.extras["state_serialization_version"])
            if meta.extras.get("state_serialization_version") is not None
            else None,
        )
        return clone_recurrent_state(restored)

    @staticmethod
    def _validate_fingerprint(
        actual: object,
        expected: str | None,
        label: str,
    ) -> None:
        if expected is None:
            return
        if actual is None:
            raise StateParkingCompatibilityError(
                f"parked state has no {label} fingerprint"
            )
        if str(actual) != str(expected):
            raise StateParkingCompatibilityError(
                f"state {label} fingerprint mismatch"
            )

    def disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*/*.snapshot"))


__all__ = [
    "HierarchicalStateStore",
    "StateParkingCompatibilityError",
    "StateParkingStats",
    "clone_recurrent_state",
    "state_nbytes",
]
