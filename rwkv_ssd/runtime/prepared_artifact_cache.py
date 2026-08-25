"""Disposable content-addressed cache for machine-specific prepared payloads."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_MAGIC = b"PAC\x01"


@dataclass
class ArtifactCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corruptions: int = 0
    evictions: int = 0


class PreparedArtifactCache:
    def __init__(self, root: str | Path, *, max_bytes: int = 0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(0, int(max_bytes))
        self.stats = ArtifactCacheStats()

    @staticmethod
    def make_key(namespace: str, source: bytes, metadata: dict) -> str:
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256()
        digest.update(str(namespace).encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical)
        digest.update(b"\0")
        digest.update(source)
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
            raise ValueError("invalid prepared artifact key")
        return self.root / key[:2] / f"{key}.pac"

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            raw = path.read_bytes()
            if len(raw) < 8 or raw[:4] != _MAGIC:
                raise ValueError("bad magic")
            (header_len,) = struct.unpack_from("<I", raw, 4)
            header_end = 8 + header_len
            header = json.loads(raw[8:header_end].decode("utf-8"))
            payload = raw[header_end:]
            if int(header.get("length", -1)) != len(payload):
                raise ValueError("length mismatch")
            if hashlib.sha256(payload).hexdigest() != header.get("sha256"):
                raise ValueError("checksum mismatch")
            path.touch()
            self.stats.hits += 1
            return payload
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            self.stats.corruptions += 1
            self.stats.misses += 1
            return None

    def put(self, key: str, payload: bytes, *, metadata: dict | None = None) -> Path:
        target = self._path(key)
        header = {
            "version": 1,
            "key": key,
            "length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "metadata": metadata or {},
        }
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        encoded = _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, target)
        self.stats.writes += 1
        self._trim()
        return target

    def get_or_build(
        self,
        key: str,
        builder: Callable[[], bytes],
        *,
        metadata: dict | None = None,
    ) -> bytes:
        cached = self.get(key)
        if cached is not None:
            return cached
        payload = bytes(builder())
        self.put(key, payload, metadata=metadata)
        return payload

    def _trim(self) -> None:
        if self.max_bytes <= 0:
            return
        files = sorted(
            self.root.glob("*/*.pac"), key=lambda path: path.stat().st_mtime
        )
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= self.max_bytes:
                break
            size = path.stat().st_size
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            self.stats.evictions += 1

    def disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*/*.pac"))


__all__ = ["ArtifactCacheStats", "PreparedArtifactCache"]
