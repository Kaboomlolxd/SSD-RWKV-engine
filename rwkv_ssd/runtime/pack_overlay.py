"""Content-addressed pack/profile overlays with contiguous materialization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe overlay path: {value!r}")
    return path


class PackChunkStore:
    def __init__(self, root: str | Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.root = Path(root)
        self.chunk_bytes = int(chunk_bytes)
        self.chunks_dir = self.root / "chunks"
        self.profiles_dir = self.root / "profiles"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _mutation_lock(self, *, timeout_s: float = 10.0):
        lock = self.root / ".mutation.lock"
        deadline = time.monotonic() + max(0.0, timeout_s)
        fd = None
        while fd is None:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for chunk-store lock: {lock}")
                time.sleep(0.01)
        try:
            yield
        finally:
            os.close(fd)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def _chunk_path(self, digest: str) -> Path:
        return self.chunks_dir / digest[:2] / digest

    def _put_chunk(self, payload: bytes) -> tuple[str, bool]:
        digest = hashlib.sha256(payload).hexdigest()
        target = self._chunk_path(digest)
        if target.is_file():
            return digest, False
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)
        return digest, True

    def ingest_directory(
        self,
        source: str | Path,
        profile: str,
        *,
        include: list[str] | None = None,
    ) -> dict[str, int | str]:
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(source)
        if not profile or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in profile):
            raise ValueError("profile name contains unsafe characters")
        if include is None:
            files = sorted(
                path
                for path in source.rglob("*")
                if path.is_file()
                and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(source).parts)
            )
        else:
            files = [source / _safe_relative(name) for name in include]
            if any(not path.is_file() for path in files):
                missing = next(path for path in files if not path.is_file())
                raise FileNotFoundError(missing)
        with self._mutation_lock():
            manifest_files: dict[str, dict] = {}
            logical_bytes = 0
            new_bytes = 0
            new_chunks = 0
            reused_chunks = 0
            for path in files:
                relative = path.relative_to(source).as_posix()
                size = path.stat().st_size
                logical_bytes += size
                chunks: list[str] = []
                with path.open("rb") as handle:
                    while True:
                        payload = handle.read(self.chunk_bytes)
                        if not payload:
                            break
                        digest, created = self._put_chunk(payload)
                        chunks.append(digest)
                        if created:
                            new_chunks += 1
                            new_bytes += len(payload)
                        else:
                            reused_chunks += 1
                manifest_files[relative] = {"size": size, "chunks": chunks}
            overlay = {
                "version": 1,
                "profile": profile,
                "chunk_bytes": self.chunk_bytes,
                "files": manifest_files,
            }
            identity_payload = json.dumps(
                overlay, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            overlay["profile_sha256"] = hashlib.sha256(identity_payload).hexdigest()
            profile_path = self.profiles_dir / f"{profile}.json"
            tmp = profile_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, profile_path)
            metadata_bytes = profile_path.stat().st_size
        return {
            "profile": profile,
            "files": len(files),
            "logical_bytes": logical_bytes,
            "new_bytes": new_bytes,
            "unique_bytes_added": new_bytes,
            "new_chunks": new_chunks,
            "reused_chunks": reused_chunks,
            "metadata_bytes": metadata_bytes,
            "profile_sha256": overlay["profile_sha256"],
        }

    def load_profile(self, profile: str) -> dict:
        path = self.profiles_dir / f"{profile}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw.get("version", 0)) != 1:
            raise ValueError("unsupported pack overlay version")
        expected = str(raw.get("profile_sha256", ""))
        unsigned = dict(raw)
        unsigned.pop("profile_sha256", None)
        actual = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if expected and actual != expected:
            raise ValueError("pack overlay profile checksum mismatch")
        return raw

    def remove_profile(self, profile: str) -> bool:
        if not profile or Path(profile).name != profile:
            raise ValueError("unsafe profile name")
        with self._mutation_lock():
            path = self.profiles_dir / f"{profile}.json"
            if not path.is_file():
                return False
            path.unlink()
            return True

    def garbage_collect(self) -> dict[str, int]:
        """Delete immutable chunks unreferenced by every committed profile."""
        with self._mutation_lock():
            referenced: set[str] = set()
            for path in self.profiles_dir.glob("*.json"):
                raw = json.loads(path.read_text(encoding="utf-8"))
                for spec in raw.get("files", {}).values():
                    referenced.update(str(value) for value in spec.get("chunks", []))
            removed_chunks = 0
            removed_bytes = 0
            for path in self.chunks_dir.rglob("*"):
                if not path.is_file() or path.name in referenced:
                    continue
                removed_bytes += path.stat().st_size
                path.unlink()
                removed_chunks += 1
            return {
                "referenced_chunks": len(referenced),
                "removed_chunks": removed_chunks,
                "removed_bytes": removed_bytes,
            }

    def store_stats(self) -> dict[str, int | float]:
        """Report repository-wide logical, unique, and metadata bytes."""
        profiles = sorted(self.profiles_dir.glob("*.json"))
        logical_bytes = 0
        references = 0
        for path in profiles:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for spec in raw.get("files", {}).values():
                logical_bytes += int(spec.get("size", 0))
                references += len(spec.get("chunks", []))
        chunks = [path for path in self.chunks_dir.rglob("*") if path.is_file()]
        unique_bytes = sum(path.stat().st_size for path in chunks)
        metadata_bytes = sum(path.stat().st_size for path in profiles)
        return {
            "profiles": len(profiles),
            "chunk_references": references,
            "unique_chunks": len(chunks),
            "logical_bytes": logical_bytes,
            "unique_bytes": unique_bytes,
            "metadata_bytes": metadata_bytes,
            "dedup_ratio": (
                logical_bytes / unique_bytes if unique_bytes > 0 else 0.0
            ),
        }

    def read_file(self, profile: str, relative_path: str) -> "OverlayFile":
        raw = self.load_profile(profile)
        safe = _safe_relative(relative_path).as_posix()
        spec = raw.get("files", {}).get(safe)
        if spec is None:
            raise FileNotFoundError(safe)
        return OverlayFile(self, spec, int(raw["chunk_bytes"]))

    def materialize(self, profile: str, output_dir: str | Path) -> Path:
        raw = self.load_profile(profile)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, spec in raw["files"].items():
            relative = _safe_relative(name)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            with tmp.open("wb") as handle:
                for digest in spec["chunks"]:
                    payload = self._chunk_path(str(digest)).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != digest:
                        raise ValueError(f"overlay chunk checksum mismatch: {digest}")
                    handle.write(payload)
                handle.truncate(int(spec["size"]))
            os.replace(tmp, target)
        return output


class OverlayFile:
    def __init__(self, store: PackChunkStore, spec: dict, chunk_bytes: int) -> None:
        self.store = store
        self.size = int(spec["size"])
        self.chunks = tuple(str(value) for value in spec["chunks"])
        self.chunk_bytes = int(chunk_bytes)

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("overlay read exceeds file bounds")
        if length == 0:
            return b""
        first = offset // self.chunk_bytes
        last = (offset + length - 1) // self.chunk_bytes
        slab = bytearray()
        for index in range(first, last + 1):
            digest = self.chunks[index]
            payload = self.store._chunk_path(digest).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"overlay chunk checksum mismatch: {digest}")
            slab.extend(payload)
        relative = offset - first * self.chunk_bytes
        return bytes(slab[relative : relative + length])


class ChunkStoreWeightStore(WeightStore):
    """Read a profile's contiguous weights file without materializing it."""

    def __init__(
        self,
        store: PackChunkStore,
        profile: str,
        *,
        relative_path: str = "weights.bin",
    ) -> None:
        self._file = store.read_file(profile, relative_path)
        self.read_calls = 0
        self.bytes_read = 0
        self.closed = False

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self.read_bytes_span(entry.offset, entry.length)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        if self.closed:
            raise ValueError("chunk store is closed")
        payload = self._file.read(offset, length)
        self.read_calls += 1
        self.bytes_read += len(payload)
        return payload

    def close(self) -> None:
        self.closed = True


__all__ = ["ChunkStoreWeightStore", "OverlayFile", "PackChunkStore"]
