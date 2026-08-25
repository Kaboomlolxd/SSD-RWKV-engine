"""Persistent on-disk cache of decoded layer bf16 bytes (builds on first use)."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import TensorEntry

_CACHE_MAGIC = b"RDC\x01"
_CACHE_MAGIC_ZLIB = b"RDC\x02"
_CACHE_VERSION = 1


def cache_write_stats() -> tuple[int, float]:
    """Return ``(writes_submitted, total_async_wait_ms)`` for the current process.

    ``total_async_wait_ms`` is the time the calling thread spent waiting
    on the disk-cache write (the encode + the actual write completion).
    Near zero means writes fully overlap with the next layer's compute.
    The two components are tracked separately as ``encode_ms`` (CPU
    work, sync on the calling thread) and ``write_ms`` (disk write,
    async via the thread pool).
    """
    with _WRITE_STATS_LOCK:
        return (
            _WRITE_STATS["submits"],
            float(_WRITE_STATS["encode_ms"] + _WRITE_STATS["write_ms"]),
        )


def cache_write_stats_detail() -> dict[str, float]:
    """Detailed stats: ``submits``, ``encode_ms`` (sync, calling thread),
    ``write_ms`` (async, completed in the thread pool)."""
    with _WRITE_STATS_LOCK:
        return {
            "submits": _WRITE_STATS["submits"],
            "encode_ms": float(_WRITE_STATS["encode_ms"]),
            "write_ms": float(_WRITE_STATS["write_ms"]),
        }


def reset_cache_write_stats() -> None:
    with _WRITE_STATS_LOCK:
        _WRITE_STATS["submits"] = 0
        _WRITE_STATS["encode_ms"] = 0.0
        _WRITE_STATS["write_ms"] = 0.0


_WRITE_STATS: dict[str, float] = {"submits": 0, "encode_ms": 0.0, "write_ms": 0.0}
_WRITE_STATS_LOCK = threading.Lock()


def _cache_compress_enabled() -> bool:
    raw = os.environ.get("RWKV_DECODE_CACHE_COMPRESS", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    return raw in ("", "auto")


def _cache_dir(pack_dir: Path) -> Path:
    return Path(pack_dir) / ".decode_cache"


def _weights_key(meta: dict) -> str:
    by_file = meta.get("weights_sha256_by_file")
    if isinstance(by_file, dict) and by_file:
        # Sharded packs intentionally have no single weights_sha256.  A
        # canonical per-file digest map still gives the cache a stable identity
        # and invalidates decoded layers when any shard is rebuilt in place.
        return "sharded:" + json.dumps(
            by_file, sort_keys=True, separators=(",", ":")
        )
    return str(meta.get("weights_sha256") or meta.get("source_checkpoint") or "unknown")


def _layer_cache_path(pack_dir: Path, layer_id: int, weights_key: str) -> Path:
    digest = hashlib.sha256(weights_key.encode()).hexdigest()[:16]
    return _cache_dir(pack_dir) / f"layer_{layer_id}_{digest}.bin"


def cache_enabled(env: str | None) -> bool:
    raw = (env or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return raw in ("", "auto")


def _encode_layer_payload(
    entries: list[TensorEntry],
    tensors: dict[str, torch.Tensor],
) -> bytes:
    ordered = sorted(entries, key=lambda e: e.offset)
    parts: list[bytes] = []
    for entry in ordered:
        t = tensors[entry.name]
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        packed = t.contiguous().view(torch.uint16)
        # The disk cache is a host-side artifact, but decoded weights may
        # live on CUDA/XPU when model computation is accelerator-resident.
        # ``Tensor.numpy()`` only accepts CPU tensors; make the transfer
        # explicit so enabling the optional cache cannot abort accelerator
        # inference.
        if packed.device.type != "cpu":
            packed = packed.cpu()
        parts.append(packed.numpy().tobytes())
    payload = b"".join(parts)
    if _cache_compress_enabled():
        import zlib

        payload = zlib.compress(payload, level=3)
        header = _CACHE_MAGIC_ZLIB + struct.pack(
            "<III", _CACHE_VERSION, len(ordered), len(payload)
        )
    else:
        header = _CACHE_MAGIC + struct.pack(
            "<III", _CACHE_VERSION, len(ordered), len(payload)
        )
    return header + payload


def _decode_layer_payload(
    blob: bytes | memoryview,
    entries: list[TensorEntry],
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if len(blob) < 16:
        return None
    magic = blob[:4]
    if magic not in (_CACHE_MAGIC, _CACHE_MAGIC_ZLIB):
        return None
    ver, n_tensors, payload_len = struct.unpack("<III", blob[4:16])
    if ver != _CACHE_VERSION or n_tensors != len(entries):
        return None
    ordered = sorted(entries, key=lambda e: e.offset)
    expected = sum(e.numel * 2 for e in ordered)
    raw_payload = blob[16 : 16 + payload_len]
    if len(raw_payload) != payload_len:
        return None
    if magic == _CACHE_MAGIC_ZLIB:
        import zlib

        try:
            payload = zlib.decompress(raw_payload)
        except zlib.error:
            return None
    else:
        payload = raw_payload
    if len(payload) != expected:
        return None
    flat = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
    out: dict[str, torch.Tensor] = {}
    pos = 0
    for entry in ordered:
        n = entry.numel
        t = flat[pos : pos + n].reshape(entry.shape)
        pos += n
        out[entry.name] = t if device.type == "cpu" else t.to(device=device)
    return out


def warm_disk_cache_layers(
    disk_cache: DecodeDiskCache,
    provider: object,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
) -> int:
    """
    Decode streamed layers once at load and write bf16 blobs to ``.decode_cache/``.

    Skips layers already cached or resident in ``z``. Returns count written.
    """
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    if not isinstance(provider, ManifestWeightProvider):
        return 0
    warmed = 0
    with torch.no_grad():
        for layer_id in layer_ids:
            entries = by_layer.get(layer_id, [])
            if not entries or not provider.layer_has_streamed_tensors(entries):
                continue
            if provider._layer_resident_in_z(layer_id):
                continue
            if disk_cache.try_load_layer(layer_id, entries, provider._device):
                warmed += 1
                continue
            load_fn = getattr(provider, "load_layer_tensors_materialized", None)
            if load_fn is not None:
                tensors = load_fn(entries)
            else:
                tensors = provider.load_layer_tensors(entries)
            disk_cache.store_layer(layer_id, entries, tensors, async_write=False)
            provider.evict_streamed_layer(layer_id, force=True)
            warmed += 1
    disk_cache.flush()
    return warmed


class DecodeDiskCache:
    """Store concatenated bf16 layer bytes keyed by pack weights hash + layer id."""

    _MMAP_CACHE_MAX = 32

    def __init__(self, pack_dir: Path, meta: dict) -> None:
        self._pack_dir = Path(pack_dir)
        self._weights_key = _weights_key(meta)
        self._dir = _cache_dir(self._pack_dir)
        self._index_path = self._dir / "index.json"
        self._index: dict[str, str] = {}
        self._index_lock = threading.Lock()
        self._writer = ThreadPoolExecutor(
            max_workers=max(1, int(os.environ.get("RWKV_DECODE_CACHE_WRITERS", "2"))),
            thread_name_prefix="decode_cache",
        )
        self._pending: dict[int, Future[None]] = {}
        self._mmap_cache: dict[int, tuple[object, int]] = {}
        self._mmap_lru: list[int] = []
        self._mmap_lock = threading.Lock()
        if self._index_path.is_file():
            try:
                self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._index = {}

    def _path_for(self, layer_id: int) -> Path:
        return _layer_cache_path(self._pack_dir, layer_id, self._weights_key)

    def _get_mmap(self, path: Path) -> tuple[object, int] | None:
        """Return a cached read-only mmap for ``path``; reuses across loads (DNV-3)."""
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size <= 0:
            return None
        layer_id = int(path.stem.split("_")[1]) if path.stem.startswith("layer_") else -1
        key = str(path)
        with self._mmap_lock:
            cached = self._mmap_cache.get(layer_id)
            if cached is not None and cached[1] == size:
                if layer_id in self._mmap_lru:
                    self._mmap_lru.remove(layer_id)
                self._mmap_lru.append(layer_id)
                return cached
            # Evict oldest if cache is full
            while len(self._mmap_lru) >= self._MMAP_CACHE_MAX:
                old_id = self._mmap_lru.pop(0)
                old_entry = self._mmap_cache.pop(old_id, None)
                if old_entry is not None:
                    try:
                        old_entry[0].close()
                    except (OSError, ValueError):
                        pass
            try:
                f = open(path, "rb")
            except OSError:
                return None
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                f.close()
                return None
            entry = (mm, size)
            self._mmap_cache[layer_id] = entry
            self._mmap_lru.append(layer_id)
            return entry

    def _evict_mmap(self, layer_id: int) -> None:
        path = self._path_for(layer_id)
        with self._mmap_lock:
            entry = self._mmap_cache.pop(layer_id, None)
            if layer_id in self._mmap_lru:
                self._mmap_lru.remove(layer_id)
        if entry is not None:
            mm, _size = entry
            try:
                mm.close()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        self.flush()
        self._writer.shutdown(wait=True)
        with self._mmap_lock:
            entries = list(self._mmap_cache.items())
            self._mmap_cache.clear()
            self._mmap_lru.clear()
        for _key, (mm, _size) in entries:
            try:
                mm.close()
            except (OSError, ValueError):
                pass

    def flush(self) -> None:
        for fut in list(self._pending.values()):
            try:
                fut.result(timeout=120)
            except Exception:
                pass
        self._pending.clear()

    def try_load_layer(
        self,
        layer_id: int,
        entries: list[TensorEntry],
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        pending = self._pending.get(layer_id)
        if pending is not None:
            try:
                pending.result(timeout=5)
            except Exception:
                return None
        path = self._path_for(layer_id)
        if not path.is_file():
            return None
        cached = self._get_mmap(path)
        if cached is None:
            return None
        mm, _size = cached
        try:
            out = _decode_layer_payload(memoryview(mm), entries, device)
        except (OSError, ValueError):
            self._evict_mmap(layer_id)
            return None
        if out is not None:
            return out
        return None

    def _write_layer_file(self, layer_id: int, blob: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(layer_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(blob)
        # A same-size replacement must not reuse the old read-only mapping;
        # _get_mmap caches by layer id and size for the hot read path.
        self._evict_mmap(layer_id)
        os.replace(tmp, path)
        with self._index_lock:
            self._index[str(layer_id)] = path.name
            index_tmp = self._index_path.with_suffix(".json.tmp")
            index_tmp.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
            os.replace(index_tmp, self._index_path)

    def store_layer(
        self,
        layer_id: int,
        entries: list[TensorEntry],
        tensors: dict[str, torch.Tensor],
        *,
        async_write: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        blob = _encode_layer_payload(entries, tensors)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        with _WRITE_STATS_LOCK:
            _WRITE_STATS["submits"] += 1
            _WRITE_STATS["encode_ms"] += encode_ms
        if async_write:
            prev = self._pending.pop(layer_id, None)
            if prev is not None:
                # Writes for one layer share the destination path.  Waiting
                # here prevents two workers from racing on the same temporary
                # file and lets the newest blob replace the older one safely.
                try:
                    prev.result()
                except Exception:
                    pass
            fut = self._writer.submit(_timed_write_layer_file, self, layer_id, blob)
            self._pending[layer_id] = fut
            return
        # Sync path: call directly so write_ms is recorded.
        t_write0 = time.perf_counter()
        self._write_layer_file(layer_id, blob)
        with _WRITE_STATS_LOCK:
            _WRITE_STATS["write_ms"] += (time.perf_counter() - t_write0) * 1000.0


def _timed_write_layer_file(
    cache: "DecodeDiskCache", layer_id: int, blob: bytes
) -> None:
    """Async-write wrapper that records the actual write time."""
    t_write0 = time.perf_counter()
    try:
        cache._write_layer_file(layer_id, blob)
    finally:
        with _WRITE_STATS_LOCK:
            _WRITE_STATS["write_ms"] += (time.perf_counter() - t_write0) * 1000.0
