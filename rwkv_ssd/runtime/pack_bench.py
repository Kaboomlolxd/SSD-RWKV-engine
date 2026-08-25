"""Pack size and read throughput helpers for M5 / Trinity experiments."""

from __future__ import annotations

import time
from pathlib import Path

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store


def _weight_paths(manifest: Manifest) -> list[Path]:
    paths = [manifest.weights_path]
    paths.extend(path for path in manifest.shard_files if path not in paths)
    return paths


def _open_pack_store(manifest: Manifest, io_backend: str):
    if manifest.is_sharded():
        from rwkv_ssd.runtime.weight_store_sharded import open_sharded_weight_store

        return open_sharded_weight_store(
            manifest, parallel_workers=max(1, len(manifest.shard_files))
        )
    return open_weight_store(manifest.weights_path, backend=io_backend)


def pack_read_stats(pack_dir: Path, io_backend: str = "mmap") -> dict:
    manifest = Manifest.load(pack_dir)
    weights_mb = sum(path.stat().st_size for path in _weight_paths(manifest)) / (
        1024 * 1024
    )
    codecs = sorted({t.dequant for t in manifest.tensors})
    dtypes = sorted({t.dtype for t in manifest.tensors})
    streamed = manifest.streamed_tensors() or manifest.tensors
    total_bytes = sum(t.length for t in streamed)

    t0 = time.perf_counter()
    with _open_pack_store(manifest, io_backend) as store:
        for t in streamed:
            store.read_bytes(t)
    read_s = time.perf_counter() - t0
    gbs = (total_bytes / (1024**3)) / read_s if read_s > 0 else 0.0

    return {
        "pack": str(pack_dir),
        "weights_mb": round(weights_mb, 2),
        "tensors": len(manifest.tensors),
        "dequant_codecs": codecs,
        "dtypes": dtypes,
        "read_s": round(read_s, 3),
        "read_gbs": round(gbs, 2),
        "io_backend": io_backend,
    }


def _safe_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _dir_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def pack_full_stats(pack_dir: Path, io_backend: str = "mmap") -> dict:
    """Full pack directory byte breakdown for honest compression reporting.

    A pack on disk is more than ``weights.bin``: shadow sidecars, persistent
    ``.decode_cache/``, ``.state_cache/``, manifest/meta JSONs, and any
    sidecar files (bf16 fallback, LUT2 CLUT, etc.) all consume disk and
    affect cold-load time. The legacy ``storage_ratio`` only compares
    ``weights_mb``; ``storage_ratio_adjusted`` compares **total on-disk
    bytes** for an honest number a reviewer can audit.
    """
    pack_dir = Path(pack_dir)
    try:
        manifest = Manifest.load(pack_dir)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        manifest = None

    if manifest is not None:
        weight_paths = _weight_paths(manifest)
        shadow_path = manifest.shadow_path() or pack_dir / "shadow.bin"
    else:
        weight_paths = [pack_dir / "weights.bin"]
        shadow_path = pack_dir / "shadow.bin"

    weights_mb = sum(_safe_mb(path) for path in weight_paths)
    shadow_mb = _safe_mb(shadow_path)
    manifest_json_mb = _safe_mb(pack_dir / "manifest.json")
    meta_json_mb = _safe_mb(pack_dir / "meta.json")
    decode_cache_mb = _dir_mb(pack_dir / ".decode_cache")
    state_cache_mb = _dir_mb(pack_dir / ".state_cache")

    classified = tuple(
        [*weight_paths, shadow_path]
        + [
            pack_dir / "manifest.json",
            pack_dir / "meta.json",
            pack_dir / ".decode_cache",
            pack_dir / ".state_cache",
        ]
    )

    sidecar_mb = 0.0
    seen: set[Path] = set()
    classified_dirs: set[Path] = set()
    for p in classified:
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        seen.add(resolved)
        if p.is_dir():
            classified_dirs.add(resolved)
    if pack_dir.is_dir():
        for p in pack_dir.rglob("*"):
            if p.is_dir():
                continue
            try:
                rp = p.resolve()
            except OSError:
                continue
            if rp in seen or any(directory in rp.parents for directory in classified_dirs):
                continue
            sidecar_mb += _safe_mb(p)

    total_mb = (
        weights_mb
        + shadow_mb
        + manifest_json_mb
        + meta_json_mb
        + decode_cache_mb
        + state_cache_mb
        + sidecar_mb
    )

    if manifest is None:
        return {
            "pack": str(pack_dir),
            "weights_bin_mb": round(weights_mb, 2),
            "shadow_bin_mb": round(shadow_mb, 2),
            "manifest_json_mb": round(manifest_json_mb, 2),
            "meta_json_mb": round(meta_json_mb, 2),
            "decode_cache_mb": round(decode_cache_mb, 2),
            "state_cache_mb": round(state_cache_mb, 2),
            "sidecar_mb": round(sidecar_mb, 2),
            "total_mb": round(total_mb, 2),
            "tensors": 0,
            "dequant_codecs": [],
            "dtypes": [],
            "read_s": 0.0,
            "read_gbs": 0.0,
            "io_backend": io_backend,
            "weights_mb": round(weights_mb, 2),
        }

    codecs = sorted({t.dequant for t in manifest.tensors})
    dtypes = sorted({t.dtype for t in manifest.tensors})
    streamed = manifest.streamed_tensors() or manifest.tensors
    total_bytes = sum(t.length for t in streamed)

    t0 = time.perf_counter()
    with _open_pack_store(manifest, io_backend) as store:
        for t in streamed:
            store.read_bytes(t)
    read_s = time.perf_counter() - t0
    gbs = (total_bytes / (1024**3)) / read_s if read_s > 0 else 0.0

    return {
        "pack": str(pack_dir),
        "weights_bin_mb": round(weights_mb, 2),
        "shadow_bin_mb": round(shadow_mb, 2),
        "manifest_json_mb": round(manifest_json_mb, 2),
        "meta_json_mb": round(meta_json_mb, 2),
        "decode_cache_mb": round(decode_cache_mb, 2),
        "state_cache_mb": round(state_cache_mb, 2),
        "sidecar_mb": round(sidecar_mb, 2),
        "total_mb": round(total_mb, 2),
        "tensors": len(manifest.tensors),
        "dequant_codecs": codecs,
        "dtypes": dtypes,
        "read_s": round(read_s, 3),
        "read_gbs": round(gbs, 2),
        "io_backend": io_backend,
        "weights_mb": round(weights_mb, 2),
    }
