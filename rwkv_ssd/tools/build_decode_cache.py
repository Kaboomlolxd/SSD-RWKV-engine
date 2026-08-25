#!/usr/bin/env python3
"""
Pre-built ``.decode_cache/`` for a runtime pack — decode streamed layers to bf16
once at build time so the engine never pays the LUT/zlib decode tax.

Usage:
  python -m rwkv_ssd.tools.build_decode_cache --pack ./my_pack
  python -m rwkv_ssd.tools.build_decode_cache --pack ./my_pack --compress --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.decode_disk_cache import (
    DecodeDiskCache,
    _weights_key,
    warm_disk_cache_layers,
)
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.provider_factory import create_weight_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_decode_cache(
    pack_dir: Path,
    *,
    compress: bool = False,
    verify: bool = False,
    quiet: bool = False,
    device: str = "cpu",
) -> int:
    if not pack_dir.is_dir():
        raise FileNotFoundError(f"pack directory not found: {pack_dir}")

    manifest = Manifest.load(pack_dir)
    layer_ids = manifest_block_layers(manifest)
    if not layer_ids:
        logger.error("no block layers found in manifest")
        return 1

    if not quiet:
        logger.info(
            "building decode cache for %s (%d layers, %d tensors) ...",
            pack_dir,
            len(layer_ids),
            len(manifest.tensors),
        )

    config = EngineConfig(
        pack_dir=pack_dir,
        mode="streaming",
        backend="chatrwkv",
        device=device,
        max_tokens=1,
        stream_layer_cache=False,
        max_layers_in_z=0,
        skeleton_load=False,
    )
    if compress:
        import os

        os.environ["RWKV_DECODE_CACHE_COMPRESS"] = "1"

    if manifest.is_sharded():
        from rwkv_ssd.runtime.weight_store_sharded import open_sharded_weight_store

        store = open_sharded_weight_store(
            manifest, parallel_workers=max(1, len(manifest.shard_files))
        )
    else:
        store = open_weight_store(manifest.weights_path, backend="mmap")
    shadow_store = None
    shadow_path = manifest.shadow_path()
    if shadow_path is not None:
        shadow_store = open_weight_store(shadow_path, backend="mmap")

    metrics = MetricsCollector()
    device_obj = torch.device(device)
    provider = create_weight_provider(
        config,
        store,
        manifest.tensors,
        device_obj,
        metrics,
        shadow_store=shadow_store,
        pack_dir=pack_dir,
        manifest_meta=manifest.meta,
    )

    disk_cache = DecodeDiskCache(pack_dir, manifest.meta)
    t0 = time.perf_counter()

    try:
        n_warmed = warm_disk_cache_layers(
            disk_cache,
            provider,
            manifest.by_layer(),
            layer_ids,
        )
    finally:
        disk_cache.close()
        provider.close()
        store.close()
        if shadow_store is not None:
            shadow_store.close()

    elapsed = time.perf_counter() - t0

    if not quiet:
        cache_dir = pack_dir / ".decode_cache"
        total_mb = 0.0
        if cache_dir.is_dir():
            total_mb = sum(f.stat().st_size for f in cache_dir.glob("*.bin")) / 1e6
        logger.info(
            "cached %d/%d layers in %.1fs (%.1f MB on disk, %.1f ms/layer)",
            n_warmed,
            len(layer_ids),
            elapsed,
            total_mb,
            (elapsed / max(n_warmed, 1)) * 1000,
        )

    if verify and n_warmed > 0:
        _verify_cache(disk_cache, manifest, layer_ids, device_obj, quiet=quiet)

    return 0 if n_warmed > 0 else 1


def _verify_cache(
    disk_cache: DecodeDiskCache,
    manifest: Manifest,
    layer_ids: list[int],
    device: torch.device,
    *,
    quiet: bool = False,
) -> None:
    by_layer = manifest.by_layer()
    ok = 0
    for lid in layer_ids:
        entries = by_layer.get(lid, [])
        if not entries:
            continue
        cached = disk_cache.try_load_layer(lid, entries, device)
        if cached is not None and len(cached) == len(entries):
            ok += 1
    if not quiet:
        logger.info("verified %d/%d cached layers", ok, len(layer_ids))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-build .decode_cache/ for a runtime pack"
    )
    p.add_argument("--pack", "-p", required=True, type=Path, help="pack directory")
    p.add_argument(
        "--compress",
        action="store_true",
        help="enable zlib compression on cache blobs (RWKV_DECODE_CACHE_COMPRESS=1)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="verify all cached layers load correctly after build",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="do not write; only verify an existing .decode_cache/ matches the pack",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="print JSON summary {layer_count, total_bytes, weights_key} and exit",
    )
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument(
        "--device",
        default="cpu",
        help="device for decode (default: cpu)",
    )
    args = p.parse_args()

    if args.print_summary:
        sys.exit(_print_summary(args.pack))

    if args.verify_only:
        sys.exit(_verify_only(args.pack, args.device, args.quiet))

    sys.exit(
        build_decode_cache(
            args.pack,
            compress=args.compress,
            verify=args.verify,
            quiet=args.quiet,
            device=args.device,
        )
    )


def _print_summary(pack_dir: Path) -> int:
    cache_dir = pack_dir / ".decode_cache"
    files = sorted(cache_dir.glob("layer_*.bin")) if cache_dir.is_dir() else []
    total_bytes = sum(f.stat().st_size for f in files)
    weights_key = "unknown"
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            weights_key = _weights_key(Manifest.load(pack_dir).meta)
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
    meta_path = pack_dir / "meta.json"
    if weights_key == "unknown" and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            weights_key = _weights_key(meta)
        except (json.JSONDecodeError, OSError):
            pass
    summary = {
        "pack": str(pack_dir),
        "weights_key": weights_key,
        "layer_count": len(files),
        "total_bytes": int(total_bytes),
        "total_mb": round(total_bytes / 1e6, 2),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _verify_only(pack_dir: Path, device: str, quiet: bool) -> int:
    manifest = Manifest.load(pack_dir)
    layer_ids = manifest_block_layers(manifest)
    if not layer_ids:
        logger.error("no block layers found in manifest")
        return 1
    disk_cache = DecodeDiskCache(pack_dir, manifest.meta)
    try:
        _verify_cache(
            disk_cache, manifest, layer_ids, torch.device(device), quiet=quiet
        )
    finally:
        disk_cache.close()
    return 0


if __name__ == "__main__":
    main()
