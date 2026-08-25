#!/usr/bin/env python3
"""Verify optimization improvements with before/after measurements."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACK_SHADOW = ROOT / "test_model/trinity_eval/trinity_lut2_shadow_0.1b"
PACK_LUT2 = ROOT / "test_model/trinity_eval/trinity_lut2_0.1b"


def shadow_read_span_sum(manifest, min_numel: int = 0) -> int:
    from rwkv_ssd.runtime.decode_shadow import (
        entries_shadow_layer_read_span,
        split_shadow_lut_entries,
    )

    total = 0
    layer_ids = sorted(
        {
            e.layer_id
            for e in manifest.tensors
            if 0 <= e.layer_id < 9000
        }
    )
    for lid in layer_ids:
        entries = [e for e in manifest.tensors if e.layer_id == lid]
        if min_numel > 0:
            shadow, _ = split_shadow_lut_entries(entries)
            span = entries_shadow_layer_read_span(shadow) if shadow else None
        else:
            span = entries_shadow_layer_read_span(
                [e for e in entries if e.fast_offset >= 0]
            )
        if span:
            total += span[1]
    return total


def bench_lut_decode() -> dict:
    from rwkv_ssd.runtime.layer_io import entries_layer_read_span
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.trinity_decode_fast import decode_lut2_layer_cpu_fast
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(PACK_LUT2)
    entries = [e for e in manifest.tensors if e.layer_id == 0]
    base, total = entries_layer_read_span(entries)
    with open_weight_store(manifest.weights_path, backend="mmap") as store:
        raw = bytes(store.read_memoryview_span(base, total))

    os.environ["RWKV_LUT_KERNEL"] = "native"
    os.environ["RWKV_LUT_BF16_NATIVE"] = "0"
    for _ in range(3):
        decode_lut2_layer_cpu_fast(raw, entries, base, torch.device("cpu"))
    t0 = time.perf_counter()
    for _ in range(20):
        decode_lut2_layer_cpu_fast(raw, entries, base, torch.device("cpu"))
    f32_ms = (time.perf_counter() - t0) / 20 * 1000

    os.environ["RWKV_LUT_BF16_NATIVE"] = "auto"
    for _ in range(3):
        decode_lut2_layer_cpu_fast(raw, entries, base, torch.device("cpu"))
    t0 = time.perf_counter()
    for _ in range(20):
        decode_lut2_layer_cpu_fast(raw, entries, base, torch.device("cpu"))
    bf16_ms = (time.perf_counter() - t0) / 20 * 1000

    return {
        "float32_gather_ms": round(f32_ms, 2),
        "native_bf16_gather_ms": round(bf16_ms, 2),
        "speedup_x": round(f32_ms / bf16_ms, 3) if bf16_ms > 0 else 0,
    }


def bench_prefetch(mode: str, samples: int = 3) -> dict:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine

    os.environ["RWKV_DECODE_SHADOW"] = "1"
    os.environ["RWKV_PREFETCH_IO_ONLY"] = mode
    os.environ["RWKV_DECODE_DISK_CACHE"] = "0"

    tok_s: list[float] = []
    read_ms: list[float] = []
    overlaps: list[int] = []
    for _ in range(samples):
        cfg = EngineConfig(
            pack_dir=PACK_SHADOW,
            backend="chatrwkv",
            mode="streaming",
            stream_layer_cache=False,
            warm_z=False,
            max_layers_in_z=1,
            prefetch_enabled=True,
            max_tokens=8,
            greedy=True,
            device="cpu",
        )
        eng = InferenceEngine(cfg)
        eng.load()
        eng.metrics.layers.clear()
        eng.metrics.prefetch_overlaps = 0
        t0 = time.perf_counter()
        eng.generate("Hello")
        wall = time.perf_counter() - t0
        m = eng.metrics
        tok_s.append(8 / wall)
        read_ms.append(sum(L.read_ms for L in m.layers) / 8)
        overlaps.append(m.prefetch_overlaps)
        eng.close()

    return {
        "prefetch_io_only": mode,
        "tok_s_mean": round(statistics.mean(tok_s), 2),
        "ms_per_token_read": round(statistics.mean(read_ms), 2),
        "prefetch_overlaps_mean": round(statistics.mean(overlaps), 1),
    }


def bench_shadow_decode() -> dict:
    from rwkv_ssd.runtime.decode_shadow import (
        decode_shadow_layer_from_span,
        entries_shadow_layer_read_span,
    )
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(PACK_SHADOW)
    entries = [e for e in manifest.tensors if e.layer_id == 0]
    base, span_len = entries_shadow_layer_read_span(entries)
    with open_weight_store(manifest.shadow_path(), backend="mmap") as store:
        raw = store.read_bytearray_span(base, span_len)
    for _ in range(3):
        decode_shadow_layer_from_span(raw, entries, base, torch.device("cpu"))
    t0 = time.perf_counter()
    for _ in range(20):
        decode_shadow_layer_from_span(raw, entries, base, torch.device("cpu"))
    return {
        "shadow_decode_ms": round((time.perf_counter() - t0) / 20 * 1000, 2),
        "shadow_read_span_kib": round(span_len / 1024, 1),
    }


def bench_async_cache() -> dict:
    import shutil
    import tempfile

    from rwkv_ssd.runtime.decode_disk_cache import DecodeDiskCache
    from rwkv_ssd.runtime.manifest import Manifest, TensorEntry

    manifest = Manifest.load(PACK_LUT2)
    entries = [e for e in manifest.tensors if e.layer_id == 0][:1]
    tensors = {
        entries[0].name: torch.randn(*entries[0].shape, dtype=torch.bfloat16),
    }
    tmp = Path(tempfile.mkdtemp())
    try:
        cache = DecodeDiskCache(tmp, {"weights_sha256": "bench"})
        t0 = time.perf_counter()
        cache.store_layer(0, entries, tensors, async_write=False)
        sync_ms = (time.perf_counter() - t0) * 1000

        cache2 = DecodeDiskCache(tmp, {"weights_sha256": "bench2"})
        t0 = time.perf_counter()
        cache2.store_layer(0, entries, tensors, async_write=True)
        async_return_ms = (time.perf_counter() - t0) * 1000
        cache2.flush()
        cache2.close()
        cache.close()
        return {
            "sync_write_ms": round(sync_ms, 3),
            "async_return_ms": round(async_return_ms, 3),
            "async_faster_return_x": round(sync_ms / max(async_return_ms, 1e-6), 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    from rwkv_ssd.runtime.manifest import Manifest

    results: dict = {}

    if PACK_SHADOW.is_dir():
        m = Manifest.load(PACK_SHADOW)
        shadow_path = m.shadow_path()
        full_read = shadow_read_span_sum(m, 0)
        sel_read = shadow_read_span_sum(m, 4096)
        results["selective_shadow"] = {
            "shadow_bin_mib": round(shadow_path.stat().st_size / 1024**2, 2)
            if shadow_path
            else 0,
            "full_read_span_kib": round(full_read / 1024, 1),
            "selective_read_span_kib_min4096": round(sel_read / 1024, 1),
            "read_span_reduction_pct": round((1 - sel_read / max(full_read, 1)) * 100, 1),
            "note": "Repack with --shadow-min-numel 4096 to apply on disk",
        }
        results["shadow_decode"] = bench_shadow_decode()

    if PACK_LUT2.is_dir():
        results["lut_bf16_native"] = bench_lut_decode()

    if PACK_SHADOW.is_dir():
        try:
            results["prefetch"] = {
                "auto": bench_prefetch("auto"),
                "off": bench_prefetch("0"),
            }
            auto = results["prefetch"]["auto"]
            off = results["prefetch"]["off"]
            results["prefetch"]["delta"] = {
                "tok_s_gain_pct": round(
                    (auto["tok_s_mean"] / max(off["tok_s_mean"], 1e-6) - 1) * 100, 1
                ),
                "read_ms_saved": round(off["ms_per_token_read"] - auto["ms_per_token_read"], 2),
            }
        except Exception as exc:
            results["prefetch"] = {"error": str(exc)}

    results["async_decode_cache"] = bench_async_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
