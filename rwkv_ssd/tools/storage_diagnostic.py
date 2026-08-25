"""CPU/Windows-safe diagnostics for runtime-pack storage layouts.

This command measures file sizes, sequential tensor reads, per-layer read
latency, and the available shard concurrency.  Repeated reads are reported as
an OS/page-cache heuristic; they are not a claim about a physical SSD cache.
It works with legacy single-file packs, layer-affinity shards, and manifest-v2
striped packs.

Example::

    python -m rwkv_ssd.tools.storage_diagnostic \
      --pack ./runtime_pack_striped --workers 4 --json-out diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from rwkv_ssd.runtime.layer_io import entries_layer_read_span
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.weight_store_base import WeightStore
from rwkv_ssd.runtime.weight_store_sharded import (
    ShardedWeightStore,
    open_sharded_weight_store,
)


def _throughput_mbps(byte_count: int, elapsed_s: float) -> float:
    return (float(byte_count) / 1_000_000.0 / elapsed_s) if elapsed_s > 0 else 0.0


def _read_layer(store: WeightStore, entries: list[TensorEntry]) -> int:
    """Read one layer and return logical tensor bytes consumed."""
    gather = getattr(store, "read_layer_coalesced", None)
    if callable(gather):
        gathered = gather(entries)
        return sum(len(payload) for payload in gathered.values())

    span = entries_layer_read_span(entries)
    read_span = getattr(store, "read_bytes_span", None)
    if span is not None and callable(read_span):
        read_span(*span)
    else:
        for entry in entries:
            store.read_bytes(entry)
    return sum(entry.length for entry in entries)


def _layer_diagnostics(
    store: WeightStore,
    by_layer: dict[int, list[TensorEntry]],
    *,
    repeats: int,
    max_layers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_ids = sorted(by_layer)
    if max_layers > 0:
        layer_ids = layer_ids[:max_layers]
    for layer_id in layer_ids:
        entries = by_layer[layer_id]
        timings: list[float] = []
        logical_bytes = sum(entry.length for entry in entries)
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            _read_layer(store, entries)
            timings.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "layer_id": layer_id,
                "tensor_count": len(entries),
                "logical_bytes": logical_bytes,
                "read_ms": round(statistics.median(timings), 6),
                "read_ms_first": round(timings[0], 6),
                "read_ms_last": round(timings[-1], 6),
                "read_mbps": round(
                    _throughput_mbps(logical_bytes, statistics.median(timings) / 1000.0),
                    6,
                ),
            }
        )
    return rows


def run_storage_diagnostic(
    pack_dir: Path,
    *,
    backend: str = "pread",
    repeats: int = 2,
    shard_workers: int = 0,
    max_layers: int = 0,
) -> dict[str, Any]:
    """Measure a runtime pack without requiring Linux or accelerator hardware."""
    manifest = Manifest.load(pack_dir)
    is_sharded = manifest.is_sharded()
    if is_sharded:
        workers = shard_workers if shard_workers > 0 else max(1, len(manifest.shard_files))
        store: WeightStore = open_sharded_weight_store(
            manifest, parallel_workers=workers
        )
    else:
        workers = 1
        store = open_weight_store(manifest.weights_path, backend=backend)

    files = list(manifest.shard_files or [manifest.weights_path])
    shadow_path = manifest.shadow_path()
    if shadow_path is not None and shadow_path not in files:
        files.append(shadow_path)
    file_rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    entries = list(manifest.tensors)
    logical_bytes = sum(entry.length for entry in entries)
    sequential_timings: list[float] = []
    try:
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            for entry in entries:
                store.read_bytes(entry)
            sequential_timings.append(time.perf_counter() - started)

        layer_rows = _layer_diagnostics(
            store,
            manifest.by_layer(),
            repeats=max(1, repeats),
            max_layers=max_layers,
        )

        concurrency: dict[str, Any] = {
            "supported": bool(is_sharded),
            "shard_count": len(manifest.shard_files) if is_sharded else 1,
            "parallel_workers": workers,
        }
        if is_sharded:
            one_per_shard: list[TensorEntry] = []
            for shard in manifest.shard_files:
                match = next(
                    (
                        entry
                        for entry in entries
                        if entry.shard_file.replace("\\", "/") == shard.relative_to(
                            manifest.pack_dir or manifest.weights_path.parent
                        ).as_posix()
                    ),
                    None,
                )
                if match is None:
                    match = next(
                        (
                            entry
                            for entry in entries
                            if any(
                                stripe["shard_file"].replace("\\", "/")
                                == shard.relative_to(
                                    manifest.pack_dir
                                    or manifest.weights_path.parent
                                ).as_posix()
                                for stripe in entry.stripes
                            )
                        ),
                        None,
                    )
                if match is not None:
                    one_per_shard.append(match)
            started = time.perf_counter()
            read_many = getattr(store, "read_bytes_many", None)
            if callable(read_many):
                read_many(one_per_shard)
            else:
                for entry in one_per_shard:
                    store.read_bytes(entry)
            parallel_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            for entry in one_per_shard:
                store.read_bytes(entry)
            sequential_ms = (time.perf_counter() - started) * 1000.0
            concurrency.update(
                {
                    "shards_sampled": len(one_per_shard),
                    "parallel_read_ms": round(parallel_ms, 6),
                    "sequential_read_ms": round(sequential_ms, 6),
                    "parallel_speedup": round(
                        sequential_ms / parallel_ms if parallel_ms > 0 else 0.0,
                        6,
                    ),
                    "physical_scaling_claim": False,
                }
            )

        first_s = sequential_timings[0]
        last_s = sequential_timings[-1]
        return {
            "schema_version": 1,
            "pack": str(pack_dir),
            "manifest_version": manifest.version,
            "layout": manifest.meta.get("shard_strategy", "single_file"),
            "sharded": is_sharded,
            "files": file_rows,
            "file_size_bytes": sum(row["bytes"] for row in file_rows),
            "tensor_count": len(entries),
            "logical_tensor_bytes": logical_bytes,
            "sequential_tensor_read": {
                "repeats": len(sequential_timings),
                "median_ms": round(statistics.median(sequential_timings) * 1000.0, 6),
                "first_ms": round(first_s * 1000.0, 6),
                "last_ms": round(last_s * 1000.0, 6),
                "median_mbps": round(
                    _throughput_mbps(
                        logical_bytes, statistics.median(sequential_timings)
                    ),
                    6,
                ),
            },
            "per_layer": layer_rows,
            "shard_concurrency": concurrency,
            "cache_behavior": {
                "kind": "OS/page-cache heuristic",
                "repeat_reads": len(sequential_timings),
                "first_read_ms": round(first_s * 1000.0, 6),
                "repeat_read_ms": round(last_s * 1000.0, 6),
                "warm_speedup": round(first_s / last_s if last_s > 0 else 0.0, 6),
                "likely_warm_cache": bool(len(sequential_timings) > 1 and last_s <= first_s),
                "provider_cache_hits": None,
            },
            "hardware_validation": {
                "cpu_windows_safe": True,
                "physical_multi_ssd_validated": False,
                "gds_or_io_uring_used": False,
            },
        }
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--backend", default="pread")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = run_storage_diagnostic(
        args.pack,
        backend=args.backend,
        repeats=args.repeats,
        shard_workers=args.workers,
        max_layers=args.max_layers,
    )
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
