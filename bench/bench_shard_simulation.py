#!/usr/bin/env python3
"""
Multi-SSD simulation bench: compare sharded vs single-SSD throughput.

Since most dev machines have one SSD, this bench uses a bandwidth-throttled
weight store to simulate the per-SSD bandwidth of a multi-SSD setup. It
compares:

  1. Single SSD at X MB/s (baseline)
  2. K=2 SSDs, each at X MB/s (aggregate 2X)
  3. K=4 SSDs, each at X MB/s (aggregate 4X)

The throttled store sleeps for ``bytes / bandwidth`` before returning,
simulating a slower disk. The engine's prefetch path issues parallel
reads across shards, so the aggregate bandwidth should approach K×X.

Usage:
  python bench/bench_shard_simulation.py --pack test_model/trinity_eval/trinity_grouped_0.1b
  python bench/bench_shard_simulation.py --pack test_model/trinity_eval/trinity_grouped_0.1b --per-shard-mbps 200
  python bench/bench_shard_simulation.py --pack test_model/trinity_eval/trinity_grouped_0.1b --shards 1 2 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_synthetic_pack(pack_dir: Path, n_layer: int = 32, n_embd: int = 1024) -> Path:
    """Build a synthetic pack large enough to exceed the page cache.

    32 layers × 1024 embd × 4 tensors × 2 bytes = ~16 MB.
    Use n_embd=4096 for 7B-like geometry: ~250 MB.
    """
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    return create_synthetic_pack(
        pack_dir,
        n_layer=n_layer,
        n_embd=n_embd,
        vocab_size=65536,
        seed=42,
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        quiet=True,
    )


def _build_sharded_pack(src_pack: Path, n_shards: int) -> Path:
    """Shard an existing pack into n_shards files."""
    from rwkv_ssd.tools.shard_pack import shard_pack

    dst = src_pack.parent / f"{src_pack.name}_sharded_{n_shards}"
    if (dst / "manifest.json").is_file() and len(list(dst.glob("weights.shard.*.bin"))) == n_shards:
        return dst
    shard_pack(src_pack, dst, n_shards=n_shards)
    return dst


def _build_throttled_store(pack_dir: Path, n_shards: int, per_shard_mbps: float):
    """Build a weight store that simulates n_shards SSDs at per_shard_mbps."""
    from rwkv_ssd.runtime.io_throttled import (
        ThrottledShardedStore,
        ThrottledWeightStore,
        throttled_sharded_store,
    )
    from rwkv_ssd.runtime.io_mmap import MmapWeightStore
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store_sharded import ShardedWeightStore

    manifest = Manifest.load(pack_dir)
    if n_shards == 1:
        inner = MmapWeightStore(manifest.weights_path)
        return ThrottledShardedStore(
            ThrottledWeightStore(inner, bandwidth_mbps=per_shard_mbps, store_id=0),
            n_shards=1,
        )
    inner = ShardedWeightStore(manifest, parallel_workers=n_shards)
    throttled: dict[str, ThrottledWeightStore] = {}
    for i, shard_path in enumerate(manifest.shard_files):
        throttled[shard_path.name] = ThrottledWeightStore(
            inner._stores[shard_path.name],
            bandwidth_mbps=per_shard_mbps,
            store_id=i,
        )
    inner._stores = throttled
    return ThrottledShardedStore(inner, n_shards=n_shards)


def _run_throughput_bench(
    pack_dir: Path,
    n_shards: int,
    per_shard_mbps: float,
    n_layer: int,
    max_tokens: int,
) -> dict:
    """Run a single throughput bench with a throttled store."""
    from rwkv_ssd.runtime.io_throttled import ThrottledShardedStore

    store = _build_throttled_store(pack_dir, n_shards, per_shard_mbps)
    try:
        t0 = time.perf_counter()
        total_bytes = _simulate_layer_reads(store, n_layer, max_tokens)
        wall = time.perf_counter() - t0
        tok_s = max_tokens / wall if wall > 0 else 0.0
        return {
            "n_shards": n_shards,
            "per_shard_mbps": per_shard_mbps,
            "aggregate_mbps": n_shards * per_shard_mbps,
            "max_tokens": max_tokens,
            "wall_s": round(wall, 4),
            "tok_s": round(tok_s, 2),
            "total_bytes_read": total_bytes,
            "throttle_stats": store.aggregate_stats(),
        }
    finally:
        store.close()


def _simulate_layer_reads(store, n_layer: int, max_tokens: int) -> int:
    """Simulate the engine reading all layers once per token.

    For a strict-fused / no-cache path, every token reads all n_layer
    layers from disk. This mimics the F1 access pattern.
    """
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.layer_keys import manifest_block_layers
    from rwkv_ssd.runtime.layer_io import entries_layer_read_span

    # Load the manifest from the store
    pack_dir = None
    if hasattr(store, "_inner"):
        inner = store._inner
        if hasattr(inner, "_manifest"):
            manifest = inner._manifest
        elif hasattr(inner, "_stores"):
            first_store = next(iter(inner._stores.values()))
            if hasattr(first_store, "_inner"):
                # ThrottledWeightStore wrapping MmapWeightStore
                pass
    # Just use the manifest from the pack dir via global
    return 0


def _measure_raw_read_throughput(
    pack_dir: Path,
    n_shards: int,
    per_shard_mbps: float,
    n_tokens: int = 10,
) -> dict:
    """Measure raw per-layer read time with throttling.

    Reads all block layers once per token, K times. With throttling,
    the per-shard read time = layer_bytes / per_shard_mbps. With
    parallel cross-shard reads, the wall time is max(per_shard_time)
    rather than sum(per_shard_time).
    """
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.layer_keys import manifest_block_layers
    from rwkv_ssd.runtime.layer_io import entries_layer_read_span

    manifest = Manifest.load(pack_dir)
    block_layers = manifest_block_layers(manifest)
    if not block_layers:
        return {"error": "no block layers in pack"}

    by_layer = manifest.by_layer()
    layer_specs: list[tuple[str, int, int]] = []
    for lid in block_layers:
        group = by_layer.get(lid, [])
        if not group:
            continue
        span = entries_layer_read_span(group)
        if span is None:
            continue
        base, total = span
        shard_name = group[0].shard_file
        if not shard_name:
            continue
        layer_specs.append((shard_name, base, total))

    if not layer_specs:
        return {"error": "no layer specs (pack may not be sharded)"}

    total_bytes_per_token = sum(length for _, _, length in layer_specs)

    from rwkv_ssd.runtime.io_throttled import (
        ThrottledShardedStore,
        ThrottledWeightStore,
    )
    from rwkv_ssd.runtime.io_mmap import MmapWeightStore
    from rwkv_ssd.runtime.weight_store_sharded import ShardedWeightStore

    if n_shards == 1:
        inner = MmapWeightStore(manifest.weights_path)
        store = ThrottledShardedStore(
            ThrottledWeightStore(inner, bandwidth_mbps=per_shard_mbps, store_id=0),
            n_shards=1,
        )
    else:
        inner = ShardedWeightStore(manifest, parallel_workers=n_shards)
        throttled: dict[str, ThrottledWeightStore] = {}
        for i, shard_path in enumerate(manifest.shard_files):
            throttled[shard_path.name] = ThrottledWeightStore(
                inner._stores[shard_path.name],
                bandwidth_mbps=per_shard_mbps,
                store_id=i,
            )
        inner._stores = throttled
        store = ThrottledShardedStore(inner, n_shards=n_shards)

    try:
        # Warmup
        if hasattr(store, "read_layer_spans_parallel"):
            store._inner.read_layer_spans_parallel(layer_specs[:2])

        t0 = time.perf_counter()
        for _ in range(n_tokens):
            if hasattr(store, "read_layer_spans_parallel"):
                store._inner.read_layer_spans_parallel(layer_specs)
            else:
                for name, off, length in layer_specs:
                    store._inner.read_bytes_for_shard(name, off, length)
        wall = time.perf_counter() - t0

        tok_s = n_tokens / wall if wall > 0 else 0.0
        stats = store.aggregate_stats()
        return {
            "n_shards": n_shards,
            "per_shard_mbps": per_shard_mbps,
            "aggregate_mbps": n_shards * per_shard_mbps,
            "n_tokens": n_tokens,
            "wall_s": round(wall, 4),
            "tok_s": round(tok_s, 2),
            "bytes_per_token": total_bytes_per_token,
            "expected_wall_single_ssd": round(
                total_bytes_per_token / (per_shard_mbps * 1e6), 4
            ),
            "expected_wall_parallel_ssd": round(
                max(
                    sum(
                        length
                        for name, _, length in layer_specs
                        if hash(name) % n_shards == shard_id
                    )
                    / (per_shard_mbps * 1e6)
                    for shard_id in range(n_shards)
                ),
                4,
            ) if n_shards > 1 else None,
            "throttle_stats": stats,
        }
    finally:
        store.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-SSD simulation bench")
    p.add_argument("--pack", type=Path, help="Existing pack to use")
    p.add_argument(
        "--build-synthetic",
        action="store_true",
        help="Build a synthetic pack (32 layers, 4096 embd, ~250 MB)",
    )
    p.add_argument(
        "--synthetic-dir",
        type=Path,
        default=ROOT / "_bench_pack" / "shard_sim",
        help="Where to build the synthetic pack",
    )
    p.add_argument("--n-layer", type=int, default=32, help="Layers for synthetic pack")
    p.add_argument("--n-embd", type=int, default=4096, help="Embedding dim for synthetic pack")
    p.add_argument(
        "--shards", type=int, nargs="+", default=[1, 2, 4],
        help="Shard counts to test",
    )
    p.add_argument(
        "--per-shard-mbps", type=float, default=200.0,
        help="Simulated per-SSD bandwidth in MB/s",
    )
    p.add_argument(
        "--max-tokens", type=int, default=10,
        help="Number of token-equivalent reads (all layers per token)",
    )
    p.add_argument("--json-out", type=Path, help="Write results JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.build_synthetic or not args.pack:
        if not args.quiet:
            print(f"Building synthetic pack: {args.n_layer} layers × {args.n_embd} embd")
        pack_dir = _build_synthetic_pack(
            args.synthetic_dir,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
    else:
        pack_dir = args.pack

    if not args.quiet:
        print(f"\nMulti-SSD simulation bench")
        print(f"  pack: {pack_dir}")
        print(f"  per-shard bandwidth: {args.per_shard_mbps} MB/s")
        print(f"  shard counts: {args.shards}")
        print(f"  tokens: {args.max_tokens}\n")

    results = []
    for n_shards in args.shards:
        if n_shards > 1:
            if not args.quiet:
                print(f"Sharding pack into {n_shards} files...")
            sharded = _build_sharded_pack(pack_dir, n_shards)
            test_pack = sharded
        else:
            test_pack = pack_dir

        if not args.quiet:
            print(f"Running bench: n_shards={n_shards}...")
        result = _measure_raw_read_throughput(
            test_pack,
            n_shards=n_shards,
            per_shard_mbps=args.per_shard_mbps,
            n_tokens=args.max_tokens,
        )
        results.append(result)
        if not args.quiet:
            print(f"  tok/s = {result['tok_s']:.2f}")
            print(f"  wall = {result['wall_s']:.4f}s")
            if result.get("expected_wall_single_ssd"):
                print(f"  expected (sequential): {result['expected_wall_single_ssd']:.4f}s")
            if result.get("expected_wall_parallel_ssd"):
                print(f"  expected (parallel):   {result['expected_wall_parallel_ssd']:.4f}s")
            print()

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"{'shards':>8} {'agg MB/s':>10} {'tok/s':>8} {'speedup':>8}")
        print(f"{'-'*60}")
        baseline = results[0]["tok_s"] if results else 1.0
        for r in results:
            speedup = r["tok_s"] / baseline if baseline > 0 else 0.0
            print(
                f"{r['n_shards']:>8} {r['aggregate_mbps']:>10.0f} "
                f"{r['tok_s']:>8.2f} {speedup:>7.2f}x"
            )
        print(f"{'='*60}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
