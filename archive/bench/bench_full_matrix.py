#!/usr/bin/env python3
"""
Full residency matrix: resident / partial / strict streaming / bounded cache.

One matched ChatRWKV config (cpu bf16, mmap, prefetch) across modes so comparisons
are fair. Emits JSON + CSV + printed analysis (time breakdown, disk vs decode).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.stream_cache_policy import resolve_max_layers_in_z
from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "bench" / "profiles"

CKPT_0_01B = ROOT / "test_model/rwkv7-g1d-0.01b-bench.pth"
CKPT_0_1B = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"


@dataclass(frozen=True)
class PackSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class Scenario:
    key: str
    mode: str
    stream_layer_cache: bool = False
    warm_z: bool = False
    max_layers_in_z: int = 1
    partial_profile: Path | None = None


def _pack_disk_mb(pack_dir: Path) -> dict[str, float]:
    weights = pack_dir / "weights.bin"
    shadow = pack_dir / "shadow.bin"
    w = weights.stat().st_size / 1e6 if weights.is_file() else 0.0
    s = shadow.stat().st_size / 1e6 if shadow.is_file() else 0.0
    return {
        "weights_mb": round(w, 2),
        "shadow_mb": round(s, 2),
        "total_disk_mb": round(w + s, 2),
    }


def _timing_breakdown(m: Any, wall_s: float, max_tokens: int) -> dict[str, Any]:
    layers = m.layers
    read = sum(L.read_ms for L in layers)
    staging = sum(L.staging_ms for L in layers)
    compute = sum(L.compute_ms for L in layers)
    h2d = sum(L.h2d_ms for L in layers)
    bubble = sum(L.bubble_ms for L in layers)
    prefetch_w = sum(L.prefetch_wait_ms for L in layers)
    tracked = read + staging + compute + h2d + bubble
    wall_ms = wall_s * 1000.0
    other_ms = max(0.0, wall_ms - tracked)
    denom = tracked if tracked > 1e-6 else wall_ms
    per_tok = max(max_tokens, 1)

    return {
        "total_read_ms": round(read, 1),
        "total_staging_ms": round(staging, 1),
        "total_compute_ms": round(compute, 1),
        "total_h2d_ms": round(h2d, 1),
        "total_bubble_ms": round(bubble, 1),
        "prefetch_wait_ms": round(prefetch_w, 1),
        "orchestration_ms": round(other_ms, 1),
        "ms_per_token_read": round(read / per_tok, 2),
        "ms_per_token_staging": round(staging / per_tok, 2),
        "ms_per_token_compute": round(compute / per_tok, 2),
        "ms_per_token_wall": round(wall_ms / per_tok, 2),
        "pct_read": round(100.0 * read / denom, 1) if denom else 0.0,
        "pct_staging": round(100.0 * staging / denom, 1) if denom else 0.0,
        "pct_compute": round(100.0 * compute / denom, 1) if denom else 0.0,
        "pct_orchestration": round(100.0 * other_ms / wall_ms, 1) if wall_ms else 0.0,
        "layer_steps": len(layers),
        "chunk_reads": sum(L.chunk_reads for L in layers),
        "layer_cache_hits": sum(L.layer_cache_hits for L in layers),
        "shadow_hits": sum(getattr(L, "shadow_hits", 0) for L in layers),
        "disk_cache_hits": sum(getattr(L, "disk_cache_hits", 0) for L in layers),
        "prefetch_hits": sum(L.prefetch_hits for L in layers),
    }


def _run_once(
    pack: Path,
    ckpt: Path,
    scenario: Scenario,
    *,
    max_tokens: int,
    strategy: str,
    io_backend: str,
    prefetch: bool,
) -> dict[str, Any]:
    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt),
        backend="chatrwkv",
        mode=scenario.mode,
        strategy=strategy,
        device="cpu",
        max_tokens=max_tokens,
        stream_layer_cache=scenario.stream_layer_cache,
        warm_z=scenario.warm_z,
        max_layers_in_z=scenario.max_layers_in_z,
        greedy=True,
        skeleton_load=True,
        prefetch_enabled=prefetch,
        io_backend=io_backend,
        residency_profile=scenario.partial_profile,
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        eng.metrics.layers.clear()
        t0 = time.perf_counter()
        eng.generate("Throughput bench prompt for residency matrix.")
        wall = time.perf_counter() - t0
        m = eng.metrics
        model = getattr(eng.backend, "_model", None)
        layers_in_z = (
            len(block_layer_ids_in_z(model.z))
            if model is not None and hasattr(model, "z")
            else 0
        )
        tok_s = max_tokens / wall if wall > 0 else 0.0
        row = {
            "tok_s": round(tok_s, 2),
            "wall_s": round(wall, 3),
            "layers_in_z": layers_in_z,
            "z_mb": round(m.z_bytes / 1e6, 2) if m.z_bytes else 0.0,
            "provider_cache_mb": round(m.provider_cache_bytes / 1e6, 2),
            "prefetch_overlaps": m.prefetch_overlaps,
        }
        row.update(_timing_breakdown(m, wall, max_tokens))
        return row
    finally:
        eng.close()


def _median_runs(
    pack: Path,
    ckpt: Path,
    scenario: Scenario,
    *,
    max_tokens: int,
    samples: int,
    warmup: int,
    strategy: str,
    io_backend: str,
    prefetch: bool,
) -> dict[str, Any]:
    numeric_keys = (
        "tok_s",
        "wall_s",
        "total_read_ms",
        "total_staging_ms",
        "total_compute_ms",
        "ms_per_token_staging",
        "ms_per_token_read",
        "ms_per_token_compute",
    )
    buckets: dict[str, list[float]] = {k: [] for k in numeric_keys}
    last: dict[str, Any] = {}
    for attempt in range(warmup + samples):
        last = _run_once(
            pack,
            ckpt,
            scenario,
            max_tokens=max_tokens,
            strategy=strategy,
            io_backend=io_backend,
            prefetch=prefetch,
        )
        if attempt >= warmup:
            for k in numeric_keys:
                if k in last:
                    buckets[k].append(float(last[k]))
    out = dict(last)
    out["tok_s_median"] = round(statistics.median(buckets["tok_s"]), 2) if buckets["tok_s"] else 0.0
    out["tok_s_min"] = round(min(buckets["tok_s"]), 2) if buckets["tok_s"] else 0.0
    out["tok_s_max"] = round(max(buckets["tok_s"]), 2) if buckets["tok_s"] else 0.0
    out["tok_s"] = out["tok_s_median"]
    out["samples"] = len(buckets["tok_s"])
    return out


def _scenarios_for_pack(pack: Path, size_key: str) -> list[Scenario]:
    manifest = Manifest.load(pack)
    n_block = len(manifest_block_layers(manifest))
    max_z = resolve_max_layers_in_z(2, n_block, stream_layer_cache=True, warm_z=False)
    partial = PROFILES / f"partial_{size_key}.json"
    if not partial.is_file():
        partial = None
    return [
        Scenario("resident", "resident"),
        Scenario("partial", "partial", partial_profile=partial),
        Scenario("streaming_strict", "streaming", stream_layer_cache=False),
        Scenario(
            "streaming_cache",
            "streaming",
            stream_layer_cache=True,
            max_layers_in_z=max_z,
        ),
    ]


def _analyze(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Time breakdown (median run)",
        "",
        "Staging ~ dequant/LUT/shadow inject; read = disk/mmap; compute = RWKV kernels.",
        "",
    ]
    by_size: dict[str, list[dict]] = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)

    for size, group in by_size.items():
        lines.append(f"### {size}")
        lines.append("")
        lines.append(
            "| Pack | Mode | tok/s | %read | %staging | %compute | "
            "read ms/tok | staging ms/tok | vs resident |"
        )
        lines.append("|------|------|-------|-------|----------|----------|"
                     "-----------|----------------|-------------|")
        residents = {
            r["pack"]: r["tok_s"]
            for r in group
            if r["scenario"] == "resident"
        }
        for r in sorted(group, key=lambda x: (x["pack"], x["scenario"])):
            base = residents.get(r["pack"], 0.0)
            ratio = f"{100.0 * r['tok_s'] / base:.0f}%" if base > 0 else "n/a"
            lines.append(
                f"| {r['pack']} | {r['scenario']} | {r['tok_s']:.1f} | "
                f"{r['pct_read']:.0f}% | {r['pct_staging']:.0f}% | {r['pct_compute']:.0f}% | "
                f"{r['ms_per_token_read']:.2f} | {r['ms_per_token_staging']:.2f} | {ratio} |"
            )
        lines.append("")

        lines.append("**Disk vs I/O (Trinity hypothesis)**")
        lines.append("")
        for pack in sorted({r["pack"] for r in group}):
            sub = [r for r in group if r["pack"] == pack and r["scenario"] == "streaming_strict"]
            if not sub:
                continue
            r = sub[0]
            lines.append(
                f"- **{pack}**: on-disk {r['weights_mb']:.1f} MiB "
                f"(+shadow {r['shadow_mb']:.1f}) → read {r['ms_per_token_read']:.2f} ms/tok, "
                f"staging {r['ms_per_token_staging']:.2f} ms/tok "
                f"(disk is smaller but decode dominates)."
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    if find_chatrwkv_root() is None:
        raise SystemExit("ChatRWKV not found")

    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--size", choices=("0.01b", "0.1b", "both"), default="both")
    p.add_argument(
        "--strategy",
        default="cpu bf16",
        help="Matched ChatRWKV strategy for all modes (default: cpu bf16)",
    )
    p.add_argument("--io-backend", default="mmap")
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "test_model/trinity_eval/matrix",
    )
    args = p.parse_args()

    suites: list[tuple[str, Path, Path, list[PackSpec]]] = []
    if args.size in ("0.01b", "both"):
        suites.append(
            (
                "0.01b",
                "0.01B",
                CKPT_0_01B,
                [
                    PackSpec("FP16", ROOT / "test_model/runtime_pack_0.01b"),
                    PackSpec("trinity_lut2", ROOT / "test_model/trinity_eval/trinity_lut2"),
                    PackSpec(
                        "trinity_lut2+shadow",
                        ROOT / "test_model/trinity_eval/trinity_lut2_shadow",
                    ),
                ],
            )
        )
    if args.size in ("0.1b", "both"):
        suites.append(
            (
                "0.1b",
                "0.1B",
                CKPT_0_1B,
                [
                    PackSpec("FP16", ROOT / "test_model/runtime_pack"),
                    PackSpec("trinity_lut2", ROOT / "test_model/trinity_eval/trinity_lut2_0.1b"),
                    PackSpec(
                        "trinity_lut2+shadow",
                        ROOT / "test_model/trinity_eval/trinity_lut2_shadow_0.1b",
                    ),
                ],
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    print(
        f"Matrix bench  strategy={args.strategy!r}  io={args.io_backend}  "
        f"prefetch={not args.no_prefetch}  tokens={args.max_tokens}  "
        f"samples={args.samples}\n"
    )

    for size_key, size_label, ckpt, packs in suites:
        if not ckpt.is_file():
            print(f"SKIP {size_label}: missing {ckpt}")
            continue
        print(f"=== {size_label} ===")
        for spec in packs:
            if not spec.path.is_dir():
                print(f"  SKIP {spec.label}")
                continue
            disk = _pack_disk_mb(spec.path)
            scenarios = _scenarios_for_pack(spec.path, size_key)
            for scen in scenarios:
                print(f"  {spec.label:22} {scen.key:18} ...", flush=True)
                r = _median_runs(
                    spec.path,
                    ckpt,
                    scen,
                    max_tokens=args.max_tokens,
                    samples=args.samples,
                    warmup=args.warmup,
                    strategy=args.strategy,
                    io_backend=args.io_backend,
                    prefetch=not args.no_prefetch,
                )
                row = {
                    "size": size_label,
                    "pack": spec.label,
                    "scenario": scen.key,
                    "mode": scen.mode,
                    "max_layers_in_z": scen.max_layers_in_z if scen.stream_layer_cache else 0,
                    **disk,
                    **r,
                }
                results.append(row)
                print(
                    f"    -> {r['tok_s']:.1f} tok/s  "
                    f"read {r['pct_read']:.0f}%  staging {r['pct_staging']:.0f}%  "
                    f"compute {r['pct_compute']:.0f}%  "
                    f"({r['ms_per_token_read']:.2f}+{r['ms_per_token_staging']:.2f}+"
                    f"{r['ms_per_token_compute']:.2f} ms/tok)"
                )
        print()

    json_path = args.out_dir / "full_matrix.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if results:
        fieldnames = list(results[0].keys())
        csv_path = args.out_dir / "full_matrix.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    md_path = args.out_dir / "ANALYSIS.md"
    md_path.write_text(_analyze(results), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {args.out_dir / 'full_matrix.csv'}")
    print(f"Wrote {md_path}")
    print()
    text = md_path.read_text(encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
