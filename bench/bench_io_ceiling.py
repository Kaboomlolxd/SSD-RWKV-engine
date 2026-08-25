#!/usr/bin/env python3
"""
RAM -> tok/s frontier bench.

Reports Pareto-best presets at each ``z`` budget (not every env combination).
Inf-compute ceiling: ``1000 / (read_ms + staging_ms)`` per token.
Legacy subsumed scenarios live in ``archive/bench/io_ceiling_legacy_scenarios.py``.

Examples:

  python bench/bench_io_ceiling.py --heavy
  python bench/bench_io_ceiling.py --heavy --full
  python bench/bench_io_ceiling.py --heavy --diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench._bench_profiles import resolve_profile, frontier_filter
from bench._ram_frontier import (
    annotate_gap_to_resident,
    diagnostic_scenarios,
    format_frontier_table,
    frontier_scenarios,
    model_scale_stats,
    pack_lut2,
    pack_shadow_sel,
    pareto_frontier,
    resolve_pack,
)
from archive.bench.io_ceiling_legacy_scenarios import LEGACY_SCENARIOS

CKPT_0_1B = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"


def _raw_gbs(pack: Path, io_backend: str, trials: int = 2) -> float | None:
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(pack)
    streamed = manifest.streamed_tensors() or manifest.tensors
    total = sum(t.length for t in streamed)
    if total <= 0:
        return None
    times: list[float] = []
    for _ in range(trials):
        with open_weight_store(manifest.weights_path, backend=io_backend) as store:
            t0 = time.perf_counter()
            for entry in streamed:
                store.read_bytes(entry)
            times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    if avg <= 0:
        return None
    return round((total / (1024**3)) / avg, 2)


def _apply_preset_flags(cfg, manifest, kw: dict) -> None:
    from rwkv_ssd.runtime.throughput_defaults import (
        apply_bounded_fused_defaults,
        apply_bounded_stream_defaults,
        apply_partial_fused_defaults,
        apply_partial_hot4_ssd_defaults,
        apply_partial_ssd_tier_defaults,
        apply_promote_max_defaults,
        apply_ssd_tier_fused_defaults,
        apply_stacked_defaults,
        apply_stacked_strict_defaults,
        apply_streaming_defaults,
    )

    if kw.get("apply_bounded"):
        apply_bounded_stream_defaults(cfg, manifest)
    elif kw.get("apply_bounded_fused"):
        apply_bounded_fused_defaults(cfg, manifest)
    elif kw.get("apply_partial_fused"):
        apply_partial_fused_defaults(cfg, manifest)
        os.environ["RWKV_PARTIAL_FUSED"] = "1"
        os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    elif kw.get("apply_partial_ssd_tier"):
        apply_partial_ssd_tier_defaults(cfg, manifest)
        os.environ["RWKV_PARTIAL_SSD_TIER"] = "1"
        os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    elif kw.get("apply_partial_hot4"):
        apply_partial_hot4_ssd_defaults(cfg, manifest)
        os.environ["RWKV_PARTIAL_SSD_TIER"] = "1"
        os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    elif kw.get("apply_ssd_tier"):
        apply_ssd_tier_fused_defaults(cfg, manifest)
    elif kw.get("apply_promote_max"):
        apply_promote_max_defaults(cfg, manifest)
    elif kw.get("apply_stacked"):
        apply_stacked_defaults(cfg, manifest)
    elif kw.get("apply_stacked_strict"):
        apply_stacked_strict_defaults(cfg, manifest)
    elif kw.get("apply_stream_defaults"):
        apply_streaming_defaults(cfg, manifest)


def _build_scenario_engine(
    pack: Path,
    ckpt: Path,
    *,
    label: str,
    max_tokens: int,
    frontier_candidate: bool = True,
    skip_warm_disk_cache: bool = False,
    **kw: object,
) -> tuple[Any, dict]:
    """Build the engine for a scenario. Returns (engine, ctx). Engine is reused
    across warmup+samples so cold-start cost is paid once per scenario, not
    per measurement."""
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.manifest import Manifest

    stream_layer_cache = bool(kw.get("stream_layer_cache", False))
    fused_gemm = bool(kw.get("fused_gemm", False))
    mmap_dontneed = bool(kw.get("mmap_dontneed", False))
    decode_disk_cache = str(kw.get("decode_disk_cache", "auto"))
    io_backend = str(kw.get("io_backend", "mmap"))
    promote_full_z = kw.get("promote_full_z")
    decode_cache_compress = kw.get("decode_cache_compress")

    for key in (
        "RWKV_PREFER_FUSED_LUT",
        "RWKV_SSD_TIER",
        "RWKV_PACK_PROFILE",
        "RWKV_PROMOTE_FULL_Z",
        "RWKV_DECODE_CACHE_COMPRESS",
        "RWKV_DECODE_SHADOW",
        "RWKV_LUT_BF16_NATIVE",
        "RWKV_WARM_PROVIDER_CACHE",
        "RWKV_PARTIAL_FUSED",
        "RWKV_PARTIAL_SSD_TIER",
        "RWKV_LUT_GEMM_FUSED",
        "RWKV_BOUNDED_STREAM",
    ):
        os.environ.pop(key, None)

    os.environ.pop("RWKV_WARM_DISK_CACHE", None)
    if skip_warm_disk_cache:
        os.environ["RWKV_WARM_DISK_CACHE"] = "0"

    os.environ["RWKV_STREAM_LAYER_CACHE"] = "1" if stream_layer_cache else "0"
    if stream_layer_cache:
        os.environ.pop("RWKV_WARM_PROVIDER_CACHE", None)
    else:
        os.environ["RWKV_WARM_PROVIDER_CACHE"] = "0"
    os.environ["RWKV_LUT_GEMM_FUSED"] = "1" if fused_gemm else "0"
    if promote_full_z is not None:
        os.environ["RWKV_PROMOTE_FULL_Z"] = str(promote_full_z)
    if decode_cache_compress is not None:
        os.environ["RWKV_DECODE_CACHE_COMPRESS"] = str(decode_cache_compress)

    mode = str(kw.get("mode", "streaming"))
    skeleton_load = kw.get("skeleton_load")
    if skeleton_load is None:
        skeleton_load = mode != "resident"

    preset_kw = {
        k: kw[k]
        for k in (
            "apply_bounded",
            "apply_bounded_fused",
            "apply_partial_fused",
            "apply_partial_ssd_tier",
            "apply_partial_hot4",
            "apply_ssd_tier",
            "apply_promote_max",
            "apply_stacked",
            "apply_stacked_strict",
            "apply_stream_defaults",
        )
        if k in kw
    }

    z_cap = kw.get("max_layers_in_z")
    if z_cap is None:
        if preset_kw.get("apply_stacked") or preset_kw.get("apply_promote_max"):
            # F5 (promote-max) / stacked need warm_z to bypass LRU; cap is a
            # placeholder — the apply_*_defaults() calls below may override it.
            z_cap = 0
        elif preset_kw.get("apply_stream_defaults"):
            z_cap = 1
        else:
            z_cap = 0 if not stream_layer_cache else 1

    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt),
        backend="chatrwkv",
        mode=mode,
        strategy="cpu bf16",
        device="cpu",
        max_tokens=max_tokens,
        greedy=True,
        skeleton_load=bool(skeleton_load),
        stream_layer_cache=stream_layer_cache,
        max_layers_in_z=int(z_cap) if z_cap is not None else 1,
        max_provider_cache_layers=int(kw.get("max_provider_cache_layers", 0)),
        decouple_provider_cache=bool(kw.get("decouple_provider_cache", False)),
        io_backend=io_backend,
        mmap_dontneed=mmap_dontneed,
        decode_disk_cache=decode_disk_cache if mode != "resident" else "0",
        ram_budget_gb=kw.get("ram_budget_gb"),
        # F5 (apply_promote_max) and F5s (apply_stacked) both need warm_z=True
        # so the ZLayerRetention bypasses the LRU cap and the engine can use
        # native model.forward() across the whole pack. Without this the F5
        # preset silently runs as F2 with cap=1.
        warm_z=bool(
            preset_kw.get("apply_promote_max") or preset_kw.get("apply_stacked")
        ),
    )
    manifest = Manifest.load(pack)
    if mode != "resident":
        _apply_preset_flags(cfg, manifest, preset_kw)
    eng = InferenceEngine(cfg)
    eng.load()
    ctx = {
        "label": label,
        "frontier_candidate": frontier_candidate,
        "frontier_id": kw.get("frontier_id"),
        "manifest": manifest,
    }
    return eng, ctx


def _run_one_generate(
    eng: Any,
    ctx: dict,
    *,
    max_tokens: int,
    snapshot_layers: bool,
) -> tuple[float, dict]:
    """Run one generate() on a pre-built engine, return (wall_s, metrics_dict).

    Clears the engine's layer rows before each call so per-sample metrics
    aren't accumulated (was doubling compute_ms on ``samples=2`` — every
    layer row is appended, not replaced, so two samples gave a 2x sum).
    The bench takes the median tok/s of all samples, not the sum, so
    accumulated metrics would have hidden steady-state regressions.
    """
    eng.metrics.layers.clear()
    t0 = time.perf_counter()
    eng.generate("RAM frontier bench prompt for throughput measurement")
    wall = time.perf_counter() - t0
    m = eng.metrics
    if snapshot_layers:
        # Slice off everything before this run so warmup's compute/staging is
        # not double-counted. Layers are appended chronologically.
        m.layers = m.layers[snapshot_layers:]
    return wall, m


def _summarize_row(
    wall: float,
    m: Any,
    *,
    label: str,
    max_tokens: int,
    frontier_candidate: bool,
    frontier_id: str | None,
    eng: Any = None,
) -> dict:
    """Build the bench row dict from a single generate() result."""
    from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z

    n_layer = max((L.layer_id for L in m.layers if L.layer_id >= 0), default=-1) + 1
    if n_layer <= 0:
        # F6 resident / Fb promote record one row per token at
        # layer_id=-1 (native forward path). Treat all rows as decode.
        n_layer = 0
    read_ms = sum(L.read_ms for L in m.layers)
    staging_ms = sum(L.staging_ms for L in m.layers)
    compute_ms = sum(L.compute_ms for L in m.layers)
    rest_io = sum(L.read_ms + L.staging_ms for L in m.layers[n_layer:])
    steady_tokens = max(max_tokens - 1, 0)
    io_ms = (read_ms + staging_ms) / max_tokens
    steady_io_ms = rest_io / steady_tokens if steady_tokens > 0 else io_ms
    compute_ms_pt = compute_ms / max_tokens
    tok_s = max_tokens / wall if wall > 0 else 0.0
    model = getattr(eng.backend, "_model", None) if eng is not None else None
    layers_in_z = (
        len(block_layer_ids_in_z(model.z))
        if model is not None and hasattr(model, "z")
        else 0
    )
    from rwkv_ssd.runtime.throughput_defaults import capture_active_toggles

    toggles = capture_active_toggles()
    wall_ms_pt = (wall * 1000.0) / max_tokens if max_tokens > 0 else 0.0
    # bridge = wall - read - compute - staging - h2d (Python overhead, kernel
    # launches, and unclassified per-token work). ``compute_ms`` is required:
    # omitting it double-counts the dominant cost and makes every tier look
    # bridge-bound.
    classified_pt = (
        (read_ms + staging_ms + compute_ms + sum(L.h2d_ms for L in m.layers))
        / max_tokens
        if max_tokens > 0
        else 0.0
    )
    bridge_ms = round(max(0.0, wall_ms_pt - classified_pt), 2)
    bridge_pct = round(bridge_ms / wall_ms_pt, 3) if wall_ms_pt > 1e-6 else 0.0
    return {
        "scenario": label,
        "frontier_id": frontier_id,
        "tok_s": round(tok_s, 2),
        "steady_io_ms_per_token": round(steady_io_ms, 2),
        "read_ms_per_token": round(read_ms / max_tokens, 2),
        "staging_ms_per_token": round(staging_ms / max_tokens, 2),
        "compute_ms_per_token": round(compute_ms_pt, 2),
        "io_ms_per_token": round(io_ms, 2),
        "layers_in_z": layers_in_z,
        "z_mb": round(m.z_bytes / 1e6, 2) if m.z_bytes else None,
        "provider_mb": round(m.provider_cache_bytes / 1e6, 2),
        "frontier_candidate": frontier_candidate,
        "bridge_ms_per_token": bridge_ms,
        "bridge_pct": bridge_pct,
        "shadow_active": toggles["shadow_active"],
        "decode_disk_cache_active": toggles["decode_disk_cache_active"],
        "fused_gemm_active": toggles["fused_gemm_active"],
        "stacked_count": toggles["stacked_count"],
        "stacked": toggles["stacked"],
    }


def _run_scenario(
    pack: Path,
    ckpt: Path,
    *,
    label: str,
    max_tokens: int,
    warmup: int = 0,
    samples: int = 1,
    frontier_candidate: bool = True,
    skip_warm_disk_cache: bool = False,
    **kw: object,
) -> dict:
    """Build the engine once, do warmup + samples, return the median row."""
    eng, ctx = _build_scenario_engine(
        pack,
        ckpt,
        label=label,
        max_tokens=max_tokens,
        frontier_candidate=frontier_candidate,
        skip_warm_disk_cache=skip_warm_disk_cache,
        **kw,
    )
    try:
        # Warmup pays the cold-start cost once: first-touch memory + lazy
        # decode-cache + provider-cache build. Without this every tier looks
        # 25-40% slower than steady-state (the F5=3.79 vs documented 8-10
        # tok/s gap).
        for _ in range(max(warmup, 0)):
            eng.generate("RAM frontier warmup prompt for cold-start amortization")
        # Drop the warmup layers from the metrics so the row is steady-state
        # only (otherwise warmup's compute + staging get folded into the
        # measurement and F1-F3 — which don't actually win from warm — end
        # up reporting inflated compute_ms).
        eng.metrics.layers.clear()
        rates: list[float] = []
        last_row: dict = {}
        for _ in range(max(samples, 1)):
            wall, m = _run_one_generate(
                eng, ctx, max_tokens=max_tokens, snapshot_layers=False
            )
            row = _summarize_row(
                wall,
                m,
                label=label,
                max_tokens=max_tokens,
                frontier_candidate=frontier_candidate,
                frontier_id=ctx["frontier_id"],
                eng=eng,
            )
            last_row = row
            rates.append(row["tok_s"])
        if rates:
            last_row["tok_s"] = round(statistics.median(rates), 2)
        return last_row
    finally:
        eng.close()


def _resolve_legacy_pack(key: str | None, default: Path) -> Path:
    if key == "shadow_sel" and pack_shadow_sel().is_dir():
        return pack_shadow_sel()
    return default


def _run_batch(
    scenarios: list[tuple[str, dict, Path, bool, str | None]],
    ckpt: Path,
    max_tokens: int,
    warmup: int,
    samples: int,
    skip_warm_disk_cache: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for label, kw, run_pack, frontier_candidate, frontier_id in scenarios:
        kw_run = {k: v for k, v in kw.items() if k != "pack_override_key"}
        kw_run["frontier_id"] = frontier_id
        # _run_scenario builds the engine once, does warmup internally, and
        # runs ``samples`` measurement calls — so cold-start cost is paid once
        # per scenario (was paid once per measurement before this refactor).
        last = _run_scenario(
            run_pack,
            ckpt,
            label=label,
            max_tokens=max_tokens,
            warmup=warmup,
            samples=samples,
            frontier_candidate=frontier_candidate,
            skip_warm_disk_cache=skip_warm_disk_cache,
            **kw_run,
        )
        rows.append(last)
        print(json.dumps(last))
    frontier_ids = {r["scenario"] for r in pareto_frontier(rows)}
    for r in rows:
        if r.get("frontier_candidate"):
            r["on_frontier"] = r["scenario"] in frontier_ids
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="RAM -> tok/s frontier bench")
    p.add_argument("--pack", type=Path, default=pack_lut2())
    p.add_argument("--checkpoint", type=Path, default=CKPT_0_1B)
    p.add_argument(
        "--heavy",
        action="store_true",
        help="0.1B Trinity pack (uses quick profile unless --full)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="F1+F3+F5 smoke: 12 tokens, skip load-time disk-cache warm (faster)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Full frontier: all F scenarios, 48 tokens, 3 samples, load-time warm cache",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Run archived subsumed scenarios (regression)",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also run D0/D1 diagnostic baselines",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None, help="Override profile token count"
    )
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "test_model/trinity_eval/ram_frontier.json",
    )
    args = p.parse_args()

    pack = args.pack
    ckpt = args.checkpoint
    if args.heavy:
        pack = pack_lut2()
        ckpt = CKPT_0_1B

    if not pack.is_dir():
        raise SystemExit(f"pack missing: {pack}")
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    profile = resolve_profile(quick=args.quick, full=args.full, heavy=args.heavy)
    max_tokens = args.max_tokens if args.max_tokens is not None else profile.max_tokens
    samples = args.samples if args.samples is not None else profile.samples
    warmup = args.warmup if args.warmup is not None else profile.warmup

    scenario_tuples: list[tuple[str, dict, Path, bool, str | None]] = []

    for fs in frontier_scenarios():
        if not frontier_filter(profile, fs.id):
            continue
        run_pack = resolve_pack(fs.pack_key) or pack
        scenario_tuples.append(
            (fs.label, dict(fs.kwargs), run_pack, fs.frontier, fs.id)
        )

    if args.diagnostics:
        for ds in diagnostic_scenarios():
            scenario_tuples.append((ds.label, dict(ds.kwargs), pack, False, ds.id))

    if args.legacy:
        for label, kw in LEGACY_SCENARIOS:
            run_pack = _resolve_legacy_pack(kw.get("pack_override_key"), pack)
            legacy_kw = {k: v for k, v in kw.items() if k != "pack_override_key"}
            scenario_tuples.append((label, legacy_kw, run_pack, True, None))

    raw_mmap = None
    if not profile.skip_raw_gbs:
        raw_mmap = _raw_gbs(pack, "mmap", trials=1)
    rows = _run_batch(
        scenario_tuples,
        ckpt,
        max_tokens,
        warmup,
        samples,
        skip_warm_disk_cache=profile.skip_warm_disk_cache,
    )
    annotate_gap_to_resident(rows)
    model_scale = model_scale_stats(pack, ckpt, rows)
    print(format_frontier_table(rows, raw_mmap, model_scale))

    out = {
        "pack": str(pack),
        "checkpoint": str(ckpt),
        "max_tokens": max_tokens,
        "samples": samples,
        "warmup": warmup,
        "bench_profile": profile.name,
        "mode": "legacy" if args.legacy else "frontier",
        "raw_read_gbs_mmap": raw_mmap,
        "model_scale": model_scale,
        "rows": rows,
        "pareto": [r["scenario"] for r in pareto_frontier(rows)],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
