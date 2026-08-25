#!/usr/bin/env python3
"""
Throughput comparison: resident vs partial vs streaming vs bounded streaming+cache.

Default model: 0.01B bench pack (fast). Use ``--heavy`` for 0.1B (more realistic).

The default rwkv.cpp backend uses the native GGML graph. Streaming+cache keeps
only ``max_layers_in_z`` block layers in ``z`` (plus skeleton globals). Use
``--warm-z`` to opt into full preload (throughput experiments, not M3 memory story).
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

from bench._bench_profiles import resolve_profile
from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z

BENCH_PACK_0_01B = Path("test_model/runtime_pack_0.01b")
BENCH_CKPT_0_01B = Path("test_model/rwkv7-g1d-0.01b-bench.pth")
HEAVY_PACK = Path("test_model/runtime_pack")
HEAVY_CKPT = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
BENCH_GGML_0_01B = Path("test_model/rwkv7-g1d-0.01b-bench-FP16.bin")
HEAVY_GGML = Path("test_model/rwkv7-g1d-0.1b-FP16.bin")


def _resolved_max_z(pack: Path, configured: int, *, stream_layer_cache: bool) -> int:
    from rwkv_ssd.runtime.layer_keys import manifest_block_layers
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.stream_cache_policy import resolve_max_layers_in_z

    manifest = Manifest.load(pack)
    n_block = len(manifest_block_layers(manifest))
    return resolve_max_layers_in_z(
        configured, n_block, stream_layer_cache=stream_layer_cache, warm_z=False
    )


def _pack_weights_mb(pack: Path) -> float:
    weights = pack / "weights.bin"
    if weights.is_file():
        return round(weights.stat().st_size / 1e6, 1)
    return 0.0


def _layer_io_stats(
    metrics: object,
    *,
    backend: str,
    tok_s: float = 0.0,
    resident_tok_s: float | None = None,
    wall_s: float = 0.0,
    max_tokens: int = 0,
) -> dict:
    from rwkv_ssd.runtime.metrics import MetricsCollector

    m = metrics
    assert isinstance(m, MetricsCollector)
    total_read = sum(L.read_ms for L in m.layers)
    total_compute = sum(L.compute_ms for L in m.layers)
    total_staging = sum(L.staging_ms for L in m.layers)
    total_h2d = sum(L.h2d_ms for L in m.layers)
    n_layers = len(m.layers)
    denom = total_read + total_compute + total_staging
    io_frac = (total_read / denom) if denom > 1e-6 else None
    stream_ratio = (
        (tok_s / resident_tok_s)
        if resident_tok_s and resident_tok_s > 0 and tok_s > 0
        else None
    )
    ms_per_tok = (
        (wall_s * 1000.0 / max_tokens) if max_tokens > 0 and wall_s > 0 else None
    )
    layer_ms_per_tok = (denom / max_tokens) if max_tokens > 0 and denom > 0 else None
    orchestration_ms = None
    if ms_per_tok is not None and layer_ms_per_tok is not None:
        orchestration_ms = round(max(0.0, ms_per_tok - layer_ms_per_tok), 2)
    bridge_ms: float | None = None
    bridge_pct: float | None = None
    if ms_per_tok is not None:
        classified = total_read + total_compute + total_staging + total_h2d
        bridge_ms = round(max(0.0, ms_per_tok - classified / max_tokens), 2)
        if ms_per_tok > 1e-6:
            bridge_pct = round(bridge_ms / ms_per_tok, 3)

    if backend == "synthetic":
        bottleneck = "n/a (toy matmul; RAM/IO columns not meaningful)"
    elif stream_ratio is not None and stream_ratio < 0.85:
        tax = "large" if stream_ratio < 0.5 else "moderate"
        if io_frac is not None and io_frac >= 0.35:
            bottleneck = f"{tax} streaming tax (disk/io + inject)"
        elif io_frac is not None and io_frac < 0.12:
            bottleneck = f"{tax} streaming tax (inject path; read_ms understated)"
        else:
            bottleneck = f"{tax} streaming tax (mixed)"
    elif denom < 0.5:
        bottleneck = "n/a (no per-layer metrics)"
    elif io_frac is not None and io_frac >= 0.5:
        bottleneck = "disk/io (layer timers)"
    elif total_compute > total_read * 1.5:
        bottleneck = "kernels (layer timers)"
    else:
        bottleneck = "mixed (layer timers)"
    return {
        "total_read_ms": round(total_read, 2),
        "total_compute_ms": round(total_compute, 2),
        "total_staging_ms": round(total_staging, 2),
        "layer_steps": n_layers,
        "io_fraction": round(io_frac, 3) if io_frac is not None else None,
        "stream_ratio": round(stream_ratio, 3) if stream_ratio is not None else None,
        "ms_per_token": round(ms_per_tok, 2) if ms_per_tok is not None else None,
        "orchestration_ms_per_token": orchestration_ms,
        "bottleneck": bottleneck,
        "avg_read_ms": round(total_read / n_layers, 3) if n_layers else 0.0,
        "avg_compute_ms": round(total_compute / n_layers, 3) if n_layers else 0.0,
        "bridge_ms_per_token": bridge_ms,
        "bridge_pct": bridge_pct,
    }


def _streaming_bottleneck_label(
    backend: str,
    stream_ratio: float | None,
    io_fraction: float | None,
) -> str:
    if backend == "synthetic":
        return "n/a (toy matmul)"
    if stream_ratio is None or stream_ratio >= 0.85:
        if io_fraction is not None and io_fraction >= 0.5:
            return "disk/io (layer timers)"
        return "kernels (layer timers)"
    tax = "large" if stream_ratio < 0.5 else "moderate"
    if io_fraction is not None and io_fraction >= 0.35:
        return f"{tax} streaming tax (disk/io + inject)"
    if io_fraction is not None and io_fraction < 0.12:
        return f"{tax} streaming tax (inject path; read_ms understated)"
    return f"{tax} streaming tax (mixed)"


def _reclassify_bottlenecks(rows: list[dict], *, backend: str, max_tokens: int) -> None:
    resident = next((r["tok_s"] for r in rows if r.get("mode") == "resident"), None)
    if not resident:
        return
    for r in rows:
        r["max_tokens"] = max_tokens
        if r.get("mode") == "resident":
            r["stream_ratio"] = 1.0
            continue
        ratio = float(r["tok_s"]) / float(resident)
        r["stream_ratio"] = round(ratio, 3)
        r["bottleneck"] = _streaming_bottleneck_label(
            backend, ratio, r.get("io_fraction")
        )


def _run(
    pack: Path,
    ckpt: Path | None,
    *,
    backend: str,
    mode: str,
    strategy: str,
    max_tokens: int,
    io_backend: str,
    io_chunk_bytes: int,
    prefetch_policy: str,
    stream_layer_cache: bool,
    warm_z: bool,
    max_layers_in_z: int,
    residency_profile: Path | None,
    pipeline_resident: bool,
    trinity_decode_device: str = "auto",
) -> dict:
    from rwkv_ssd.runtime.throughput_defaults import capture_active_toggles

    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt) if ckpt else None,
        backend=backend,
        mode=mode,
        strategy=strategy,
        device="cpu",
        max_tokens=max_tokens,
        verify_hash=False,
        io_backend=io_backend,
        io_chunk_bytes=io_chunk_bytes,
        prefetch_policy=prefetch_policy,
        stream_layer_cache=stream_layer_cache,
        warm_z=warm_z,
        max_layers_in_z=max_layers_in_z,
        residency_profile=residency_profile,
        greedy=not pipeline_resident,
        trinity_decode_device=trinity_decode_device,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        if pipeline_resident and mode == "resident" and backend == "chatrwkv":
            t0 = time.perf_counter()
            text = engine.backend.generate_simple(  # type: ignore[union-attr]
                "Throughput bench prompt",
                max_tokens,
                greedy=True,
            )
            wall = time.perf_counter() - t0
            _ = text
            z_mb = 0.0
            layers_in_z = 0
            provider_cache_mb = 0.0
        else:
            t0 = time.perf_counter()
            engine.generate("Throughput bench prompt")
            wall = time.perf_counter() - t0
            model = getattr(engine.backend, "_model", None)
            if model is not None and hasattr(model, "z"):
                layers_in_z = len(block_layer_ids_in_z(model.z))
            else:
                layers_in_z = 0
            m = engine.metrics
            z_mb = round(m.z_bytes / 1e6, 2) if m.z_bytes else 0.0
            provider_cache_mb = round(m.provider_cache_bytes / 1e6, 2)
        m = engine.metrics
        tok_s = round(max_tokens / wall, 2) if wall > 0 else 0.0
        io = _layer_io_stats(
            m,
            backend=backend,
            tok_s=tok_s,
            wall_s=wall,
            max_tokens=max_tokens,
        )
        toggles = capture_active_toggles()
        return {
            "backend": backend,
            "mode": mode,
            "max_tokens": max_tokens,
            "stream_layer_cache": stream_layer_cache,
            "warm_z": warm_z,
            "max_layers_in_z": max_layers_in_z if stream_layer_cache else 0,
            "wall_s": round(wall, 3),
            "tok_s": tok_s,
            "avg_read_ms": io["avg_read_ms"],
            "avg_compute_ms": io["avg_compute_ms"],
            "total_read_ms": io["total_read_ms"],
            "total_compute_ms": io["total_compute_ms"],
            "io_fraction": io["io_fraction"],
            "bottleneck": io["bottleneck"],
            "prefetch_overlaps": m.prefetch_overlaps,
            "z_mb": z_mb,
            "layers_in_z": layers_in_z,
            "provider_cache_mb": provider_cache_mb,
            "pack_on_disk_mb": _pack_weights_mb(pack),
            "metrics_summary": m.summary() if m.layers else "",
            "bridge_ms_per_token": io["bridge_ms_per_token"],
            "bridge_pct": io["bridge_pct"],
            "shadow_active": toggles["shadow_active"],
            "decode_disk_cache_active": toggles["decode_disk_cache_active"],
            "fused_gemm_active": toggles["fused_gemm_active"],
            "stacked_count": toggles["stacked_count"],
            "stacked": toggles["stacked"],
        }
    finally:
        engine.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare inference modes (throughput plan)")
    p.add_argument(
        "--model",
        default=str(BENCH_PACK_0_01B),
        help=f"Runtime pack (default: {BENCH_PACK_0_01B})",
    )
    p.add_argument(
        "--checkpoint",
        default=str(BENCH_CKPT_0_01B),
        help=f"ChatRWKV checkpoint (default: {BENCH_CKPT_0_01B})",
    )
    p.add_argument(
        "--ggml",
        type=Path,
        help="rwkv.cpp GGML model; defaults to the matching bundled FP16 file",
    )
    p.add_argument(
        "--heavy",
        action="store_true",
        help=f"Use 0.1B pack ({HEAVY_PACK}) and checkpoint ({HEAVY_CKPT.name})",
    )
    p.add_argument(
        "--backend",
        default="rwkvcpp",
        choices=["rwkvcpp", "chatrwkv", "synthetic"],
        help="rwkvcpp for real packs (default); chatrwkv for compatibility; synthetic for demo/temp packs",
    )
    p.add_argument("--strategy", default="cpu bf16")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Resident + streaming+cache only; 12 tokens (faster)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="All modes, 48 tokens, 3 samples (legacy depth)",
    )
    p.add_argument(
        "--trinity-decode-device",
        default="auto",
        help="Trinity LUT decode: auto | cpu | xpu (Intel iGPU; ChatRWKV stays CPU)",
    )
    p.add_argument("--io-backend", default="mmap")
    p.add_argument("--io-chunk-bytes", type=int, default=0)
    p.add_argument(
        "--prefetch-policy",
        default="layer_aware",
        help="layer_aware recommended for streaming (P2.a)",
    )
    p.add_argument(
        "--partial-profile", type=Path, help="JSON residency profile for partial"
    )
    p.add_argument(
        "--no-layer-cache",
        action="store_true",
        help="Skip streaming+cache and streaming+warm-z scenarios",
    )
    p.add_argument(
        "--warm-z",
        action="store_true",
        help="Also run streaming with full z preload (opt-in throughput mode)",
    )
    p.add_argument(
        "--max-layers-in-z",
        type=int,
        default=1,
        help="Max streamed block layers retained in z (streaming+cache; default 1)",
    )
    p.add_argument(
        "--compare-io-backends",
        action="store_true",
        help="Also run streaming+cache for each io backend",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Repeat each scenario N times; report median tok/s",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Extra discarded runs per scenario before the sampled runs",
    )
    p.add_argument(
        "--pipeline-resident",
        action="store_true",
        help="Resident uses pipeline.generate instead of native forward (legacy)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--json-out", type=Path, help="Write results JSON")
    p.add_argument(
        "--print-bridge",
        action="store_true",
        help="Print only bridge_ms_per_token and bottleneck columns (suppresses other output)",
    )
    args = p.parse_args()

    pack = Path(args.model)
    ckpt = Path(args.checkpoint) if args.backend in {"chatrwkv", "rwkvcpp"} else None
    if args.heavy:
        pack = HEAVY_PACK
        ckpt = HEAVY_CKPT if args.backend in {"chatrwkv", "rwkvcpp"} else None
    if args.backend == "rwkvcpp":
        native_ggml = args.ggml or (HEAVY_GGML if args.heavy else BENCH_GGML_0_01B)
        if not native_ggml.is_file():
            raise SystemExit(
                f"rwkv.cpp GGML model not found: {native_ggml}; pass --ggml"
            )
        import os

        os.environ["RWKVCPP_GGML_PATH"] = str(native_ggml)
    if args.backend == "chatrwkv" and find_chatrwkv_root() is None:
        raise SystemExit("ChatRWKV not found")
    if args.backend == "chatrwkv" and ckpt is not None and not ckpt.is_file():
        raise SystemExit(
            f"checkpoint missing: {ckpt}\n"
            "Place the .pth under test_model/ or pass --checkpoint (see test_model/README.md)."
        )

    bench_profile = resolve_profile(quick=args.quick, full=args.full, heavy=args.heavy)
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else bench_profile.max_tokens
    )
    samples = args.samples if args.samples is not None else bench_profile.samples
    warmup = args.warmup if args.warmup is not None else bench_profile.warmup
    quick_modes_only = bench_profile.name == "quick" and not args.full

    def run_scenario(
        *,
        mode: str,
        io_backend: str,
        stream_layer_cache: bool,
        warm_z: bool,
        residency_profile: Path | None,
        label: str | None = None,
    ) -> dict:
        attempts = max(1, samples) + max(0, warmup)
        tok_rates: list[float] = []
        result: dict = {}
        for attempt in range(attempts):
            result = _run(
                pack,
                ckpt,
                backend=args.backend,
                mode=mode,
                strategy=args.strategy,
                max_tokens=max_tokens,
                io_backend=io_backend,
                io_chunk_bytes=args.io_chunk_bytes,
                prefetch_policy=args.prefetch_policy,
                stream_layer_cache=stream_layer_cache,
                warm_z=warm_z,
                max_layers_in_z=args.max_layers_in_z,
                residency_profile=residency_profile,
                pipeline_resident=args.pipeline_resident,
                trinity_decode_device=args.trinity_decode_device,
            )
            result["io_backend"] = io_backend
            if label:
                result["mode"] = label
            if attempt >= warmup:
                tok_rates.append(float(result["tok_s"]))
        if tok_rates:
            result["tok_s"] = round(statistics.median(tok_rates), 2)
            result["tok_s_mean"] = round(statistics.mean(tok_rates), 2)
            result["tok_s_min"] = round(min(tok_rates), 2)
            result["tok_s_max"] = round(max(tok_rates), 2)
            result["samples"] = len(tok_rates)
        return result

    rows = []
    mode_list: tuple[str, ...] = (
        ("resident", "streaming")
        if quick_modes_only
        else ("resident", "partial", "streaming")
    )
    for mode in mode_list:
        residency_profile = args.partial_profile if mode == "partial" else None
        rows.append(
            run_scenario(
                mode=mode,
                io_backend=args.io_backend,
                stream_layer_cache=False,
                warm_z=False,
                residency_profile=residency_profile,
            )
        )
    if not args.no_layer_cache:
        cap = _resolved_max_z(pack, args.max_layers_in_z, stream_layer_cache=True)
        rows.append(
            run_scenario(
                mode="streaming",
                io_backend=args.io_backend,
                stream_layer_cache=True,
                warm_z=False,
                residency_profile=None,
                label=f"streaming+cache (max_z={cap})",
            )
        )
        if args.warm_z:
            rows.append(
                run_scenario(
                    mode="streaming",
                    io_backend=args.io_backend,
                    stream_layer_cache=True,
                    warm_z=True,
                    residency_profile=None,
                    label="streaming+warm-z",
                )
            )
    if args.compare_io_backends:
        cap = args.max_layers_in_z
        for io_backend in ("mmap", "pread", "threaded"):
            if io_backend == args.io_backend and not args.no_layer_cache:
                continue
            rows.append(
                run_scenario(
                    mode="streaming",
                    io_backend=io_backend,
                    stream_layer_cache=True,
                    warm_z=False,
                    residency_profile=None,
                    label=f"streaming+cache ({io_backend}, max_z={cap})",
                )
            )

    _reclassify_bottlenecks(rows, backend=args.backend, max_tokens=max_tokens)

    out_doc = {
        "pack": str(pack),
        "checkpoint": str(ckpt) if ckpt else None,
        "backend": args.backend,
        "max_tokens": max_tokens,
        "bench_profile": bench_profile.name,
        "heavy": args.heavy,
        "rows": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out_doc, indent=2))
        return

    model_note = "0.1B heavy" if args.heavy else "0.01B default"
    decode_note = (
        "pipeline.generate"
        if args.pipeline_resident
        else "native forward (resident / warm-z)"
    )
    pack_mb = _pack_weights_mb(pack)
    print(
        f"backend={args.backend} model={model_note} pack={pack.name} "
        f"weights.bin={pack_mb}MB max_tokens={max_tokens} io={args.io_backend} "
        f"prefetch={args.prefetch_policy} samples={max(1, samples)} profile={bench_profile.name} "
        f"resident_decode={decode_note}"
    )
    if args.backend == "synthetic":
        print(
            "  NOTE: synthetic backend uses a tiny in-memory pack — tok/s and RAM are NOT "
            "representative of RWKV. Use --backend chatrwkv for disk/CPU diagnosis."
        )
    print(
        "  z_mb = ChatRWKV model.z; provider_mb = decoded tensor cache; "
        "io_fraction = read/(read+compute+staging); bottleneck from layer metrics CSV path"
    )
    for r in rows:
        if args.print_bridge:
            bn = r.get("bottleneck", "")
            bridge = r.get("bridge_ms_per_token")
            ratio = r.get("stream_ratio")
            ratio_note = f" vs_resident={ratio:.0%}" if ratio is not None else ""
            bridge_note = (
                f" bridge={bridge:.2f}ms ({r.get('bridge_pct', 0):.0%})"
                if bridge is not None
                else ""
            )
            print(
                f"  {r['mode']:36} {r['tok_s']:8.2f} tok/s{ratio_note}"
                f"{bridge_note}  [{bn}]"
            )
            continue
        io_note = ""
        if r.get("io_backend") and r["io_backend"] != args.io_backend:
            io_note = f" io={r['io_backend']}"
        sample_note = ""
        if r.get("samples", 1) > 1:
            sample_note = (
                f"  (median of {r['samples']}: "
                f"min={r.get('tok_s_min')} max={r.get('tok_s_max')} mean={r.get('tok_s_mean')})"
            )
        z_note = f" z={r.get('z_mb', 0):.2f}MB({r.get('layers_in_z', 0)}L)"
        prov_note = f" prov={r.get('provider_cache_mb', 0):.2f}MB"
        io_frac = r.get("io_fraction")
        frac_note = f" io%={io_frac:.0%}" if io_frac is not None else ""
        ratio = r.get("stream_ratio")
        ratio_note = f" vs_resident={ratio:.0%}" if ratio is not None else ""
        bn = r.get("bottleneck", "")
        stacked = r.get("stacked_count", 0)
        stacked_note = f" stacked={stacked}" if stacked else ""
        bridge = r.get("bridge_ms_per_token")
        bridge_note = (
            f" bridge={bridge:.1f}ms" if bridge is not None and bridge > 0.5 else ""
        )
        print(
            f"  {r['mode']:36} {r['tok_s']:8.2f} tok/s{ratio_note}  "
            f"read={r['avg_read_ms']:.3f}ms compute={r['avg_compute_ms']:.3f}ms{frac_note}  "
            f"{z_note}{prov_note}  [{bn}]{io_note}{sample_note}{stacked_note}{bridge_note}"
        )
    if args.json_out:
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
