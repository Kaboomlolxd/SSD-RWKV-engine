#!/usr/bin/env python3
"""
Memory-tier throughput: how much tok/s you lose streaming from SSD vs keeping weights in RAM.

Analog mapping (engine modes → hardware story):

  resident              → DDR5 / HBM (all block weights resident in RAM)
  partial               → hybrid: hot layers pinned in RAM, rest on SSD
  streaming (strict)    → SSD every token: reload + decode/inject all layers
  streaming+cache       → SSD + small rolling RAM cache (thesis bounded-z path)
  streaming+cache warm  → after promote: most layers in RAM (steady chat)

This is intentionally simpler than ``bench_tok_s_compare.py``: one table, vs_resident,
and ms/token breakdown (read / staging / compute).

Quick start:

  # Fast synthetic I/O model (~seconds)
  python bench/bench_memory_tier.py --synthetic --n-layer 16 --n-embd 128

  # Real RWKV-7 forward (0.01B default)
  python bench/bench_memory_tier.py

  # Realistic 0.1B
  python bench/bench_memory_tier.py --heavy
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCH_PACK_0_01B = ROOT / "test_model/runtime_pack_0.01b"
BENCH_CKPT_0_01B = ROOT / "test_model/rwkv7-g1d-0.01b-bench.pth"
HEAVY_PACK = ROOT / "test_model/runtime_pack"
HEAVY_CKPT = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"

TIER_LABELS = {
    "resident": "DDR5/HBM (resident)",
    "partial": "hybrid (partial pin)",
    "streaming": "SSD strict (no cache)",
    "streaming+cache": "SSD + RAM cache",
    "streaming+warm-z": "SSD + full z warm",
    "streaming+cold": "SSD strict + DONTNEED (cold-ish)",
}


def _pack_weights_mb(pack: Path) -> float:
    weights = pack / "weights.bin"
    if weights.is_file():
        return round(weights.stat().st_size / 1e6, 2)
    return 0.0


def _raw_disk_gbs(pack: Path, backend: str, trials: int = 2) -> float | None:
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(pack)
    streamed = manifest.streamed_tensors() or manifest.tensors
    total = sum(t.length for t in streamed)
    if total <= 0:
        return None
    times: list[float] = []
    for _ in range(trials):
        with open_weight_store(manifest.weights_path, backend=backend) as store:
            t0 = time.perf_counter()
            for entry in streamed:
                store.read_bytes(entry)
            times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    if avg <= 0:
        return None
    return round((total / (1024**3)) / avg, 2)


def _run_once(
    pack: Path,
    ckpt: Path | None,
    *,
    backend: str,
    mode: str,
    strategy: str,
    max_tokens: int,
    io_backend: str,
    stream_layer_cache: bool,
    warm_z: bool,
    max_layers_in_z: int,
    residency_profile: Path | None,
    decode_disk_cache: str,
    mmap_dontneed: bool = False,
) -> dict:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z

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
        stream_layer_cache=stream_layer_cache,
        warm_z=warm_z,
        max_layers_in_z=max_layers_in_z,
        residency_profile=residency_profile,
        decode_disk_cache=decode_disk_cache,
        greedy=True,
        mmap_dontneed=mmap_dontneed,
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        t0 = time.perf_counter()
        eng.generate("Memory tier bench prompt")
        wall = time.perf_counter() - t0
        m = eng.metrics
        tok_s = round(max_tokens / wall, 2) if wall > 0 else 0.0
        read_ms = sum(L.read_ms for L in m.layers)
        staging_ms = sum(L.staging_ms for L in m.layers)
        compute_ms = sum(L.compute_ms for L in m.layers)
        model = getattr(eng.backend, "_model", None)
        layers_in_z = (
            len(block_layer_ids_in_z(model.z))
            if model is not None and hasattr(model, "z")
            else 0
        )
        io_ms_per_token = (read_ms + staging_ms) / max_tokens
        io_ceiling_tok_s = (
            round(1000.0 / io_ms_per_token, 2) if io_ms_per_token > 1e-6 else None
        )
        denom = read_ms + staging_ms + compute_ms
        return {
            "mode": mode,
            "tok_s": tok_s,
            "io_ceiling_tok_s": io_ceiling_tok_s,
            "wall_s": round(wall, 3),
            "ms_per_token": round(wall * 1000 / max_tokens, 2),
            "read_ms_per_token": round(read_ms / max_tokens, 2),
            "staging_ms_per_token": round(staging_ms / max_tokens, 2),
            "compute_ms_per_token": round(compute_ms / max_tokens, 2),
            "io_fraction": round(read_ms / denom, 3) if denom > 1e-6 else None,
            "layers_in_z": layers_in_z,
            "z_mb": round(m.z_bytes / 1e6, 2) if m.z_bytes else 0.0,
            "provider_mb": round(m.provider_cache_bytes / 1e6, 2),
            "prefetch_overlaps": m.prefetch_overlaps,
        }
    finally:
        eng.close()


def _median_run(
    pack: Path,
    ckpt: Path | None,
    *,
    backend: str,
    mode: str,
    label: str,
    strategy: str,
    max_tokens: int,
    io_backend: str,
    stream_layer_cache: bool,
    warm_z: bool,
    max_layers_in_z: int,
    residency_profile: Path | None,
    decode_disk_cache: str,
    samples: int,
    warmup: int,
    mmap_dontneed: bool = False,
) -> dict:
    rates: list[float] = []
    last: dict = {}
    for i in range(warmup + samples):
        last = _run_once(
            pack,
            ckpt,
            backend=backend,
            mode=mode,
            strategy=strategy,
            max_tokens=max_tokens,
            io_backend=io_backend,
            stream_layer_cache=stream_layer_cache,
            warm_z=warm_z,
            max_layers_in_z=max_layers_in_z,
            residency_profile=residency_profile,
            decode_disk_cache=decode_disk_cache,
            mmap_dontneed=mmap_dontneed,
        )
        if i >= warmup:
            rates.append(float(last["tok_s"]))
    if rates:
        last["tok_s"] = round(statistics.median(rates), 2)
        last["tok_s_samples"] = rates
    last["tier_label"] = label
    last["memory_tier"] = TIER_LABELS.get(label, label)
    return last


def _build_synthetic_pack(
    tmp: Path,
    *,
    n_layer: int,
    n_embd: int,
    pack_codec: str,
) -> Path:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack_dir = tmp / f"syn_{n_layer}L_{n_embd}d_{pack_codec}"
    create_synthetic_pack(
        pack_dir,
        n_layer=n_layer,
        n_embd=n_embd,
        pack_codec=pack_codec,
        pack_layout="layer_grouped",
        quiet=True,
    )
    return pack_dir


def _print_table(rows: list[dict], resident_tok_s: float | None) -> None:
    print("\n=== Memory tier vs throughput ===\n")
    print(
        f"{'Tier':<28} {'tok/s':>8} {'I/O cap':>8} {'vs RAM':>8} "
        f"{'read':>8} {'stage':>8} {'compute':>8} {'z':>4}"
    )
    print(
        f"{'':28} {'':>8} {'(∞CPU)':>8} {'':>8} "
        f"{'ms/tok':>8} {'ms/tok':>8} {'ms/tok':>8} {'layers':>4}"
    )
    for r in rows:
        vs = (
            f"{100 * r['tok_s'] / resident_tok_s:.0f}%"
            if resident_tok_s and resident_tok_s > 0
            else "—"
        )
        cap = r.get("io_ceiling_tok_s")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        tax = (
            f"{100 - 100 * r['tok_s'] / resident_tok_s:.0f}% loss"
            if resident_tok_s and resident_tok_s > 0 and r["tok_s"] < resident_tok_s
            else ""
        )
        label = r.get("memory_tier", r.get("tier_label", r["mode"]))
        print(
            f"{label:<28} {r['tok_s']:8.2f} {cap_s:>8} {vs:>8} "
            f"{r['read_ms_per_token']:8.2f} {r['staging_ms_per_token']:8.2f} "
            f"{r['compute_ms_per_token']:8.2f} {r['layers_in_z']:4d}"
        )
        if tax:
            print(f"    SSD tax vs resident: {tax}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="SSD vs RAM (resident) throughput tax")
    p.add_argument("--model", type=Path, default=BENCH_PACK_0_01B)
    p.add_argument("--checkpoint", type=Path, default=BENCH_CKPT_0_01B)
    p.add_argument("--heavy", action="store_true", help="Use 0.1B pack + checkpoint")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Build temp synthetic pack (I/O path only; toy matmul)",
    )
    p.add_argument("--n-layer", type=int, default=16)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument(
        "--pack-codec",
        default="none",
        choices=["none", "trinity_lut2"],
        help="Synthetic pack codec (none=FP16-ish raw bytes)",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "synthetic", "chatrwkv"],
    )
    p.add_argument("--strategy", default="cpu bf16")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--io-backend", default="mmap", choices=["mmap", "pread", "threaded", "cold"])
    p.add_argument("--max-layers-in-z", type=int, default=1)
    p.add_argument(
        "--partial-profile",
        type=Path,
        default=ROOT / "bench/profiles/partial_0.1b.json",
    )
    p.add_argument("--skip-partial", action="store_true")
    p.add_argument("--skip-warm-z", action="store_true")
    p.add_argument(
        "--decode-disk-cache",
        default="0",
        help="0 for fair strict SSD; auto for production-like revisit",
    )
    p.add_argument(
        "--cold-read",
        action="store_true",
        help="Add strict streaming row with mmap DONTNEED on layer release (less OS cache)",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    pack = args.model
    ckpt = args.checkpoint
    if args.heavy:
        pack = HEAVY_PACK
        ckpt = HEAVY_CKPT

    tmp_ctx = None
    if args.synthetic:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="rwkv_memtier_")
        pack = _build_synthetic_pack(
            Path(tmp_ctx.name),
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            pack_codec=args.pack_codec,
        )

    backend = args.backend
    if backend == "auto":
        backend = "synthetic" if args.synthetic else "chatrwkv"

    if backend == "chatrwkv":
        from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root

        if find_chatrwkv_root() is None:
            raise SystemExit("ChatRWKV not found; use --synthetic or --backend synthetic")
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint missing: {ckpt}")

    disk_gbs = _raw_disk_gbs(pack, args.io_backend)
    pack_mb = _pack_weights_mb(pack)

    meta = {
        "pack": str(pack),
        "pack_on_disk_mb": pack_mb,
        "raw_read_gbs": disk_gbs,
        "backend": backend,
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "io_backend": args.io_backend,
        "note": (
            "resident = DDR5/HBM ceiling; streaming strict = SSD tax; "
            "stream+cache = bounded RAM + SSD rotation"
        ),
    }
    print(json.dumps(meta, indent=2))

    scenarios: list[tuple[str, str, bool, bool, Path | None, bool]] = [
        ("resident", "resident", False, False, None, False),
        ("streaming", "streaming", False, False, None, False),
        (
            "streaming+cache",
            "streaming+cache",
            True,
            False,
            None,
            False,
        ),
    ]
    if not args.skip_partial and args.partial_profile.is_file():
        scenarios.insert(
            1, ("partial", "partial", False, False, args.partial_profile, False)
        )
    if not args.skip_warm_z:
        scenarios.append(("streaming+warm-z", "streaming+warm-z", False, True, None, False))
    if args.cold_read:
        scenarios.insert(
            2,
            (
                "streaming+cold",
                "streaming+cold",
                False,
                False,
                None,
                True,
            ),
        )
        scenarios.insert(
            3,
            (
                "streaming+cold-io",
                "streaming+cold-io",
                False,
                False,
                None,
                False,
            ),
        )

    cold_io = args.io_backend
    if args.cold_read and cold_io == "mmap":
        cold_io = "cold"

    rows: list[dict] = []
    for label, mode, cache, warm_z, profile, mmap_dontneed in scenarios:
        eff_mode = "streaming" if mode.startswith("streaming") else mode
        io_backend = cold_io if "cold" in label else args.io_backend
        row = _median_run(
            pack,
            ckpt if backend == "chatrwkv" else None,
            backend=backend,
            mode=eff_mode,
            label=label,
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            io_backend=io_backend,
            stream_layer_cache=cache,
            warm_z=warm_z,
            max_layers_in_z=args.max_layers_in_z if cache else 0,
            residency_profile=profile,
            decode_disk_cache=args.decode_disk_cache,
            samples=args.samples,
            warmup=args.warmup,
            mmap_dontneed=mmap_dontneed,
        )
        rows.append(row)

    resident_tok_s = next(
        (r["tok_s"] for r in rows if r.get("tier_label") == "resident"),
        None,
    )
    for r in rows:
        if resident_tok_s and r["tok_s"] > 0:
            r["vs_resident"] = round(r["tok_s"] / resident_tok_s, 3)
            r["ssd_tax_pct"] = round(100 * (1 - r["tok_s"] / resident_tok_s), 1)

    _print_table(rows, resident_tok_s)

    if disk_gbs and resident_tok_s and pack_mb:
        strict = next((r for r in rows if r.get("tier_label") == "streaming"), None)
        if strict:
            # Rough: bytes touched per token ≈ full pack sweep for strict streaming
            bytes_per_tok = pack_mb * 1e6
            est_read_ms = (bytes_per_tok / (disk_gbs * 1e9)) * 1000
            print(
                f"Rough SSD read budget (strict, cold): about {pack_mb:.2f} MB/tok "
                f"at {disk_gbs:.1f} GB/s gives about {est_read_ms:.0f} ms/tok read alone "
                f"(actual read_ms/tok={strict['read_ms_per_token']:.1f}; "
                "0 means OS page cache warm)"
            )

    out_path = args.json_out or (ROOT / "test_model/trinity_eval/memory_tier.json")
    payload = {"meta": meta, "results": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
