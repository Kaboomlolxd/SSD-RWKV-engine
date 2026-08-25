#!/usr/bin/env python3
"""
Synthetic pack benchmark — resident / partial / streaming tok/s and RAM.

Toy matmul backend; use for **streaming tax ratio**, not absolute ChatRWKV tok/s.
For real model numbers use ``bench_throughput.py``. For inf-compute frontier use
``bench_io_ceiling.py``.
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

from bench._bench_profiles import DEFAULT, SYNTH_FULL, SYNTH_QUICK
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack


def bench_mode(pack: Path, mode: str, max_tokens: int, prompt: str) -> dict:
    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode=mode,
        device="cpu",
        max_tokens=max_tokens,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        t0 = time.perf_counter()
        engine.generate(prompt)
        wall = time.perf_counter() - t0
        tok_s = max_tokens / wall if wall > 0 else 0.0
        m = engine.metrics
        read_ms = sum(x.read_ms for x in m.layers)
        staging_ms = sum(x.staging_ms for x in m.layers)
        compute_ms = sum(x.compute_ms for x in m.layers)
        io_ms = (read_ms + staging_ms) / max_tokens if max_tokens else 0.0
        io_ceiling = 1000.0 / io_ms if io_ms > 1e-6 else None
        return {
            "mode": mode,
            "tok_s": round(tok_s, 2),
            "wall_s": round(wall, 4),
            "z_mb": round(m.z_bytes / 1e6, 3) if m.z_bytes else 0.0,
            "provider_mb": round(m.provider_cache_bytes / 1e6, 3),
            "read_ms_per_token": round(read_ms / max_tokens, 2) if max_tokens else 0.0,
            "staging_ms_per_token": round(staging_ms / max_tokens, 2) if max_tokens else 0.0,
            "compute_ms_per_token": round(compute_ms / max_tokens, 2) if max_tokens else 0.0,
            "io_ceiling_tok_s": round(io_ceiling, 2) if io_ceiling else None,
            "prefetch_overlaps": m.prefetch_overlaps,
        }
    finally:
        engine.close()


def _clear_trinity_bench_env() -> None:
    """Avoid frontier preset env leaking into synthetic / toy runs."""
    import os

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
        "RWKV_WARM_DISK_CACHE",
        "RWKV_STRICT_FUSED_RETAIN",
    ):
        os.environ.pop(key, None)


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic pack mode comparison")
    p.add_argument("--model", help="Runtime pack directory")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="8 tokens, 1 sample (fast smoke)")
    p.add_argument("--full", action="store_true", help="32 tokens, 2 samples (legacy depth)")
    p.add_argument("--prompt", default="benchmark")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    p.add_argument("--json-out", type=Path, help="Write JSON results")
    args = p.parse_args()
    _clear_trinity_bench_env()

    if args.full:
        profile = SYNTH_FULL
    elif args.quick:
        profile = SYNTH_QUICK
    else:
        profile = DEFAULT
    max_tokens = args.max_tokens if args.max_tokens is not None else profile.max_tokens
    samples = args.samples if args.samples is not None else profile.samples

    pack = Path(args.model) if args.model else Path("_bench_pack")
    if not args.model:
        create_synthetic_pack(pack, n_layer=4, n_embd=32)

    modes = ("resident", "partial", "streaming")
    rows: list[dict] = []
    for mode in modes:
        rates: list[float] = []
        last: dict = {}
        for _ in range(samples):
            last = bench_mode(pack, mode, max_tokens, args.prompt)
            rates.append(last["tok_s"])
        if rates:
            last["tok_s"] = round(statistics.median(rates), 2)
        rows.append(last)

    resident = next((r["tok_s"] for r in rows if r["mode"] == "resident"), None)
    for r in rows:
        if resident and resident > 0:
            r["vs_resident"] = round(r["tok_s"] / resident, 3)

    out = {
        "pack": str(pack),
        "max_tokens": max_tokens,
        "bench_profile": profile.name,
        "backend": "synthetic",
        "rows": rows,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"synthetic bench  pack={pack}  max_tokens={max_tokens}  profile={profile.name}")
    print(
        f"{'mode':<12} {'tok/s':>8} {'z_mb':>8} {'io_cap':>8} "
        f"{'read':>7} {'stage':>7} {'vs_res':>7}"
    )
    for r in rows:
        cap = r.get("io_ceiling_tok_s")
        cap_s = f"{cap:.0f}" if cap else "—"
        vs = r.get("vs_resident")
        vs_s = f"{vs:.2f}" if vs is not None else "—"
        print(
            f"{r['mode']:<12} {r['tok_s']:>8.2f} {r['z_mb']:>8.3f} {cap_s:>8} "
            f"{r['read_ms_per_token']:>7.2f} {r['staging_ms_per_token']:>7.2f} {vs_s:>7}"
        )
    if args.json_out:
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
