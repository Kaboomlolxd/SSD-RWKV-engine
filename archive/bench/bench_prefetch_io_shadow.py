#!/usr/bin/env python3
"""Compare RWKV_PREFETCH_IO_ONLY=auto vs 0 on shadow strict streaming."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_once(
    pack: Path,
    *,
    prefetch_io: str,
    max_tokens: int,
    warmup: int,
) -> dict:
    os.environ["RWKV_DECODE_SHADOW"] = "1"
    os.environ["RWKV_PREFETCH_IO_ONLY"] = prefetch_io
    os.environ.pop("RWKV_DECODE_DISK_CACHE", None)

    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.metrics import MetricsCollector

    cfg = EngineConfig(
        pack_dir=pack,
        backend="chatrwkv",
        mode="streaming",
        stream_layer_cache=False,
        warm_z=False,
        max_layers_in_z=1,
        prefetch_enabled=True,
        max_tokens=max_tokens,
        greedy=True,
        device="cpu",
    )
    metrics = MetricsCollector()
    eng = InferenceEngine(cfg, metrics=metrics)
    prompt = "Hello"
    for _ in range(warmup):
        eng.generate(prompt, max_tokens=2)
    metrics.layers.clear()
    metrics.prefetch_overlaps = 0
    t0 = time.perf_counter()
    eng.generate(prompt, max_tokens=max_tokens)
    wall = time.perf_counter() - t0
    read_ms = sum(L.read_ms for L in metrics.layers)
    staging_ms = sum(L.staging_ms for L in metrics.layers)
    shadow_hits = sum(getattr(L, "shadow_hits", 0) for L in metrics.layers)
    prefetch_wait = sum(L.prefetch_wait_ms for L in metrics.layers)
    tok_s = max_tokens / wall if wall > 0 else 0.0
    eng.close()
    return {
        "prefetch_io_only": prefetch_io,
        "tok_s": round(tok_s, 2),
        "ms_per_token_wall": round(wall * 1000 / max_tokens, 2),
        "ms_per_token_read": round(read_ms / max_tokens, 2) if max_tokens else 0,
        "ms_per_token_staging": round(staging_ms / max_tokens, 2) if max_tokens else 0,
        "prefetch_overlaps": metrics.prefetch_overlaps,
        "prefetch_wait_ms": round(prefetch_wait, 2),
        "shadow_hits": shadow_hits,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark shadow I/O prefetch overlap")
    p.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "test_model/trinity_eval/trinity_lut2_shadow_0.1b",
    )
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not args.pack.is_dir():
        raise SystemExit(f"pack not found: {args.pack}")

    rows: list[dict] = []
    for mode in ("auto", "0"):
        sample_tok: list[float] = []
        sample_read: list[float] = []
        last: dict = {}
        for _ in range(args.samples):
            last = _run_once(
                args.pack,
                prefetch_io=mode,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
            )
            sample_tok.append(last["tok_s"])
            sample_read.append(last["ms_per_token_read"])
        rows.append(
            {
                **last,
                "tok_s_mean": round(statistics.mean(sample_tok), 2),
                "ms_per_token_read_mean": round(statistics.mean(sample_read), 2),
            }
        )

    print(json.dumps(rows, indent=2))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
