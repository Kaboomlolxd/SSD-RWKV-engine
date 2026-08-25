#!/usr/bin/env python3
"""Cold SSD throughput: strict streaming with reopen-per-read I/O backend."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "test_model/trinity_eval/trinity_lut2_0.1b"
CKPT = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"


def _run(
    pack: Path,
    ckpt: Path,
    *,
    io_backend: str,
    fused: bool,
    max_tokens: int,
) -> dict:
    import os

    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine

    os.environ["RWKV_STREAM_LAYER_CACHE"] = "0"
    os.environ["RWKV_WARM_PROVIDER_CACHE"] = "0"
    os.environ["RWKV_LUT_GEMM_FUSED"] = "1" if fused else "0"
    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt),
        backend="chatrwkv",
        mode="streaming",
        strategy="cpu bf16",
        device="cpu",
        max_tokens=max_tokens,
        greedy=True,
        skeleton_load=True,
        stream_layer_cache=False,
        max_layers_in_z=0,
        io_backend=io_backend,
        mmap_dontneed=io_backend == "mmap",
        decode_disk_cache="0",
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        t0 = time.perf_counter()
        eng.generate("Cold SSD bench prompt")
        wall = time.perf_counter() - t0
        m = eng.metrics
        read_ms = sum(L.read_ms for L in m.layers)
        staging_ms = sum(L.staging_ms for L in m.layers)
        return {
            "io_backend": io_backend,
            "fused_gemm": fused,
            "tok_s": round(max_tokens / wall, 2) if wall > 0 else 0.0,
            "wall_s": round(wall, 3),
            "read_ms": round(read_ms, 1),
            "staging_ms": round(staging_ms, 1),
        }
    finally:
        eng.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Cold SSD strict streaming bench")
    p.add_argument("--pack", type=Path, default=PACK)
    p.add_argument("--checkpoint", type=Path, default=CKPT)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument(
        "--io-backend",
        default="cold",
        choices=["cold", "cold_pread", "pread", "mmap"],
    )
    p.add_argument("--with-fused", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    rows: list[dict] = []
    for fused in (False, True) if args.with_fused else (False,):
        rates: list[float] = []
        last: dict = {}
        for _ in range(args.samples):
            last = _run(
                args.pack,
                args.checkpoint,
                io_backend=args.io_backend,
                fused=fused,
                max_tokens=args.max_tokens,
            )
            rates.append(last["tok_s"])
        last["tok_s"] = round(statistics.median(rates), 2)
        rows.append(last)
        print(json.dumps(last))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
