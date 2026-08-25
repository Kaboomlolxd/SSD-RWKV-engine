#!/usr/bin/env python3
"""Run throughput comparison and print recommended flags (wrapper around bench_throughput)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_PACK_0_01B = "test_model/runtime_pack_0.01b"
BENCH_CKPT_0_01B = "test_model/rwkv7-g1d-0.01b-bench.pth"
HEAVY_PACK = "test_model/runtime_pack"
HEAVY_CKPT = "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"


def main() -> None:
    p = argparse.ArgumentParser(description="Throughput tuning helper")
    p.add_argument("--model", default=BENCH_PACK_0_01B)
    p.add_argument("--checkpoint", default=BENCH_CKPT_0_01B)
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--partial-profile", type=Path, default=Path("deploy/rwkv7_0.1b_partial_hot7.json"))
    p.add_argument(
        "--warm-z",
        action="store_true",
        help="Also bench streaming+warm-z (full z preload)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    bench = root / "bench" / "bench_throughput.py"
    model = HEAVY_PACK if args.heavy else args.model
    ckpt = HEAVY_CKPT if args.heavy else args.checkpoint
    cmd = [
        sys.executable,
        str(bench),
        "--backend",
        "chatrwkv",
        "--model",
        str(model),
        "--checkpoint",
        str(ckpt),
        "--max-tokens",
        str(args.max_tokens),
        "--samples",
        str(args.samples),
        "--prefetch-policy",
        "layer_aware",
        "--strategy",
        "cpu bf16",
    ]
    if args.heavy and args.partial_profile.is_file():
        cmd.extend(["--partial-profile", str(args.partial_profile)])
    if args.heavy:
        cmd.append("--heavy")
    if args.warm_z:
        cmd.append("--warm-z")
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
