#!/usr/bin/env python3
"""Benchmark ChatRWKV resident decode (real checkpoint) — M4 performance table."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def main() -> None:
    p = argparse.ArgumentParser(description="ChatRWKV resident tok/s (requires checkpoint + pack)")
    p.add_argument("--model", default="test_model/runtime_pack")
    p.add_argument("--checkpoint", default="test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    p.add_argument("--strategy", default="cpu bf16")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if find_chatrwkv_root() is None:
        raise SystemExit("ChatRWKV not found — clone into test_model/ChatRWKV or set CHATRWKV_ROOT")

    cfg = EngineConfig(
        pack_dir=Path(args.model),
        checkpoint_path=Path(args.checkpoint),
        backend="chatrwkv",
        mode="resident",
        strategy=args.strategy,
        device="cpu",
        max_tokens=args.max_tokens,
        verify_hash=False,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        t0 = time.perf_counter()
        engine.generate(args.prompt)
        wall = time.perf_counter() - t0
        tok_s = args.max_tokens / wall if wall > 0 else 0.0
        row = {
            "backend": "chatrwkv",
            "mode": "resident",
            "max_tokens": args.max_tokens,
            "wall_s": round(wall, 4),
            "tok_s": round(tok_s, 2),
            "strategy": args.strategy,
        }
        if args.json:
            print(json.dumps(row, indent=2))
        else:
            print(
                f"chatrwkv resident: {tok_s:.2f} tok/s "
                f"({args.max_tokens} tok in {wall:.2f}s) strategy={args.strategy!r}"
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
