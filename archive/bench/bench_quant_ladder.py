#!/usr/bin/env python3
"""
M5 quantization ladder — document pack size and read latency per codec.

Compares FP16/BF16 runtime packs (and optional external quant artifacts when present).
Does not claim stacked speedups; prints one row per codec.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rwkv_ssd.runtime.pack_bench import pack_read_stats


def main() -> None:
    p = argparse.ArgumentParser(description="M5 quant / pack ladder benchmark")
    p.add_argument("--model", action="append", required=True, help="Runtime pack dir (repeatable)")
    p.add_argument("--label", action="append", help="Row label per --model (same order)")
    p.add_argument("--io-backend", default="mmap", choices=["mmap", "pread", "threaded"])
    p.add_argument("--json-out", help="Write results JSON")
    args = p.parse_args()

    labels = args.label or []
    rows = []
    for i, pack in enumerate(args.model):
        label = labels[i] if i < len(labels) else Path(pack).name
        stats = pack_read_stats(Path(pack), args.io_backend)
        stats["label"] = label
        rows.append(stats)
        print(
            f"{label:20} {stats['weights_mb']:8.2f} MB  "
            f"codecs={stats['dequant_codecs']}  read={stats['read_gbs']:.2f} GB/s"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
