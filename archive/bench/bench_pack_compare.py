#!/usr/bin/env python3
"""Compare pack size and read throughput across pack variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.pack_bench import pack_read_stats


def main() -> None:
    p = argparse.ArgumentParser(description="Compare runtime packs (M5 / P2.b)")
    p.add_argument("--pack", action="append", required=True, help="Pack directory")
    p.add_argument("--label", action="append", help="Labels (same order as --pack)")
    p.add_argument("--io-backend", default="mmap")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    rows = []
    for i, pack in enumerate(args.pack):
        label = args.label[i] if args.label and i < len(args.label) else Path(pack).name
        stats = pack_read_stats(Path(pack), args.io_backend)
        stats["label"] = label
        rows.append(stats)
        print(
            f"{label:16} {stats['weights_mb']:8.2f} MB  "
            f"codecs={stats['dequant_codecs']}  read={stats['read_gbs']:.2f} GB/s"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
