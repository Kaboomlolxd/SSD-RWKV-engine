#!/usr/bin/env python3
"""Suggest hot layers for partial residency from a metrics CSV (P2.c)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Suggest resident_layer_ids from metrics CSV")
    p.add_argument("metrics_csv", type=Path)
    p.add_argument(
        "--top", type=int, default=2, help="How many layers to pin by I/O cost"
    )
    p.add_argument(
        "--metric",
        choices=["read", "io_total"],
        default="io_total",
        help="Rank by read_ms or read_ms+prefetch_wait_ms (default)",
    )
    p.add_argument("--json-out", type=Path, help="Write partial profile JSON")
    args = p.parse_args()

    by_layer: dict[int, list[float]] = defaultdict(list)
    with args.metrics_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = int(row["layer_id"])
            read_ms = float(row.get("read_ms", 0))
            wait_ms = float(row.get("prefetch_wait_ms", 0))
            cost = read_ms if args.metric == "read" else read_ms + wait_ms
            by_layer[lid].append(cost)

    ranked = sorted(
        ((lid, sum(v) / len(v)) for lid, v in by_layer.items() if 0 <= lid < 9000),
        key=lambda x: x[1],
        reverse=True,
    )
    top_ids = [lid for lid, _ in ranked[: args.top]]
    label = "read_ms" if args.metric == "read" else "read_ms+prefetch_wait_ms"
    print(f"Layers ranked by avg {label} (highest first):")
    for lid, avg in ranked[:8]:
        mark = " <-- pin" if lid in top_ids else ""
        print(f"  layer {lid}: {avg:.2f} ms{mark}")

    if args.json_out:
        import json

        profile = {
            "resident_layer_ids": top_ids,
            "always_resident_tensors": ["embed", "head", "ln_out"],
        }
        args.json_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
