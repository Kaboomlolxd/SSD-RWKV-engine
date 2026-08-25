#!/usr/bin/env python3
"""Compare raw weights.bin read throughput across I/O backends."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store


def main() -> None:
    p = argparse.ArgumentParser(description="Raw pack I/O backend comparison")
    p.add_argument("--model", default="test_model/runtime_pack")
    p.add_argument(
        "--backends",
        default="mmap,pread,threaded",
        help="Comma-separated backends",
    )
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max streamed tensors to read (0 = all)",
    )
    args = p.parse_args()

    manifest = Manifest.load(args.model)
    streamed = manifest.streamed_tensors() or manifest.tensors
    if args.limit > 0:
        streamed = streamed[: args.limit]
    total_bytes = sum(t.length for t in streamed)

    print(f"pack={args.model} tensors={len(streamed)} bytes={total_bytes}")
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        times: list[float] = []
        for trial in range(1, args.trials + 1):
            with open_weight_store(manifest.weights_path, backend=backend) as store:
                t0 = time.perf_counter()
                for entry in streamed:
                    store.read_bytes(entry)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                gbs = (total_bytes / (1024**3)) / elapsed if elapsed > 0 else 0.0
                print(f"  {backend:10} trial={trial} time_s={elapsed:.3f} bw_GB_s={gbs:.2f}")
        avg = sum(times) / len(times)
        gbs = (total_bytes / (1024**3)) / avg if avg > 0 else 0.0
        print(f"  {backend:10} avg_GB_s={gbs:.2f}")


if __name__ == "__main__":
    main()
