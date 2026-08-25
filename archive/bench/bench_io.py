#!/usr/bin/env python3
"""Measure raw mmap read throughput for a runtime pack (no model)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rwkv_ssd.runtime.io_mmap import MmapWeightStore
from rwkv_ssd.runtime.manifest import Manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Runtime pack directory")
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    m = Manifest.load(args.model)
    streamed = m.streamed_tensors() or m.tensors
    total_bytes = sum(t.length for t in streamed)

    with MmapWeightStore(m.weights_path) as store:
        buf = bytearray(max(t.length for t in streamed))
        for trial in range(1, args.trials + 1):
            t0 = time.perf_counter()
            for t in streamed:
                store.read_into(t, memoryview(buf)[: t.length])
            elapsed = time.perf_counter() - t0
            gbs = (total_bytes / (1024**3)) / elapsed
            print(f"trial={trial} bytes={total_bytes} time_s={elapsed:.3f} bw_GB_s={gbs:.2f}")


if __name__ == "__main__":
    main()
