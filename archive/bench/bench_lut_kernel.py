#!/usr/bin/env python3
"""Compare LUT gather kernels + CPU vs XPU on synthetic and real packs."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _bench_gather(n: int, kernel: str, iters: int) -> float:
    os.environ["RWKV_LUT_KERNEL"] = kernel
    from rwkv_ssd.runtime.lut_gather_kernel import gather_lut2_packed

    codebook = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    packed = np.random.randint(0, 256, size=(n + 3) // 4, dtype=np.uint8)
    flat = np.empty(n, dtype=np.float32)
    for _ in range(3):
        gather_lut2_packed(flat, 0, codebook, packed, n)
    t0 = time.perf_counter()
    for _ in range(iters):
        gather_lut2_packed(flat, 0, codebook, packed, n)
    return (time.perf_counter() - t0) / iters


def _bench_xpu_vs_cpu(n: int, iters: int) -> None:
    import torch

    if not torch.xpu.is_available():
        print("XPU not available — skip GPU comparison")
        return
    os.environ["RWKV_LUT_KERNEL"] = "numba"
    from rwkv_ssd.runtime.trinity_accel import decode_lut2_layer_on_accel
    from rwkv_ssd.runtime.trinity_codec import encode_trinity_lut2
    from rwkv_ssd.runtime.manifest import TensorEntry

    import torch as th

    side = int(n**0.5)
    t = th.randn(side, side, dtype=th.bfloat16)
    raw = encode_trinity_lut2(t)
    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=list(t.shape),
        offset=0,
        length=len(raw),
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    blobs = [(raw, entry)]

    for _ in range(2):
        decode_lut2_layer_on_accel(
            blobs=blobs, decode_device=torch.device("cpu"), output_device=torch.device("cpu")
        )
    t0 = time.perf_counter()
    for _ in range(iters):
        decode_lut2_layer_on_accel(
            blobs=blobs, decode_device=torch.device("cpu"), output_device=torch.device("cpu")
        )
    cpu_s = (time.perf_counter() - t0) / iters

    for _ in range(2):
        decode_lut2_layer_on_accel(
            blobs=blobs, decode_device=torch.device("xpu"), output_device=torch.device("cpu")
        )
    t0 = time.perf_counter()
    for _ in range(iters):
        decode_lut2_layer_on_accel(
            blobs=blobs, decode_device=torch.device("xpu"), output_device=torch.device("cpu")
        )
    xpu_s = (time.perf_counter() - t0) / iters

    print(
        f"  layer n={n:,}  cpu={cpu_s*1000:.2f} ms  xpu={xpu_s*1000:.2f} ms  "
        f"ratio={cpu_s/max(xpu_s,1e-9):.2f}x ({'XPU' if xpu_s<cpu_s else 'CPU'} wins)"
    )


def _bench_pack_layer(pack: Path, iters: int) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "bench"))
    from bench_trinity_per_token import _one_layer_ms

    for lid in (0, 6):
        try:
            r = _one_layer_ms(pack, lid, iters=iters)
            print(
                f"  pack={pack.name} layer={r['layer_id']} span={r['span_kb']}KB "
                f"read={r['read_ms']:.2f}ms decode={r['decode_ms']:.2f}ms "
                f"total={r['total_ms']:.2f}ms (~{r['tok_s_ceiling']:.0f} tok/s cap)"
            )
        except SystemExit:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--numel", type=int, default=4_194_304, help="synthetic gather size")
    p.add_argument("--pack", type=Path, action="append")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--skip-xpu", action="store_true")
    args = p.parse_args()

    print(f"LUT kernel (numel={args.numel:,}, iters={args.iters})")
    for k in ("numpy", "numba", "native", "torch"):
        try:
            t = _bench_gather(args.numel, k, args.iters)
            print(f"  {k:6} {t*1000:.2f} ms")
        except Exception as exc:
            print(f"  {k:6} failed: {exc}")

    if not args.skip_xpu:
        print("\nCPU vs XPU full layer decode (synthetic one tensor):")
        for n in (262_144, 1_048_576, 4_194_304):
            if n <= args.numel or n == 262_144:
                side = int(n**0.5)
                _bench_xpu_vs_cpu(side * side, max(3, args.iters // 3))

    if args.pack:
        print("\nReal pack per-token layer (engine path):")
        for pack in args.pack:
            _bench_pack_layer(pack, args.iters)


if __name__ == "__main__":
    main()
