#!/usr/bin/env python3
"""
Build throughput-optimized runtime packs from a checkpoint.

Creates up to three variants under --output:
  fp16/           default layout, no codec
  grouped/        layer_grouped + sector padding (P2.b)
  scale_u8/       UINT8 quant (M5)
  scale_u4/       4-bit quant (M5, smallest)
  trinity_fast/   trinity layer_grouped, TCL\\x02 (no zlib — fastest Trinity decode)

Usage:
  python -m rwkv_ssd.tools.build_optimized_pack \\
    --input test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth \\
    --output test_model/packs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rwkv_ssd.runtime.pack_layout import DEFAULT_SECTOR_BYTES
from rwkv_ssd.tools.pack_runtime import pack


def main() -> None:
    p = argparse.ArgumentParser(description="Build M5/P2.b pack variants")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-family", default="rwkv7")
    p.add_argument("--sector-bytes", type=int, default=DEFAULT_SECTOR_BYTES)
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-grouped", action="store_true")
    p.add_argument("--skip-quant", action="store_true")
    p.add_argument(
        "--shadow-min-numel",
        type=int,
        default=4096,
        help="for shadow variants: only shadow tensors with numel >= N (0=all)",
    )
    p.add_argument("--skip-shadow", action="store_true")
    args = p.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_fp16:
        dest = out / "fp16"
        print(f"pack fp16 -> {dest}")
        pack(args.input, dest, model_family=args.model_family, pack_codec="none")

    if not args.skip_grouped:
        dest = out / "grouped"
        print(f"pack layer_grouped -> {dest}")
        pack(
            args.input,
            dest,
            model_family=args.model_family,
            pack_codec="none",
            pack_layout="layer_grouped",
            sector_bytes=args.sector_bytes,
        )

    if not args.skip_quant:
        for codec in (
            "scale_u8",
            "scale_u4",
            "trinity_lut2",
            "trinity",
        ):
            dest = out / codec
            print(f"pack {codec} -> {dest}")
            pack(
                args.input,
                dest,
                model_family=args.model_family,
                pack_codec=codec,
                pack_layout="layer_grouped",
                sector_bytes=args.sector_bytes,
            )
        dest = out / "trinity_fast"
        print(f"pack trinity (no layer zlib) -> {dest}")
        pack(
            args.input,
            dest,
            model_family=args.model_family,
            pack_codec="trinity",
            pack_layout="layer_grouped",
            sector_bytes=args.sector_bytes,
            trinity_layer_compress=False,
        )

    if not args.skip_shadow and not args.skip_quant:
        dest = out / "trinity_lut2_shadow"
        print(f"pack trinity_lut2 + full shadow -> {dest}")
        pack(
            args.input,
            dest,
            model_family=args.model_family,
            pack_codec="trinity_lut2",
            pack_layout="layer_grouped",
            sector_bytes=args.sector_bytes,
            bf16_shadow=True,
            shadow_min_numel=0,
        )
        if args.shadow_min_numel > 0:
            dest = out / "trinity_lut2_shadow_sel"
            print(
                f"pack trinity_lut2 + selective shadow (min_numel={args.shadow_min_numel}) -> {dest}"
            )
            pack(
                args.input,
                dest,
                model_family=args.model_family,
                pack_codec="trinity_lut2",
                pack_layout="layer_grouped",
                sector_bytes=args.sector_bytes,
                bf16_shadow=True,
                shadow_min_numel=args.shadow_min_numel,
            )

    print("done. Compare packs with bench/bench_io_ceiling.py --heavy")


if __name__ == "__main__":
    main()
