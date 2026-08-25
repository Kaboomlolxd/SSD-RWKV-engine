#!/usr/bin/env python3
"""Build 0.1B Trinity LUT2 + selective bf16 shadow (hybrid pack)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rwkv_ssd.tools.pack_runtime import pack

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
DEFAULT_OUT = ROOT / "test_model/_rebuilt/trinity_lut2_shadow_sel_0.1b"


def main() -> None:
    p = argparse.ArgumentParser(description="0.1B Trinity LUT2 + selective shadow")
    p.add_argument("--input", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--shadow-min-numel",
        type=int,
        default=4096,
        help="Shadow only tensors with numel >= N (large weight mats)",
    )
    args = p.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"checkpoint missing: {args.input}")
    print(
        f"Packing trinity_lut2 + selective shadow (min_numel={args.shadow_min_numel}) "
        f"-> {args.output}"
    )
    pack(
        args.input,
        args.output,
        model_family="rwkv7",
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        bf16_shadow=True,
        shadow_min_numel=args.shadow_min_numel,
    )
    print("done")


if __name__ == "__main__":
    main()
