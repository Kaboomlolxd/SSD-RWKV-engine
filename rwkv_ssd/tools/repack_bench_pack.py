#!/usr/bin/env python3
"""Repack bench runtime packs with layer_grouped layout (thesis Ch.10)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rwkv_ssd.tools.pack_runtime import pack
from rwkv_ssd.tools.verify_pack import verify

REPO = Path(__file__).resolve().parents[2]
PRESETS = {
    "0.01b": (
        REPO / "test_model" / "rwkv7-g1d-0.01b-bench.pth",
        REPO / "test_model" / "runtime_pack_0.01b",
    ),
    "0.1b": (
        REPO / "test_model" / "rwkv7-g1d-0.1b-20260129-ctx8192.pth",
        REPO / "test_model" / "runtime_pack",
    ),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Repack bench models as layer_grouped")
    p.add_argument(
        "preset",
        nargs="?",
        choices=list(PRESETS),
        default="0.01b",
        help="which bench checkpoint/pack to rebuild",
    )
    p.add_argument("--input", type=Path, help="override checkpoint .pth")
    p.add_argument("--output", type=Path, help="override output pack directory (final path, not *_tmp)")
    p.add_argument(
        "--sector-bytes",
        type=int,
        default=256 * 1024,
        help="pad between layer groups (0=off)",
    )
    p.add_argument("--in-place", action="store_true", help="replace output via .tmp swap")
    args = p.parse_args()

    ckpt_default, out_default = PRESETS[args.preset]
    ckpt = args.input or ckpt_default
    out = args.output or out_default
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    tmp = out.parent / f"{out.name}_grouped_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"Packing {ckpt.name} -> {tmp} (layer_grouped, sector={args.sector_bytes})")
    pack(
        ckpt,
        tmp,
        model_family="rwkv7",
        pack_layout="layer_grouped",
        sector_bytes=args.sector_bytes,
        quiet=False,
    )
    if not verify(tmp):
        raise SystemExit("verify failed on grouped pack")

    if args.in_place:
        backup = out.parent / f"{out.name}_default_backup"
        if out.exists() and not backup.exists():
            shutil.copytree(out, backup)
            print(f"Backed up previous pack to {backup}")
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(tmp), str(out))
        print(f"Installed grouped pack at {out}")
    elif args.output and out != out_default:
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(tmp), str(out))
        print(f"Installed grouped pack at {out}")
    else:
        print(f"Grouped pack ready at {tmp}")
        print(f"Bench with: --model {tmp}")


if __name__ == "__main__":
    main()
