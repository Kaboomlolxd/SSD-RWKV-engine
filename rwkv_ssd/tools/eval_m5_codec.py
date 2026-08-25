#!/usr/bin/env python3
"""
M5 codec promotion gate (IDEAS / THROUGHPUT_PLAN P2.b).

Compares pack size and sequential read throughput. Promotion requires:
  - storage_ratio >= 1.3× vs FP16/none baseline (smaller weights.bin)
  - golden resident == streaming (run pytest test_pack_codec_golden)

Usage:
  python -m rwkv_ssd.tools.eval_m5_codec \\
    --pack test_model/runtime_pack_0.01b \\
    --pack path/to/scale_u8 --pack path/to/scale_u4

  python -m rwkv_ssd.tools.eval_m5_codec \\
    --build test_model/rwkv7-g1d-0.01b-bench.pth --output /tmp/m5_001b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rwkv_ssd.runtime.pack_bench import pack_read_stats
from rwkv_ssd.tools.pack_runtime import pack

STORAGE_GATE = 1.3


def _build_variants(checkpoint: Path, output: Path, model_family: str) -> list[tuple[str, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, Path]] = []
    for label, codec in (
        ("none", "none"),
        ("scale_u8", "scale_u8"),
        ("scale_u4", "scale_u4"),
    ):
        dest = output / label
        print(f"pack {codec} -> {dest}")
        pack(
            checkpoint,
            dest,
            model_family=model_family,
            pack_codec=codec,
            pack_layout="layer_grouped",
            quiet=True,
        )
        variants.append((label, dest))
    return variants


def eval_packs(
    packs: list[tuple[str, Path]],
    *,
    io_backend: str,
    storage_gate: float,
) -> list[dict]:
    if not packs:
        raise SystemExit("no packs to evaluate")
    baseline_mb = pack_read_stats(packs[0][1], io_backend)["weights_mb"]
    rows: list[dict] = []
    for label, path in packs:
        stats = pack_read_stats(path, io_backend)
        mb = float(stats["weights_mb"])
        ratio = (baseline_mb / mb) if mb > 0 else 0.0
        passes = ratio >= storage_gate if label != packs[0][0] else True
        row = {
            "label": label,
            "pack": str(path),
            "weights_mb": mb,
            "storage_ratio_vs_baseline": round(ratio, 2),
            "read_gbs": stats["read_gbs"],
            "dequant_codecs": stats["dequant_codecs"],
            "storage_gate": storage_gate,
            "passes_storage_gate": passes,
        }
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="M5 codec size/read gate (P2.b)")
    p.add_argument("--pack", action="append", type=Path, help="Pack dir (first = baseline)")
    p.add_argument("--label", action="append", help="Labels aligned with --pack")
    p.add_argument(
        "--build",
        type=Path,
        metavar="CHECKPOINT",
        help="Build none/u8/u4 layer_grouped variants under --output",
    )
    p.add_argument("--output", type=Path, help="Output dir for --build")
    p.add_argument("--model-family", default="rwkv7")
    p.add_argument("--io-backend", default="mmap")
    p.add_argument("--storage-gate", type=float, default=STORAGE_GATE)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.build:
        if not args.output:
            p.error("--output required with --build")
        if not args.build.is_file():
            raise SystemExit(f"checkpoint missing: {args.build}")
        packs = _build_variants(args.build, args.output, args.model_family)
    elif args.pack:
        packs = []
        for i, path in enumerate(args.pack):
            label = (
                args.label[i]
                if args.label and i < len(args.label)
                else path.name
            )
            packs.append((label, path))
    else:
        p.error("pass --pack and/or --build")

    rows = eval_packs(
        packs, io_backend=args.io_backend, storage_gate=args.storage_gate
    )
    baseline_label = rows[0]["label"]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(
            f"M5 eval (baseline={baseline_label}, gate={args.storage_gate}× smaller):"
        )
        for r in rows:
            gate = (
                "baseline"
                if r["label"] == baseline_label
                else ("PASS" if r["passes_storage_gate"] else "FAIL")
            )
            print(
                f"  {r['label']:12} {r['weights_mb']:8.2f} MB  "
                f"ratio={r['storage_ratio_vs_baseline']:.2f}×  "
                f"read={r['read_gbs']:.2f} GB/s  codecs={r['dequant_codecs']}  [{gate}]"
            )
        print(
            "\nCorrectness: pytest tests/test_pack_codec_golden.py "
            "(resident == streaming on quantized packs)."
        )

    failed = [
        r["label"]
        for r in rows[1:]
        if not r["passes_storage_gate"]
    ]
    if failed:
        print(f"storage gate failed: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
