#!/usr/bin/env python3
"""Compare ChatRWKV tok/s across FP16 vs Trinity packs (what matters for inference)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "bench_throughput.py"
CKPT = ROOT / "test_model" / "rwkv7-g1d-0.01b-bench.pth"
PACKS = [
    ("fp16_default", ROOT / "test_model" / "runtime_pack_0.01b"),
    ("fp16_grouped", ROOT / "test_model" / "trinity_eval" / "fp16_grouped"),
    ("trinity_lut2", ROOT / "test_model" / "trinity_eval" / "trinity_lut2"),
    ("trinity_layer", ROOT / "test_model" / "trinity_eval" / "trinity_layer"),
]
MODES_OF_INTEREST = ("resident", "streaming", "streaming+cache")


def run_pack(
    label: str,
    pack: Path,
    *,
    samples: int,
    max_tokens: int,
    heavy: bool,
) -> list[dict]:
    if not pack.is_dir():
        raise FileNotFoundError(pack)
    cmd = [
        sys.executable,
        str(BENCH),
        "--backend",
        "chatrwkv",
        "--model",
        str(pack),
        "--max-tokens",
        str(max_tokens),
        "--samples",
        str(samples),
        "--prefetch-policy",
        "layer_aware",
        "--strategy",
        "cpu bf16",
    ]
    if heavy:
        cmd.append("--heavy")
    else:
        cmd.extend(["--checkpoint", str(CKPT)])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "RWKV_JIT_ON": "0"},
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"bench failed for {label}: {proc.returncode}")
    rows = json.loads(proc.stdout) if proc.stdout.strip().startswith("[") else []
    if not rows:
        # bench prints table not json — parse from stderr/stdout fallback
        raise RuntimeError(f"re-run with --json support needed for {label}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    all_rows: list[dict] = []
    for label, pack in PACKS:
        if not pack.exists():
            print(f"skip missing pack {label}: {pack}")
            continue
        cmd = [
            sys.executable,
            str(BENCH),
            "--backend",
            "chatrwkv",
            "--model",
            str(pack),
            "--max-tokens",
            str(args.max_tokens),
            "--samples",
            str(args.samples),
            "--json",
        ]
        if args.heavy:
            cmd.append("--heavy")
        else:
            cmd.extend(["--checkpoint", str(CKPT)])
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**dict(__import__("os").environ), "RWKV_JIT_ON": "0"},
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            sys.exit(proc.returncode)
        rows = json.loads(proc.stdout)
        for r in rows:
            r["pack_label"] = label
            r["weights_mb"] = round(
                (pack / "weights.bin").stat().st_size / 1e6, 2
            ) if (pack / "weights.bin").is_file() else 0
        all_rows.extend(rows)

    baseline = next(
        (r for r in all_rows if r["pack_label"] == "fp16_grouped" and r["mode"] == "resident"),
        next((r for r in all_rows if r["mode"] == "resident"), None),
    )
    base_tok = float(baseline["tok_s"]) if baseline else 1.0

    print(
        f"tok/s comparison (max_tokens={args.max_tokens}, samples={args.samples})\n"
        f"{'pack':14} {'mode':28} {'tok/s':>8} {'vs_fp16':>8} {'z_mb':>7} {'prov_mb':>7} disk_mb"
    )
    print("-" * 90)
    for label, _ in PACKS:
        pack_rows = [r for r in all_rows if r["pack_label"] == label]
        if not pack_rows:
            continue
        disk = pack_rows[0].get("weights_mb", 0)
        for mode in MODES_OF_INTEREST:
            r = next((x for x in pack_rows if mode in x.get("mode", "")), None)
            if not r:
                continue
            ratio = float(r["tok_s"]) / base_tok
            print(
                f"{label:14} {r['mode']:28} {r['tok_s']:8.2f} {ratio:7.0%} "
                f"{r.get('z_mb', 0):7.2f} {r.get('provider_cache_mb', 0):7.2f} {disk:7.2f}"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print()
    print("VRAM note: z_mb/prov_mb are CPU RAM for ChatRWKV model.z + provider cache.")
    print("Decoded layer weights are full bf16 size once in z — Trinity shrinks SSD, not tensor width.")


if __name__ == "__main__":
    main()
