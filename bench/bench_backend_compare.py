#!/usr/bin/env python3
"""Compare the selected F-tier backend with the native rwkv.cpp resident path."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.bench_f1_f3 import CLEAR_KEYS, scenario_env, run_once  # noqa: E402
from rwkv_ssd.runtime.config import EngineConfig  # noqa: E402
from rwkv_ssd.runtime.engine import InferenceEngine  # noqa: E402

PACK_0_1B = Path("test_model/runtime_pack")
CKPT_0_1B = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
GGML_0_1B = Path("test_model/rwkv7-g1d-0.1b-FP16.bin")


def run_rwkvcpp_once(
    pack: Path,
    ggml: Path,
    *,
    label: str,
    max_tokens: int,
    prompt: str,
    samples: int = 1,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _ in range(samples):
        cfg = EngineConfig(
            pack_dir=pack,
            checkpoint_path=str(ggml),
            backend="rwkvcpp",
            mode="resident",
            device="cpu",
            max_tokens=max_tokens,
            greedy=True,
            skeleton_load=False,
        )
        engine = InferenceEngine(cfg)
        t_load0 = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - t_load0
        try:
            engine.generate(prompt)
            engine.metrics.layers.clear()
            engine.metrics.prefill_wall_s = 0.0
            t0 = time.perf_counter()
            engine.generate(prompt)
            wall_s = time.perf_counter() - t0
            m = engine.metrics
            prefill_s = float(getattr(m, "prefill_wall_s", 0.0) or 0.0)
            decode_s = float(getattr(m, "decode_wall_s", 0.0) or 0.0)
            if decode_s <= 0:
                decode_s = max(1e-9, wall_s - prefill_s) if prefill_s > 0 else wall_s
            rows.append(
                {
                    "label": label,
                    "backend": "rwkvcpp",
                    "mode": "resident",
                    "tok_s": max_tokens / decode_s if decode_s > 0 else 0.0,
                    "tok_s_wall": max_tokens / wall_s if wall_s > 0 else 0.0,
                    "wall_s": wall_s,
                    "decode_s": decode_s,
                    "prefill_s": prefill_s,
                    "load_s": load_s,
                    "cpu_threads": torch.get_num_threads(),
                }
            )
        finally:
            engine.close()
    out = dict(rows[-1])
    rates = [float(r["tok_s"]) for r in rows]
    out["tok_s"] = statistics.median(rates)
    out["tok_s_min"] = min(rates)
    out["tok_s_max"] = max(rates)
    out["samples"] = samples
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK_0_1B)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_0_1B)
    parser.add_argument("--ggml", type=Path, default=GGML_0_1B)
    parser.add_argument(
        "--f-tier-backend",
        choices=["rwkvcpp", "chatrwkv"],
        default="rwkvcpp",
        help="backend used for F6/F1/F3 rows (default: rwkvcpp)",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prompt", default="Throughput bench prompt")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads for both backends (1 is fastest for the 0.1B CPU model)",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.pack.is_dir():
        raise SystemExit(f"pack not found: {args.pack}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.ggml.is_file():
        raise SystemExit(f"ggml bin not found: {args.ggml} (convert with convert_pytorch_to_ggml.py)")

    if args.f_tier_backend == "rwkvcpp":
        os.environ["RWKVCPP_GGML_PATH"] = str(args.ggml)

    os.environ["RWKV_CPU_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    tier_backend = args.f_tier_backend
    tier_label = "rwkvcpp" if tier_backend == "rwkvcpp" else "ChatRWKV"
    scenarios: list[dict[str, Any]] = [
        {
            "label": f"F6 {tier_label} resident",
            "backend": tier_backend,
            "mode": "resident",
            "overrides": {},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        {
            "label": f"F1 {tier_label} streaming",
            "backend": tier_backend,
            "mode": "streaming",
            "overrides": {"RWKV_SSD_TIER": "1"},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        {
            "label": f"F3 {tier_label} partial-hot3",
            "backend": tier_backend,
            "mode": "partial",
            "overrides": {"RWKV_PARTIAL_SSD_TIER": "1"},
            "warm_z": False,
            "stream_layer_cache": False,
        },
    ]
    if tier_backend != "rwkvcpp":
        scenarios.append(
            {
                "label": "rwkvcpp resident (ggml FP16)",
                "backend": "rwkvcpp",
                "mode": "resident",
                "native_resident": True,
                "overrides": {},
                "warm_z": False,
                "stream_layer_cache": False,
            }
        )

    results: list[dict[str, Any]] = []
    for spec in scenarios:
        required = {"label", "backend", "mode"}
        missing = sorted(required.difference(spec))
        if missing:
            raise ValueError(
                f"invalid benchmark scenario {spec.get('label', '<unnamed>')!r}; "
                f"missing fields: {', '.join(missing)}"
            )
        if spec.get("native_resident"):
            row = run_rwkvcpp_once(
                args.pack,
                args.ggml,
                label=spec["label"],
                max_tokens=args.max_tokens,
                prompt=args.prompt,
                samples=args.samples,
            )
            results.append(row)
            print(
                f"{row['label']:32s}  {row['tok_s']:.2f} tok/s  "
                f"(decode {row.get('decode_s', 0):.2f}s, load {row.get('load_s', 0):.1f}s)"
            )
            continue
        row = run_once(
            args.pack,
            args.checkpoint,
            backend=spec["backend"],
            label=spec["label"],
            mode=spec["mode"],
            overrides=spec.get("overrides", {}),
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            strategy="cpu bf16",
            io_backend="mmap",
            decode_disk_cache="0",
            warm_z=spec.get("warm_z", False),
            stream_layer_cache=spec.get("stream_layer_cache"),
        )
        if args.samples > 1:
            # run_once is single-sample; repeat for median
            rows = [
                run_once(
                    args.pack,
                    args.checkpoint,
                    backend=spec["backend"],
                    label=spec["label"],
                    mode=spec["mode"],
                    overrides=spec.get("overrides", {}),
                    max_tokens=args.max_tokens,
                    prompt=args.prompt,
                    strategy="cpu bf16",
                    io_backend="mmap",
                    decode_disk_cache="0",
                    warm_z=spec.get("warm_z", False),
                    stream_layer_cache=spec.get("stream_layer_cache"),
                )
                for _ in range(args.samples)
            ]
            rates = [float(r["tok_s"]) for r in rows]
            row = dict(rows[-1])
            row["tok_s"] = statistics.median(rates)
            row["tok_s_min"] = min(rates)
            row["tok_s_max"] = max(rates)
            row["samples"] = args.samples
        results.append(row)
        print(
            f"{row['label']:32s}  {row['tok_s']:.2f} tok/s  "
            f"(decode {row.get('decode_s', 0):.2f}s, load {row.get('load_s', 0):.1f}s)"
        )

    payload = {
        "pack": str(args.pack),
        "checkpoint": str(args.checkpoint),
        "ggml": str(args.ggml),
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "threads": args.threads,
        "f_tier_backend": tier_backend,
        "note": "rwkvcpp is the default F-tier backend; use --f-tier-backend chatrwkv for the PyTorch compatibility comparison",
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
