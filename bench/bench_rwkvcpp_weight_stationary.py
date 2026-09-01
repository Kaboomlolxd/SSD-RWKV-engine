#!/usr/bin/env python3
"""Compare native rwkv.cpp independent generation with shared CPU sweeps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def _config(pack: Path, checkpoint: Path, max_tokens: int) -> EngineConfig:
    return EngineConfig(
        pack_dir=Path(pack),
        checkpoint_path=str(checkpoint),
        backend="rwkvcpp",
        mode="streaming",
        device="cpu",
        max_tokens=int(max_tokens),
        greedy=True,
        verify_hash=False,
        prefetch_enabled=False,
        cache_format="none",
    )


def run(
    pack: Path,
    checkpoint: Path,
    prompts: list[str],
    *,
    max_tokens: int = 8,
    samples: int = 3,
) -> dict[str, object]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if max_tokens < 0 or samples <= 0:
        raise ValueError("max_tokens must be non-negative and samples positive")

    independent_times: list[float] = []
    batch_times: list[float] = []
    independent_outputs: list[str] = []
    batch_outputs: list[str] = []
    batch_metrics: dict[str, object] = {}

    for _ in range(samples):
        with InferenceEngine(_config(pack, checkpoint, max_tokens)) as independent:
            started = time.perf_counter()
            independent_outputs = [
                independent.generate(prompt) for prompt in prompts
            ]
            independent_times.append(time.perf_counter() - started)

        with InferenceEngine(_config(pack, checkpoint, max_tokens)) as batched:
            started = time.perf_counter()
            batch_outputs = batched.generate_batch(
                prompts, max_tokens=max_tokens
            )
            batch_times.append(time.perf_counter() - started)
            batch_metrics = batched.metrics.to_dict()

    if independent_outputs != batch_outputs:
        raise AssertionError("rwkv.cpp batch output differs from independent output")

    total_tokens = len(prompts) * int(max_tokens)
    independent_wall = statistics.median(independent_times)
    batch_wall = statistics.median(batch_times)
    return {
        "schema_version": 1,
        "backend": "rwkvcpp",
        "cpu_only": True,
        "batch_size": len(prompts),
        "max_tokens": int(max_tokens),
        "samples": int(samples),
        "outputs_match": True,
        "tokens_generated": total_tokens,
        "independent_wall_s_median": independent_wall,
        "batch_wall_s_median": batch_wall,
        "independent_tok_s": total_tokens / independent_wall
        if independent_wall
        else 0.0,
        "batch_tok_s": total_tokens / batch_wall if batch_wall else 0.0,
        "wall_speedup": independent_wall / batch_wall if batch_wall else 0.0,
        "batch_weight_sweeps": batch_metrics.get("weight_sweeps", 0),
        "batch_weight_layer_loads": batch_metrics.get("weight_layer_loads", 0),
        "batch_prefill_s": batch_metrics.get("batch_prefill_wall_s", 0.0),
        "batch_decode_s": batch_metrics.get("batch_decode_wall_s", 0.0),
        "scope": (
            "native rwkv.cpp CPU layer-outer/session-inner prefill and decode; "
            "normal-file/page-cache evidence, not physical SSD scaling"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", default="Hello,Hi")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    prompts = [value for value in args.prompts.split(",") if value]
    result = run(
        args.pack,
        args.checkpoint,
        prompts,
        max_tokens=args.max_tokens,
        samples=args.samples,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
