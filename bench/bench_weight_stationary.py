#!/usr/bin/env python3
"""Measure synthetic weight-stationary batching against independent sweeps.

This is a CPU/Windows research benchmark.  It proves scheduler parity and
weight-load amortization; it does not claim that ChatRWKV or rwkv.cpp already
support batched recurrent-state kernels.
"""

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


def _prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt.encode("utf-8")))


def run_weight_stationary_benchmark(
    pack: Path,
    prompts: list[str],
    *,
    max_tokens: int = 8,
    samples: int = 3,
) -> dict[str, object]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if max_tokens < 0 or samples <= 0:
        raise ValueError("max_tokens must be non-negative and samples positive")
    cfg = EngineConfig(
        pack_dir=Path(pack),
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=int(max_tokens),
        cache_format="none",
        prefetch_enabled=False,
    )
    independent_times: list[float] = []
    batch_times: list[float] = []
    independent_outputs: list[str] = []
    batch_outputs: list[str] = []
    batch_metrics: dict[str, object] = {}
    with InferenceEngine(cfg) as independent:
        for _ in range(samples):
            started = time.perf_counter()
            independent_outputs = [independent.generate(prompt) for prompt in prompts]
            independent_times.append(time.perf_counter() - started)
    with InferenceEngine(cfg) as batched:
        for _ in range(samples):
            started = time.perf_counter()
            batch_outputs = batched.generate_batch(prompts, max_tokens=max_tokens)
            batch_times.append(time.perf_counter() - started)
            batch_metrics = batched.metrics.to_dict()
        n_layer = batched.backend.num_layers

    if independent_outputs != batch_outputs:
        raise AssertionError("weight-stationary output differs from independent generation")
    independent_sweeps = sum(_prompt_tokens(prompt) + max_tokens for prompt in prompts)
    batch_sweeps = int(batch_metrics.get("weight_sweeps", 0))
    independent_layer_loads = independent_sweeps * n_layer
    batch_layer_loads = int(batch_metrics.get("weight_layer_loads", 0))
    total_tokens = len(prompts) * max_tokens
    independent_wall = statistics.median(independent_times)
    batch_wall = statistics.median(batch_times)
    return {
        "schema_version": 1,
        "backend": "synthetic",
        "cpu_only": True,
        "batch_size": len(prompts),
        "max_tokens": max_tokens,
        "samples": samples,
        "outputs_match": True,
        "tokens_generated": total_tokens,
        "independent_wall_s_median": independent_wall,
        "batch_wall_s_median": batch_wall,
        "independent_tok_s": total_tokens / independent_wall if independent_wall else 0.0,
        "batch_tok_s": total_tokens / batch_wall if batch_wall else 0.0,
        "independent_weight_sweeps": independent_sweeps,
        "batch_weight_sweeps": batch_sweeps,
        "independent_weight_layer_loads": independent_layer_loads,
        "batch_weight_layer_loads": batch_layer_loads,
        "layer_load_amortization": (
            independent_layer_loads / batch_layer_loads if batch_layer_loads else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", default="a,longer prompt,third")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    prompts = [value for value in args.prompts.split(",") if value]
    result = run_weight_stationary_benchmark(
        args.model,
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
