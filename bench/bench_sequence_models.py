#!/usr/bin/env python3
"""Small fixed-token benchmark matrix for packed Mamba/Transformer paths.

This intentionally avoids tokenizer dependencies.  It compares resident,
partial, and streaming execution on the same pack and prints metrics that are
meaningful for sequence models (state/KV bytes, layer loads, and tok/s).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--backend", required=True, choices=["mamba2", "mamba", "transformer", "llama"])
    parser.add_argument("--modes", default="resident,partial,streaming")
    parser.add_argument("--prompt-ids", default="1,7,3,11,4")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mamba weight-stationary batch size; Transformer batching is unsupported",
    )
    parser.add_argument(
        "--provider-cache",
        action="store_true",
        help="retain streamed dense layers in the bounded provider LRU",
    )
    parser.add_argument(
        "--provider-cache-layers",
        type=int,
        default=0,
        help="provider-cache layer cap (0 selects the normal throughput default)",
    )
    parser.add_argument(
        "--sdpa",
        choices=["auto", "on", "off"],
        default="auto",
        help="Transformer scaled-dot-product attention selection",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.batch_size > 1 and args.backend not in {"mamba", "mamba2"}:
        parser.error("--batch-size > 1 is currently supported only by Mamba2")

    if args.sdpa != "auto":
        os.environ["RWKV_SEQUENCE_SDPA"] = "1" if args.sdpa == "on" else "0"

    token_ids = [int(value.strip()) for value in args.prompt_ids.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    for mode in [value.strip() for value in args.modes.split(",") if value.strip()]:
        engine = InferenceEngine(
            EngineConfig(
                pack_dir=args.pack,
                backend=args.backend,
                mode=mode,
                device="cpu",
                max_tokens=args.max_tokens,
                verify_hash=False,
                stream_layer_cache=args.provider_cache,
                max_provider_cache_layers=args.provider_cache_layers,
            )
        )
        engine.load()
        try:
            provider = engine._get_or_create_pack_provider()
            started = time.perf_counter()
            output: list[int] = []
            output_tokens = 0
            for _ in range(max(1, args.repeat)):
                if args.batch_size == 1:
                    output = engine.backend.generate_greedy_ids(
                        token_ids, provider, args.max_tokens, engine.metrics
                    )
                    output_tokens = len(output)
                else:
                    output_batches = engine.backend.generate_greedy_ids_batch(
                        [token_ids] * args.batch_size,
                        provider,
                        args.max_tokens,
                        engine.metrics,
                    )
                    output = output_batches[0] if output_batches else []
                    output_tokens = sum(len(row) for row in output_batches)
            elapsed = time.perf_counter() - started
            record = engine.metrics.to_dict()
            provider_stats = provider.cache_stats()
            layer_rows = record.get("layers", [])
            layer_cache_hits = sum(
                int(row.get("layer_cache_hits", 0))
                for row in layer_rows
                if isinstance(row, dict)
            )
            read_ms = sum(
                float(row.get("read_ms", 0.0))
                for row in layer_rows
                if isinstance(row, dict)
            )
            record.update(
                {
                    "backend": args.backend,
                    "mode": mode,
                    "prompt_tokens": len(token_ids),
                    "output_tokens": int(output_tokens),
                    "batch_size_measured": int(args.batch_size),
                    "batch_output_tokens": int(output_tokens),
                    "provider_cache_enabled": bool(args.provider_cache),
                    "provider_cache_layers": int(args.provider_cache_layers),
                    "provider_cache_bytes_measured": int(
                        provider_stats.get("provider_cache_bytes", 0)
                    ),
                    "provider_cache_evictions_measured": int(
                        provider_stats.get("provider_cache_evictions", 0)
                    ),
                    "layer_cache_hits": layer_cache_hits,
                    "read_ms_sum": read_ms,
                    "sdpa": args.sdpa,
                    "wall_s_measured": elapsed,
                    "tok_s_measured": output_tokens * max(1, args.repeat) / elapsed
                    if elapsed > 0
                    else 0.0,
                }
            )
            rows.append(record)
        finally:
            engine.close()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
