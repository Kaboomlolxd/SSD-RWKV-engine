#!/usr/bin/env python3
"""Controlled cache-format and storage-throttle benchmark matrix.

The benchmark is intentionally CPU/Windows friendly.  Artificial I/O caps
are recorded as experimental ceilings; they are not interpreted as physical
SSD scaling.  Use a 0.1B pack for correctness, warm-cache, and compute-bound
smoke runs, a 2.9B pack for RAM/cache accounting, and Linux/NVIDIA hardware
for any real 7B+ storage-scaling claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def _percentile(values: Iterable[float], percentile: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sample_row(metrics: dict[str, Any], wall_s: float) -> dict[str, Any]:
    tokens = int(metrics.get("tokens_generated", 0))
    divisor = float(tokens) if tokens > 0 else 1.0
    layers = metrics.get("layers") or []
    read_ms = sum(float(layer.get("read_ms", 0.0)) for layer in layers)
    staging_ms = sum(float(layer.get("staging_ms", 0.0)) for layer in layers)
    compute_ms = sum(float(layer.get("compute_ms", 0.0)) for layer in layers)
    layer_hits = sum(int(layer.get("layer_cache_hits", 0)) for layer in layers)
    prefetch_hits = sum(int(layer.get("prefetch_hits", 0)) for layer in layers)
    disk_hits = sum(int(layer.get("disk_cache_hits", 0)) for layer in layers)
    tok_s = tokens / wall_s if wall_s > 0 and tokens > 0 else 0.0
    packed_evictions = int(metrics.get("packed_cache_evictions", 0))
    provider_evictions = int(metrics.get("provider_cache_evictions", 0))
    return {
        "wall_s": round(wall_s, 6),
        "tokens": tokens,
        "tok_s": round(tok_s, 6),
        "read_ms_per_token": round(read_ms / divisor, 6),
        "staging_ms_per_token": round(staging_ms / divisor, 6),
        "compute_ms_per_token": round(compute_ms / divisor, 6),
        "layer_cache_hits": layer_hits,
        "prefetch_hits": prefetch_hits,
        "disk_cache_hits": disk_hits,
        "cache_hits": layer_hits + prefetch_hits + disk_hits,
        "packed_cache_evictions": packed_evictions,
        "provider_cache_evictions": provider_evictions,
        "cache_evictions": packed_evictions + provider_evictions,
        "cache_format": str(metrics.get("cache_format", "auto")),
        "provider_cache_bytes": int(metrics.get("provider_cache_bytes", 0)),
        "packed_cache_bytes": int(metrics.get("packed_cache_bytes", 0)),
        "prepared_cache_bytes": int(metrics.get("prepared_cache_bytes", 0)),
        "lut2_index_cache_bytes": int(metrics.get("lut2_index_cache_bytes", 0)),
        "z_bytes": int(metrics.get("z_bytes", 0)),
        "cmix_zero_fraction": float(metrics.get("cmix_zero_fraction", 0.0)),
        "cmix_active_fraction": float(metrics.get("cmix_active_fraction", 0.0)),
    }


def run_scenario(
    pack: Path,
    *,
    backend: str,
    checkpoint: Path | None,
    cache_format: str,
    io_cap_mbps: float,
    max_tokens: int,
    prompt: str,
    device: str,
    strategy: str,
    packed_cache_bytes: int,
    prepared_cache_bytes: int,
    warmup_tokens: int = 0,
    samples: int = 1,
    profile: str = "default",
) -> dict[str, Any]:
    """Run one matrix cell and return a stable schema-v2 result row."""
    old_cap = os.environ.get("RWKV_SSD_IO_CAP_MBPS")
    try:
        if io_cap_mbps > 0:
            os.environ["RWKV_SSD_IO_CAP_MBPS"] = str(io_cap_mbps)
        else:
            os.environ.pop("RWKV_SSD_IO_CAP_MBPS", None)
        cfg = EngineConfig(
            pack_dir=pack,
            backend=backend,
            checkpoint_path=str(checkpoint) if checkpoint else None,
            mode="streaming",
            device=device,
            strategy=strategy,
            max_tokens=max_tokens,
            cache_format=cache_format,
            packed_cache_bytes=packed_cache_bytes,
            prepared_cache_bytes=prepared_cache_bytes,
            stream_layer_cache=cache_format in {"prepared", "dense"},
            warm_z=cache_format == "dense",
        )
        with InferenceEngine(cfg) as engine:
            if warmup_tokens > 0:
                original_max_tokens = engine.config.max_tokens
                engine.config.max_tokens = int(warmup_tokens)
                engine.generate(prompt)
                engine.config.max_tokens = original_max_tokens

            sample_rows: list[dict[str, Any]] = []
            for _ in range(max(1, int(samples))):
                started = time.perf_counter()
                engine.generate(prompt)
                wall = time.perf_counter() - started
                sample_rows.append(_sample_row(engine.metrics.to_dict(), wall))

        tok_rates = [float(row["tok_s"]) for row in sample_rows]
        wall_values = [float(row["wall_s"]) for row in sample_rows]
        last = sample_rows[-1]
        return {
            "schema_version": 2,
            "profile": profile,
            "cache_format": cache_format,
            "selected_cache_format": last["cache_format"],
            "io_cap_mbps": io_cap_mbps,
            "io_cap_kind": "artificial" if io_cap_mbps > 0 else "none",
            "physical_ssd_scaling_claim": False,
            "warmup_tokens": int(warmup_tokens),
            "sample_count": len(sample_rows),
            "wall_s": round(statistics.median(wall_values), 6),
            "tok_s": round(statistics.median(tok_rates), 6),
            "tok_s_median": round(statistics.median(tok_rates), 6),
            "tok_s_p95": round(_percentile(tok_rates, 95.0), 6),
            "read_ms_per_token": round(
                statistics.median(
                    float(row["read_ms_per_token"]) for row in sample_rows
                ),
                6,
            ),
            "staging_ms_per_token": round(
                statistics.median(
                    float(row["staging_ms_per_token"]) for row in sample_rows
                ),
                6,
            ),
            "compute_ms_per_token": round(
                statistics.median(
                    float(row["compute_ms_per_token"]) for row in sample_rows
                ),
                6,
            ),
            "cache_hits": sum(int(row["cache_hits"]) for row in sample_rows),
            "layer_cache_hits": sum(
                int(row["layer_cache_hits"]) for row in sample_rows
            ),
            "prefetch_hits": sum(int(row["prefetch_hits"]) for row in sample_rows),
            "disk_cache_hits": sum(int(row["disk_cache_hits"]) for row in sample_rows),
            "cache_evictions": int(last["cache_evictions"]),
            "packed_cache_evictions": int(last["packed_cache_evictions"]),
            "provider_cache_evictions": int(last["provider_cache_evictions"]),
            "provider_cache_bytes": int(last["provider_cache_bytes"]),
            "packed_cache_bytes": int(last["packed_cache_bytes"]),
            "prepared_cache_bytes": int(last["prepared_cache_bytes"]),
            "lut2_index_cache_bytes": int(last["lut2_index_cache_bytes"]),
            "z_bytes": int(last["z_bytes"]),
            "cmix_zero_fraction": float(last["cmix_zero_fraction"]),
            "cmix_active_fraction": float(last["cmix_active_fraction"]),
            "samples": sample_rows,
        }
    finally:
        if old_cap is None:
            os.environ.pop("RWKV_SSD_IO_CAP_MBPS", None)
        else:
            os.environ["RWKV_SSD_IO_CAP_MBPS"] = old_cap


def _csv_values(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def write_results_csv(output: dict[str, Any], path: Path) -> None:
    """Write scalar row fields to CSV; nested per-sample details stay in JSON."""
    rows = list(output.get("rows", []))
    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
    )
    fieldnames = ["model", "backend", "device", *scalar_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": output.get("model", ""),
                    "backend": output.get("backend", ""),
                    "device": output.get("device", ""),
                    **row,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="synthetic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strategy", default="cpu bf16")
    parser.add_argument("--profile", choices=("default", "correctness_only"), default="default")
    parser.add_argument("--cache-formats", default="none,packed,prepared,dense")
    parser.add_argument("--io-caps-mbps", default="0,250,500")
    parser.add_argument("--packed-cache-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--prepared-cache-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prompt", default="streaming benchmark")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    formats = _csv_values(args.cache_formats, str)
    caps = _csv_values(args.io_caps_mbps, float)
    max_tokens = max(1, int(args.max_tokens))
    warmup_tokens = max(0, int(args.warmup_tokens))
    samples = max(1, int(args.samples))
    if args.profile == "correctness_only":
        # This profile is a deterministic small-model smoke check.  It avoids
        # turning a correctness run into a cache-format performance claim.
        formats = ["none"]
        caps = [0.0]
        max_tokens = min(max_tokens, 2)
        warmup_tokens = 0
        samples = 1

    rows = [
        run_scenario(
            args.model,
            backend=args.backend,
            checkpoint=args.checkpoint,
            cache_format=fmt,
            io_cap_mbps=cap,
            max_tokens=max_tokens,
            prompt=args.prompt,
            device=args.device,
            strategy=args.strategy,
            packed_cache_bytes=args.packed_cache_bytes,
            prepared_cache_bytes=args.prepared_cache_bytes,
            warmup_tokens=warmup_tokens,
            samples=samples,
            profile=args.profile,
        )
        for cap in caps
        for fmt in formats
    ]
    output: dict[str, Any] = {
        "schema_version": 2,
        "profile": args.profile,
        "model": str(args.model),
        "backend": args.backend,
        "device": args.device,
        "warmup_tokens": warmup_tokens,
        "samples": samples,
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if args.csv_out:
        write_results_csv(output, args.csv_out)
    print(rendered)


if __name__ == "__main__":
    main()
