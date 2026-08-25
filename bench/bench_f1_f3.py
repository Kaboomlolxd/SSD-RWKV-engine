#!/usr/bin/env python3
"""Native rwkv.cpp F1-F5 acceptance benchmark with a compatibility F6 row.

Compares pure ChatRWKV resident against the main SSD-tier presets:

  * F1: ``RWKV_SSD_TIER=1`` strict fused streaming (min RAM)
  * F3: ``RWKV_PARTIAL_SSD_TIER=1`` partial hot3
  * F5: ``RWKV_PROMOTE_FULL_Z=1`` + warm-z promote (full native graph)
  * F6: resident backend reference (legacy comparison row)

F1-F4 are required to compare against F5.  The row reports both cold and
warm request observations; ``F6`` is retained for historical reports but is
not the low-RAM acceptance baseline.
"""

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

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.memory_usage import process_rss_bytes

PACK_0_01B = Path("test_model/runtime_pack_0.01b")
CKPT_0_01B = Path("test_model/rwkv7-g1d-0.01b-bench.pth")
GGML_0_01B = Path("test_model/rwkv7-g1d-0.01b-bench-FP16.bin")
PACK_0_1B = Path("test_model/trinity_eval/trinity_grouped_0.1b")
SHADOW_PACK_0_1B = Path("test_model/trinity_eval/trinity_safe_0.1b")
RAW_PACK_0_1B = Path("test_model/runtime_pack")
CKPT_0_1B = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
GGML_0_1B = Path("test_model/rwkv7-g1d-0.1b-FP16.bin")

CLEAR_KEYS = [
    "RWKV_SSD_TIER",
    "RWKV_PARTIAL_SSD_TIER",
    "RWKV_BOUNDED_STREAM",
    "RWKV_PARTIAL_FUSED",
    "RWKV_PARTIAL_HOT4",
    "RWKV_PROMOTE_FULL_Z",
    "RWKV_LUT_GEMM_FUSED",
    "RWKV_LUT_FUSED_ADAPTERS",
    "RWKV_PREFER_FUSED_LUT",
    "RWKV_DECODE_CACHE_COMPRESS",
    "RWKV_WARM_PROVIDER_CACHE",
    "RWKV_STRICT_FUSED_RETAIN",
    "RWKV_STRICT_FUSED_LEAN_Z",
    "RWKV_DECODE_SHADOW",
    "RWKV_WARM_DISK_CACHE",
    "RWKV_PACK_PROFILE",
    "RWKV_SSD_HEALTH",
    "RWKV_STREAM_LAYER_CACHE",
    "RWKVCPP_SYNC_EVERY_TOKEN",
    "RWKVCPP_LAYER_CACHE_BYTES",
    "RWKV_LUT_ACTIVATION_FP32",
    "RWKV_LUT_ACTIVATION_INT8",
    "RWKV_LUT_ACTIVATION_INT8_HEAD",
]


@contextlib.contextmanager
def scenario_env(overrides: dict[str, str]):
    old = {key: os.environ.get(key) for key in CLEAR_KEYS}
    for key in CLEAR_KEYS:
        os.environ.pop(key, None)
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _z_bytes(engine: InferenceEngine) -> int:
    model = getattr(engine.backend, "_model", None)
    z = getattr(model, "z", None) if model is not None else None
    if not isinstance(z, dict):
        return 0
    return int(sum(getattr(t, "nbytes", 0) for t in z.values()))


def _native_model_bytes(engine: InferenceEngine) -> int:
    """Account for a resident native GGML model hidden from ``model.z``."""
    path = getattr(engine.backend, "_ggml_path", None)
    if isinstance(path, Path) and path.is_file():
        return int(path.stat().st_size)
    return 0


def _layer_metric_sums(metrics: dict[str, Any]) -> dict[str, float]:
    """Return stage totals from one request's immutable metric snapshot."""
    rows = metrics.get("layers", [])
    if not isinstance(rows, list):
        rows = []
    return {
        name: sum(float(row.get(name, 0.0) or 0.0) for row in rows if isinstance(row, dict))
        for name in ("read_ms", "compute_ms", "staging_ms")
    }


def run_once(
    pack: Path,
    checkpoint: Path,
    *,
    backend: str = "rwkvcpp",
    label: str,
    mode: str,
    overrides: dict[str, str],
    max_tokens: int,
    prompt: str,
    strategy: str,
    io_backend: str,
    decode_disk_cache: str,
    warm_z: bool = False,
    stream_layer_cache: bool | None = None,
    ram_budget_gb: float | None = None,
    cache_budget_gb: float | None = None,
    provider_cache_budget_bytes: int | None = None,
    native_cache_budget_bytes: int | None = None,
) -> dict[str, Any]:
    effective_overrides = dict(overrides)
    if native_cache_budget_bytes is not None:
        effective_overrides["RWKVCPP_LAYER_CACHE_BYTES"] = str(
            max(0, int(native_cache_budget_bytes))
        )
    with scenario_env(effective_overrides):
        # This is deliberately measured after imports and before the engine
        # is constructed.  It is an interpreter/runtime baseline, not a
        # claim about the model's file size or its OS page-cache residency.
        baseline_rss_bytes = process_rss_bytes()
        cfg = EngineConfig(
            pack_dir=pack,
            checkpoint_path=str(checkpoint),
            backend=backend,
            mode=mode,
            strategy=strategy,
            device="cpu",
            max_tokens=max_tokens,
            greedy=True,
            skeleton_load=mode != "resident" and backend == "chatrwkv",
            io_backend=io_backend,
            decode_disk_cache=decode_disk_cache if mode != "resident" else "0",
            warm_z=warm_z,
            stream_layer_cache=(
                bool(stream_layer_cache) if stream_layer_cache is not None else False
            ),
            ram_budget_gb=ram_budget_gb,
            cache_budget_gb=cache_budget_gb,
            max_provider_cache_bytes=max(0, int(provider_cache_budget_bytes or 0)),
        )
        engine = InferenceEngine(cfg)
        t_load0 = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - t_load0
        after_load_rss_bytes = process_rss_bytes()
        try:
            # The first request is retained as the cold/page-cache observation;
            # the second is the warm steady-state observation.  Both use the
            # same loaded engine and are reported instead of silently
            # publishing only the favorable warm result.
            cold_t0 = time.perf_counter()
            engine.generate(prompt)
            cold_wall_s = time.perf_counter() - cold_t0
            cold_metrics = engine.metrics.to_dict()

            t0 = time.perf_counter()
            engine.generate(prompt)
            warm_wall_s = time.perf_counter() - t0

            m = engine.metrics
            warm_metrics = m.to_dict()
            cold_stage = _layer_metric_sums(cold_metrics)
            warm_stage = _layer_metric_sums(warm_metrics)
            layers = list(m.layers)
            # Prefill: layer_id=-1 (resident) or metrics.prefill_wall_s (streaming).
            # Decode: layer_id=-2 (resident) or layer_id>=0 (streaming layers).
            prefill_s = float(getattr(m, "prefill_wall_s", 0.0) or 0.0)
            decode_layers = [
                x
                for x in layers
                if getattr(x, "layer_id", 0) >= 0 or getattr(x, "layer_id", 0) == -2
            ]
            if prefill_s > 0:
                decode_s = max(1e-9, warm_wall_s - prefill_s)
            else:
                # Fall back to wall time (short prompts make prefill small).
                decode_s = warm_wall_s
            tok_s = max_tokens / decode_s if decode_s > 0 else 0.0
            cold_prefill_s = float(cold_metrics.get("prefill_wall_s", 0.0) or 0.0)
            cold_decode_s = (
                max(1e-9, cold_wall_s - cold_prefill_s)
                if cold_prefill_s > 0
                else cold_wall_s
            )
            cold_tok_s = max_tokens / cold_decode_s if cold_decode_s > 0 else 0.0
            z_mb = _z_bytes(engine) / 1e6
            provider_mb = m.provider_cache_bytes / 1e6
            native_mb = _native_model_bytes(engine) / 1e6
            native_file_bytes = _native_model_bytes(engine)
            cold_rss_peak = int(cold_metrics.get("process_rss_peak_bytes", 0) or 0)
            warm_rss_peak = int(warm_metrics.get("process_rss_peak_bytes", 0) or 0)
            rss_highwater_bytes = max(
                cold_rss_peak,
                warm_rss_peak,
                after_load_rss_bytes,
            )
            rss_highwater_delta_bytes = max(
                0,
                rss_highwater_bytes - baseline_rss_bytes,
            )
            configured_ram_budget_bytes = int(float(ram_budget_gb or 0.0) * 1e9)
            configured_provider_budget_bytes = max(
                int(provider_cache_budget_bytes or 0),
                int(float(cache_budget_gb or 0.0) * 1e9),
            )
            configured_native_budget_bytes = int(
                native_cache_budget_bytes
                if native_cache_budget_bytes is not None
                else int(os.environ.get("RWKVCPP_LAYER_CACHE_BYTES", "0") or 0)
            )
            provider_cache_highwater_bytes = max(
                int(cold_metrics.get("provider_cache_bytes", 0) or 0),
                int(warm_metrics.get("provider_cache_bytes", 0) or 0),
            )
            native_cache_highwater_bytes = max(
                int(cold_metrics.get("native_layer_cache_bytes", 0) or 0),
                int(warm_metrics.get("native_layer_cache_bytes", 0) or 0),
            )
            rss_budget_passed = (
                None
                if configured_ram_budget_bytes <= 0
                else rss_highwater_delta_bytes <= configured_ram_budget_bytes
            )
            provider_budget_passed = (
                None
                if configured_provider_budget_bytes <= 0
                else provider_cache_highwater_bytes <= configured_provider_budget_bytes
            )
            native_budget_passed = (
                None
                if configured_native_budget_bytes <= 0
                else native_cache_highwater_bytes <= configured_native_budget_bytes
            )
            memory_budget_exceeded = bool(
                cold_metrics.get("memory_budget_exceeded", False)
                or warm_metrics.get("memory_budget_exceeded", False)
                or rss_budget_passed is False
                or provider_budget_passed is False
                or native_budget_passed is False
            )
            memory_checks = [
                value
                for value in (rss_budget_passed, provider_budget_passed, native_budget_passed)
                if value is not None
            ]
            return {
                "label": label,
                "mode": engine.config.mode,
                "tok_s": tok_s,
                "tok_s_wall": max_tokens / warm_wall_s if warm_wall_s > 0 else 0.0,
                "wall_s": warm_wall_s,
                "decode_s": decode_s,
                "prefill_s": prefill_s,
                "warm_tok_s": tok_s,
                "warm_wall_s": warm_wall_s,
                "warm_decode_s": decode_s,
                "warm_prefill_s": prefill_s,
                "cold_tok_s": cold_tok_s,
                "cold_wall_s": cold_wall_s,
                "cold_decode_s": cold_decode_s,
                "cold_prefill_s": cold_prefill_s,
                "load_s": load_s,
                "z_mb": z_mb,
                "provider_mb": provider_mb,
                "provider_layer_cache_mb": float(
                    m.provider_layer_cache_bytes / 1e6
                ),
                "provider_resident_mb": float(m.provider_resident_bytes / 1e6),
                "provider_mmap_mb": float(m.provider_mmap_bytes / 1e6),
                # This is a file-size observation, not process RAM.  Keep it
                # separate from RSS so mmap/page-cache effects are not
                # reported as a false fixed overhead.
                "native_model_mb": native_mb,
                "native_model_file_mb": native_mb,
                "native_model_file_bytes": native_file_bytes,
                "fixed_process_overhead_mb": baseline_rss_bytes / 1e6,
                "fixed_process_overhead_bytes": baseline_rss_bytes,
                "process_rss_baseline_bytes": baseline_rss_bytes,
                "process_rss_after_load_bytes": after_load_rss_bytes,
                "rss_highwater_bytes": rss_highwater_bytes,
                "rss_highwater_delta_bytes": rss_highwater_delta_bytes,
                "ram_budget_bytes": configured_ram_budget_bytes,
                "ram_budget_passed": rss_budget_passed,
                "provider_cache_budget_bytes": configured_provider_budget_bytes,
                "provider_cache_highwater_bytes": provider_cache_highwater_bytes,
                "provider_layer_cache_highwater_bytes": max(
                    int(cold_metrics.get("provider_layer_cache_bytes", 0) or 0),
                    int(warm_metrics.get("provider_layer_cache_bytes", 0) or 0),
                ),
                "provider_resident_highwater_bytes": max(
                    int(cold_metrics.get("provider_resident_bytes", 0) or 0),
                    int(warm_metrics.get("provider_resident_bytes", 0) or 0),
                ),
                "provider_mmap_highwater_bytes": max(
                    int(cold_metrics.get("provider_mmap_bytes", 0) or 0),
                    int(warm_metrics.get("provider_mmap_bytes", 0) or 0),
                ),
                "provider_cache_budget_passed": provider_budget_passed,
                "native_cache_budget_bytes": configured_native_budget_bytes,
                "native_cache_highwater_bytes": native_cache_highwater_bytes,
                "native_cache_budget_passed": native_budget_passed,
                "memory_budget_exceeded": memory_budget_exceeded,
                "memory_gate_configured": bool(memory_checks),
                "memory_gate_passed": all(memory_checks) if memory_checks else None,
                "resident_weight_mb": z_mb + provider_mb,
                "tracked_weight_mb": z_mb + provider_mb,
                "tracked_weight_plus_native_file_mb": z_mb + provider_mb + native_mb,
                "read_ms": warm_stage["read_ms"],
                "compute_ms": warm_stage["compute_ms"],
                "staging_ms": warm_stage["staging_ms"],
                "cold_read_ms": cold_stage["read_ms"],
                "cold_compute_ms": cold_stage["compute_ms"],
                "cold_staging_ms": cold_stage["staging_ms"],
                "warm_read_ms": warm_stage["read_ms"],
                "warm_compute_ms": warm_stage["compute_ms"],
                "warm_staging_ms": warm_stage["staging_ms"],
                "streamed_bytes": int(warm_metrics.get("streamed_bytes", 0) or 0),
                "native_upload_bytes": int(warm_metrics.get("native_upload_bytes", 0) or 0),
                "native_packed_bytes": int(warm_metrics.get("native_packed_bytes", 0) or 0),
                "native_active_bytes": int(warm_metrics.get("native_active_bytes", 0) or 0),
                "native_decoded_cache_hits": int(warm_metrics.get("native_decoded_cache_hits", 0) or 0),
                "native_evictions": int(warm_metrics.get("native_evictions", 0) or 0),
                "native_queue_wait_ms": float(warm_metrics.get("native_queue_wait_ms", 0.0) or 0.0),
                "native_layer_cache_bytes": int(warm_metrics.get("native_layer_cache_bytes", 0) or 0),
                "native_layer_cache_hits": int(warm_metrics.get("native_layer_cache_hits", 0) or 0),
                "native_layer_cache_misses": int(warm_metrics.get("native_layer_cache_misses", 0) or 0),
                "native_layer_cache_evictions": int(warm_metrics.get("native_layer_cache_evictions", 0) or 0),
                "cache_hits": int(warm_metrics.get("cache_hits", 0) or 0),
                "process_rss_bytes": int(warm_metrics.get("process_rss_bytes", 0) or 0),
                "process_rss_peak_bytes": int(warm_metrics.get("process_rss_peak_bytes", 0) or 0),
                "median_token_latency_ms": float(warm_metrics.get("median_token_latency_ms", 0.0) or 0.0),
                "p95_token_latency_ms": float(warm_metrics.get("p95_token_latency_ms", 0.0) or 0.0),
                # Preserve both request snapshots.  Warm bytes can be zero
                # after the cold request populated a native/provider cache;
                # that is useful evidence, not a missing value.
                "cold_streamed_bytes": int(cold_metrics.get("streamed_bytes", 0) or 0),
                "warm_streamed_bytes": int(warm_metrics.get("streamed_bytes", 0) or 0),
                "cold_native_upload_bytes": int(cold_metrics.get("native_upload_bytes", 0) or 0),
                "warm_native_upload_bytes": int(warm_metrics.get("native_upload_bytes", 0) or 0),
                "cold_native_packed_bytes": int(cold_metrics.get("native_packed_bytes", 0) or 0),
                "warm_native_packed_bytes": int(warm_metrics.get("native_packed_bytes", 0) or 0),
                "cold_native_active_bytes": int(cold_metrics.get("native_active_bytes", 0) or 0),
                "warm_native_active_bytes": int(warm_metrics.get("native_active_bytes", 0) or 0),
                "cold_native_decoded_cache_hits": int(cold_metrics.get("native_decoded_cache_hits", 0) or 0),
                "warm_native_decoded_cache_hits": int(warm_metrics.get("native_decoded_cache_hits", 0) or 0),
                "cold_native_evictions": int(cold_metrics.get("native_evictions", 0) or 0),
                "warm_native_evictions": int(warm_metrics.get("native_evictions", 0) or 0),
                "cold_native_queue_wait_ms": float(cold_metrics.get("native_queue_wait_ms", 0.0) or 0.0),
                "warm_native_queue_wait_ms": float(warm_metrics.get("native_queue_wait_ms", 0.0) or 0.0),
                "cold_native_layer_cache_bytes": int(cold_metrics.get("native_layer_cache_bytes", 0) or 0),
                "warm_native_layer_cache_bytes": int(warm_metrics.get("native_layer_cache_bytes", 0) or 0),
                "cold_native_layer_cache_hits": int(cold_metrics.get("native_layer_cache_hits", 0) or 0),
                "warm_native_layer_cache_hits": int(warm_metrics.get("native_layer_cache_hits", 0) or 0),
                "cold_native_layer_cache_misses": int(cold_metrics.get("native_layer_cache_misses", 0) or 0),
                "warm_native_layer_cache_misses": int(warm_metrics.get("native_layer_cache_misses", 0) or 0),
                "cold_native_layer_cache_evictions": int(cold_metrics.get("native_layer_cache_evictions", 0) or 0),
                "warm_native_layer_cache_evictions": int(warm_metrics.get("native_layer_cache_evictions", 0) or 0),
                "cold_cache_hits": int(cold_metrics.get("cache_hits", 0) or 0),
                "warm_cache_hits": int(warm_metrics.get("cache_hits", 0) or 0),
                "cold_provider_cache_bytes": int(cold_metrics.get("provider_cache_bytes", 0) or 0),
                "warm_provider_cache_bytes": int(warm_metrics.get("provider_cache_bytes", 0) or 0),
                "cold_process_rss_bytes": int(cold_metrics.get("process_rss_bytes", 0) or 0),
                "cold_process_rss_peak_bytes": int(cold_metrics.get("process_rss_peak_bytes", 0) or 0),
                "warm_process_rss_bytes": int(warm_metrics.get("process_rss_bytes", 0) or 0),
                "warm_process_rss_peak_bytes": int(warm_metrics.get("process_rss_peak_bytes", 0) or 0),
                "cold_memory_budget_exceeded": bool(cold_metrics.get("memory_budget_exceeded", False)),
                "warm_memory_budget_exceeded": bool(warm_metrics.get("memory_budget_exceeded", False)),
                "cold_median_token_latency_ms": float(cold_metrics.get("median_token_latency_ms", 0.0) or 0.0),
                "cold_p95_token_latency_ms": float(cold_metrics.get("p95_token_latency_ms", 0.0) or 0.0),
                "warm_median_token_latency_ms": float(warm_metrics.get("median_token_latency_ms", 0.0) or 0.0),
                "warm_p95_token_latency_ms": float(warm_metrics.get("p95_token_latency_ms", 0.0) or 0.0),
                "layer_steps": len(decode_layers),
                "cpu_threads": torch.get_num_threads(),
            }
        finally:
            engine.close()


def run_scenario(
    pack: Path,
    checkpoint: Path,
    *,
    backend: str = "rwkvcpp",
    label: str,
    mode: str,
    overrides: dict[str, str],
    max_tokens: int,
    prompt: str,
    strategy: str,
    io_backend: str,
    decode_disk_cache: str,
    samples: int,
    warm_z: bool = False,
    stream_layer_cache: bool | None = None,
    ram_budget_gb: float | None = None,
    cache_budget_gb: float | None = None,
    provider_cache_budget_bytes: int | None = None,
    native_cache_budget_bytes: int | None = None,
) -> dict[str, Any]:
    rows = [
        run_once(
            pack,
            checkpoint,
            backend=backend,
            label=label,
            mode=mode,
            overrides=overrides,
            max_tokens=max_tokens,
            prompt=prompt,
            strategy=strategy,
            io_backend=io_backend,
            decode_disk_cache=decode_disk_cache,
            warm_z=warm_z,
            stream_layer_cache=stream_layer_cache,
            ram_budget_gb=ram_budget_gb,
            cache_budget_gb=cache_budget_gb,
            provider_cache_budget_bytes=provider_cache_budget_bytes,
            native_cache_budget_bytes=native_cache_budget_bytes,
        )
        for _ in range(samples)
    ]
    out = dict(rows[-1])
    rates = [float(row["tok_s"]) for row in rows]
    out["tok_s"] = statistics.median(rates)
    # Keep the published row representative of the sampled process rather
    # than accidentally mixing last-sample memory with median throughput.
    # Every numeric telemetry field is safe to aggregate; labels/modes and
    # the explicit min/max fields remain structural.
    for key in rows[0]:
        if key in {"tok_s", "tok_s_min", "tok_s_max", "samples"}:
            continue
        values = [row.get(key) for row in rows]
        if values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            out[key] = statistics.median(float(value) for value in values)
    for key in ("warm_tok_s", "cold_tok_s"):
        out[key] = statistics.median(float(row[key]) for row in rows)
    out["tok_s_min"] = min(rates)
    out["tok_s_max"] = max(rates)
    out["cold_tok_s_min"] = min(float(row["cold_tok_s"]) for row in rows)
    out["cold_tok_s_max"] = max(float(row["cold_tok_s"]) for row in rows)
    out["warm_tok_s_min"] = min(float(row["warm_tok_s"]) for row in rows)
    out["warm_tok_s_max"] = max(float(row["warm_tok_s"]) for row in rows)
    out["samples"] = samples
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true", help="Use the 0.01B bench model")
    parser.add_argument(
        "--backend",
        choices=["chatrwkv", "rwkvcpp"],
        default="rwkvcpp",
        help="Native compute backend for the same F-tier matrix (default: rwkvcpp)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use the raw bf16 0.1B pack instead of the Trinity/LUT pack",
    )
    parser.add_argument(
        "--shadow-sel",
        action="store_true",
        help="Use the selective-shadow Trinity 0.1B pack",
    )
    parser.add_argument("--pack", type=Path, help="Runtime pack directory")
    parser.add_argument("--checkpoint", type=Path, help="ChatRWKV .pth checkpoint")
    parser.add_argument(
        "--ggml",
        type=Path,
        help="rwkv.cpp GGML model; sets RWKVCPP_GGML_PATH when supplied",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--prompt", default="Throughput bench prompt")
    parser.add_argument("--strategy", default="cpu bf16")
    parser.add_argument("--io-backend", default="mmap")
    parser.add_argument(
        "--ram-budget-gb",
        type=float,
        default=None,
        help="tier RAM budget used by the optional memory acceptance gate",
    )
    parser.add_argument(
        "--cache-budget-gb",
        type=float,
        default=None,
        help="provider/cache budget used by the optional memory acceptance gate",
    )
    parser.add_argument(
        "--provider-cache-budget-bytes",
        type=int,
        default=None,
        help="explicit provider cache budget for the tier run",
    )
    parser.add_argument(
        "--native-cache-budget-bytes",
        type=int,
        default=None,
        help="explicit rwkv.cpp native decoded-layer cache budget",
    )
    parser.add_argument(
        "--decode-disk-cache",
        default="0",
        choices=["0", "1", "auto"],
        help="Persistent decoded layer cache; default off for wiring-speed benches",
    )
    parser.add_argument(
        "--tiers",
        default="F1,F2,F3,F4,F5,F6",
        help="Comma-separated tiers: F1,F2,F3,F4,F5,F6 (F6 = resident)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--enforce-f-tier-gates",
        action="store_true",
        help="fail unless F1/F2/F3/F4 meet their cold and warm ratios versus F5",
    )
    parser.add_argument(
        "--enforce-f-tier-memory-gates",
        action="store_true",
        help="fail unless configured RAM/provider/native-cache budgets pass",
    )
    args = parser.parse_args()

    if args.ggml is not None:
        if not args.ggml.is_file():
            raise SystemExit(f"ggml model not found: {args.ggml}")
        os.environ["RWKVCPP_GGML_PATH"] = str(args.ggml)
    elif args.backend == "rwkvcpp":
        default_ggml = GGML_0_01B if args.small else GGML_0_1B
        if default_ggml.is_file():
            os.environ["RWKVCPP_GGML_PATH"] = str(default_ggml)
        else:
            raise SystemExit(
                f"rwkv.cpp default GGML model not found: {default_ggml}; "
                "pass --ggml or build the matching native model"
            )

    if args.raw and args.shadow_sel:
        raise SystemExit("--raw and --shadow-sel are mutually exclusive")
    if args.raw:
        default_0_1b_pack = RAW_PACK_0_1B
    elif args.shadow_sel:
        default_0_1b_pack = SHADOW_PACK_0_1B
    elif args.backend == "rwkvcpp":
        # The native bridge consumes dense tensors.  Keep the default native
        # benchmark on the raw pack so it measures GGML/provider performance
        # rather than materializing the historical Trinity/LUT artifact.
        default_0_1b_pack = RAW_PACK_0_1B
    else:
        default_0_1b_pack = PACK_0_1B
    pack = args.pack or (PACK_0_01B if args.small else default_0_1b_pack)
    checkpoint = args.checkpoint or (CKPT_0_01B if args.small else CKPT_0_1B)
    if not pack.is_dir():
        raise SystemExit(f"pack not found: {pack}")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    tier_specs: dict[str, dict[str, Any]] = {
        "F6": {
            "label": f"F6 resident reference {args.backend}",
            "mode": "resident",
            "overrides": {},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        "F1": {
            "label": "F1 ssd-tier-min",
            "mode": "streaming",
            "overrides": {"RWKV_SSD_TIER": "1"},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        "F2": {
            "label": "F2 bounded-fused",
            "mode": "streaming",
            "overrides": {"RWKV_BOUNDED_STREAM": "1"},
            "warm_z": False,
            "stream_layer_cache": True,
        },
        "F3": {
            "label": "F3 partial-hot3",
            "mode": "partial",
            "overrides": {"RWKV_PARTIAL_SSD_TIER": "1"},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        "F4": {
            "label": "F4 partial-hot4",
            "mode": "partial",
            "overrides": {"RWKV_PARTIAL_HOT4": "1"},
            "warm_z": False,
            "stream_layer_cache": False,
        },
        "F5": {
            "label": "F5 promote-max native graph",
            "mode": "streaming",
            # F5 keeps the public streaming/provider mode for compatibility,
            # but the rwkv.cpp loader selects its full resident GGML graph
            # when this promote profile is active.
            "overrides": {"RWKV_PROMOTE_FULL_Z": "1"},
            "warm_z": True,
            "stream_layer_cache": True,
        },
    }
    wanted = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in tier_specs]
    if unknown:
        raise SystemExit(f"unknown tiers {unknown}; choose from {list(tier_specs)}")
    # Always put F6 first when present so vs_normal is vs resident.
    if "F6" in wanted:
        wanted = ["F6"] + [t for t in wanted if t != "F6"]

    rows = [
        run_scenario(
            pack,
            checkpoint,
            backend=args.backend,
            label=str(tier_specs[tid]["label"]),
            mode=str(tier_specs[tid]["mode"]),
            overrides=dict(tier_specs[tid]["overrides"]),
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            strategy=args.strategy,
            io_backend=args.io_backend,
            decode_disk_cache=args.decode_disk_cache,
            samples=max(1, args.samples),
            warm_z=bool(tier_specs[tid]["warm_z"]),
            stream_layer_cache=tier_specs[tid]["stream_layer_cache"],
            ram_budget_gb=args.ram_budget_gb,
            cache_budget_gb=args.cache_budget_gb,
            provider_cache_budget_bytes=args.provider_cache_budget_bytes,
            native_cache_budget_bytes=args.native_cache_budget_bytes,
        )
        for tid in wanted
    ]
    for row, tid in zip(rows, wanted):
        row["frontier_id"] = tid
    baseline = float(rows[0]["tok_s"])
    for row in rows:
        row["vs_normal"] = float(row["tok_s"]) / baseline if baseline > 0 else None

    by_frontier = {str(row["frontier_id"]): row for row in rows}
    gate_rows: list[dict[str, Any]] = []
    gate_passed = True
    memory_gate_rows: list[dict[str, Any]] = []
    memory_gate_passed = True
    f5 = by_frontier.get("F5")
    if f5 is not None:
        for frontier_id, minimum in {"F1": 0.60, "F2": 0.80, "F3": 0.80, "F4": 0.80}.items():
            row = by_frontier.get(frontier_id)
            if row is None:
                continue
            checks = {}
            for measurement in ("cold", "warm"):
                ratio = float(row[f"{measurement}_tok_s"]) / max(
                    1e-12, float(f5[f"{measurement}_tok_s"])
                )
                row[f"{measurement}_vs_f5"] = ratio
                checks[measurement] = {
                    "ratio": ratio,
                    "minimum": minimum,
                    "passed": ratio >= minimum,
                }
                gate_passed = gate_passed and bool(checks[measurement]["passed"])
            row["throughput_gate"] = checks
            gate_rows.append({"frontier_id": frontier_id, **checks})

    if f5 is not None:
        for frontier_id in ("F1", "F2", "F3", "F4"):
            row = by_frontier.get(frontier_id)
            if row is None:
                continue
            checks = {
                "ram": {
                    "configured_bytes": int(row.get("ram_budget_bytes", 0) or 0),
                    "highwater_delta_bytes": int(
                        row.get("rss_highwater_delta_bytes", 0) or 0
                    ),
                    "passed": row.get("ram_budget_passed"),
                },
                "provider_cache": {
                    "configured_bytes": int(
                        row.get("provider_cache_budget_bytes", 0) or 0
                    ),
                    "highwater_bytes": int(
                        row.get("provider_cache_highwater_bytes", 0) or 0
                    ),
                    "passed": row.get("provider_cache_budget_passed"),
                },
                "native_cache": {
                    "configured_bytes": int(
                        row.get("native_cache_budget_bytes", 0) or 0
                    ),
                    "highwater_bytes": int(
                        row.get("native_cache_highwater_bytes", 0) or 0
                    ),
                    "passed": row.get("native_cache_budget_passed"),
                },
            }
            applicable = [
                value["passed"]
                for value in checks.values()
                if value["passed"] is not None
            ]
            row["memory_gate"] = {
                "configured": bool(applicable),
                "passed": all(applicable) if applicable else None,
                "checks": checks,
            }
            if applicable:
                memory_gate_passed = memory_gate_passed and all(applicable)
            memory_gate_rows.append({"frontier_id": frontier_id, **row["memory_gate"]})

    result = {
        "pack": str(pack),
        "checkpoint": str(checkpoint),
        "max_tokens": args.max_tokens,
        "samples": max(1, args.samples),
        "strategy": args.strategy,
        "io_backend": args.io_backend,
        "decode_disk_cache": args.decode_disk_cache,
        "configured_budgets": {
            "ram_budget_gb": args.ram_budget_gb,
            "cache_budget_gb": args.cache_budget_gb,
            "provider_cache_budget_bytes": args.provider_cache_budget_bytes,
            "native_cache_budget_bytes": args.native_cache_budget_bytes,
        },
        "measurement_semantics": {
            "cold": "first request in a fresh engine process after load; OS/page cache is not forcibly dropped",
            "warm": "second request on the same loaded engine",
            "fixed_process_overhead": "RSS before engine construction; interpreter/runtime baseline only",
            "native_model_file": "GGML file size, reported separately from process RSS",
        },
        "tiers": wanted,
        "rows": rows,
        "f_tier_gate": {
            "baseline": "F5",
            "required_ratios": {"F1": 0.60, "F2": 0.80, "F3": 0.80, "F4": 0.80},
            "measurements": ["cold", "warm"],
            "rows": gate_rows,
            "passed": gate_passed if f5 is not None else None,
        },
        "f_tier_memory_gate": {
            "baseline": "configured per-tier budgets",
            "rows": memory_gate_rows,
            "passed": memory_gate_passed if f5 is not None else None,
            "enforced": bool(args.enforce_f_tier_memory_gates),
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(
        f"F1/F2/F3/F4/F5/F6 {args.backend} bench: pack={pack} max_tokens={args.max_tokens} "
        f"samples={max(1, args.samples)} strategy={args.strategy} "
        f"decode_disk_cache={args.decode_disk_cache} "
        f"(tok/s = decode-only; excludes prompt prefill)"
    )
    print(
        f"{'scenario':28} {'mode':10} {'tok/s':>8} {'vs F6':>10} "
        f"{'z_mb':>9} {'prov_mb':>9} {'native':>9} {'tracked':>9} "
        f"{'read_ms':>9} {'compute_ms':>10} "
        f"{'stage_ms':>9} {'decode_s':>8} {'thr':>3}"
    )
    for row in rows:
        print(
            f"{row['label']:28} {row['mode']:10} {float(row['tok_s']):8.2f} "
            f"{float(row['vs_normal'] or 0):10.2f} {float(row['z_mb']):9.1f} "
            f"{float(row['provider_mb']):9.1f} "
            f"{float(row.get('native_model_mb', 0.0)):9.1f} "
            f"{float(row.get('tracked_weight_mb', row['z_mb'] + row['provider_mb'])):9.1f} "
            f"{float(row['read_ms']):9.1f} "
            f"{float(row['compute_ms']):10.1f} {float(row['staging_ms']):9.1f} "
            f"{float(row.get('decode_s', row['wall_s'])):8.3f} "
            f"{int(row.get('cpu_threads', 0)):3d}"
        )
    if f5 is not None:
        print("F-tier acceptance (ratio vs F5, cold/warm):")
        for item in gate_rows:
            print(
                f"  {item['frontier_id']}: cold={item['cold']['ratio']:.3f} "
                f"({'PASS' if item['cold']['passed'] else 'FAIL'}), "
                f"warm={item['warm']['ratio']:.3f} "
                f"({'PASS' if item['warm']['passed'] else 'FAIL'})"
            )
        if memory_gate_rows:
            print("F-tier memory acceptance (configured budgets):")
            for item in memory_gate_rows:
                status = item.get("passed")
                print(
                    f"  {item['frontier_id']}: "
                    f"{('PASS' if status else 'FAIL') if status is not None else 'NOT_CONFIGURED'}"
                )
    if args.enforce_f_tier_gates and (f5 is None or not gate_passed):
        raise SystemExit("F1-F4 throughput gate failed; inspect cold/warm ratios in the JSON output")
    if args.enforce_f_tier_memory_gates and (
        f5 is None or not memory_gate_passed or not any(row.get("configured") for row in memory_gate_rows)
    ):
        raise SystemExit(
            "F1-F4 memory gate failed or no budgets were configured; "
            "inspect f_tier_memory_gate in the JSON output"
        )
    if args.json_out:
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
