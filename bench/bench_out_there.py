#!/usr/bin/env python3
"""
Afternoon experiments — "out there" ideas outside Trinity LUT compression.

Runs engine smoke tests (0.1B when available), pytest gates, and thesis simulation
benches. Writes ``test_model/trinity_eval/out_there.json``.

Usage:
  python bench/bench_out_there.py --quick
  python bench/bench_out_there.py --full   # more tokens + all sim benches
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT_0_1B = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
PACK_LUT2 = ROOT / "test_model/trinity_eval/trinity_grouped_0.1b"
PACK_SHADOW = ROOT / "test_model/trinity_eval/trinity_safe_0.1b"
PACK_TIERED = ROOT / "test_model/trinity_eval/trinity_tiered_hot3_0.1b"
PACK_GROUPED = ROOT / "test_model/trinity_eval/fp16_grouped_0.1b"
PACK_001 = ROOT / "test_model/runtime_pack_0.01b"
CKPT_001 = ROOT / "test_model/rwkv7-g1d-0.1b-bench.pth"
PACK_U8 = ROOT / "test_model/packs_0.01b/scale_u8"
PACK_U4 = ROOT / "test_model/packs_0.01b/scale_u4"

TRINITY_ENV_KEYS = (
    "RWKV_PREFER_FUSED_LUT",
    "RWKV_SSD_TIER",
    "RWKV_PACK_PROFILE",
    "RWKV_PROMOTE_FULL_Z",
    "RWKV_DECODE_CACHE_COMPRESS",
    "RWKV_DECODE_SHADOW",
    "RWKV_LUT_BF16_NATIVE",
    "RWKV_WARM_PROVIDER_CACHE",
    "RWKV_PARTIAL_FUSED",
    "RWKV_PARTIAL_SSD_TIER",
    "RWKV_LUT_GEMM_FUSED",
    "RWKV_BOUNDED_STREAM",
    "RWKV_WARM_DISK_CACHE",
    "RWKV_STRICT_FUSED_RETAIN",
    "RWKV_STRICT_FUSED_LEAN_Z",
)


def _clear_trinity_env() -> None:
    for key in TRINITY_ENV_KEYS:
        os.environ.pop(key, None)


@dataclass
class Experiment:
    id: str
    area: str
    label: str
    kind: str  # engine | pytest | sim | skip
    fn: Callable[[], dict[str, Any]] | None = None
    skip_reason: str | None = None


def _run_engine_chat(
    pack: Path,
    ckpt: Path,
    *,
    label: str,
    max_tokens: int,
    mode: str = "streaming",
    env: dict[str, str] | None = None,
    config_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.throughput_defaults import (
        apply_partial_ssd_tier_defaults,
        apply_promote_max_defaults,
        apply_ssd_tier_fused_defaults,
    )
    from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z

    if not pack.is_dir() or not ckpt.is_file():
        return {"status": "skip", "reason": f"missing pack={pack} or ckpt={ckpt}"}

    _clear_trinity_env()
    if env:
        for k, v in env.items():
            os.environ[k] = v

    cfg_kw = dict(config_kw or {})
    manifest = Manifest.load(pack)
    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(ckpt),
        backend="chatrwkv",
        mode=mode,
        strategy="cpu bf16",
        device="cpu",
        max_tokens=max_tokens,
        greedy=True,
        skeleton_load=True,
        stream_layer_cache=bool(cfg_kw.pop("stream_layer_cache", False)),
        max_layers_in_z=int(cfg_kw.pop("max_layers_in_z", 0)),
        decode_disk_cache=str(cfg_kw.pop("decode_disk_cache", "auto")),
        **cfg_kw,
    )
    preset = env.get("RWKV_PRESET", "") if env else ""
    if preset == "f1":
        apply_ssd_tier_fused_defaults(cfg, manifest)
        os.environ["RWKV_SSD_TIER"] = "1"
    elif preset == "f3":
        apply_partial_ssd_tier_defaults(cfg, manifest)
        os.environ["RWKV_PARTIAL_SSD_TIER"] = "1"
    elif preset == "f5":
        apply_promote_max_defaults(cfg, manifest)

    eng = InferenceEngine(cfg)
    eng.load()
    try:
        t0 = time.perf_counter()
        eng.generate("Out-there bench prompt for throughput measurement.")
        wall = time.perf_counter() - t0
        m = eng.metrics
        n_tok = max_tokens
        read_ms = sum(L.read_ms for L in m.layers)
        staging_ms = sum(L.staging_ms for L in m.layers)
        compute_ms = sum(L.compute_ms for L in m.layers)
        io_ms = (read_ms + staging_ms) / n_tok
        tok_s = n_tok / wall if wall > 0 else 0.0
        model = getattr(eng.backend, "_model", None)
        layers_in_z = (
            len(block_layer_ids_in_z(model.z))
            if model is not None and hasattr(model, "z")
            else 0
        )
        return {
            "status": "ok",
            "scenario": label,
            "tok_s": round(tok_s, 2),
            "z_mb": round(m.z_bytes / 1e6, 2) if m.z_bytes else None,
            "provider_mb": round(m.provider_cache_bytes / 1e6, 2),
            "layers_in_z": layers_in_z,
            "read_ms_per_token": round(read_ms / n_tok, 2),
            "staging_ms_per_token": round(staging_ms / n_tok, 2),
            "compute_ms_per_token": round(compute_ms / n_tok, 2),
            "io_ms_per_token": round(io_ms, 2),
            "io_ceiling_tok_s": round(1000.0 / io_ms, 1) if io_ms > 1e-6 else None,
            "state_cache_hit": m.state_cache_hit,
            "prefill_wall_s": round(m.prefill_wall_s, 4),
            "mtp_gate_open": m.mtp_gate_open,
            "prefetch_overlaps": m.prefetch_overlaps,
        }
    finally:
        eng.close()


def _raw_gbs(pack: Path) -> float | None:
    from bench.bench_io_ceiling import _raw_gbs

    try:
        return _raw_gbs(pack, "mmap", trials=1)
    except Exception as exc:
        return None


def _run_pytest(test_path: str, timeout: int = 120) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", test_path, "-q", "--tb=line"]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    tail = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    summary = lines[-1] if lines else ""
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "wall_s": round(wall, 2),
        "summary": summary,
    }


def _run_sim(script_rel: str, timeout: int = 90) -> dict[str, Any]:
    path = ROOT / script_rel
    if not path.is_file():
        return {"status": "skip", "reason": "script missing"}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": timeout}
    wall = time.perf_counter() - t0
    out = (proc.stdout or "")[-2000:]
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "wall_s": round(wall, 2),
        "tail": out.strip(),
    }


def build_experiments(max_tokens: int, run_sims: bool) -> list[Experiment]:
    ex: list[Experiment] = []

    # --- Compute / layout (not Trinity codec) ---
    ex.append(
        Experiment(
            "E01",
            "compute_layout",
            "lut2 strict F1 baseline",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="lut2 F1", max_tokens=max_tokens,
                env={"RWKV_PRESET": "f1", "RWKV_WARM_DISK_CACHE": "0"},
            ),
        )
    )
    if PACK_SHADOW.is_dir():
        ex.append(
            Experiment(
                "E02",
                "compute_layout",
                "shadow_sel strict F1",
                "engine",
                lambda: _run_engine_chat(
                    PACK_SHADOW, CKPT_0_1B, label="shadow_sel F1", max_tokens=max_tokens,
                    env={"RWKV_PRESET": "f1", "RWKV_WARM_DISK_CACHE": "0"},
                ),
            )
        )
    if PACK_TIERED.is_dir():
        ex.append(
            Experiment(
                "E03",
                "compute_layout",
                "tiered_hot3 F3",
                "engine",
                lambda: _run_engine_chat(
                    PACK_TIERED, CKPT_0_1B, label="tiered F3", max_tokens=max_tokens,
                    env={"RWKV_PRESET": "f3", "RWKV_WARM_DISK_CACHE": "0"},
                ),
            )
        )
    ex.append(
        Experiment(
            "E04",
            "compute_layout",
            "F1 lean_z off (skeleton in z)",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="F1 lean_z=0", max_tokens=max_tokens,
                env={
                    "RWKV_PRESET": "f1",
                    "RWKV_WARM_DISK_CACHE": "0",
                    "RWKV_STRICT_FUSED_LEAN_Z": "0",
                },
            ),
        )
    )
    if PACK_GROUPED.is_dir():
        ex.append(
            Experiment(
                "E05",
                "compute_layout",
                "fp16_grouped raw mmap GB/s",
                "engine",
                lambda: {
                    "status": "ok",
                    "raw_read_gbs_mmap": _raw_gbs(PACK_GROUPED),
                    "pack_mb": round((PACK_GROUPED / "weights.bin").stat().st_size / 1e6, 2)
                    if (PACK_GROUPED / "weights.bin").is_file()
                    else None,
                },
            )
        )

    # --- RWKV schedule / prefetch ---
    ex.append(
        Experiment(
            "E06",
            "rwkv_schedule",
            "prefetch gate policy F1",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="gate prefetch", max_tokens=max_tokens,
                env={"RWKV_PRESET": "f1", "RWKV_WARM_DISK_CACHE": "0"},
                config_kw={"prefetch_policy": "gate"},
            ),
        )
    )
    ex.append(
        Experiment(
            "E07",
            "rwkv_schedule",
            "prefetch layer (no lookahead) F1",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="layer prefetch", max_tokens=max_tokens,
                env={"RWKV_PRESET": "f1", "RWKV_WARM_DISK_CACHE": "0"},
                config_kw={"prefetch_policy": "layer"},
            ),
        )
    )
    ex.append(
        Experiment(
            "E08",
            "rwkv_schedule",
            "ngram_weight_cache F1",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="ngram cache", max_tokens=max_tokens,
                env={"RWKV_PRESET": "f1", "RWKV_WARM_DISK_CACHE": "0"},
                config_kw={"ngram_weight_cache": True},
            ),
        )
    )

    # --- Serving / state ---
    long_prefix = "SYSTEM:" + (" context block. " * 64)
    ex.append(
        Experiment(
            "E09",
            "serving",
            "prefix state cache F5",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="state cache F5", max_tokens=max_tokens,
                env={"RWKV_PRESET": "f5", "RWKV_WARM_DISK_CACHE": "0"},
                config_kw={
                    "state_cache": True,
                    "system_prefix": long_prefix,
                    "stream_layer_cache": True,
                    "max_layers_in_z": 1,
                },
            ),
        )
    )
    ex.append(
        Experiment(
            "E10",
            "serving",
            "mtp_speculative gate F5",
            "engine",
            lambda: _run_engine_chat(
                PACK_LUT2, CKPT_0_1B, label="mtp gate", max_tokens=max(max_tokens, 32),
                env={"RWKV_PRESET": "f5", "RWKV_WARM_DISK_CACHE": "0"},
                config_kw={
                    "mtp_speculative": True,
                    "mtp_draft_tokens": 4,
                    "mtp_min_prefill_tokens": 8,
                    "stream_layer_cache": True,
                },
            ),
        )
    )

    # --- Ecosystem quant (0.01B) ---
    if PACK_001.is_dir() and CKPT_001.is_file():
        ex.append(
            Experiment(
                "E11",
                "ecosystem_quant",
                "0.01B resident baseline",
                "engine",
                lambda: _run_engine_chat(
                    PACK_001, CKPT_001, label="0.01b resident", max_tokens=max_tokens,
                    mode="resident", env={}, config_kw={"skeleton_load": False},
                ),
            )
        )
    if PACK_U8.is_dir():
        ex.append(
            Experiment(
                "E12",
                "ecosystem_quant",
                "0.01B scale_u8 stream",
                "engine",
                lambda: _run_engine_chat(
                    PACK_U8, CKPT_001, label="scale_u8", max_tokens=max_tokens,
                    mode="streaming", env={"RWKV_WARM_DISK_CACHE": "0"},
                ),
            )
        )
    if PACK_U4.is_dir():
        ex.append(
            Experiment(
                "E13",
                "ecosystem_quant",
                "0.01B scale_u4 stream",
                "engine",
                lambda: _run_engine_chat(
                    PACK_U4, CKPT_001, label="scale_u4", max_tokens=max_tokens,
                    mode="streaming", env={"RWKV_WARM_DISK_CACHE": "0"},
                ),
            )
        )

    # --- Pytest gates ---
    ex.append(
        Experiment(
            "T01",
            "pytest",
            "gate prefetch greedy parity",
            "pytest",
            lambda: _run_pytest("tests/test_gate_prefetch.py"),
        )
    )
    ex.append(
        Experiment(
            "T02",
            "pytest",
            "trinity overhead elimination",
            "pytest",
            lambda: _run_pytest("tests/test_trinity_overhead_elimination.py"),
        )
    )
    ex.append(
        Experiment(
            "T03",
            "pytest",
            "state cache",
            "pytest",
            lambda: _run_pytest("tests/test_state_cache.py"),
        )
    )

    # --- Skipped (need GPU / multi-disk / not in repo) ---
    for sid, area, label, reason in [
        ("S01", "gpu", "GDS NVMe→VRAM (M6c)", "requires Linux + NVIDIA + GDS"),
        ("S02", "gpu", "FLUTE CUDA GEMV (M6b)", "not implemented in engine"),
        ("S03", "io_topology", "dual-NVMe pack/cache split", "single drive on dev machine"),
        ("S04", "model", "low-rank factored stream pack", "no pack tooling yet"),
        ("S05", "model", "HRWKV7 hybrid layers", "M8 deferred"),
        ("S06", "model", "LoRA delta stream", "sim only — engine slot not wired"),
        ("S07", "model", "Engram / ROSA / DeepEmbed", "deferred — model contract change"),
        ("S08", "io_topology", "shared decode cache multi-process", "manual ops test"),
    ]:
        ex.append(Experiment(sid, area, label, "skip", skip_reason=reason))

    if run_sims:
        sim_map: list[tuple[str, str, str, str, int]] = [
            ("SIM01", "rwkv_schedule", "gate_prefetch_bench", "simulations/benches/gate_prefetch_bench.py", 90),
            ("SIM02", "rwkv_schedule", "layer_aware_prefetch", "simulations/benches/layer_aware_prefetch_schedule_bench.py", 90),
            ("SIM03", "serving", "prefix_state_library", "simulations/benches/prefix_state_library_bench.py", 90),
            ("SIM04", "serving", "ngram_weight_cache", "simulations/benches/ngram_weight_cache_bench.py", 90),
            ("SIM05", "rwkv_schedule", "ssm_speculation_cache", "simulations/benches/ssm_speculation_cache_bench.py", 90),
            ("SIM06", "serving", "mtp_ssd_speculation", "simulations/benches/mtp_ssd_speculation_bench.py", 90),
            ("SIM07", "io_reliability", "erasure_coding", "simulations/benches/erasure_coding_bench.py", 90),
            ("SIM08", "io_reliability", "read_disturb_rotation", "simulations/benches/read_disturb_rotation_bench.py", 90),
            ("SIM09", "io_reliability", "pslc_state_endurance", "simulations/benches/pslc_state_endurance_bench.py", 90),
            ("SIM10", "codec", "compression_path_comparison", "simulations/benches/compression_path_comparison_bench.py", 90),
            ("SIM11", "codec", "compression_trinity", "simulations/benches/compression_trinity_bench.py", 300),
            ("SIM12", "io_schedule", "heterogeneous_chunk_schedule", "simulations/benches/heterogeneous_chunk_schedule_bench.py", 90),
            ("SIM13", "io_schedule", "micro_pipeline", "simulations/benches/micro_pipeline_bench.py", 90),
            ("SIM14", "serving", "batch_scaling", "simulations/benches/batch_scaling_bench.py", 90),
            ("SIM15", "model", "lora_delta_streaming", "simulations/benches/lora_delta_streaming_bench.py", 90),
            ("SIM16", "model", "engram_architecture", "simulations/benches/engram_architecture_bench.py", 90),
            ("SIM17", "gpu", "kernel_bypass_gds", "simulations/benches/kernel_bypass_gds_bench.py", 90),
            ("SIM18", "io_topology", "numa_topology", "simulations/benches/numa_topology_bench.py", 90),
            ("SIM19", "io_topology", "asymmetric_raid_power", "simulations/benches/asymmetric_raid_power_bench.py", 90),
            ("SIM20", "io_topology", "die_thermal_balance", "simulations/benches/die_thermal_balance_bench.py", 90),
            ("SIM21", "codec", "temporal_weight_locality", "simulations/benches/temporal_weight_locality_bench.py", 90),
        ]
        for sid, area, label, script, sim_timeout in sim_map:
            ex.append(
                Experiment(
                    sid,
                    area,
                    label,
                    "sim",
                    lambda s=script, t=sim_timeout: _run_sim(s, timeout=t),
                )
            )

    return ex


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Out-there idea afternoon experiments")
    p.add_argument("--quick", action="store_true", help="12 tokens, engine + pytest only")
    p.add_argument("--full", action="store_true", help="24 tokens + all simulation benches")
    p.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "test_model/trinity_eval/out_there.json",
    )
    args = p.parse_args()

    # Default: 12 tokens, engine + pytest only (same as --quick).
    # --full: 24 tokens + all simulation benches.
    max_tokens = 24 if args.full else 12
    run_sims = bool(args.full)

    experiments = build_experiments(max_tokens, run_sims=run_sims)
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for exp in experiments:
        print(f"[{exp.id}] {exp.label} ...", flush=True)
        if exp.kind == "skip":
            row = {
                "id": exp.id,
                "area": exp.area,
                "label": exp.label,
                "kind": exp.kind,
                "status": "skip",
                "reason": exp.skip_reason,
            }
        else:
            try:
                result = exp.fn() if exp.fn else {"status": "skip", "reason": "no fn"}
                row = {
                    "id": exp.id,
                    "area": exp.area,
                    "label": exp.label,
                    "kind": exp.kind,
                    **result,
                }
            except Exception as exc:
                row = {
                    "id": exp.id,
                    "area": exp.area,
                    "label": exp.label,
                    "kind": exp.kind,
                    "status": "error",
                    "error": str(exc),
                }
        rows.append(row)
        print(json.dumps(row, default=str))

    out = {
        "max_tokens": max_tokens,
        "wall_s": round(time.perf_counter() - t_all, 1),
        "machine_note": "Windows dev CPU; GPU/dual-NVMe items skipped or sim-only",
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
