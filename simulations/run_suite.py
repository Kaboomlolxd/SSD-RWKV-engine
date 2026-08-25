#!/usr/bin/env python3
"""
Legacy thesis / feasibility simulation suite (not the inference engine).

Usage (from repo root):
    python simulations/run_suite.py
    python simulations/run_suite.py --list

Outputs:
    simulations/suite_output.log
    simulations/suite_summary.csv
    simulations/results/*_metrics.csv  (per-bench cwd)
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS = [
    "universal_architecture_bench.py",
    "compression_trinity_bench.py",
    "micro_pipeline_bench.py",
    "mtp_ssd_speculation_bench.py",
    "thermal_duty_cycle_bench.py",
    "io_uring_simulation_bench.py",
    "e2e_token_throughput_bench.py",
    "ssd_native_optimizations_bench.py",
    "engram_architecture_bench.py",
    "engram_nine_ssd_bench.py",
    "engram_write_buffer_bench.py",
    "pcie_duplex_bench.py",
    "numa_topology_bench.py",
    "semantic_prefetch_bench.py",
    "ouroboros_mamba_bench.py",
    "csd_in_situ_bench.py",
    "unified_decompression_bench.py",
    "gate_prefetch_bench.py",
    "die_thermal_balance_bench.py",
    "ecc_bypass_bench.py",
    "ngram_weight_cache_bench.py",
    "asymmetric_raid_power_bench.py",
    "lora_delta_streaming_bench.py",
    "erasure_coding_bench.py",
    "temporal_weight_locality_bench.py",
    "ssd_speculative_prefetch_bench.py",
    "expert_prefetch_bench.py",
    "ssm_speculation_cache_bench.py",
    "residual_channel_prefetch_bench.py",
    "activation_sparse_skip_bench.py",
    "async_layer_execution_bench.py",
    "compression_path_comparison_bench.py",
    "h100_vram_baseline_bench.py",
    "mcts_state_tree_bench.py",
    "csd_zts_scan_bench.py",
    "cow_state_fork_bench.py",
    "delta_log_state_bench.py",
    "pslc_state_endurance_bench.py",
    "read_disturb_rotation_bench.py",
    "mimo_ssm_pruning_bench.py",
    "hilos_validation_bench.py",
    "expert_trajectory_bench.py",
    "kernel_bypass_gds_bench.py",
    "byte_express_inline_bench.py",
    "plane_parallelism_bench.py",
    "yggdrasil_tree_decode_bench.py",
    "mamba2_fused_kernel_bench.py",
    "batch_scaling_bench.py",
    "prefix_state_library_bench.py",
    "layer_aware_prefetch_schedule_bench.py",
    "heterogeneous_chunk_schedule_bench.py",
]

SUITE_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SUITE_DIR / "benches"
RESULTS_DIR = SUITE_DIR / "results"
LOG_FILE = SUITE_DIR / "suite_output.log"
SUMMARY_CSV = SUITE_DIR / "suite_summary.csv"


def list_benchmarks() -> None:
    print(f"{'Benchmark':<45} {'Status':<10}")
    print("-" * 55)
    for bench in BENCHMARKS:
        path = BENCH_DIR / bench
        exists = "FOUND" if path.exists() else "MISSING"
        print(f"  {bench:<45} {exists:<10}")
    found = sum(1 for b in BENCHMARKS if (BENCH_DIR / b).exists())
    print(f"\nTotal: {len(BENCHMARKS)} | Found: {found}")


def run_benchmark(bench_name: str, log_fh) -> tuple[bool, float]:
    bench_path = BENCH_DIR / bench_name
    if not bench_path.exists():
        msg = f"[SKIP] {bench_name}: file not found"
        print(msg)
        log_fh.write(msg + "\n\n")
        return False, 0.0

    print(f"\n{'='*80}\n  RUNNING: {bench_name}\n{'='*80}")
    log_fh.write(f"\n{'='*80}\n  RUNNING: {bench_name}\n{'='*80}\n")

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(bench_path)],
            cwd=str(RESULTS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )
        duration = time.time() - start
        if proc.stdout:
            log_fh.write(proc.stdout)
            for line in proc.stdout.strip().split("\n")[-20:]:
                print(line)
        if proc.stderr:
            log_fh.write(f"\n--- STDERR ---\n{proc.stderr}\n")
        success = proc.returncode == 0
        status = "PASS" if success else f"FAIL (rc={proc.returncode})"
        print(f"  [{status}] {bench_name} ({duration:.1f}s)")
        log_fh.write(f"\n  [{status}] {bench_name} ({duration:.1f}s)\n")
        return success, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"[TIMEOUT] {bench_name}")
        return False, duration
    except Exception as e:
        duration = time.time() - start
        print(f"[ERROR] {bench_name}: {e}")
        return False, duration


def main() -> None:
    if "--list" in sys.argv:
        list_benchmarks()
        return

    skip_list = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--skip="):
            skip_list.add(arg.split("=", 1)[1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("SSD feasibility simulation suite (not the RWKV engine)")
    print(f"  benches: {BENCH_DIR}")
    print(f"  results: {RESULTS_DIR}")

    results = []
    total_start = time.time()
    with open(LOG_FILE, "w", encoding="utf-8") as log_fh:
        for bench in BENCHMARKS:
            if bench in skip_list:
                results.append((bench, "SKIPPED", 0.0))
                continue
            ok, duration = run_benchmark(bench, log_fh)
            results.append((bench, "PASS" if ok else "FAIL", duration))

    total_duration = time.time() - total_start
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Benchmark", "Status", "Duration_s"])
        for bench, status, duration in results:
            w.writerow([bench, status, f"{duration:.2f}"])

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\nDone: {passed} passed, {failed} failed, {total_duration:.1f}s")
    print(f"  log: {LOG_FILE}")
    print(f"  summary: {SUMMARY_CSV}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
