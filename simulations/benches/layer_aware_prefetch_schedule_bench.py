import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: LAYER-AWARE GLOBAL PREFETCH SCHEDULE (Inspired)
# =====================================================================
# Inspired by: PreScope (layer-aware prefetch + cross-layer scheduling),
# Layered Prefill, AMoE async patterns. This bench compares:
#   A) Naive: prefetch next layer only when current layer finishes (local)
#   B) Global: scheduler prefetches layer L+k based on predicted critical path
#
# Simplified SSD model: bandwidth shared across concurrent prefetches.
#
# CONFLICTS AND NON-STACKING:
#   - Does NOT multiply with micro_pipeline_bench.py: that bench is sub-layer
#     overlap; this one is layer-granularity. One SSD budget; model contention.
#   - Overlaps conceptually with async_layer_execution_bench.py; use one story
#     for a given thesis section, not two independent multipliers.
#   - Do not stack with gate_prefetch / speculative prefetch as independent
#     speedups without a unified bandwidth-limited schedule.
#
# SELF-CRITIQUE:
#   - overlap_efficiency is hand-tuned, not measured.
#   - No GPU compute time; no io_uring queue depth; no multi-stream realism.
# =====================================================================

np.random.seed(42)

NUM_LAYERS = 80
LAYER_WEIGHT_MB = 25.6 / 10.0
SSD_BW_GBS = 22.0
LOOKAHEAD_LAYERS = 3  # global scheduler issues reads for L+1..L+lookahead


def layer_read_time_mb(mb: float) -> float:
    return (mb / 1024.0) / SSD_BW_GBS


def simulate_naive(num_tokens: int = 100):
    """One token: strictly sequential layer loads; no overlap across layers beyond one stream."""
    total_s = 0.0
    for _ in range(num_tokens):
        for _ in range(NUM_LAYERS):
            total_s += layer_read_time_mb(LAYER_WEIGHT_MB)
    return total_s


def simulate_global_lookahead(num_tokens: int = 100, lookahead: int = LOOKAHEAD_LAYERS):
    """
    Pipelined prefetch: while GPU works on layer L, SSD streams up to `lookahead`
    future layers. Effective time per layer approaches max(io_one_layer, compute)
    when bandwidth is not saturated; here compute is tiny so overlap reduces
    idle gaps between layer boundaries (modeled as fractional overlap).
    """
    # Overlap factor: not 1:1 with lookahead due to bandwidth sharing
    overlap_efficiency = 0.55 + 0.08 * min(lookahead, 4)
    per_layer_effective = layer_read_time_mb(LAYER_WEIGHT_MB) * (1.0 - overlap_efficiency * 0.35)
    total_s = 0.0
    for _ in range(num_tokens):
        total_s += NUM_LAYERS * per_layer_effective
    return total_s


def run_layer_aware_prefetch_schedule_benchmark():
    print("=" * 90)
    print(" THESIS: LAYER-AWARE GLOBAL PREFETCH SCHEDULE (Inspired)")
    print(" Naive sequential layer I/O vs lookahead prefetch scheduling")
    print("=" * 90)

    naive_s = simulate_naive(50)
    global_s = simulate_global_lookahead(50, LOOKAHEAD_LAYERS)
    speedup = naive_s / max(1e-12, global_s)

    print(f"\n  Layers: {NUM_LAYERS}, layer weight (compressed): {LAYER_WEIGHT_MB:.2f} MB")
    print(f"  SSD sequential BW: {SSD_BW_GBS} GB/s")
    print(f"  Lookahead depth: {LOOKAHEAD_LAYERS}")
    print(f"\n  Total I/O time (50 tokens, model):")
    print(f"    Naive sequential:     {naive_s*1000:.2f} ms")
    print(f"    Global lookahead:     {global_s*1000:.2f} ms")
    print(f"    Speedup:              {speedup:.3f}x")

    rows = [
        ("num_layers", NUM_LAYERS, ""),
        ("layer_weight_mb_compressed", f"{LAYER_WEIGHT_MB:.4f}", ""),
        ("ssd_bw_gbs", SSD_BW_GBS, ""),
        ("lookahead_layers", LOOKAHEAD_LAYERS, ""),
        ("total_ms_naive_50_tokens", f"{naive_s*1000:.4f}", ""),
        ("total_ms_global_50_tokens", f"{global_s*1000:.4f}", ""),
        ("speedup_global_vs_naive", f"{speedup:.4f}", ""),
        ("evidence_tier", "Simulated / inspired", "Abstract scheduler; not a measured runtime"),
    ]

    with open("layer_aware_prefetch_schedule_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        for r in rows:
            w.writerow(r)

    print("\n[+] Wrote layer_aware_prefetch_schedule_metrics.csv")


if __name__ == "__main__":
    run_layer_aware_prefetch_schedule_benchmark()
