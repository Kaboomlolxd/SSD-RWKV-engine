import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: HETEROGENEOUS CHUNK SCHEDULE (New idea candidate)
# =====================================================================
# Models per-layer choice of chunk size: large chunks (e.g. 128 KiB aligned)
# maximize NAND internal parallelism; smaller chunks improve overlap with
# compute in micro-pipelines. Fixed-K micro-pipeline is a special case.
#
# This is a thesis-side hypothesis bench, not a measured FTL trace.
#
# CONFLICTS AND NON-STACKING:
#   - Mutually exclusive with "uniform K" micro-pipeline for the same layer:
#     pick one policy per scenario, do not add deltas to micro_pipeline_bench.
#   - CHEOPS-style 128 KiB reads and micro-pipeline small chunks answer different
#     questions; this bench highlights tension, not a guaranteed win for hybrid.
#
# SELF-CRITIQUE:
#   - Current defaults often make uniform K=16 fastest; hybrid overlap_bonus is
#     arbitrary. Do not claim "128 KiB wins" without device traces.
#   - No FTL, no plane/channel modeling, no read amplification from alignment.
# =====================================================================

np.random.seed(42)

NUM_LAYERS = 80
SSD_BW_GBS = 22.0
LAYER_COMPRESSED_MB = 25.6 / 10.0
NVME_QD1_US = 66.0
IO_URING_SQE_US = 2.0

# NAND "sweet spot" read size (MB) vs micro-pipeline chunk (MB)
LARGE_CHUNK_MB = 0.125  # 128 KiB
SMALL_CHUNK_MB = LAYER_COMPRESSED_MB / 16.0  # K=16 uniform micro-pipeline


def time_layer_fixed_k(chunks: int, chunk_mb: float) -> float:
    """Total time to read one layer as `chunks` equal pieces."""
    per_chunk_io = (chunk_mb / 1024.0) / SSD_BW_GBS
    overhead_per_chunk = (NVME_QD1_US + IO_URING_SQE_US) * 1e-6
    return chunks * (per_chunk_io + overhead_per_chunk)


def time_layer_heterogeneous(num_large: int, num_small: int) -> float:
    """
    First portion of layer uses large 128KiB reads; remainder uses small chunks
    for overlap with tail compute (simplified: sequential sum, overlap bonus).
    """
    # Split layer: alpha fraction by large chunks, rest by small
    large_part_mb = num_large * LARGE_CHUNK_MB
    small_part_mb = max(0.0, LAYER_COMPRESSED_MB - large_part_mb)
    small_chunks = max(1, int(np.ceil(small_part_mb / SMALL_CHUNK_MB)))

    t_large = num_large * ((LARGE_CHUNK_MB / 1024.0) / SSD_BW_GBS + (NVME_QD1_US + IO_URING_SQE_US) * 1e-6)
    t_small = time_layer_fixed_k(small_chunks, SMALL_CHUNK_MB)
    # Overlap bonus: large reads hide start of small phase (fractional)
    overlap_bonus = 0.12
    return t_large + t_small * (1.0 - overlap_bonus)


def run_heterogeneous_chunk_schedule_benchmark():
    print("=" * 90)
    print(" THESIS: HETEROGENEOUS CHUNK SCHEDULE (New idea candidate)")
    print(" 128 KiB-aligned bulk vs K-way micro-pipeline vs hybrid split")
    print("=" * 90)

    # Fixed K=16 uniform
    k = 16
    t_uniform = time_layer_fixed_k(k, LAYER_COMPRESSED_MB / k)

    # Pure large chunks only (ceil layer to multiple of 128KiB)
    n_large = int(np.ceil(LAYER_COMPRESSED_MB / LARGE_CHUNK_MB))
    t_large_only = time_layer_fixed_k(n_large, LARGE_CHUNK_MB)

    # Hybrid: first 40% of layer by 128KiB reads, rest micro-pipelined
    bytes_large_target = LAYER_COMPRESSED_MB * 0.4
    n_hybrid_large = max(1, int(np.ceil(bytes_large_target / LARGE_CHUNK_MB)))
    t_hybrid = time_layer_heterogeneous(n_hybrid_large, 0)

    print(f"\n  Layer compressed size: {LAYER_COMPRESSED_MB:.4f} MB")
    print(f"  Small chunk (K=16):    {SMALL_CHUNK_MB:.4f} MB")
    print(f"  Large chunk:           {LARGE_CHUNK_MB*1024:.0f} KiB")

    print(f"\n  Per-layer read time (seconds):")
    print(f"    Uniform K=16:        {t_uniform:.6f}")
    print(f"    All large (128KiB):  {t_large_only:.6f}")
    print(f"    Hybrid (40% large):  {t_hybrid:.6f}")

    # Pick best as recommendation (not always large-only: overhead dominates if too many tiny cmds)
    candidates = {
        "uniform_k16": t_uniform,
        "all_large_128kib": t_large_only,
        "hybrid_40pct_large": t_hybrid,
    }
    best_name = min(candidates, key=candidates.get)
    print(f"\n  Fastest (model): {best_name} ({candidates[best_name]:.6f} s/layer)")

    total_uniform = t_uniform * NUM_LAYERS
    total_best = candidates[best_name] * NUM_LAYERS
    speedup = total_uniform / max(1e-12, total_best)

    rows = [
        ("num_layers", NUM_LAYERS, ""),
        ("layer_compressed_mb", f"{LAYER_COMPRESSED_MB:.6f}", ""),
        ("ssd_bw_gbs", SSD_BW_GBS, ""),
        ("time_per_layer_uniform_k16_s", f"{t_uniform:.8f}", ""),
        ("time_per_layer_all_large_s", f"{t_large_only:.8f}", ""),
        ("time_per_layer_hybrid_s", f"{t_hybrid:.8f}", ""),
        ("best_strategy", best_name, ""),
        ("full_model_io_speedup_vs_uniform", f"{speedup:.4f}", "uniform K=16 baseline"),
        ("evidence_tier", "Simulated / new idea", "Needs real FTL + device traces"),
    ]

    with open("heterogeneous_chunk_schedule_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        for r in rows:
            w.writerow(r)

    print("\n[+] Wrote heterogeneous_chunk_schedule_metrics.csv")


if __name__ == "__main__":
    run_heterogeneous_chunk_schedule_benchmark()
