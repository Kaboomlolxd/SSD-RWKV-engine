import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: PREFIX-STATE LIBRARY (Inspired)
# =====================================================================
# Inspired by: Marconi (hybrid LLM prefix caching), vLLM Mamba prefix
# caching, and recurrent-state checkpointing discussions for exact-match
# reuse of common system prompts.
#
# This bench does NOT claim novelty. It models cross-request savings when
# identical prompt prefixes recur and the server can restore a saved
# recurrent state (Mamba/RWKV) instead of re-scanning the prefix from SSD.
#
# METRICS:
#   - effective_tokens_saved: tokens not recomputed on cache hit
#   - hit_rate: fraction of requests with exact prefix match in library
#   - latency_ms_no_library vs latency_ms_with_library
#
# CONFLICTS AND NON-STACKING (read before citing numbers):
#   - Do NOT multiply these savings into e2e_token_throughput_bench.py steady-decode
#     tok/s unless the scenario explicitly defines repeated-prefix traffic.
#   - Not the same as "instant state parking" / COW fork / delta logs: those are
#     session lifecycle; this bench is cross-request prefix reuse (Marconi-style).
#   - Prior art: cite Marconi, vLLM Mamba prefix caching, hybrid prefix caching.
#
# SELF-CRITIQUE:
#   - Hit rate and FRACTION_REPEATED_PROMPTS are synthetic, not trace-driven.
#   - Prefill cost model is simplistic (full layer sweep per token of prefill).
#   - No eviction, admission, or memory budget for the library.
# =====================================================================

np.random.seed(42)

NUM_LAYERS = 80
COMPRESSED_LAYER_MB = 25.6 / 10.0  # ~2.56 MB per layer after 10x trinity
SSD_BW_GBS = 22.0
STATE_SIZE_MB = 16.0  # representative Mamba recurrent state snapshot
TOKENS_PER_LAYER_PREFILL = 32  # rough: prefill work scales with prefix length

# Library parameters
LIBRARY_SIZE_ENTRIES = 512  # max distinct cached prefixes
MEAN_PREFIX_TOKENS = 256
PREFIX_TOKEN_STDDEV = 128
FRACTION_REPEATED_PROMPTS = 0.35  # share of traffic that reuses a known prefix


def simulate_request(prefix_tokens: int, cache_hit: bool):
    """Time to serve one decode step after prefill; prefill cost if miss."""
    # Full prefill: read all layer weights sequentially for prefix (simplified)
    prefill_io_s = (COMPRESSED_LAYER_MB * NUM_LAYERS / 1024.0) / SSD_BW_GBS
    prefill_compute_s = prefill_io_s * 0.4  # overlap not modeled here for clarity

    if cache_hit:
        # Restore state from RAM: one memcpy-scale cost, no full SSD sweep for prefix
        restore_s = (STATE_SIZE_MB / 1024.0) / (SSD_BW_GBS * 4) + 0.0001  # tiny
        prefill_total_s = restore_s
        tokens_saved = prefix_tokens
    else:
        prefill_total_s = prefill_io_s + prefill_compute_s
        tokens_saved = 0

    # Single decode step (one token) — same either way
    decode_s = (COMPRESSED_LAYER_MB / 1024.0) / SSD_BW_GBS / 16.0  # micro-pipelined layer chunk
    return prefill_total_s, decode_s, tokens_saved


def run_prefix_state_library_benchmark(num_requests: int = 2000):
    print("=" * 90)
    print(" THESIS: PREFIX-STATE LIBRARY (Inspired - Marconi / vLLM APC lineage)")
    print(" Cross-request reuse of recurrent state for repeated system prompts")
    print("=" * 90)

    hits = 0
    total_tokens_saved = 0
    latency_no_lib = []
    latency_with_lib = []

    # Simulate: each request draws prefix length; with prob FRACTION_REPEATED, hits library
    for i in range(num_requests):
        prefix_tokens = max(32, int(np.random.normal(MEAN_PREFIX_TOKENS, PREFIX_TOKEN_STDDEV)))
        # Hash bucket: repeated traffic collides with probability related to library + repeat rate
        cache_hit = np.random.random() < (
            FRACTION_REPEATED_PROMPTS * min(1.0, LIBRARY_SIZE_ENTRIES / 4096.0)
        )
        if cache_hit:
            hits += 1

        pre_a, dec_a, _ = simulate_request(prefix_tokens, cache_hit=False)
        pre_b, dec_b, saved = simulate_request(prefix_tokens, cache_hit=cache_hit)

        latency_no_lib.append((pre_a + dec_a) * 1000)
        latency_with_lib.append((pre_b + dec_b) * 1000)
        total_tokens_saved += saved

    hit_rate = hits / num_requests
    mean_lat_no = float(np.mean(latency_no_lib))
    mean_lat_yes = float(np.mean(latency_with_lib))
    speedup = mean_lat_no / max(1e-9, mean_lat_yes)

    print(f"\n  Requests simulated:     {num_requests}")
    print(f"  Library capacity:       {LIBRARY_SIZE_ENTRIES} prefix entries")
    print(f"  Cache hit rate (model): {hit_rate*100:.2f}%")
    print(f"  Total prefix tokens saved (sum): {total_tokens_saved:,}")
    print(f"  Mean latency no library:  {mean_lat_no:.2f} ms (prefill+1 decode)")
    print(f"  Mean latency w/ library:  {mean_lat_yes:.2f} ms")
    print(f"  Speedup (mean latency):   {speedup:.2f}x")

    rows = [
        ("num_requests", num_requests, ""),
        ("library_capacity", LIBRARY_SIZE_ENTRIES, ""),
        ("cache_hit_rate", f"{hit_rate:.4f}", ""),
        ("total_tokens_saved_sum", total_tokens_saved, ""),
        ("mean_latency_ms_no_library", f"{mean_lat_no:.4f}", ""),
        ("mean_latency_ms_with_library", f"{mean_lat_yes:.4f}", ""),
        ("mean_latency_speedup", f"{speedup:.4f}", ""),
        ("evidence_tier", "Simulated / inspired", "Not deployment measurement"),
    ]

    with open("prefix_state_library_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        for r in rows:
            w.writerow(r)

    print("\n[+] Wrote prefix_state_library_metrics.csv")
    return {r[0]: r[1] for r in rows}


if __name__ == "__main__":
    run_prefix_state_library_benchmark()
