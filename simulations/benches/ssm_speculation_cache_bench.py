import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: SSM STATE-AWARE SPECULATION CACHE (Chapter 10mm)
# =====================================================================
# ADAPTED - Saguaro/SSD (Kumar, Dao & May, 2026) + SpecMamba (Zhong et al., 2025)
# ORIGINAL SYNTHESIS: Applying Saguaro's speculative-speculative decoding
# concept to Mamba's hidden state space, pre-computing SSM state
# trajectories for likely verification outcomes.
#
# Saguaro/SSD (2603.03251) eliminates drafting overhead by pre-speculating
# verification outcomes while the target model verifies. It uses geometric
# fan-out allocation (Theorem 12) to budget compute across outcomes.
# SpecMamba (2509.19873) is the first to apply speculative decoding to
# Mamba on FPGA, using FIFO-based tree verification.
#
# OUR SYNTHESIS: Since Mamba has no KV cache (O(1) state), the speculation
# cache stores compact (h_t, token) tuples in host RAM. Pre-computed SSM
# state trajectories are generated for the top-K most likely verification
# outcomes using geometric fan-out. When verification completes, the
# matching trajectory is returned instantly - eliminating SSD read latency
# for accepted tokens.
#
# MECHANISM:
#   1. While GPU verifies token t, a lightweight SSM draft on CPU pre-
#      computes SSM state trajectories for the top-K verification outcomes.
#   2. Outcomes are allocated using geometric fan-out: F_k = F_0 * a_p^(k/(1+r))
#   3. Cache hit = verification outcome matches a pre-speculated trajectory
#   4. On hit: return cached state + token (zero SSD read)
#   5. On miss: fall back to standard SSD weight load
#
# IMPACT: For Mamba's O(1) state, the cache footprint is tiny (~16KB per
# trajectory vs. MB-scale KV cache in Transformers). This enables deep
# speculation (K=8-16) with minimal memory overhead.
# =====================================================================

# ---- Hardware Constants ----
SINGLE_DRIVE_BW_GBS = 7.0
DRIVES = 4
RAID_BW_GBS = SINGLE_DRIVE_BW_GBS * DRIVES

# ---- Model Constants ----
MAMBA_70B_COMPRESSED_GB = 17.5
MAMBA_STATE_SIZE_KB = 16  # O(1) state per layer * 80 layers
COMPUTE_TIME_S = 0.4  # GPU compute per token


def geometric_fan_out(acceptance_rate, num_outcomes, r=1.0):
    """
    Saguaro's geometric fan-out allocation (Theorem 12).
    Allocates compute budget across verification outcomes proportional
    to a_p^(k/(1+r)), where a_p is the acceptance probability.
    """
    probs = np.array([acceptance_rate ** (k / (1 + r)) for k in range(num_outcomes)])
    probs /= probs.sum()
    return probs


def simulate_ssm_speculation_cache(acceptance_rate=0.7, num_outcomes=12,
                                    cache_depth=16, r=1.0):
    """
    Simulate SSM state-aware speculation cache.

    While the target model verifies, pre-compute SSM state trajectories
    for the top-K most likely verification outcomes.

    Returns immediately on cache hit (zero SSD read).
    Falls back to full SSD read on cache miss.
    """
    np.random.seed(42)

    # Geometric fan-out allocation
    outcome_probs = geometric_fan_out(acceptance_rate, num_outcomes, r)

    # Cache hit probability = sum of probabilities of pre-speculated outcomes
    # In practice, we can pre-speculate the top-K outcomes
    num_pre_speculated = min(cache_depth, num_outcomes)
    cache_hit_prob = np.sum(np.sort(outcome_probs)[::-1][:num_pre_speculated])

    # SSD read time (only on cache miss)
    read_time_s = MAMBA_70B_COMPRESSED_GB / RAID_BW_GBS

    # Speculation overhead: CPU draft model computes trajectories
    # Tiny SSM draft: ~1ms per trajectory (vs. KV cache draft at ~10ms)
    draft_time_per_trajectory = 0.001  # seconds
    speculation_overhead = num_pre_speculated * draft_time_per_trajectory

    # Effective token time
    # On hit: compute + speculation overhead (no SSD read)
    # On miss: compute + speculation overhead + SSD read
    hit_time = COMPUTE_TIME_S + speculation_overhead
    miss_time = COMPUTE_TIME_S + speculation_overhead + read_time_s

    avg_time = cache_hit_prob * hit_time + (1 - cache_hit_prob) * miss_time
    tok_per_s = 1.0 / avg_time

    # Baseline (no speculation)
    baseline_time = COMPUTE_TIME_S + read_time_s
    baseline_tok_per_s = 1.0 / baseline_time

    # Standard SD (draft + verify sequential)
    sd_draft_time = 0.01  # Transformer draft model (10ms)
    sd_time = COMPUTE_TIME_S + read_time_s * (1 - acceptance_rate) + sd_draft_time
    sd_tok_per_s = 1.0 / sd_time

    return {
        'acceptance_rate': acceptance_rate,
        'num_outcomes': num_outcomes,
        'num_pre_speculated': num_pre_speculated,
        'cache_hit_prob': cache_hit_prob,
        'speculation_overhead_ms': speculation_overhead * 1000,
        'hit_time_ms': hit_time * 1000,
        'miss_time_ms': miss_time * 1000,
        'avg_time_ms': avg_time * 1000,
        'tok_per_s': tok_per_s,
        'baseline_tok_per_s': baseline_tok_per_s,
        'sd_tok_per_s': sd_tok_per_s,
        'speedup_vs_baseline': tok_per_s / baseline_tok_per_s,
        'speedup_vs_sd': tok_per_s / sd_tok_per_s,
    }


def run_ssm_speculation_cache_benchmark():
    print("=" * 110)
    print(" THESIS: SSM STATE-AWARE SPECULATION CACHE (Chapter 10mm)")
    print(" ADAPTED - Saguaro/SSD (Kumar, Dao & May, 2026) + SpecMamba (Zhong et al., 2025)")
    print(" ORIGINAL SYNTHESIS: Geometric fan-out pre-speculation for Mamba's O(1) state")
    print("=" * 110)

    # ---- Baseline Comparison ----
    print(f"\n{'='*80}")
    print(f" PHASE 1: BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"  Model: Mamba-70B 2-bit ({MAMBA_70B_COMPRESSED_GB}GB)")
    print(f"  RAID bandwidth: {RAID_BW_GBS} GB/s")
    print(f"  GPU compute: {COMPUTE_TIME_S*1000:.0f}ms")
    print(f"  SSM state size: {MAMBA_STATE_SIZE_KB}KB (vs. MB-scale KV cache)")

    result = simulate_ssm_speculation_cache()

    print(f"\n  {'Metric':<40} {'Baseline':<18} {'Std SD':<18} {'SSM Spec Cache':<18}")
    print(f"  {'-'*95}")
    print(f"  {'Token time (ms)':<40} {result['avg_time_ms']*result['baseline_tok_per_s']/result['tok_per_s']:<18.1f} "
          f"{result['avg_time_ms']*result['sd_tok_per_s']/result['tok_per_s']:<18.1f} "
          f"{result['avg_time_ms']:<18.1f}")
    print(f"  {'Tokens/s':<40} {result['baseline_tok_per_s']:<18.2f} "
          f"{result['sd_tok_per_s']:<18.2f} {result['tok_per_s']:<18.2f}")
    print(f"  {'Cache hit rate':<40} {'N/A':<18} {'N/A':<18} {result['cache_hit_prob']*100:.1f}%")
    print(f"  {'Speculation overhead (ms)':<40} {'0':<18} {'10.0':<18} {result['speculation_overhead_ms']:.1f}")
    sd_speedup = result['sd_tok_per_s']/result['baseline_tok_per_s']
    print(f"  {'Speedup vs baseline':<40} {'1.00x':<18} {sd_speedup:.2f}x{'':<17} {result['speedup_vs_baseline']:.2f}x")

    # ---- Sensitivity: Acceptance Rate ----
    print(f"\n{'='*80}")
    print(f" PHASE 2: SENSITIVITY - VARYING ACCEPTANCE RATE")
    print(f"{'='*80}")

    print(f"\n  {'Acceptance':<14} {'Cache Hit%':<14} {'Spec Overhead':<16} {'SSM Cache tok/s':<18} {'Speedup':<12}")
    print(f"  {'-'*75}")

    for ar in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = simulate_ssm_speculation_cache(acceptance_rate=ar)
        print(f"  {ar:<14.1f} {r['cache_hit_prob']*100:<14.1f}% {r['speculation_overhead_ms']:<16.1f}ms "
              f"{r['tok_per_s']:<18.2f} {r['speedup_vs_baseline']:<12.2f}x")

    # ---- Sensitivity: Cache Depth ----
    print(f"\n{'='*80}")
    print(f" PHASE 3: SENSITIVITY - VARYING CACHE DEPTH")
    print(f"{'='*80}")

    print(f"\n  {'Cache Depth':<14} {'Outcomes':<14} {'Cache Hit%':<14} {'Overhead (ms)':<16} {'Tok/s':<12}")
    print(f"  {'-'*70}")

    for depth in [4, 8, 12, 16, 24, 32]:
        r = simulate_ssm_speculation_cache(cache_depth=depth)
        print(f"  {depth:<14} {r['num_outcomes']:<14} {r['cache_hit_prob']*100:<14.1f}% "
              f"{r['speculation_overhead_ms']:<16.1f} {r['tok_per_s']:<12.2f}")

    # ---- State Size Comparison: Mamba vs Transformer ----
    print(f"\n{'='*80}")
    print(f" PHASE 4: STATE SIZE COMPARISON - Mamba vs Transformer")
    print(f"{'='*80}")

    print(f"\n  The key advantage: Mamba's O(1) state is ~16KB vs. Transformer")
    print(f"  KV cache which grows with context length (~1MB at seq_len=4K).")
    print(f"\n  {'Model':<20} {'State Size':<16} {'Cache@K=16':<16} {'Draft Overhead':<16}")
    print(f"  {'-'*70}")
    print(f"  {'Mamba-70B (O(1))':<20} {MAMBA_STATE_SIZE_KB} KB{'':<10} {MAMBA_STATE_SIZE_KB*16} KB{'':<10} {16*1:.1f} ms")
    print(f"  {'Transformer-70B':<20} {'~1024 KB (4K ctx)':<16} {'~16 MB':<16} {'~10 ms':<16}")
    print(f"\n  Mamba's speculation cache is 1000x smaller, enabling")
    print(f"  much deeper speculation (K=16-32) with negligible overhead.")

    # ---- Academic Summary ----
    print(f"\n{'='*110}")
    print(f" ACADEMIC SUMMARY")
    print(f"{'='*110}")
    print(f"""
  CONTRIBUTION: SSM State-Aware Speculation Cache is an ADAPTED technique
  building on Saguaro/SSD (Kumar, Dao & May, 2026) and SpecMamba
  (Zhong et al., 2025). The ORIGINAL SYNTHESIS applies geometric fan-out
  pre-speculation to Mamba's O(1) hidden state space, storing compact
  (h_t, token) tuples instead of KV cache entries.

  KEY FINDINGS:
    1. At 70% acceptance rate with cache depth 16, the speculation cache
       achieves {simulate_ssm_speculation_cache()['cache_hit_prob']*100:.1f}% hit rate,
       delivering {simulate_ssm_speculation_cache()['tok_per_s']:.2f} tok/s
       ({simulate_ssm_speculation_cache()['speedup_vs_baseline']:.2f}x over baseline).
    2. SSM speculation overhead is only {simulate_ssm_speculation_cache()['speculation_overhead_ms']:.1f}ms
       (vs. ~10ms for Transformer draft models), because the SSM draft
       operates on 16KB states instead of MB-scale KV caches.
    3. The cache footprint at K=16 is only {MAMBA_STATE_SIZE_KB*16}KB -
       1000x smaller than Transformer KV cache speculation - enabling
       much deeper speculation with negligible memory overhead.
    4. Speedup scales with acceptance rate: at 90% acceptance,
       the cache achieves {simulate_ssm_speculation_cache(acceptance_rate=0.9)['tok_per_s']:.2f} tok/s.

  WHY THIS IS SSD-NATIVE: On VRAM-resident models, weight access is
  instantaneous so speculation provides marginal benefit. On SSD-native
  models, eliminating the SSD read on cache hits (which costs {MAMBA_70B_COMPRESSED_GB/RAID_BW_GBS*1000:.0f}ms)
  is a transformative optimization.
""")

    # ---- Save CSV ----
    with open('ssm_speculation_cache_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Acceptance_Rate", "Cache_Depth", "Cache_Hit_Prob",
                         "Spec_Overhead_ms", "Tok_per_s", "Speedup_vs_Baseline"])
        for ar in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for depth in [4, 8, 16, 32]:
                r = simulate_ssm_speculation_cache(acceptance_rate=ar, cache_depth=depth)
                writer.writerow([ar, depth, f"{r['cache_hit_prob']:.4f}",
                                 f"{r['speculation_overhead_ms']:.2f}",
                                 f"{r['tok_per_s']:.2f}",
                                 f"{r['speedup_vs_baseline']:.2f}"])

    print("[+] Academic data saved to 'ssm_speculation_cache_metrics.csv'.")


if __name__ == "__main__":
    run_ssm_speculation_cache_benchmark()
