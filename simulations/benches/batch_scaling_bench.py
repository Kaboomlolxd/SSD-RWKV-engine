import pandas as pd
import numpy as np

def simulate_batch_scaling():
    print("Simulating Batch Scaling for SSD-Native Mamba Inference...")
    print("(Total system tok/s vs per-user tok/s across batch sizes)\n")
    
    # Model: 70B Mamba, Q4 compressed + 10x compression trinity
    model_compressed_gb = 14.0  # Q4
    effective_payload_gb = 1.4  # After 10x compression
    
    # Hardware: 4x Gen5 RAID
    ssd_bw_gbps = 22.0  # Sustained
    gpu_tflops = 989.0  # H100 FP16
    
    # MIMO g=4
    mimo_g = 4
    
    # Fused kernel speedup (from mamba2_fused_kernel_bench.py)
    fused_decode_speedup = 1.28
    fused_prefill_speedup = 2.09
    
    # Base times (from E2E model)
    base_io_ms = (effective_payload_gb / ssd_bw_gbps) * 1000 / mimo_g  # ~15.9ms
    base_compute_ms = 3.5  # Compute per token on H100
    base_overhead_ms = 1.6  # Kernel launches, sync, etc.
    
    # With fused kernel
    fused_compute_ms = base_compute_ms / fused_decode_speedup
    fused_overhead_ms = base_overhead_ms / fused_decode_speedup
    
    print("=" * 90)
    print("Batch Scaling: SSD-Native Mamba 70B (4x Gen5 RAID, H100, MIMO g=4)")
    print("=" * 90)
    print(f"{'Batch':<8} {'Total tok/s':<14} {'Per-User tok/s':<16} {'Latency (ms)':<14} {'Bottleneck':<20} {'GPU Util %':<10}")
    print("-" * 90)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []
    
    for batch in batch_sizes:
        # IO time: batched reads don't help much (still need to stream all weights)
        # But with MIMO, we amortize across g tokens per batch element
        io_ms = base_io_ms  # IO is per-sweep, not per-batch (weights shared)
        
        # Compute time: scales linearly with batch (more tokens to process)
        # But GPU parallelism gives sublinear scaling up to a point
        if batch <= 8:
            compute_scaling = batch ** 0.85  # Good parallelism
        elif batch <= 64:
            compute_scaling = batch ** 0.92  # Diminishing returns
        else:
            compute_scaling = batch ** 0.97  # Near-linear (memory bandwidth bound)
        
        compute_ms = fused_compute_ms * compute_scaling
        
        # Overhead: kernel launches scale sublinearly (batched kernels)
        overhead_ms = fused_overhead_ms * (batch ** 0.5)
        
        # Total time per sweep
        total_ms = io_ms + compute_ms + overhead_ms
        
        # Tokens per sweep = batch * MIMO group
        tokens_per_sweep = batch * mimo_g
        
        # Total tok/s
        total_tok_s = tokens_per_sweep / (total_ms / 1000.0)
        
        # Per-user tok/s
        per_user_tok_s = total_tok_s / batch
        
        # Latency per token (time to generate one token for one user)
        latency_ms = total_ms / mimo_g
        
        # GPU utilization
        gpu_util = min(95.0, (compute_ms / total_ms) * 100)
        
        # Bottleneck
        if io_ms > compute_ms * 1.5:
            bottleneck = "IO-bound"
        elif compute_ms > io_ms * 1.5:
            bottleneck = "Compute-bound"
        else:
            bottleneck = "Balanced"
        
        print(f"{batch:<8} {total_tok_s:<14.1f} {per_user_tok_s:<16.1f} {latency_ms:<14.1f} {bottleneck:<20} {gpu_util:<10.0f}")
        
        results.append({
            "batch_size": batch,
            "total_tok_s": total_tok_s,
            "per_user_tok_s": per_user_tok_s,
            "latency_ms": latency_ms,
            "bottleneck": bottleneck,
            "gpu_util_pct": gpu_util,
            "io_ms": io_ms,
            "compute_ms": compute_ms,
            "overhead_ms": overhead_ms,
        })
    
    # Also show without fused kernel for comparison
    print(f"\n{'='*90}")
    print("Impact of Fused Kernels on Batch Scaling")
    print("=" * 90)
    print(f"{'Batch':<8} {'Total (Unfused)':<16} {'Total (Fused)':<16} {'Speedup':<10} {'Per-User (Fused)':<18}")
    print("-" * 90)
    
    for r in results:
        batch = r["batch_size"]
        # Unfused compute and overhead
        unfused_compute_ms = base_compute_ms * (batch ** (0.85 if batch <= 8 else 0.92 if batch <= 64 else 0.97))
        unfused_overhead_ms = base_overhead_ms * (batch ** 0.5)
        unfused_total_ms = base_io_ms + unfused_compute_ms + unfused_overhead_ms
        unfused_tok_s = (batch * mimo_g) / (unfused_total_ms / 1000.0)
        
        speedup = r["total_tok_s"] / unfused_tok_s
        print(f"{batch:<8} {unfused_tok_s:<16.1f} {r['total_tok_s']:<16.1f} {speedup:<10.2f}x {r['per_user_tok_s']:<18.1f}")
    
    df = pd.DataFrame(results)
    df.to_csv("batch_scaling_metrics.csv", index=False)
    print(f"\n[+] Metrics saved to batch_scaling_metrics.csv")
    
    # Key insights
    print(f"\n{'='*90}")
    print("KEY INSIGHTS")
    print("=" * 90)
    
    # Find peak total tok/s
    peak_idx = df["total_tok_s"].idxmax()
    peak = df.iloc[peak_idx]
    print(f"  Peak total throughput: {peak['total_tok_s']:.0f} tok/s at batch={int(peak['batch_size'])}")
    print(f"  Per-user at peak batch: {peak['per_user_tok_s']:.1f} tok/s")
    print(f"  Latency at peak batch: {peak['latency_ms']:.1f} ms")
    
    # Find best per-user tok/s
    best_user_idx = df["per_user_tok_s"].idxmax()
    best_user = df.iloc[best_user_idx]
    print(f"  Best per-user throughput: {best_user['per_user_tok_s']:.1f} tok/s at batch={int(best_user['batch_size'])}")
    print(f"  Total at best per-user batch: {best_user['total_tok_s']:.1f} tok/s")
    
    # Interactive threshold (20 tok/s per user)
    interactive = df[df["per_user_tok_s"] >= 20]
    if len(interactive) > 0:
        max_interactive_batch = int(interactive["batch_size"].max())
        print(f"  Max batch for interactive (>20 tok/s per user): {max_interactive_batch}")

if __name__ == "__main__":
    simulate_batch_scaling()
