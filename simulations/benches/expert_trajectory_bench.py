import pandas as pd

def simulate_expert_trajectory_prediction():
    print("Simulating Expert Trajectory Prediction for SSD-Native MoE Inference...")
    print("(Inspired by Expert Streaming, arXiv 2603.27624)\n")
    
    # MoE model parameters
    total_experts = 128
    active_experts_per_token = 8
    expert_size_mb = 50.0  # Each expert is 50MB
    layers = 64
    
    # Baseline: Fetch experts on-demand per layer (one at a time, no overlap)
    baseline_fetches = layers * active_experts_per_token
    baseline_io_mb = baseline_fetches * expert_size_mb
    baseline_latency_ms = baseline_io_mb / 7.0  # Gen4 NVMe
    
    # Naive prefetch: Prefetch all experts for next token (over-fetches 50%)
    naive_prefetch_experts = int(active_experts_per_token * 1.5)  # 50% over-fetch
    naive_io_mb = layers * naive_prefetch_experts * expert_size_mb
    naive_latency_ms = naive_io_mb / 7.0
    
    # Expert Trajectory Prediction (Expert Streaming + our Graph-Partitioned RAID)
    # Predict the entire expert path across all layers from previous token's routing
    prediction_accuracy = 0.85  # 85% accuracy (from Expert Streaming paper)
    # With 85% accuracy, we only fetch 85% * active experts upfront
    # The remaining 15% are fetched on-demand (miss penalty)
    predicted_hit_experts = int(active_experts_per_token * prediction_accuracy)
    miss_experts = active_experts_per_token - predicted_hit_experts
    
    # IO = prefetch (hit experts) + on-demand (miss experts)
    trajectory_io_mb = (layers * predicted_hit_experts * expert_size_mb +
                        layers * miss_experts * expert_size_mb)
    # But prefetch happens in parallel across SSDs (Graph-Partitioned RAID)
    # So effective latency is reduced by parallelism factor
    raid_parallelism = 4.0  # 4 SSDs in RAID
    trajectory_latency_ms = trajectory_io_mb / (7.0 * raid_parallelism)
    
    print("=" * 70)
    print("Expert Fetching Strategies (64-layer, 128-expert MoE)")
    print("=" * 70)
    print(f"{'Strategy':<35} {'IO (MB)':<15} {'Latency (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'On-Demand Fetch':<35} {baseline_io_mb:<15.0f} {baseline_latency_ms:<15.1f} {'1.00x':<10}")
    print(f"{'Naive Prefetch (1.5x)':<35} {naive_io_mb:<15.0f} {naive_latency_ms:<15.1f} {baseline_latency_ms/naive_latency_ms:.2f}x")
    print(f"{'Trajectory Prediction (85%)':<35} {trajectory_io_mb:<15.0f} {trajectory_latency_ms:<15.1f} {baseline_latency_ms/trajectory_latency_ms:.2f}x")
    
    results = [
        {"Strategy": "On-Demand Fetch", "IO_MB": baseline_io_mb, "Latency_ms": baseline_latency_ms, "Speedup": 1.0},
        {"Strategy": "Naive Prefetch (1.5x)", "IO_MB": naive_io_mb, "Latency_ms": naive_latency_ms, "Speedup": baseline_latency_ms/naive_latency_ms},
        {"Strategy": "Trajectory Prediction (85%)", "IO_MB": trajectory_io_mb, "Latency_ms": trajectory_latency_ms, "Speedup": baseline_latency_ms/trajectory_latency_ms}
    ]
    
    df = pd.DataFrame(results)
    df.to_csv("expert_trajectory_metrics.csv", index=False)
    print(f"\n[+] Metrics saved to expert_trajectory_metrics.csv")

if __name__ == "__main__":
    simulate_expert_trajectory_prediction()
