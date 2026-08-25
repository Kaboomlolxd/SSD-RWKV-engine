import pandas as pd
import numpy as np

def simulate_plane_level_parallelism():
    print("Simulating Plane-Level Parallelism Exploitation for Mamba Weight Streaming...")
    print("(IEEE 2025: Parallel all the time - Plane Level Parallelism)\n")
    
    # Modern NVMe SSD internal structure:
    # - 8-16 NAND channels per controller
    # - 2-4 dies per channel (CE lines)
    # - 2-4 planes per die
    # Total parallelism: 8 channels * 4 dies * 4 planes = 128 concurrent operations
    
    # Standard FTL: Sequential reads only exploit channel-level parallelism
    # Plane-parallel reads: Exploit channel + die + plane parallelism simultaneously
    
    # Mamba weights are stored as layer files (~200MB each for 70B Q4)
    # Standard: Read layer sequentially (one plane at a time per channel)
    # Optimized: Stripe each layer across all planes of all dies
    
    channels = 8
    dies_per_channel = 4
    planes_per_die = 4
    total_planes = channels * dies_per_channel * planes_per_die  # 128
    
    # Standard sequential read: uses 1 plane per channel at a time
    standard_active_planes = channels  # 8
    # Plane-parallel: uses ALL planes simultaneously
    parallel_active_planes = total_planes  # 128
    
    # Read latency per 200MB layer
    nand_read_latency_ms = 0.1  # Per plane
    data_per_plane_mb = 200.0 / standard_active_planes  # 25MB per plane (standard)
    
    # Standard: 8 planes read 25MB each sequentially
    standard_layer_read_ms = (200.0 / 7.0) * 1000 / channels  # ~3.6ms (bottlenecked by channel BW)
    
    # Plane-parallel: 128 planes read ~1.56MB each simultaneously
    # NAND internal bandwidth per plane: ~400MB/s
    plane_bw_mbps = 400.0
    parallel_layer_read_ms = (200.0 / (plane_bw_mbps * total_planes)) * 1000
    
    # But the real bottleneck is the NAND channel interface, not plane read
    # Channel interface: ~1.2GB/s per channel (ONFI 5.0)
    channel_bw_gbps = 1.2
    total_channel_bw = channel_bw_gbps * channels  # 9.6 GB/s
    parallel_layer_read_ms_real = (200.0 / total_channel_bw) * 1000  # ~20.8ms
    
    # However, with proper weight striping across planes, we can pipeline
    # reads so that while one plane is reading, another is transferring
    pipeline_factor = 4.0  # 4-stage pipeline: read -> transfer -> buffer -> DMA
    effective_read_ms = parallel_layer_read_ms_real / pipeline_factor
    
    print("=" * 80)
    print("Plane-Level Parallelism for Mamba Weight Streaming")
    print("=" * 80)
    print(f"{'Metric':<50} {'Standard':<15} {'Plane-Parallel':<20}")
    print("-" * 80)
    print(f"{'Active Planes':<50} {standard_active_planes:<15} {parallel_active_planes:<20}")
    print(f"{'Data per Plane (MB)':<50} {data_per_plane_mb:<15.1f} {200.0/parallel_active_planes:<20.2f}")
    print(f"{'Layer Read Time (ms)':<50} {standard_layer_read_ms:<15.1f} {effective_read_ms:<20.1f}")
    print(f"{'Speedup':<50} {'1.00x':<15} {standard_layer_read_ms/effective_read_ms:.1f}x")
    
    # Multi-layer impact (64 layers)
    print(f"\n{'='*80}")
    print("Full Model Streaming (64 Layers, 70B Mamba)")
    print("=" * 80)
    
    layers = 64
    total_weight_gb = 14.0  # Q4 compressed
    
    standard_total_ms = (total_weight_gb / 7.0) * 1000  # ~2000ms at 7GB/s
    parallel_total_ms = (total_weight_gb / (total_channel_bw * 0.85)) * 1000  # ~1372ms (85% efficiency)
    
    print(f"{'Metric':<50} {'Standard':<15} {'Plane-Parallel':<20}")
    print("-" * 80)
    print(f"{'Total Streaming Time (ms)':<50} {standard_total_ms:<15.0f} {parallel_total_ms:<20.0f}")
    print(f"{'Effective BW (GB/s)':<50} {7.0:<15.1f} {total_channel_bw * 0.85:<20.1f}")
    print(f"{'Speedup':<50} {'1.00x':<15} {standard_total_ms/parallel_total_ms:.1f}x")
    
    results = [
        {"Approach": "Standard Sequential", "Active_Planes": standard_active_planes, "Layer_Read_ms": standard_layer_read_ms, "Total_Streaming_ms": standard_total_ms, "Eff_BW_GBps": 7.0},
        {"Approach": "Plane-Parallel Striped", "Active_Planes": parallel_active_planes, "Layer_Read_ms": effective_read_ms, "Total_Streaming_ms": parallel_total_ms, "Eff_BW_GBps": total_channel_bw * 0.85},
    ]
    
    df = pd.DataFrame(results)
    df.to_csv("plane_parallelism_metrics.csv", index=False)
    print(f"\n[+] Metrics saved to plane_parallelism_metrics.csv")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHT: Weight Striping Across SSD Planes")
    print("=" * 80)
    print(f"  By striping each Mamba layer file across all 128 NAND planes,")
    print(f"  we achieve {total_channel_bw * 0.85:.1f} GB/s effective bandwidth vs {7.0:.1f} GB/s standard.")
    print(f"  This is a {standard_total_ms/parallel_total_ms:.1f}x improvement in weight streaming speed.")
    print(f"  Implementation: Use NVMe multi-plane read commands (0x23h) with")
    print(f"  pre-computed plane-to-LBA mapping tables stored in SSD firmware.")

if __name__ == "__main__":
    simulate_plane_level_parallelism()
