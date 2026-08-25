import pandas as pd
import numpy as np

def simulate_inline_metadata_transfer():
    print("Simulating Inline Metadata Transfer (ByteExpress-inspired) for Mamba SSM...")
    print("(HotStorage 2025: Co-locating small payloads with NVMe command metadata)\n")
    
    # In Mamba inference, each layer needs:
    # - Large weight matrix read (7GB total, streamed)
    # - Small per-token data: token embedding (4KB), state update params (2KB), routing info (1KB)
    # Currently: Each small payload requires a separate NVMe command (submission queue entry)
    # This adds ~5us of NVMe command overhead per small read
    
    layers = 64
    tokens_generated = 10000
    
    # Standard approach: separate reads for weights + small data
    standard_weight_reads = layers * tokens_generated
    standard_small_reads = layers * tokens_generated  # Token embeddings, state params
    standard_command_overhead_us = (standard_weight_reads + standard_small_reads) * 5.0  # 5us per command
    
    # ByteExpress approach: inline small data in the same NVMe command as weight reads
    # NVMe spec allows up to 4KB of metadata per command (PRP/SGL inline)
    # We pack token embedding + state params into the weight read's metadata field
    inline_small_reads = 0  # Eliminated!
    inline_command_overhead_us = standard_weight_reads * 5.0  # Only weight reads need commands
    
    # Time savings
    time_saved_ms = (standard_command_overhead_us - inline_command_overhead_us) / 1000.0
    
    # Per-token impact
    per_token_saved_us = time_saved_ms / tokens_generated * 1000
    
    print("=" * 80)
    print("Inline Metadata Transfer (ByteExpress-Inspired)")
    print("=" * 80)
    print(f"{'Metric':<50} {'Standard':<15} {'Inline (ByteExpress)':<25}")
    print("-" * 80)
    print(f"{'Total NVMe Commands':<50} {standard_weight_reads + standard_small_reads:<15,} {standard_weight_reads:<25,}")
    print(f"{'Command Overhead (ms)':<50} {standard_command_overhead_us/1000:<15.1f} {inline_command_overhead_us/1000:<25.1f}")
    print(f"{'Time Saved (ms)':<50} {'N/A':<15} {time_saved_ms:<25.1f}")
    print(f"{'Per-Token Latency Saved (us)':<50} {'N/A':<15} {per_token_saved_us:<25.1f}")
    
    # Additional benefit: NVMe queue depth utilization
    # Standard: 50% of SQ entries wasted on small reads
    # Inline: 100% of SQ entries carry useful weight data
    sq_utilization_standard = 0.50
    sq_utilization_inline = 1.00
    
    # Effective bandwidth improvement
    nvme_bw_gbps = 14.0
    effective_bw_standard = nvme_bw_gbps * sq_utilization_standard
    effective_bw_inline = nvme_bw_gbps * sq_utilization_inline
    
    print(f"\n{'='*80}")
    print("NVMe Queue Efficiency")
    print("=" * 80)
    print(f"{'Metric':<50} {'Standard':<15} {'Inline':<25}")
    print("-" * 80)
    print(f"{'SQ Entry Utilization':<50} {sq_utilization_standard*100:<15.0f}% {sq_utilization_inline*100:<25.0f}%")
    print(f"{'Effective BW (GB/s per drive)':<50} {effective_bw_standard:<15.1f} {effective_bw_inline:<25.1f}")
    print(f"{'BW Improvement':<50} {'N/A':<15} {effective_bw_inline/effective_bw_standard:.1f}x")
    
    results = [
        {"Approach": "Standard Separate Reads", "NVMe_Commands": standard_weight_reads + standard_small_reads, "Overhead_ms": standard_command_overhead_us/1000, "SQ_Util": sq_utilization_standard, "Eff_BW_GBps": effective_bw_standard},
        {"Approach": "Inline Metadata (ByteExpress)", "NVMe_Commands": standard_weight_reads, "Overhead_ms": inline_command_overhead_us/1000, "SQ_Util": sq_utilization_inline, "Eff_BW_GBps": effective_bw_inline},
    ]
    
    df = pd.DataFrame(results)
    df.to_csv("byte_express_inline_metrics.csv", index=False)
    print(f"\n[+] Metrics saved to byte_express_inline_metrics.csv")

if __name__ == "__main__":
    simulate_inline_metadata_transfer()
