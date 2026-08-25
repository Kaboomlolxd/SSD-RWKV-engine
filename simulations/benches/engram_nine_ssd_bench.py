import numpy as np
import pandas as pd
import argparse

def simulate_engram_nine(seq_len, ssd_iops, ssd_bw_gb_s, trials=5):
    """
    Simulates the Engram-Nine architecture on SSDs, contrasting naive 
    implementation with Sector-Packed and Bloom Filter optimizations.
    """
    results = []
    
    # Engram parameters
    engram_size_bytes = 256
    ssd_sector_size_bytes = 4096
    
    # Queue Depth Reality: Local LLMs are QD1. Real NVMe QD1 random read is ~15,000 IOPS, not 1M.
    qd1_ssd_iops = 15000 
    
    # SSD Latency per random read at QD1 (approx 66 microseconds)
    ssd_latency_s = 1.0 / qd1_ssd_iops
    
    # Compute time per token
    compute_time_s = 0.005 
    
    # Probabilities
    collision_prob = 0.40 # 40% chance we need to check multiple slots
    avg_probes_on_collision = 4.5 # Average probes needed if a collision occurs
    empty_slot_prob = 0.30 # 30% chance the queried engram concept doesn't exist yet

    for trial in range(trials):
        
        # 1. Naive Engram-1 (Original Paper, assumes 1 probe, fails on collision)
        time_engram_1 = seq_len * (compute_time_s + ssd_latency_s)
        
        # 2. Naive Engram-9 (Random IOPS for every probe)
        # If no collision: 1 read. If collision: ~4.5 reads.
        total_reads_naive_9 = seq_len * ((1 - collision_prob) * 1 + collision_prob * avg_probes_on_collision)
        time_engram_9_naive = (seq_len * compute_time_s) + (total_reads_naive_9 * ssd_latency_s)
        
        # 3. Optimized: Sector-Nine Packing
        # All 9 collision slots are packed into a single 4KB sector. 
        # No matter how many collisions, it's always exactly 1 IOPS per token.
        # Bandwidth is slightly higher (reading 4KB instead of 256B), but IOPS is 1.
        time_sector_nine = seq_len * (compute_time_s + ssd_latency_s)
        
        # 4. Ultra-Optimized: Sector-Nine + VRAM Bloom Filter
        # We skip the SSD entirely if the slot is empty (30% of the time).
        # Otherwise, 1 IOPS for the packed sector.
        actual_reads_bloom = seq_len * (1.0 - empty_slot_prob)
        time_sector_bloom = (seq_len * compute_time_s) + (actual_reads_bloom * ssd_latency_s)
        
        # 5. Perfect Prefill (Existing SSM Optimization applied to Engram)
        # Batching the entire sequence into one sequential read (Bandwidth bound)
        total_data_gb = (seq_len * ssd_sector_size_bytes) / (1024**3)
        time_prefill = (seq_len * compute_time_s) + (total_data_gb / ssd_bw_gb_s)

        results.append({
            'Engram_1_Baseline': time_engram_1,
            'Engram_9_Naive': time_engram_9_naive,
            'Engram_9_SectorPacked': time_sector_nine,
            'Engram_9_BloomPacked': time_sector_bloom,
            'Engram_9_PrefillBatched': time_prefill
        })
        
    df = pd.DataFrame(results)
    
    return {
        'Engram_1_Baseline': df['Engram_1_Baseline'].mean(),
        'Engram_9_Naive': df['Engram_9_Naive'].mean(),
        'Engram_9_SectorPacked': df['Engram_9_SectorPacked'].mean(),
        'Engram_9_BloomPacked': df['Engram_9_BloomPacked'].mean(),
        'Engram_9_PrefillBatched': df['Engram_9_PrefillBatched'].mean(),
    }

if __name__ == "__main__":
    print("Running Engram-Nine SSD Optimization Benchmark...")
    print("-" * 60)
    
    seq_lens = [128, 1024, 4096] # Short to long context
    ssd_iops = 1000000
    ssd_bw = 14.0 # 14 GB/s
    
    all_res = []
    
    for sl in seq_lens:
        res = simulate_engram_nine(seq_len=sl, ssd_iops=ssd_iops, ssd_bw_gb_s=ssd_bw)
        
        print(f"\nSequence Length: {sl} tokens")
        print(f"  Naive Engram-9 (Random IOPS) : {res['Engram_9_Naive']:.4f} sec")
        print(f"  Optimized: Sector-Packed     : {res['Engram_9_SectorPacked']:.4f} sec ({(res['Engram_9_Naive']/res['Engram_9_SectorPacked']):.2f}x speedup)")
        print(f"  Optimized: + Bloom Filter    : {res['Engram_9_BloomPacked']:.4f} sec ({(res['Engram_9_Naive']/res['Engram_9_BloomPacked']):.2f}x speedup)")
        print(f"  Optimized: + Chunked Prefill : {res['Engram_9_PrefillBatched']:.4f} sec (Absolute Best)")
        
        res['Sequence_Length'] = sl
        all_res.append(res)
        
    df_out = pd.DataFrame(all_res)
    df_out.to_csv("engram_nine_ssd_results.csv", index=False)
    print("\nResults saved to engram_nine_ssd_results.csv")
