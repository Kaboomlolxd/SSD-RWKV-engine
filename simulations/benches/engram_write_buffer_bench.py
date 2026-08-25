import pandas as pd
import argparse

def simulate_engram_write_path(num_tokens, engram_size_bytes=256, ssd_page_size_bytes=4096, ssd_erase_block_bytes=2*1024*1024, random_write_iops=100000, seq_write_bw_gb_s=5.0):
    """
    Simulates the WRITE path for Engram/SSM architectures.
    When an LLM learns/updates its state, it must write back to the Engram.
    SSDs suffer from Write Amplification: writing 256 bytes forces a 4096 byte 
    Read-Modify-Write, destroying the drive's lifespan and performance.
    """
    results = []
    
    # 1. Naive Random Writes (The SSD Killer)
    # Every token generates a 256B write. SSD forces a 4KB read-modify-write.
    # Time is bottlenecked by random write IOPS.
    time_naive_write = num_tokens / random_write_iops
    write_amplification_naive = ssd_page_size_bytes / engram_size_bytes # 16x!
    bytes_written_physical_naive = num_tokens * ssd_page_size_bytes
    
    # 2. Page-Aligned MHC Interleaving (Addressing Multi-Head Cache)
    # If the model uses MHC, we pack all heads into a single 4KB page.
    # This reduces IOPS if multiple heads update simultaneously, but still suffers if random.
    # (Assuming 4 heads update at once, packed into 1 page)
    time_mhc_packed = (num_tokens / 4) / random_write_iops
    write_amplification_mhc = ssd_page_size_bytes / (engram_size_bytes * 4) # 4x
    
    # 3. Log-Structured RAM Buffer (The Ultimate Fix)
    # We never overwrite engrams in-place. We append all updates to a RAM buffer.
    # When the buffer hits 2MB (an SSD Erase Block), we flush it sequentially.
    # Time is bottlenecked by Sequential Write Bandwidth, not IOPS.
    total_data_gb = (num_tokens * engram_size_bytes) / (1024**3)
    time_log_structured = total_data_gb / seq_write_bw_gb_s
    write_amplification_log = 1.0 # Perfect 1:1, no wasted writes
    bytes_written_physical_log = num_tokens * engram_size_bytes

    results.append({
        'num_tokens': num_tokens,
        'Naive_Write_Time_s': time_naive_write,
        'MHC_Packed_Time_s': time_mhc_packed,
        'Log_Structured_Time_s': time_log_structured,
        'Naive_Wear_GB': bytes_written_physical_naive / (1024**3),
        'Log_Wear_GB': bytes_written_physical_log / (1024**3)
    })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Running Engram Write-Path (Log-Structured) Benchmark...")
    
    token_counts = [10_000, 100_000, 1_000_000] # Number of generated tokens that update the engram
    
    all_res = []
    for tc in token_counts:
        df = simulate_engram_write_path(num_tokens=tc)
        all_res.append(df)
        
    final_df = pd.concat(all_res, ignore_index=True)
    
    for _, row in final_df.iterrows():
        print(f"\nTokens Generated: {row['num_tokens']:,}")
        print(f"  Naive Write Time       : {row['Naive_Write_Time_s']:.4f} sec | SSD Wear: {row['Naive_Wear_GB']:.4f} GB")
        print(f"  MHC Packed Write Time  : {row['MHC_Packed_Time_s']:.4f} sec")
        print(f"  Log-Structured Time    : {row['Log_Structured_Time_s']:.6f} sec | SSD Wear: {row['Log_Wear_GB']:.6f} GB")
        print(f"  Speedup                : {row['Naive_Write_Time_s'] / row['Log_Structured_Time_s']:,.0f}x faster!")
        
    final_df.to_csv("engram_write_buffer_results.csv", index=False)
    print("\nResults saved to engram_write_buffer_results.csv")
