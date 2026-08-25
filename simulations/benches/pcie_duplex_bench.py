import pandas as pd
import argparse

def simulate_pcie_duplex(read_gb, write_gb, pcie_lanes_bw_gb_s=14.0):
    """
    Simulates PCIe Full-Duplex Interleaving for Continuous Learning LLMs.
    PCIe Gen 4x4 provides 14 GB/s Read AND 14 GB/s Write simultaneously
    on physically separate wire pairs (TX and RX lanes).
    """
    
    # 1. Half-Duplex / Synchronous (Standard PyTorch behavior)
    # The CPU halts reads to perform writes, sharing the time domain.
    time_read_only = read_gb / pcie_lanes_bw_gb_s
    time_write_only = write_gb / pcie_lanes_bw_gb_s
    time_half_duplex = time_read_only + time_write_only
    
    # 2. Full-Duplex / Asynchronous DMA (Optimized Rust io_uring behavior)
    # TX and RX lanes are saturated simultaneously on the PCIe bus. 
    # [FIX 5: NAND Mixed Workload Penalty]
    # While PCIe is full-duplex, consumer NVMe controllers and NAND channels are NOT.
    # Concurrent read/write workloads cause severe internal resource contention (SRAM/FTL).
    # We apply a 40% mixed-workload degradation penalty to the total time.
    time_full_duplex = max(time_read_only, time_write_only) * 1.40
    
    # Speedup calculation
    speedup = time_half_duplex / time_full_duplex
    
    return {
        'Read_GB': read_gb,
        'Write_GB': write_gb,
        'Time_Half_Duplex_s': time_half_duplex,
        'Time_Full_Duplex_s': time_full_duplex,
        'Speedup_Factor': speedup,
        'Free_Write_Bandwidth_GB': min(read_gb, write_gb) # The amount of data written basically for "free" in the background
    }

if __name__ == "__main__":
    print("Running PCIe Full-Duplex Interleaving Benchmark...")
    print("Simulating simultaneous Model Weight Streaming (Reads) and Engram Updates (Writes)")
    print("-" * 70)
    
    # Scenarios:
    # 1. Heavy Read, Light Write (Standard generation + minor engram updates)
    # 2. Balanced (Heavy processing and massive context shifting)
    # 3. Heavy Write, Light Read (Ingesting a massive document into the Engram)
    
    scenarios = [
        (10.0, 1.0),   # 10GB read, 1GB write
        (14.0, 14.0),  # 14GB read, 14GB write (Max saturation)
        (2.0, 10.0),   # 2GB read, 10GB write
    ]
    
    results = []
    for r_gb, w_gb in scenarios:
        res = simulate_pcie_duplex(read_gb=r_gb, write_gb=w_gb)
        results.append(res)
        
        print(f"Scenario: {r_gb}GB Read / {w_gb}GB Write")
        print(f"  Standard (Blocking) Time : {res['Time_Half_Duplex_s']:.3f} sec")
        print(f"  Full-Duplex (Async) Time : {res['Time_Full_Duplex_s']:.3f} sec")
        print(f"  Speedup                  : {res['Speedup_Factor']:.2f}x (Wrote {res['Free_Write_Bandwidth_GB']}GB for 'free')\n")
        
    df = pd.DataFrame(results)
    df.to_csv("pcie_duplex_results.csv", index=False)
    print("Results saved to pcie_duplex_results.csv")
