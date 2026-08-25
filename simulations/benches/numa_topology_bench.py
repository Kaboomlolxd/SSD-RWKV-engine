import pandas as pd
import argparse

def simulate_numa_bottleneck(num_drives, gb_per_drive, cross_numa_penalty=0.4):
    """
    Simulates the NUMA (Non-Uniform Memory Access) topology problem on multi-socket servers 
    (like Azure L-Series). If a CPU thread on Socket 0 reads an NVMe drive wired to Socket 1,
    the data must cross the motherboard's interconnect (AMD Infinity Fabric / Intel UPI),
    causing massive bandwidth drops and latency spikes.
    """
    # Assume each drive can do 7 GB/s
    drive_bw = 7.0 
    
    # 1. OS-Level RAID 0 (NUMA-Blind)
    # OS treats all drives as one block. Reads are scattered randomly.
    # 50% of reads will cross the NUMA boundary, incurring the penalty.
    effective_bw_blind = (drive_bw * num_drives) * (1.0 - (0.5 * cross_numa_penalty))
    time_blind = (gb_per_drive * num_drives) / effective_bw_blind
    
    # 2. Application-Level Sharded I/O (NUMA-Aware)
    # We spawn separate io_uring threads pinned to specific CPU cores.
    # Core 0 (Socket 0) only reads Drive 0 & 1. Core 16 (Socket 1) only reads Drive 2 & 3.
    # 0% NUMA crossing penalty.
    effective_bw_aware = drive_bw * num_drives
    time_aware = (gb_per_drive * num_drives) / effective_bw_aware
    
    # 3. [NEW] Consumer Chipset Bottleneck (DMI 4.0 / PCIe x4 Uplink)
    # On consumer motherboards (Z790/X670E), CPU provides 1x Gen4 NVMe.
    # The other 3 NVMe drives connect to the PCH (Chipset).
    # The Chipset uplink is DMI 4.0 x8 (15.7 GB/s) or PCIe 4.0 x4 (7.8 GB/s).
    # 3 drives attempting 21 GB/s will hard bottleneck at the uplink limit.
    cpu_direct_bw = drive_bw * 1.0 # 1 drive direct to CPU
    pch_uplink_bw_limit = 15.7 # Intel DMI 4.0 x8 limit
    pch_drives_bw = min(drive_bw * (num_drives - 1), pch_uplink_bw_limit) if num_drives > 1 else 0
    effective_bw_consumer = cpu_direct_bw + pch_drives_bw
    time_consumer = (gb_per_drive * num_drives) / effective_bw_consumer if num_drives > 0 else float('inf')
    
    return {
        'Drives': num_drives,
        'Total_GB': gb_per_drive * num_drives,
        'Blind_BW_GBs': effective_bw_blind,
        'Aware_BW_GBs': effective_bw_aware,
        'Consumer_BW_GBs': effective_bw_consumer,
        'Blind_Time_s': time_blind,
        'Aware_Time_s': time_aware,
        'Speedup': time_blind / time_aware
    }

if __name__ == "__main__":
    print("Running NUMA-Topology IO Benchmark...")
    
    scenarios = [2, 4, 8, 16] # Number of NVMe drives
    
    results = []
    for d in scenarios:
        res = simulate_numa_bottleneck(num_drives=d, gb_per_drive=50.0)
        results.append(res)
        
        print(f"\nDrives: {d} (Total {res['Total_GB']} GB payload)")
        print(f"  OS RAID (NUMA-Blind)   : {res['Blind_BW_GBs']:.2f} GB/s | {res['Blind_Time_s']:.3f} sec")
        print(f"  Thread-Pinned (Aware)  : {res['Aware_BW_GBs']:.2f} GB/s | {res['Aware_Time_s']:.3f} sec")
        print(f"  Recovered Bandwidth    : +{(res['Aware_BW_GBs'] - res['Blind_BW_GBs']):.2f} GB/s")
        
    df = pd.DataFrame(results)
    df.to_csv("numa_topology_results.csv", index=False)
    print("\nResults saved to numa_topology_results.csv")