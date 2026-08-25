import numpy as np
import pandas as pd
import time
import os

def simulate_csd_zts_scan():
    print("Simulating Zero-Transfer SSM Scan (ZTS-Scan) via CSD...")
    
    # Mamba Model parameters
    layer_count = 64
    d_model = 4096
    d_state = 128
    d_inner = d_model * 2
    
    # State size per layer: (d_inner, d_state)
    state_size_mb = (d_inner * d_state * 2) / (1024**2) # fp16
    total_state_size_mb = state_size_mb * layer_count
    
    token_embedding_size_kb = (d_model * 2) / 1024 # fp16
    
    # Hardware specs
    pcie_gen4_bw_gbps = 7.0 # GB/s
    csd_internal_bw_gbps = 25.0 # GB/s (SSD controller internal SRAM/DRAM bandwidth)
    
    # Baseline: Host reads/writes state over PCIe
    baseline_transfer_time_ms = (total_state_size_mb / 1024 / pcie_gen4_bw_gbps) * 1000 * 2 # Read + Write
    
    # ZTS-Scan: Host only sends token embedding over PCIe
    zts_transfer_time_ms = ((token_embedding_size_kb / 1024 / 1024) / pcie_gen4_bw_gbps) * 1000 * layer_count
    zts_internal_compute_time_ms = (total_state_size_mb / 1024 / csd_internal_bw_gbps) * 1000 * 2
    
    zts_total_time_ms = zts_transfer_time_ms + zts_internal_compute_time_ms
    
    results = [
        {"architecture": "Host-Driven (PCIe bound)", "state_transfer_ms": baseline_transfer_time_ms, "pcie_data_transferred_mb": total_state_size_mb * 2},
        {"architecture": "ZTS-Scan (CSD offload)", "state_transfer_ms": zts_total_time_ms, "pcie_data_transferred_mb": (token_embedding_size_kb / 1024) * layer_count}
    ]
    
    df = pd.DataFrame(results)
    df.to_csv("csd_zts_scan_metrics.csv", index=False)
    
    print("\nResults:")
    print(df.to_string())
    print(f"\nSpeedup: {baseline_transfer_time_ms / zts_total_time_ms:.2f}x")
    print(f"PCIe Bandwidth Reduction: {(total_state_size_mb * 2) / ((token_embedding_size_kb / 1024) * layer_count):.2f}x")

if __name__ == "__main__":
    simulate_csd_zts_scan()
