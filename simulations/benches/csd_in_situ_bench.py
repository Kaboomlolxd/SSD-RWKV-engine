import os
import time
import torch
import csv
import statistics

# =====================================================================
# THESIS EXPERIMENT: IN-SITU COMPUTATIONAL STORAGE (CSD)
# ACADEMIC EDITION: Moving Compute to Storage vs Storage to Compute
# =====================================================================
# This script simulates Chapter 4: "In-Situ State Propagation".
# Instead of moving 256MB of weights across the PCIe bus to the GPU,
# we move a 4MB Mamba State Vector across the PCIe bus to "Smart SSDs"
# equipped with tiny onboard NPUs.

NUM_LAYERS = 16
LAYER_WEIGHTS_MB = 256
MAMBA_STATE_MB = 4
BATCH_SIZES = [1, 16, 64, 256]
TRIALS = 5
NUM_SMART_SSDS = 8

# PCIe Gen 4/5 realistic effective bandwidths (MB/s)
PCIE_BANDWIDTH = 14000.0 

def simulate_gpu_compute(batch_size):
    """Simulates a high-end GPU doing the math (Fast)"""
    start = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.randn(batch_size, 4096, device=device)
    proj = torch.randn(4096, 4096, device=device)
    for _ in range(4): _ = torch.matmul(state, proj)
    if device == 'cuda': torch.cuda.synchronize()
    return time.perf_counter() - start

def simulate_csd_npu_compute(batch_size, num_drives):
    """
    Simulates tiny, weak AI chips inside the SSDs doing the math.
    Assume the SSD NPU is 10x slower than a desktop GPU, but the work
    is perfectly parallelized across all 8 SSDs.
    """
    # [FIX 16: The In-Situ CSD NAND Bandwidth Fallacy]
    # An SSD NPU still has to physically read the model weights from the raw NAND 
    # chips inside the drive. Consumer SSDs have an internal NAND bandwidth of ~7 GB/s.
    # The NPU cannot compute the matrix multiplication faster than the NAND feeds it.
    # We must explicitly add the internal NAND read time to the CSD compute time.
    gpu_time = simulate_gpu_compute(batch_size)
    npu_time_penalty = 10.0
    
    npu_compute_time = (gpu_time * npu_time_penalty) / num_drives
    
    # Each drive holds 1/N of the layer weights (e.g., 256MB / 8 = 32MB)
    # The drive's internal NAND bandwidth is ~7000 MB/s
    internal_nand_read_time = (LAYER_WEIGHTS_MB / num_drives) / 7000.0
    
    # CSD execution is pipelined internally: read from NAND, compute in NPU
    return max(internal_nand_read_time, npu_compute_time)

def run_csd_benchmark():
    print("="*80)
    print(" THESIS: COMPUTATIONAL STORAGE DRIVE (CSD) SIMULATION ")
    print("="*80)

    with open('csd_in_situ_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "Batch_Size", "PCIe_Payload_MB", "PCIe_Transfer_ms", "Compute_ms", "Total_Time_ms", "Tok_sec_Mean", "Tok_sec_StdDev"])

        for batch in BATCH_SIZES:
            # --- Architecture A: Standard GPU Offloading ---
            # Move 256MB weights -> Compute on Fast GPU
            std_tok_sec = []
            for _ in range(TRIALS):
                pcie_time_s = LAYER_WEIGHTS_MB / PCIE_BANDWIDTH
                compute_time_s = simulate_gpu_compute(batch)
                # Double buffering hides some latency, so we take the max
                layer_time = max(pcie_time_s, compute_time_s)
                std_tok_sec.append(batch / (layer_time * NUM_LAYERS))
            
            mean_std = statistics.mean(std_tok_sec)
            writer.writerow(["Standard_GPU_Offload", batch, LAYER_WEIGHTS_MB, f"{pcie_time_s*1000:.2f}", f"{compute_time_s*1000:.2f}", f"{layer_time*1000:.2f}", f"{mean_std:.2f}", f"{statistics.stdev(std_tok_sec):.2f}"])

            # --- Architecture B: CSD In-Situ Propagation ---
            # Move 4MB State -> Compute on Weak SSD NPUs
            csd_tok_sec = []
            for _ in range(TRIALS):
                # We only transfer the state vector!
                pcie_time_csd_s = (MAMBA_STATE_MB * batch) / PCIE_BANDWIDTH 
                compute_time_csd_s = simulate_csd_npu_compute(batch, NUM_SMART_SSDS)
                # Sequential: Transfer state, then compute, then return state
                layer_time_csd = pcie_time_csd_s + compute_time_csd_s + pcie_time_csd_s
                csd_tok_sec.append(batch / (layer_time_csd * NUM_LAYERS))

            mean_csd = statistics.mean(csd_tok_sec)
            writer.writerow(["CSD_In_Situ", batch, MAMBA_STATE_MB * batch, f"{pcie_time_csd_s*1000:.2f}", f"{compute_time_csd_s*1000:.2f}", f"{layer_time_csd*1000:.2f}", f"{mean_csd:.2f}", f"{statistics.stdev(csd_tok_sec):.2f}"])
            
            print(f"Batch {batch:<4} | Std Tok/s: {mean_std:>6.2f} | CSD Tok/s: {mean_csd:>6.2f} | Winner: {'CSD (Bandwidth Saved)' if mean_csd > mean_std else 'GPU (Compute Won)'}")

    print("[+] Academic data saved to 'csd_in_situ_metrics.csv'.")

if __name__ == "__main__":
    run_csd_benchmark()