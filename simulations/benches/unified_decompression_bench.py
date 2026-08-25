import os
import time
import torch
import csv
import platform
import statistics

# =====================================================================
# THESIS EXPERIMENT: UNIFIED NEURAL DECOMPRESSION BENCHMARK
# ACADEMIC EDITION: Includes Multi-Trial Stats & CSV Export
# =====================================================================

RAW_FP16_MB = 1024
COMPRESSED_MB = 128
TRIALS = 5 # Run multiple trials for variance tracking

RAW_FILE = "raw_layer_fp16.bin"
COMPRESSED_FILE = "compressed_layer_2bit.bin"

def create_files():
    if not os.path.exists(RAW_FILE):
        with open(RAW_FILE, "wb") as f: f.write(os.urandom(RAW_FP16_MB * 1024 * 1024))
    if not os.path.exists(COMPRESSED_FILE):
        with open(COMPRESSED_FILE, "wb") as f: f.write(os.urandom(COMPRESSED_MB * 1024 * 1024))

def test_raw_read(filepath):
    start = time.perf_counter()
    with open(filepath, "rb") as f: _ = f.read()
    return time.perf_counter() - start

def test_decompression(filepath, device_type):
    read_start = time.perf_counter()
    with open(filepath, "rb") as f: raw_bytes = f.read()
    read_time = time.perf_counter() - read_start

    decode_start = time.perf_counter()
    
    # [FIX 18: PCIe Transfer Omission Fallacy]
    # In earlier versions, packed_tensor was generated natively on the GPU via torch.randint(..., device='cuda').
    # This completely bypassed the PCIe transfer time required to copy `raw_bytes` from Host RAM to VRAM.
    # We must explicitly model the Host-To-Device (H2D) PCIe transfer, as it is often slower than the 
    # CUDA decompression kernel itself.
    num_elements = (RAW_FP16_MB * 1024 * 1024) // 2 
    
    # 1. Create tensor in CPU RAM (simulating the f.read() buffer)
    cpu_tensor = torch.randint(-2, 2, (num_elements // 4,), dtype=torch.int8, device='cpu')
    
    # 2. PCIe Transfer to GPU (This is the critical bottleneck we were missing!)
    packed_tensor = cpu_tensor.to(device_type, non_blocking=True)
    
    # 3. Simulate Decompression (expand to fp16)
    decompressed_fp16 = torch.zeros((num_elements,), dtype=torch.float16, device=device_type)
    decompressed_fp16[:len(packed_tensor)] = packed_tensor.to(torch.float16) * 0.45 
    
    if device_type == 'cuda': torch.cuda.synchronize()
    decode_time = time.perf_counter() - decode_start
    
    return read_time, decode_time

def run_unified_benchmark():
    create_files()
    
    # Open CSV for data collection
    with open('decompression_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Target_Device", "Payload_MB", "Read_Time_Mean_ms", "Decode_Time_Mean_ms", "Total_Time_Mean_ms", "Effective_BW_MBs", "StdDev_BW"])

        # --- SCENARIO A: RAW SSD READ ---
        raw_bws = []
        raw_times = []
        for _ in range(TRIALS):
            t = test_raw_read(RAW_FILE)
            raw_times.append(t * 1000)
            raw_bws.append(RAW_FP16_MB / t)
        
        mean_raw_time = statistics.mean(raw_times)
        mean_raw_bw = statistics.mean(raw_bws)
        std_raw_bw = statistics.stdev(raw_bws) if TRIALS > 1 else 0
        writer.writerow(["RAW_SSD", RAW_FP16_MB, f"{mean_raw_time:.2f}", "0.00", f"{mean_raw_time:.2f}", f"{mean_raw_bw:.2f}", f"{std_raw_bw:.2f}"])

        # --- SCENARIO B: CPU DECOMPRESSION ---
        cpu_bws, cpu_reads, cpu_decodes, cpu_totals = [], [], [], []
        for _ in range(TRIALS):
            r, d = test_decompression(COMPRESSED_FILE, 'cpu')
            tot = r + d
            cpu_reads.append(r * 1000)
            cpu_decodes.append(d * 1000)
            cpu_totals.append(tot * 1000)
            cpu_bws.append(RAW_FP16_MB / tot)
            
        writer.writerow(["CPU_AVX512", COMPRESSED_MB, f"{statistics.mean(cpu_reads):.2f}", f"{statistics.mean(cpu_decodes):.2f}", f"{statistics.mean(cpu_totals):.2f}", f"{statistics.mean(cpu_bws):.2f}", f"{statistics.stdev(cpu_bws):.2f}"])

        # --- SCENARIO C: GPU DECOMPRESSION ---
        if torch.cuda.is_available():
            gpu_bws, gpu_reads, gpu_decodes, gpu_totals = [], [], [], []
            for _ in range(TRIALS):
                r, d = test_decompression(COMPRESSED_FILE, 'cuda')
                tot = r + d
                gpu_reads.append(r * 1000)
                gpu_decodes.append(d * 1000)
                gpu_totals.append(tot * 1000)
                gpu_bws.append(RAW_FP16_MB / tot)
                
            writer.writerow(["GPU_CUDA", COMPRESSED_MB, f"{statistics.mean(gpu_reads):.2f}", f"{statistics.mean(gpu_decodes):.2f}", f"{statistics.mean(gpu_totals):.2f}", f"{statistics.mean(gpu_bws):.2f}", f"{statistics.stdev(gpu_bws):.2f}"])

    print("[+] Academic data saved to 'decompression_metrics.csv' for charting.")

if __name__ == "__main__":
    run_unified_benchmark()
