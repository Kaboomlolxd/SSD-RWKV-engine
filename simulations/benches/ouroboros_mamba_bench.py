import os
import time
import torch
import csv
import platform
import statistics

# =====================================================================
# THESIS EXPERIMENT: THE "OUROBOROS" UNIVERSAL MAMBA
# ACADEMIC EDITION: Proving O(1) Memory vs O(N) Depth
# =====================================================================
# [FIX 25: The Ouroboros Parameter Collapse Paradox]
# CRITICAL CORRECTION: This tests the radical architecture where all 100 layers 
# share the exact same physical 256MB weights in VRAM. The SSD only needs to
# stream a tiny 1KB "Conditioning Vector" for each layer step.
# THE PARADOX: If the entire model fits in 256MB of VRAM, the SSD is completely 
# irrelevant! The 100KB of conditioning vectors for a full pass can simply be 
# pinned in L1/L2 cache. Furthermore, mathematically, weight sharing across 100 layers 
# fundamentally collapses the parameter capacity of the model. You are no longer 
# running a 70B parameter model; you are running a ~100M parameter model looped 
# 100 times. Claiming this as an "SSD-native optimization for large models" is 
# physically contradictory because the model is no longer large.

SHARED_LAYER_MB = 256
COND_VECTOR_KB = 1
NUM_LAYERS = 100
BATCH_SIZES = [1, 16, 64, 256]
TRIALS = 5

SHARED_FILE = "ouroboros_shared_layer.bin"
COND_FILE = "ouroboros_cond_vector.bin"

def create_files():
    if not os.path.exists(SHARED_FILE):
        with open(SHARED_FILE, "wb") as f: f.write(os.urandom(SHARED_LAYER_MB * 1024 * 1024))
    if not os.path.exists(COND_FILE):
        with open(COND_FILE, "wb") as f: f.write(os.urandom(COND_VECTOR_KB * 1024))

def simulate_compute(batch_size):
    """Simulate applying the conditioning vector to the shared weights"""
    start_comp = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.randn(batch_size, 4096, device=device)
    proj = torch.randn(4096, 4096, device=device)
    
    for _ in range(4): _ = torch.matmul(state, proj)
    if device == 'cuda': torch.cuda.synchronize()
    return time.perf_counter() - start_comp

def run_ouroboros_benchmark():
    create_files()
    print("="*80)
    print(" THESIS: OUROBOROS UNIVERSAL MAMBA BENCHMARK ")
    print("="*80)

    with open('ouroboros_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "Batch_Size", "Total_SSD_Read_MB", "Total_Time_s", "Tok_sec_Mean", "Tok_sec_StdDev"])

        for batch in BATCH_SIZES:
            # --- Baseline (Standard SSD Streaming) ---
            std_tok_sec = []
            for _ in range(TRIALS):
                total_time = 0
                for _ in range(NUM_LAYERS):
                    total_time += (SHARED_LAYER_MB / 3000.0) # Simulate a 3GB/s SSD reading the full layer
                    total_time += simulate_compute(batch)
                std_tok_sec.append(batch / total_time)
            
            mean_std_tok = statistics.mean(std_tok_sec)
            writer.writerow(["Standard_Streaming", batch, NUM_LAYERS * SHARED_LAYER_MB, f"{total_time:.2f}", f"{mean_std_tok:.2f}", f"{statistics.stdev(std_tok_sec):.2f}"])

            # --- Ouroboros Architecture ---
            our_tok_sec = []
            for _ in range(TRIALS):
                start_total = time.perf_counter()
                
                # 1. Load the massive shared layer exactly ONCE at startup
                with open(SHARED_FILE, "rb") as f: _ = f.read()
                
                # 2. Inference Loop: Stream only the tiny 1KB vectors
                for _ in range(NUM_LAYERS):
                    with open(COND_FILE, "rb") as f: _ = f.read() # 1KB Read
                    _ = simulate_compute(batch)
                
                total_time = time.perf_counter() - start_total
                our_tok_sec.append(batch / total_time)

            mean_our_tok = statistics.mean(our_tok_sec)
            total_read_mb = SHARED_LAYER_MB + ((NUM_LAYERS * COND_VECTOR_KB) / 1024)
            writer.writerow(["Ouroboros_Shared", batch, total_read_mb, f"{total_time:.2f}", f"{mean_our_tok:.2f}", f"{statistics.stdev(our_tok_sec):.2f}"])
            
            speedup = mean_our_tok / mean_std_tok if mean_std_tok > 0 else 0
            print(f"Batch {batch:<4} | Std Tok/s: {mean_std_tok:>6.2f} | Ouroboros Tok/s: {mean_our_tok:>8.2f} | Speedup: {speedup:>5.1f}x")

    print("[+] Academic data saved to 'ouroboros_metrics.csv'.")

if __name__ == "__main__":
    run_ouroboros_benchmark()
