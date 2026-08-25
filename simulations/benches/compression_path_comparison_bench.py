import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: COMPRESSION PATH COMPARISON (Chapter 10tt)
# =====================================================================
# Compares 4 distinct compression/decompression paths for SSD-native
# Mamba inference. Each path represents a different point on the
# compression ratio vs decompression speed trade-off surface.
#
# PATH A (Baseline -- Current Compression Trinity):
#   Sparsity (FP16) -> LUT 2-bit -> ANS via nvCOMP
#   Source: QuIP#/AQLM (2-bit LUT) + nvCOMP (ANS)
#   Compression: 10x (LUT 8x * ANS 1.25x)
#   Decompress: 200 GB/s (GPU CUDA cores)
#
# PATH B (BTC-LLM -- Maximum Compression):
#   Sparsity (FP16) -> BTC-LLM 0.8-bit -> ANS via nvCOMP
#   Source: BTC-LLM (Gu et al., 2025, arxiv:2506.12040)
#   Compression: ~17x (0.8-bit = 20x raw, * 0.85 ANS = ~17x)
#   Decompress: ~150 GB/s (binary codebook lookup is more complex)
#   Accuracy: 5.83 PPL on LLaMA-2-13B @ 0.8b (FP16: 4.88)
#
# PATH C (PTQTP -- Maximum Speed):
#   PTQTP 1.58-bit ternary {-1, 0, 1}, multiplication-free
#   Source: PTQTP (Xiao et al., 2025, arxiv:2509.16989)
#   Compression: 10x (16/1.58 = ~10x)
#   Decompress: ~900 GB/s (4.63x FP16, just add/subtract)
#   Accuracy: 8.53 PPL on LLaMA3.1-8B @ 1.58b (FP16: 6.23)
#
# PATH D (VcLLM -- Zero GPU Compute):
#   Sparsity (FP16) -> LUT 2-bit -> NVDEC hardware decode
#   Source: VcLLM (Xu et al., 2024, arxiv:2407.00467)
#   Compression: ~10x (same as Path A)
#   Decompress: ~1100 MB/s (NVDEC hardware codec, zero GPU)
#   Accuracy: On-par with FP16 on commonsense tasks
# =====================================================================

# ---- Hardware Constants ----
SINGLE_DRIVE_BW_GBS = 7.0
DRIVES = 4
RAID_BW_GBS = SINGLE_DRIVE_BW_GBS * DRIVES

# ---- Model Constants ----
MAMBA_70B_FP16_GB = 140.0
MAMBA_70B_LAYERS = 80
COMPUTE_TIME_S = 0.4

# ---- Compression Path Definitions ----
COMPRESSION_PATHS = {
    'A_Compression_Trinity': {
        'label': 'A: Compression Trinity (Baseline)',
        'description': 'Sparsity -> LUT 2-bit -> ANS/nvCOMP',
        'compression_ratio': 10.0,
        'decompress_gbs': 200.0,
        'gpu_compute_for_decompress': True,
        'ppl_70b_estimate': '~5.0',
        'source': 'QuIP#/AQLM + nvCOMP',
        'arxiv': 'N/A',
    },
    'B_BTC_LLM': {
        'label': 'B: BTC-LLM (Max Compression)',
        'description': 'Sparsity -> BTC-LLM 0.8-bit -> ANS',
        'compression_ratio': 17.0,
        'decompress_gbs': 150.0,
        'gpu_compute_for_decompress': True,
        'ppl_70b_estimate': '~4.7',
        'source': 'BTC-LLM (Gu et al., 2025)',
        'arxiv': '2506.12040',
    },
    'C_PTQTP': {
        'label': 'C: PTQTP (Max Speed)',
        'description': 'Ternary 1.58-bit, multiplication-free',
        'compression_ratio': 10.0,
        'decompress_gbs': 900.0,
        'gpu_compute_for_decompress': True,
        'ppl_70b_estimate': '~8.0',
        'source': 'PTQTP (Xiao et al., 2025)',
        'arxiv': '2509.16989',
    },
    'E_DPU_Inline': {
        'label': 'E: SmartNIC/DPU Inline Decompression',
        'description': 'Hardware Decompression on NIC before PCIe',
        'compression_ratio': 10.0,
        'decompress_gbs': 200.0,  # e.g., BlueField-3 hardware decompression engines
        'gpu_compute_for_decompress': False,
        'ppl_70b_estimate': '~5.0',
        'source': 'DPU Inline (Thesis Chapter 10ss)',
        'arxiv': 'N/A',
    },
    'D_VcLLM': {
        'label': 'D: VcLLM (Zero GPU)',
        'description': 'Sparsity -> LUT 2-bit -> NVDEC hardware',
        'compression_ratio': 10.0,
        'decompress_gbs': 1.1,  # 1100 MB/s = 1.1 GB/s (but zero GPU compute)
        'gpu_compute_for_decompress': False,
        'ppl_70b_estimate': '~5.0',
        'source': 'VcLLM (Xu et al., 2024)',
        'arxiv': '2407.00467',
    },
}


def simulate_compression_path(path_name, model_fp16_gb=MAMBA_70B_FP16_GB,
                               layers=MAMBA_70B_LAYERS, compute_time_s=COMPUTE_TIME_S):
    """
    Simulate one full model sweep through a compression path.

    Returns time breakdown and throughput for the given path.
    """
    path = COMPRESSION_PATHS[path_name]

    compressed_size_gb = model_fp16_gb / path['compression_ratio']

    # SSD read time
    t_io = compressed_size_gb / RAID_BW_GBS

    # Decompression time
    t_decompress = compressed_size_gb / path['decompress_gbs']

    # Compute time (matmul on decompressed weights)
    # For PTQTP (Path C), compute is faster because it's multiplication-free
    if path_name == 'C_PTQTP':
        compute_speedup = 4.63  # PTQTP paper: 4.63x faster than FP16
        t_compute = compute_time_s / compute_speedup
    else:
        t_compute = compute_time_s

    # For Path D (VcLLM), decompression runs on NVDEC in parallel with SSD read
    # So decompression doesn't add to the critical path
    # [FIX 15: Decompression Critical Path Fallacy]
    # CRITICAL CORRECTION: Previous logic ignored `t_decompress` for Path D, assuming 
    # it operated with 0 latency. If NVDEC decodes at 1.1 GB/s, it is 10x slower than 
    # the 14 GB/s SSD, and the GPU CANNOT compute weights it hasn't received yet. 
    # Additionally, Paths A, B, C added `t_compute` sequentially, ignoring the 
    # Micro-Pipelining (Chapter 6) overlap between decompression and compute chunks.
    # In a chunked micro-pipeline, the layer latency approaches the maximum of the three 
    # pipelined stages: max(IO, Decompress, Compute), plus a ~10% pipeline bubble penalty.
    t_pipelined = max(t_io, t_decompress, t_compute) * 1.10

    tok_per_s = 1.0 / t_pipelined

    return {
        'path_name': path_name,
        'label': path['label'],
        'description': path['description'],
        'compression_ratio': path['compression_ratio'],
        'compressed_size_gb': compressed_size_gb,
        'decompress_gbs': path['decompress_gbs'],
        't_io_ms': t_io * 1000,
        't_decompress_ms': t_decompress * 1000,
        't_compute_ms': t_compute * 1000,
        't_pipelined_ms': t_pipelined * 1000,
        'tok_per_s': tok_per_s,
        'gpu_compute_for_decompress': path['gpu_compute_for_decompress'],
        'ppl_70b_estimate': path['ppl_70b_estimate'],
        'source': path['source'],
        'arxiv': path['arxiv'],
    }


def run_compression_path_benchmark():
    print("=" * 110)
    print(" THESIS: COMPRESSION PATH COMPARISON (Chapter 10tt)")
    print(" 4 Distinct Design Points on Compression Ratio vs Decompression Speed")
    print("=" * 110)

    print(f"\n  Model: Mamba-70B | FP16 Size: {MAMBA_70B_FP16_GB} GB")
    print(f"  RAID: {DRIVES}x Gen4 NVMe | Bandwidth: {RAID_BW_GBS} GB/s")
    print(f"  Compute: {COMPUTE_TIME_S*1000:.0f}ms per token (FP16 baseline)")

    # ---- Path Comparison Table ----
    print(f"\n{'='*110}")
    print(f" PHASE 1: COMPRESSION PATH COMPARISON")
    print(f"{'='*110}")

    print(f"\n  {'Path':<30} {'Ratio':<8} {'Size':<8} {'Decompress':<14} {'IO (ms)':<10} "
          f"{'Decompress (ms)':<16} {'Compute (ms)':<14} {'Total (ms)':<12} {'Tok/s':<8} {'GPU?':<6}")
    print(f"  {'-'*130}")

    results = []
    for path_name in COMPRESSION_PATHS:
        r = simulate_compression_path(path_name)
        gpu_label = 'Yes' if r['gpu_compute_for_decompress'] else 'No'
        print(f"  {r['label']:<30} {r['compression_ratio']:<8.0f}x {r['compressed_size_gb']:<8.1f}GB "
              f"{r['decompress_gbs']:<14.0f} GB/s {r['t_io_ms']:<10.1f} "
              f"{r['t_decompress_ms']:<16.1f} {r['t_compute_ms']:<14.1f} "
              f"{r['t_pipelined_ms']:<12.1f} {r['tok_per_s']:<8.2f} {gpu_label:<6}")
        results.append(r)

    # ---- Time Breakdown Pie ----
    print(f"\n{'='*110}")
    print(f" PHASE 2: TIME BREAKDOWN (WHERE DOES THE TIME GO?)")
    print(f"{'='*110}")

    for r in results:
        total = r['t_pipelined_ms']
        io_pct = r['t_io_ms'] / total * 100 if total > 0 else 0
        dec_pct = r['t_decompress_ms'] / total * 100 if total > 0 else 0
        comp_pct = r['t_compute_ms'] / total * 100 if total > 0 else 0

        print(f"\n  {r['label']}:")
        print(f"    IO:         {r['t_io_ms']:.1f}ms ({io_pct:.0f}%)")
        print(f"    Decompress: {r['t_decompress_ms']:.1f}ms ({dec_pct:.0f}%)")
        print(f"    Compute:    {r['t_compute_ms']:.1f}ms ({comp_pct:.0f}%)")
        print(f"    Bottleneck: {'IO' if io_pct > 50 else 'Compute' if comp_pct > 50 else 'Balanced'}")

    # ---- Sensitivity: Model Size ----
    print(f"\n{'='*110}")
    print(f" PHASE 3: SCALING ACROSS MODEL SIZES")
    print(f"{'='*110}")

    model_sizes = {
        'Mamba_7B': 14.0,
        'Mamba_13B': 26.0,
        'Mamba_70B': 140.0,
        'Mamba_405B': 810.0,
    }

    print(f"\n  {'Model':<16} {'Path A':<10} {'Path B':<10} {'Path C':<10} {'Path D':<10}")
    print(f"  {'-'*60}")

    for mname, fp16_gb in model_sizes.items():
        print(f"  {mname} ({fp16_gb}GB):", end='')
        for path_name in ['A_Compression_Trinity', 'B_BTC_LLM', 'C_PTQTP', 'D_VcLLM']:
            r = simulate_compression_path(path_name, model_fp16_gb=fp16_gb)
            print(f" {r['tok_per_s']:<10.2f}", end='')
        print()

    # ---- Academic Summary ----
    print(f"\n{'='*110}")
    print(f" ACADEMIC SUMMARY")
    print(f"{'='*110}")

    r_a = [r for r in results if r['path_name'] == 'A_Compression_Trinity'][0]
    r_b = [r for r in results if r['path_name'] == 'B_BTC_LLM'][0]
    r_c = [r for r in results if r['path_name'] == 'C_PTQTP'][0]
    r_d = [r for r in results if r['path_name'] == 'D_VcLLM'][0]

    print(f"""
  CONTRIBUTION: Compression Path Comparison is an ORIGINAL DESIGN SPACE
  EXPLORATION that maps 4 distinct compression/decompression strategies
  to SSD-native Mamba inference, grounded in published papers:

  PATH A (Baseline -- Current Compression Trinity):
    Sparsity -> LUT 2-bit -> ANS/nvCOMP. 10x compression, 200 GB/s
    decompress. This is the thesis baseline.

  PATH B (BTC-LLM -- Maximum Compression):
    Sparsity -> BTC-LLM 0.8-bit -> ANS. ~17x compression (vs 10x),
    reducing 70B payload from {r_a['compressed_size_gb']:.1f}GB to {r_b['compressed_size_gb']:.1f}GB.
    Tok/s improves from {r_a['tok_per_s']:.2f} to {r_b['tok_per_s']:.2f} ({r_b['tok_per_s']/r_a['tok_per_s']:.2f}x).
    Source: BTC-LLM (Gu et al., 2025, arxiv:2506.12040).
    Accuracy: 5.83 PPL on LLaMA-2-13B @ 0.8b (FP16: 4.88).

  PATH C (PTQTP -- Maximum Speed):
    Ternary 1.58-bit, multiplication-free. 10x compression, ~900 GB/s
    decompress (4.63x FP16). Tok/s: {r_c['tok_per_s']:.2f} ({r_c['tok_per_s']/r_a['tok_per_s']:.2f}x).
    Source: PTQTP (Xiao et al., 2025, arxiv:2509.16989).
    Accuracy: 8.53 PPL on LLaMA3.1-8B @ 1.58b (FP16: 6.23).

  PATH D (VcLLM -- Zero GPU Compute):
    Sparsity -> LUT 2-bit -> NVDEC hardware codec. 10x compression,
    zero GPU compute for decompression. Tok/s: {r_d['tok_per_s']:.2f} ({r_d['tok_per_s']/r_a['tok_per_s']:.2f}x).
    Source: VcLLM (Xu et al., 2024, arxiv:2407.00467).
    Runs LLaMA-3-70B @ 128k context on 4x 8GB consumer GPUs.

  KEY INSIGHT: The optimal path depends on the bottleneck:
    - IO-bound systems benefit most from Path B (smaller payload)
    - Compute-bound systems benefit most from Path C (faster decompress)
    - GPU-constrained systems benefit most from Path D (zero GPU)
""")

    # ---- Save CSV ----
    with open('compression_path_comparison_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "Description", "Compression_Ratio", "Compressed_Size_GB",
                         "Decompress_GBs", "T_IO_ms", "T_Decompress_ms", "T_Compute_ms",
                         "T_Pipelined_ms", "Tok_per_s", "GPU_For_Decompress",
                         "PPL_70B_Estimate", "Source", "arXiv"])
        for mname, fp16_gb in model_sizes.items():
            for path_name in COMPRESSION_PATHS:
                r = simulate_compression_path(path_name, model_fp16_gb=fp16_gb)
                writer.writerow([
                    r['path_name'], r['description'], r['compression_ratio'],
                    f"{r['compressed_size_gb']:.2f}", f"{r['decompress_gbs']:.0f}",
                    f"{r['t_io_ms']:.2f}", f"{r['t_decompress_ms']:.2f}",
                    f"{r['t_compute_ms']:.2f}", f"{r['t_pipelined_ms']:.2f}",
                    f"{r['tok_per_s']:.2f}", r['gpu_compute_for_decompress'],
                    r['ppl_70b_estimate'], r['source'], r['arxiv']
                ])

    print("[+] Academic data saved to 'compression_path_comparison_metrics.csv'.")


if __name__ == "__main__":
    run_compression_path_benchmark()
