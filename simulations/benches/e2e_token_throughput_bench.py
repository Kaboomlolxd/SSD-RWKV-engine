import numpy as np
import csv

# =====================================================================
# THESIS EXPERIMENT: END-TO-END TOKEN THROUGHPUT (v5 - Pure SSD-Native)
# =====================================================================
# This script computes verifiable E2E token throughput for SSD-native
# Mamba inference with ALL optimizations stacked, batched from 1 to
# practical limits, with realistic SSD bottleneck modeling.
#
# NO H100 comparisons. NO vLLM extrapolations. Pure SSD-native math.
# Core storage/compression parameters are grounded in individual benchmarks
# or published papers. Several late-stage optimization knobs remain
# exploratory thesis-side compositions rather than settled prior art.
#
# SSD BOTTLENECK REDUCTION TECHNIQUES (from literature):
#   1. LiquidGEMM (2509.01229): Fine-grained pipeline overlapping
#      weight loading + dequant + MMA across warp groups: 2.90x speedup
#   2. Cicada (2502.20959): WeightDecoupler decouples weight file
#      processing from layer construction: 26.17% improvement
#   3. MoE-Lightning (2411.11217): CPU-GPU-I/O pipelining (CGOPipe):
#      10.3x throughput on T4 for MoE
#   4. Hermes/PIPELOAD (2409.04249): Parallel model loading + dynamic
#      memory management: 4.24x speedup
#   5. FlashMem (2602.15379): Static loading schedules + dynamic
#      on-demand streaming: 1.7-75x speedup
#   6. Layered Prefill (2510.08055): Scheduling by layers not tokens,
#      eliminates chunk-induced weight reloads: 70% TTFT reduction
#   7. fastsafetensors (2505.23072): P2P DMA + GPU offloading:
#      4.8-7.5x model loading speedup
#   8. PipeBoost (2503.17707): Pipeline parallelism across loading
#      + inference: 31-50% latency reduction
# Prior-art watchlist added during the Apr 2026 originality audit:
#   - CHEOPS 2025 I/O characterization of SSD model/KV offloading
#   - LoRA-Switch / aLoRA for dynamic adapter switching and cache reuse
#   - Mamba Drafters / SpecMamba for Mamba-specific speculative decoding
#   - Dynamic Mix Precision Routing / Quamba / SFMP for adaptive precision
#   - InstInfer / HillInfer / HiFC for storage-native or in-storage inference
# These are citation targets and calibration references. They should not be
# treated as proof that the exact speedups in this script already exist for
# this full SSD-native recurrent stack.
# =====================================================================

#
# 🚨 CRITICAL SIMULATION REQUIREMENT (OPTION B): 🚨
# While this specific E2E script uses mathematical projection, the rest of the 
# benchmark suite measures actual empirical compute overhead via `torch`. 
# To guarantee validity for the thesis and prevent SSD I/O times from being 
# artificially bottlenecked by slow CPU math, the suite MUST be run on a 
# CUDA-enabled GPU instance. 
#
# ---- Hardware Tier Definitions ----
HARDWARE_TIERS = {
    'Consumer_1xGen4': {
        'label': 'Consumer (1x Gen4 NVMe, CPU-only)',
        'ssd_bw_rated_gbs': 7.0,
        'ssd_bw_sustained_gbs': 5.5,
        'num_drives': 1,
        'has_gpu': False,
        'decompress_gbs': 80.0,
        'decompress_fused_gbs': 120.0,
        'compute_tflops': 2.0,
        'sparse_compute_boost': 1.0,
        'host_overhead_us_per_layer': 15.0,
        'raid_overhead_pct': 0.0,
        'pcie_pipelining_gain': 1.10,
        'dpu_decompression_offload': False,
    },
    'Enthusiast_4xGen5': {
        'label': 'Enthusiast (4x Gen5 NVMe RAID 0, RTX 4090)',
        'ssd_bw_rated_gbs': 28.0,
        'ssd_bw_sustained_gbs': 22.0,
        'num_drives': 4,
        'has_gpu': True,
        'decompress_gbs': 200.0,
        'decompress_fused_gbs': 400.0,
        'compute_tflops': 165.0,
        'sparse_compute_boost': 2.0,
        'host_overhead_us_per_layer': 8.0,
        'raid_overhead_pct': 0.05,
        'pcie_pipelining_gain': 1.10,
        'dpu_decompression_offload': False,
    },
    'Azure_Lasv4': {
        'label': 'Azure Lasv4 (4x NVMe RAID 0, EPYC CPU, no GPU)',
        'ssd_bw_rated_gbs': 20.0,
        'ssd_bw_sustained_gbs': 16.0,
        'num_drives': 4,
        'has_gpu': False,
        'decompress_gbs': 120.0,
        'decompress_fused_gbs': 180.0,
        'compute_tflops': 4.0,
        'sparse_compute_boost': 1.0,
        'host_overhead_us_per_layer': 12.0,
        'raid_overhead_pct': 0.05,
        'pcie_pipelining_gain': 1.10,
        'dpu_decompression_offload': False,
    },
    'Decentralized_NVMe_oF': {
        'label': 'Decentralized Cluster (8x Gen4 via NVMe-oF/100GbE, RTX 4090)',
        'ssd_bw_rated_gbs': 40.0,
        'ssd_bw_sustained_gbs': 32.0,  # Accounting for network overhead
        'num_drives': 8,
        'has_gpu': True,
        'decompress_gbs': 200.0,
        'decompress_fused_gbs': 400.0,
        'compute_tflops': 165.0,
        'sparse_compute_boost': 2.0,
        'host_overhead_us_per_layer': 25.0, # Higher network orchestration overhead
        'raid_overhead_pct': 0.10, # Network striping penalty
        'pcie_pipelining_gain': 1.05, # Slightly lower due to jitter
        'dpu_decompression_offload': True, # Assuming BlueField DPUs at the network edge
        'smartssd_routing_support': True, # Distributed SmartSSDs map the MoE experts autonomously
    },
}

# ---- Model Configurations ----
MODEL_CONFIGS = {
    'Mamba_7B': {
        'fp16_size_gb': 14.0,
        'layers': 32,
        'd_model': 4096,
        'd_state': 256,
    },
    'Mamba_70B': {
        'fp16_size_gb': 140.0,
        'layers': 80,
        'd_model': 8192,
        'd_state': 512,
    },
    'Mamba_405B': {
        'fp16_size_gb': 810.0,
        'layers': 126,
        'd_model': 16384,
        'd_state': 1024,
    },
}

# =====================================================================
# OPTIMIZATION STACK
# =====================================================================

# ---- Compression Path Definitions ----
COMPRESSION_PATHS = {
    'A': {
        'label': 'Compression Trinity (Baseline)',
        'compression_ratio': 10.0,
        'decompress_gbs': 200.0,
        'compute_speedup': 1.0,
    },
    'B': {
        'label': 'BTC-LLM (Max Compression)',
        'compression_ratio': 17.0,
        'decompress_gbs': 150.0,
        'compute_speedup': 1.0,
    },
    'C': {
        'label': 'PTQTP (Max Speed)',
        'compression_ratio': 10.0,
        'decompress_gbs': 900.0,
        'compute_speedup': 4.63,
    },
    'D': {
        'label': 'VcLLM (Zero GPU)',
        'compression_ratio': 10.0,
        'decompress_gbs': 1.1,
        'compute_speedup': 1.0,
        'nvdec_parallel': True,
    },
}

# Default compression paths: C (PTQTP) for GPU (fastest decompress + compute), A (Trinity) for CPU
COMPRESSION_PATH_GPU = 'C'  # PTQTP: 900 GB/s decompress, 4.63x compute speedup
COMPRESSION_PATH_CPU = 'A'  # Compression Trinity: 200 GB/s decompress on CPU

# 2. MTP
MTP_K = 6
SSM_STATEROLL_ACC = 0.87 * 0.60
TREE_BRANCHING = 3
_THEORETICAL_TREE_ACC = 1.0 - (1.0 - SSM_STATEROLL_ACC) ** TREE_BRANCHING
TREE_CORRELATION_DISCOUNT = 0.50
MTP_TREE_ACC = SSM_STATEROLL_ACC + (_THEORETICAL_TREE_ACC - SSM_STATEROLL_ACC) * TREE_CORRELATION_DISCOUNT
MTP_TOKENS_PER_SWEEP = 1.0 + sum(MTP_TREE_ACC ** d for d in range(1, MTP_K + 1))

# 3. Micro-Pipelining
PIPELINE_EFFICIENCY = 0.90

# 4. Gate-Based Prefetch
GATE_PREFETCH_IO_REDUCTION = 0.05

# 5. N-gram Weight Cache
NGRAM_CACHE_HIT_RATE = 0.10

# 6. Residual Channel Prefetch
RESIDUAL_CHANNEL_BW_REDUCTION = 0.30

# 7. Temporal Weight Locality
TEMPORAL_LOCALITY_BW_REDUCTION = 0.175

# 8. ECC Bypass
ECC_RETRY_RATE = 0.02
ECC_MIRROR_LATENCY_US = 66

# 9. SSM Speculation Cache
SSM_SPEC_CACHE_HIT_RATE = 0.86
SSM_SPEC_CACHE_SPEEDUP = 2.02

# 10. MoE Expert Prefetching
MOE_EXPERT_PREFETCH_SPEEDUP = 8.4

# 11. MTP Head Overhead
MTP_HEAD_OVERHEAD_PCT = 0.08

# 12. Kernel Fusion
KERNEL_FUSION_ENABLED = True

# 13. Cross-Config Degradation
CROSS_CONFIG_DEGRADATION_PCT = 0.0

# 14. Activation-Sparse Weight Skipping
ACTIVATION_SPARSITY = 0.50

# 14b. Modality Sparsity Factor
# For multimodal Mamba models processing text-only prompts, visual expert 
# blocks can be dynamically skipped via scatter-gather DMA lists.
MODALITY_SPARSITY_FACTOR = 0.20 # 20% of weights are unused modality experts

# 15. KVPR Partial-Transfer Early Compute
KVPR_PARTIAL_CHUNKS = 4

# 16. AMoE Async Layer Execution
AMOE_ASYNC_ENABLED = True

# 17. [NEW: Idea 5] Graph-Partitioned RAID for MoE
# Co-activated experts are placed on physically different SSDs to ensure maximum
# parallel IO, avoiding SSD-level queue bottlenecks during MoE routing.
GRAPH_PARTITIONED_RAID_GAIN = 1.15 # 15% effective bandwidth boost for MoE models

# 18. [NEW: Idea 6] SmartSSD-Native MoE Routing
# The hidden state is sent to the SSD controller, which computes the MoE gate locally
# and self-fetches the correct experts, removing PCIe orchestration round-trips.
SMARTSSD_ROUTING_LATENCY_REDUCTION = 0.60 # 60% reduction in host orchestration overhead

# 19. [NEW: MIMO] Multi-Input Multi-Output SSM Batching
# Processes g tokens per weight fetch instead of 1. Compatible with Mamba-1/2/3, RWKV.
# MIMO group size g: 1=disabled, 2=2x IO amortization, 4=4x, 8=8x
MIMO_GROUP_SIZE = 4

# 20. [NEW: Pruning] SSM Channel Pruning (Mamba-Shedder + PerfMamba)
# Permanently removes near-zero SSM channels from SSD storage.
# Orthogonal to quantization — applied before compression.
SSM_CHANNEL_PRUNE_RATIO = 0.40  # 40% of channels pruned

# LiquidGEMM: 2.90x for fine-grained pipeline (warp-group overlap)
# Cicada WeightDecoupler: 26.17% improvement
# Hermes/PIPELOAD: 4.24x for parallel model loading
# We apply a conservative combined factor
SSD_BOTTLENECK_REDUCTION = 0.35  # 35% effective IO time reduction from literature techniques


def compute_e2e_throughput(model_config, hw_config, moe_enabled=False, batch_size=1,
                            compression_path='A'):
    """
    Compute projected E2E token throughput with ALL optimizations.

    Pipeline per token:
      1. SSD streams compressed weights (thermal-throttled, RAID overhead)
      2. Host orchestration (TaxBreak: per-layer kernel dispatch)
      3. GPU/CPU decompresses (ANS + LUT, kernel fusion)
      4. GPU/CPU computes (matmul, sparsity)
      5. MTP heads generate candidates
      6. Next sweep verifies

    With micro-pipelining: steps 1-5 overlap across chunks.
    With MTP: each sweep produces ~MTP_TOKENS_PER_SWEEP tokens.
    With batch_size > 1: compute scales sublinearly, IO is shared.
    """
    fp16_size = model_config['fp16_size_gb']
    layers = model_config['layers']
    ssd_bw = hw_config['ssd_bw_sustained_gbs']
    decompress_bw = hw_config['decompress_fused_gbs'] if KERNEL_FUSION_ENABLED else hw_config['decompress_gbs']
    compute_tflops = hw_config['compute_tflops'] * 1e12
    sparse_boost = hw_config['sparse_compute_boost']
    host_overhead_us = hw_config['host_overhead_us_per_layer']
    raid_overhead = hw_config['raid_overhead_pct']
    pipelining_gain = hw_config['pcie_pipelining_gain']

    cp = COMPRESSION_PATHS.get(compression_path, COMPRESSION_PATHS['A'])
    compression_ratio = cp['compression_ratio']
    decompress_bw = cp['decompress_gbs']
    compute_speedup = cp['compute_speedup']
    nvdec_parallel = cp.get('nvdec_parallel', False)
    
    # [FIX 1: Mutually Exclusive Hardware Acceleration]
    # PTQTP (Path C) uses multiplication-free ternary addition trees, completely bypassing 
    # the 2:4 structured sparsity Tensor Core hardware. They cannot be multiplied.
    if compression_path == 'C':
        sparse_boost = 1.0  # Disable Tensor Core sparsity boost
        activation_sparsity_boost = 1.0 # PTQTP handles sparse activations differently
    else:
        activation_sparsity_boost = 1.0 + 0.55 * ACTIVATION_SPARSITY


    # Compressed model size — apply SSM channel pruning BEFORE compression
    # Pruning permanently removes near-zero channels from storage (Mamba-Shedder)
    pruned_fp16_size = fp16_size * (1.0 - SSM_CHANNEL_PRUNE_RATIO)
    compressed_size_gb = pruned_fp16_size / compression_ratio

    # ---- Effective SSD payload after all bandwidth-reducing optimizations ----
    effective_ssd_payload = compressed_size_gb * (1.0 - NGRAM_CACHE_HIT_RATE)
    effective_ssd_payload *= (1.0 - RESIDUAL_CHANNEL_BW_REDUCTION)
    effective_ssd_payload *= (1.0 - TEMPORAL_LOCALITY_BW_REDUCTION)
    effective_ssd_payload *= (1.0 - ACTIVATION_SPARSITY)
    
    # [NEW: Idea 2] Modality Sparsity Factor (e.g. text-only prompt skipping visual expert LBAs)
    effective_ssd_payload *= (1.0 - MODALITY_SPARSITY_FACTOR)

    # Gate prefetch reduces IO submission overhead
    effective_ssd_bw = ssd_bw * (1.0 + GATE_PREFETCH_IO_REDUCTION)

    # RAID stripe overhead
    effective_ssd_bw *= (1.0 - raid_overhead)

    # Cross-config degradation
    effective_ssd_bw *= (1.0 - CROSS_CONFIG_DEGRADATION_PCT / 100.0)

    # ---- Time components for one full model sweep ----

    # 1. SSD read time
    t_io = effective_ssd_payload / effective_ssd_bw

    # SSD bottleneck reduction (LiquidGEMM, Cicada, Hermes, etc.)
    t_io *= (1.0 - SSD_BOTTLENECK_REDUCTION)

    # [NEW: MIMO] Multi-Input Multi-Output SSM Batching
    # MIMO amortizes the weight fetch across g tokens. Compatible with Mamba-1/2/3, RWKV.
    # The SSD reads weights once per group, not once per token.
    mimo_g = MIMO_GROUP_SIZE
    if mimo_g > 1:
        t_io /= mimo_g  # Weight fetch amortized across g tokens

    # [NEW: Idea 5] Graph-Partitioned RAID for MoE models
    if moe_enabled:
        effective_ssd_bw *= GRAPH_PARTITIONED_RAID_GAIN
        t_io = effective_ssd_payload / effective_ssd_bw  # Recalculate with boost
        t_io *= (1.0 - SSD_BOTTLENECK_REDUCTION)

    # 2. Host orchestration overhead
    moe_kernel_multiplier = 8.0 if moe_enabled else 1.0
    t_host = layers * host_overhead_us * moe_kernel_multiplier / 1e6
    
    # [NEW: Idea 6] SmartSSD-Native MoE Routing
    if moe_enabled and hw_config.get('smartssd_routing_support', False):
        t_host *= (1.0 - SMARTSSD_ROUTING_LATENCY_REDUCTION)
    
    # 3. [NEW: Idea 1] DPU Decompression Offload
    dpu_offload = hw_config.get('dpu_decompression_offload', False)
    if dpu_offload:
        # If the DPU decompresses inline, the host GPU/CPU spends ZERO time doing ANS decompression
        t_decompress = 0.0
    else:
        # Decompression time
        t_decompress = compressed_size_gb / decompress_bw

    # 4. Compute time — adjust for pruned channels (fewer active params)
    # Mamba-Shedder removes 40% of SSM channels; compute scales linearly with active params
    pruned_param_factor = (1.0 - SSM_CHANNEL_PRUNE_RATIO * 0.3)  # Only SSM params pruned (~30% of total FLOPs)
    num_params = fp16_size * 1e9 / 2 * pruned_param_factor
    flops_per_token = 2 * num_params / sparse_boost
    t_compute = flops_per_token / compute_tflops / activation_sparsity_boost / compute_speedup

    # MIMO increases per-sweep compute (batched matmul across g tokens)
    # but this is overlapped with IO and hidden by the memory-bound nature of decode
    if mimo_g > 1:
        t_compute *= mimo_g ** 0.3  # Sublinear compute scaling for MIMO batch

    # 5. MTP head overhead
    t_mtp_overhead = t_compute * MTP_HEAD_OVERHEAD_PCT

    # 6. ECC overhead
    ecc_overhead_s = layers * ECC_RETRY_RATE * (ECC_MIRROR_LATENCY_US / 1e6)

    # 7. [NEW: Idea 4] State Checkpointing Overhead
    # Normally, parking the Mamba state to SSD takes ~3ms per token generation sweep
    # With Shadow State Checkpointing (using MTP State-Rolling) and pSLC, we hide this latency entirely
    # Furthermore, ZTS-Scan (CSD offload) removes the PCIe state transfer overhead.
    shadow_checkpointing = True
    zts_scan_active = hw_config.get('zts_scan_active', False)
    
    if zts_scan_active:
        checkpointing_overhead_s = 0.0 # Handled internally by SSD controller
        t_io_state_pcie_overhead = 0.0 # No state transferred over PCIe
    else:
        checkpointing_overhead_s = 0.0 if shadow_checkpointing else 0.003
        t_io_state_pcie_overhead = 0.035 # Standard 35ms state transfer overhead per token (Read + Write)

    # Total overhead (non-overlappable)
    total_overhead_s = t_host + ecc_overhead_s + checkpointing_overhead_s + t_io_state_pcie_overhead

    # ---- Batch scaling ----
    # For batch_size > 1 or MTP generating multiple tokens:
    # - IO is shared (weights read once per batch)
    # - Compute scales sublinearly based on total effective tokens processed simultaneously: 
    #   (batch_size * tokens_per_sweep)^0.7
    # - Host overhead scales linearly (per-request)
    
    # [FIX 2: Zero-Cost Verification Flaw]
    # MTP verification processes all drafted tokens at once. If batch is 64 and MTP draft depth is 6,
    # the forward pass computes 384 tokens simultaneously. Compute FLOPs strictly increase.
    effective_tokens_in_forward_pass = batch_size * MTP_K if MTP_K > 1 else batch_size
    
    if effective_tokens_in_forward_pass > 1:
        compute_scaling = effective_tokens_in_forward_pass ** 0.7
        t_compute *= compute_scaling
        t_mtp_overhead *= compute_scaling
    
    if batch_size > 1:
        t_host *= batch_size

    # Without pipelining: serial
    t_serial = t_io + t_decompress + t_compute + t_mtp_overhead + total_overhead_s

    # With micro-pipelining
    t_pipelined_raw = max(t_io, t_decompress + t_compute + t_mtp_overhead + total_overhead_s) / PIPELINE_EFFICIENCY

    # KVPR partial-transfer early compute
    kvpr_reduction = 1.0 + 0.10 * (KVPR_PARTIAL_CHUNKS / 4.0)
    t_pipelined_kvpr = t_pipelined_raw / kvpr_reduction

    # AMoE async layer execution
    amoe_reduction = 1.15 if AMOE_ASYNC_ENABLED else 1.0
    t_pipelined = t_pipelined_kvpr / amoe_reduction

    # ZipFlow pipelining gain
    t_pipelined = t_pipelined / pipelining_gain

    # Determine bottleneck
    compute_total = t_decompress + t_compute + t_mtp_overhead + total_overhead_s
    if t_io > compute_total:
        bottleneck = 'IO-bound (SSD bandwidth)'
    elif t_compute > t_decompress:
        bottleneck = 'Compute-bound (matmul)'
    elif t_decompress > t_compute:
        bottleneck = 'Decompress-bound (ANS+LUT)'
    else:
        bottleneck = 'Host-bound (orchestration)'

    # Tokens per sweep
    tokens_per_sweep = MTP_TOKENS_PER_SWEEP

    # SSM Speculation Cache
    spec_cache_factor = 1.0 + (SSM_SPEC_CACHE_SPEEDUP - 1.0) * SSM_SPEC_CACHE_HIT_RATE

    # MoE Expert Prefetching
    moe_factor = MOE_EXPERT_PREFETCH_SPEEDUP if moe_enabled else 1.0

    # Final throughput
    tok_per_s_serial = 1.0 / t_serial
    tok_per_s_pipelined = 1.0 / t_pipelined
    tok_per_s_full = tokens_per_sweep / t_pipelined * spec_cache_factor * moe_factor

    # Per-user throughput
    tok_per_s_per_user = tok_per_s_full / batch_size if batch_size > 0 else 0

    # Time breakdown
    time_breakdown = {
        't_io_ms': t_io * 1000,
        't_host_ms': t_host * 1000,
        't_decompress_ms': t_decompress * 1000,
        't_compute_ms': t_compute * 1000,
        't_mtp_overhead_ms': t_mtp_overhead * 1000,
        't_ecc_ms': ecc_overhead_s * 1000,
        't_total_overhead_ms': total_overhead_s * 1000,
        't_serial_ms': t_serial * 1000,
        't_pipelined_ms': t_pipelined * 1000,
    }

    # Bandwidth breakdown
    bw_breakdown = {
        'base_fp16_gb': fp16_size,
        'base_compressed_gb': compressed_size_gb,
        'after_ngram_cache_gb': compressed_size_gb * (1.0 - NGRAM_CACHE_HIT_RATE),
        'after_residual_channel_gb': compressed_size_gb * (1.0 - NGRAM_CACHE_HIT_RATE) * (1.0 - RESIDUAL_CHANNEL_BW_REDUCTION),
        'after_temporal_locality_gb': compressed_size_gb * (1.0 - NGRAM_CACHE_HIT_RATE) * (1.0 - RESIDUAL_CHANNEL_BW_REDUCTION) * (1.0 - TEMPORAL_LOCALITY_BW_REDUCTION),
        'after_activation_sparsity_gb': effective_ssd_payload,
        'total_compression_ratio': fp16_size / effective_ssd_payload if effective_ssd_payload > 0 else 0,
        'total_bw_reduction_pct': (1.0 - effective_ssd_payload / compressed_size_gb) * 100,
    }

    # Overhead analysis
    overhead_analysis = {
        'io_pct': t_io / t_pipelined_raw * 100 if t_pipelined_raw > 0 else 0,
        'host_pct': t_host / t_pipelined_raw * 100 if t_pipelined_raw > 0 else 0,
        'decompress_pct': t_decompress / t_pipelined_raw * 100 if t_pipelined_raw > 0 else 0,
        'compute_pct': t_compute / t_pipelined_raw * 100 if t_pipelined_raw > 0 else 0,
        'mtp_overhead_pct': t_mtp_overhead / t_pipelined_raw * 100 if t_pipelined_raw > 0 else 0,
        'pipelining_gain_pct': (pipelining_gain - 1.0) * 100,
        'kernel_fused': KERNEL_FUSION_ENABLED,
    }

    return {
        'compressed_size_gb': compressed_size_gb,
        'effective_ssd_payload_gb': effective_ssd_payload,
        'ssd_bw_used_gbs': effective_ssd_bw,
        'tok_per_s_serial': tok_per_s_serial,
        'tok_per_s_pipelined': tok_per_s_pipelined,
        'tok_per_s_full_stack': tok_per_s_full,
        'tok_per_s_per_user': tok_per_s_per_user,
        'bottleneck': bottleneck,
        'tokens_per_sweep': tokens_per_sweep,
        'spec_cache_factor': spec_cache_factor,
        'moe_factor': moe_factor,
        'batch_size': batch_size,
        'time_breakdown': time_breakdown,
        'bw_breakdown': bw_breakdown,
        'overhead_analysis': overhead_analysis,
    }


def run_e2e_benchmark():
    print("=" * 110)
    print(" THESIS: END-TO-END TOKEN THROUGHPUT PROJECTION (v5)")
    print(" Pure SSD-Native, No External Comparisons, Batch Scaling 1..N")
    print("=" * 110)

    print(f"\n  Optimization Stack:")
    print(f"    Compression Path (GPU): {COMPRESSION_PATH_GPU} - {COMPRESSION_PATHS[COMPRESSION_PATH_GPU]['label']}")
    print(f"    Compression Path (CPU): {COMPRESSION_PATH_CPU} - {COMPRESSION_PATHS[COMPRESSION_PATH_CPU]['label']}")
    print(f"    Compression Ratio      : {COMPRESSION_PATHS['A']['compression_ratio']:.0f}x")
    print(f"    Decompress BW (GPU)    : {COMPRESSION_PATHS[COMPRESSION_PATH_GPU]['decompress_gbs']:.0f} GB/s")
    print(f"    Decompress BW (CPU)    : {COMPRESSION_PATHS[COMPRESSION_PATH_CPU]['decompress_gbs']:.0f} GB/s")
    print(f"    MTP Tree (k={MTP_K}, B={TREE_BRANCHING}): {MTP_TOKENS_PER_SWEEP:.2f} tokens/sweep")
    print(f"    Micro-Pipeline         : {PIPELINE_EFFICIENCY*100:.0f}% efficiency (K=16 chunks)")
    print(f"    Gate Prefetch          : +{GATE_PREFETCH_IO_REDUCTION*100:.0f}% IO sub")
    print(f"    N-gram Cache           : {NGRAM_CACHE_HIT_RATE*100:.0f}% hit rate")
    print(f"    Residual Channel       : {RESIDUAL_CHANNEL_BW_REDUCTION*100:.0f}% BW reduction")
    print(f"    Temporal Locality      : {TEMPORAL_LOCALITY_BW_REDUCTION*100:.0f}% BW reduction")
    print(f"    Activation Sparsity    : {ACTIVATION_SPARSITY*100:.0f}% weight skipping")
    print(f"    ECC Bypass             : {ECC_MIRROR_LATENCY_US}us mirror")
    print(f"    SSM Spec Cache         : {SSM_SPEC_CACHE_HIT_RATE*100:.0f}% hit, {SSM_SPEC_CACHE_SPEEDUP:.2f}x")
    print(f"    KVPR Partial-Transfer  : {KVPR_PARTIAL_CHUNKS} chunks")
    print(f"    AMoE Async Layer       : {'Enabled' if AMOE_ASYNC_ENABLED else 'Disabled'}")
    print(f"    SSD Bottleneck Reduce  : {SSD_BOTTLENECK_REDUCTION*100:.0f}% (LiquidGEMM+Cicada+Hermes)")
    print(f"    Kernel Fusion          : {'Enabled' if KERNEL_FUSION_ENABLED else 'Disabled'}")
    print(f"    PCIe Pipelining        : +10%")
    print(f"    Thermal (sustained)    : ~21% BW drop")
    print(f"    RAID Stripe            : 5% sync overhead")
    print(f"    MIMO Group Size (g)    : {MIMO_GROUP_SIZE} ({MIMO_GROUP_SIZE}x IO amortization)")
    print(f"    SSM Channel Pruning    : {SSM_CHANNEL_PRUNE_RATIO*100:.0f}% (Mamba-Shedder)")

    with open('e2e_token_throughput_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Hardware_Tier", "Model", "MoE", "Batch", "FP16_GB", "Compressed_GB",
            "Effective_Payload_GB", "SSD_BW_GBs",
            "T_IO_ms", "T_Host_ms", "T_Decompress_ms", "T_Compute_ms",
            "T_Pipelined_ms", "Bottleneck",
            "IO_Pct", "Host_Pct", "Decompress_Pct", "Compute_Pct",
            "MTP_Tokens/Sweep", "Spec_Cache_Factor", "MoE_Factor",
            "TokPerS_Total", "TokPerS_PerUser",
        ])

        for hw_name, hw_config in HARDWARE_TIERS.items():
            print(f"\n{'='*110}")
            print(f"  {hw_config['label']}")
            print(f"  SSD BW: {hw_config['ssd_bw_rated_gbs']} GB/s rated, "
                  f"{hw_config['ssd_bw_sustained_gbs']} GB/s sustained (thermal)")
            print(f"{'='*110}")

            # Batch sizes: 1 (single user) to practical limits
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            for moe in [False, True]:
                if moe:
                    print(f"\n  [MoE Mamba -- Expert Prefetching ENABLED]")

                for cp_name, cp_label in [(COMPRESSION_PATH_GPU, 'GPU Path (PTQTP)'),
                                           (COMPRESSION_PATH_CPU, 'CPU Path (Trinity)')]:
                    print(f"\n  [{cp_label}]")
                    print(f"  {'Model':<15} {'Batch':<8} {'Total tok/s':<14} {'Per-User tok/s':<16} "
                          f"{'Token Time (ms)':<18} {'Bottleneck':<25}")
                    print(f"  {'-'*100}")

                    for model_name, model_config in MODEL_CONFIGS.items():
                        for bs in batch_sizes:
                            r = compute_e2e_throughput(model_config, hw_config, moe_enabled=moe,
                                                        batch_size=bs, compression_path=cp_name)
                            tb = r['time_breakdown']

                            if bs in [1, 4, 16, 64, 256]:
                                print(f"  {model_name:<15} {bs:<8} {r['tok_per_s_full_stack']:<14.1f} "
                                      f"{r['tok_per_s_per_user']:<16.2f} {tb['t_pipelined_ms']:<18.1f} "
                                      f"{r['bottleneck']:<25}")

                            writer.writerow([
                                hw_name, model_name, moe, bs,
                                f"{model_config['fp16_size_gb']:.1f}",
                                f"{r['compressed_size_gb']:.1f}",
                                f"{r['effective_ssd_payload_gb']:.1f}",
                                f"{r['ssd_bw_used_gbs']:.1f}",
                                f"{tb['t_io_ms']:.2f}", f"{tb['t_host_ms']:.2f}",
                                f"{tb['t_decompress_ms']:.2f}", f"{tb['t_compute_ms']:.2f}",
                                f"{tb['t_pipelined_ms']:.2f}",
                                r['bottleneck'],
                                f"{r['overhead_analysis']['io_pct']:.1f}",
                                f"{r['overhead_analysis']['host_pct']:.1f}",
                                f"{r['overhead_analysis']['decompress_pct']:.1f}",
                                f"{r['overhead_analysis']['compute_pct']:.1f}",
                                f"{r['tokens_per_sweep']:.2f}",
                                f"{r['spec_cache_factor']:.2f}",
                                f"{r['moe_factor']:.2f}",
                                f"{r['tok_per_s_full_stack']:.2f}",
                                f"{r['tok_per_s_per_user']:.2f}",
                            ])

    # --- Key Findings ---
    print(f"\n{'='*110}")
    print(f" KEY FINDINGS")
    print(f"{'='*110}")

    for hw_name in HARDWARE_TIERS:
        hw = HARDWARE_TIERS[hw_name]
        # Determine which path to use for this hardware
        cp = COMPRESSION_PATH_GPU if hw['has_gpu'] else COMPRESSION_PATH_CPU
        print(f"\n  {hw['label']} (Path {cp}):")
        for mname in MODEL_CONFIGS:
            r1 = compute_e2e_throughput(MODEL_CONFIGS[mname], hw, batch_size=1, compression_path=cp)
            r4 = compute_e2e_throughput(MODEL_CONFIGS[mname], hw, batch_size=4, compression_path=cp)
            r16 = compute_e2e_throughput(MODEL_CONFIGS[mname], hw, batch_size=16, compression_path=cp)
            r64 = compute_e2e_throughput(MODEL_CONFIGS[mname], hw, batch_size=64, compression_path=cp)
            r256 = compute_e2e_throughput(MODEL_CONFIGS[mname], hw, batch_size=256, compression_path=cp)

            print(f"    {mname}:")
            print(f"      Batch=1:   {r1['tok_per_s_full_stack']:.1f} tok/s total, "
                  f"{r1['tok_per_s_per_user']:.2f}/user | {r1['bottleneck']}")
            print(f"      Batch=4:   {r4['tok_per_s_full_stack']:.1f} tok/s total, "
                  f"{r4['tok_per_s_per_user']:.2f}/user | {r4['bottleneck']}")
            print(f"      Batch=16:  {r16['tok_per_s_full_stack']:.1f} tok/s total, "
                  f"{r16['tok_per_s_per_user']:.2f}/user | {r16['bottleneck']}")
            print(f"      Batch=64:  {r64['tok_per_s_full_stack']:.1f} tok/s total, "
                  f"{r64['tok_per_s_per_user']:.2f}/user | {r64['bottleneck']}")
            print(f"      Batch=256: {r256['tok_per_s_full_stack']:.1f} tok/s total, "
                  f"{r256['tok_per_s_per_user']:.2f}/user | {r256['bottleneck']}")

    # Bandwidth reduction breakdown (show both paths)
    print(f"\n  BANDWIDTH REDUCTION BREAKDOWN (70B, Enthusiast):")
    for cp_name, cp_label in [(COMPRESSION_PATH_GPU, 'GPU Path (PTQTP)'),
                               (COMPRESSION_PATH_CPU, 'CPU Path (Trinity)')]:
        r70b = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Enthusiast_4xGen5'],
                                       compression_path=cp_name)
        bd = r70b['bw_breakdown']
        print(f"\n  [{cp_label}]")
        print(f"    Base FP16:               {bd['base_fp16_gb']:.1f} GB")
        print(f"    After Compression:       {bd['base_compressed_gb']:.1f} GB "
              f"({COMPRESSION_PATHS[cp_name]['compression_ratio']:.0f}x)")
        print(f"    After N-gram cache:      {bd['after_ngram_cache_gb']:.1f} GB "
              f"({NGRAM_CACHE_HIT_RATE*100:.0f}% from RAM)")
        print(f"    After residual channel:  {bd['after_residual_channel_gb']:.1f} GB "
              f"({RESIDUAL_CHANNEL_BW_REDUCTION*100:.0f}% predicted)")
        print(f"    After temporal locality: {bd['after_temporal_locality_gb']:.1f} GB "
              f"({TEMPORAL_LOCALITY_BW_REDUCTION*100:.0f}% Q2)")
        print(f"    After act. sparsity:     {bd['after_activation_sparsity_gb']:.1f} GB "
              f"({ACTIVATION_SPARSITY*100:.0f}% skipped)")
        print(f"    Effective compression:   {bd['total_compression_ratio']:.1f}x")
        print(f"    Total BW reduction:      {bd['total_bw_reduction_pct']:.1f}%")

    # Bottleneck analysis (show both paths)
    print(f"\n  BOTTLENECK ANALYSIS (70B, Enthusiast, Batch=1):")
    for cp_name, cp_label in [(COMPRESSION_PATH_GPU, 'GPU Path (PTQTP)'),
                               (COMPRESSION_PATH_CPU, 'CPU Path (Trinity)')]:
        r70b = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Enthusiast_4xGen5'],
                                       compression_path=cp_name)
        oa = r70b['overhead_analysis']
        print(f"\n  [{cp_label}]")
        print(f"    IO:         {oa['io_pct']:.1f}% of pipeline time")
        print(f"    Host:       {oa['host_pct']:.1f}%")
        print(f"    Decompress: {oa['decompress_pct']:.1f}%")
        print(f"    Compute:    {oa['compute_pct']:.1f}%")
        print(f"    MTP Overhead: {oa['mtp_overhead_pct']:.1f}%")

    # Sensitivity analysis
    print(f"\n{'='*110}")
    print(f" SENSITIVITY ANALYSIS: Mamba_70B on Enthusiast (Batch=1)")
    print(f"{'='*110}")

    nominal_gpu = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Enthusiast_4xGen5'],
                                          compression_path=COMPRESSION_PATH_GPU)
    nominal_cpu = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Azure_Lasv4'],
                                          compression_path=COMPRESSION_PATH_CPU)
    nominal_tok = nominal_gpu['tok_per_s_full_stack']
    print(f"\n  Nominal (GPU Path {COMPRESSION_PATH_GPU}): {nominal_tok:.2f} tok/s")
    print(f"  Nominal (CPU Path {COMPRESSION_PATH_CPU}): {nominal_cpu['tok_per_s_full_stack']:.2f} tok/s")

    params = {
        'Compression Ratio': ('_compression_ratio', [7.0, 8.5, 10.0, 11.5, 13.0]),
        'SSD Sustained BW': ('ssd_bw_sustained_gbs', [11.0, 16.5, 22.0, 27.5, 33.0]),
        'Activation Sparsity': ('ACTIVATION_SPARSITY', [0.20, 0.35, 0.50, 0.65, 0.80]),
        'Spec Cache Hit': ('SSM_SPEC_CACHE_HIT_RATE', [0.0, 0.43, 0.86, 0.93, 1.0]),
        'SSD Bottleneck Red.': ('SSD_BOTTLENECK_REDUCTION', [0.0, 0.17, 0.35, 0.50, 0.65]),
        'Pipeline Efficiency': ('PIPELINE_EFFICIENCY', [0.70, 0.80, 0.90, 0.95, 0.98]),
        'MTP Acceptance': ('MTP_TREE_ACC', [0.35, 0.45, 0.52, 0.60, 0.70]),
    }

    print(f"\n  {'Parameter':<22} {'-30%':<12} {'-15%':<12} {'Nominal':<12} {'+15%':<12} {'+30%':<12}")
    print(f"  {'-'*85}")

    for param_name, (attr, values) in params.items():
        results = []
        for val in values:
            if attr == '_compression_ratio':
                # Override compression ratio in the default path
                old_ratio = COMPRESSION_PATHS['A']['compression_ratio']
                COMPRESSION_PATHS['A']['compression_ratio'] = val
                r = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Enthusiast_4xGen5'])
                results.append(r['tok_per_s_full_stack'])
                COMPRESSION_PATHS['A']['compression_ratio'] = old_ratio
            elif attr in globals():
                old_val = globals()[attr]
                globals()[attr] = val
                r = compute_e2e_throughput(MODEL_CONFIGS['Mamba_70B'], HARDWARE_TIERS['Enthusiast_4xGen5'])
                results.append(r['tok_per_s_full_stack'])
                globals()[attr] = old_val
            else:
                results.append(nominal_tok)

        print(f"  {param_name:<22} {results[0]:<12.2f} {results[1]:<12.2f} "
              f"{results[2]:<12.2f} {results[3]:<12.2f} {results[4]:<12.2f}")

    # Honest caveats
    print(f"\n{'='*110}")
    print(f" HONEST CAVEATS")
    print(f"{'='*110}")
    print(f"""
  1. These are PROJECTIONS from component-level benchmarks and published papers.
  2. Core parameters cite a specific paper or benchmark; several late-stage
     knobs are exploratory compositions added for scenario analysis.
  3. No external GPU comparisons - these are pure SSD-native projections.
  4. Batch scaling uses sublinear compute scaling (batch^0.7) reflecting
     memory-bandwidth-bound decode phase.
  5. SSD bottleneck reduction (35%) is grounded in LiquidGEMM (2.90x),
     Cicada WeightDecoupler (26%), and Hermes (4.24x) literature.
  6. Perplexity at 2-bit is NOT measured (see QuIP#/AQLM literature).
  7. All optimizations assumed to compose without interference.
  8. Thermal throttling modeled at ~21% sustained BW drop.
""")

    print("[+] Academic data saved to 'e2e_token_throughput_metrics.csv'.")


if __name__ == "__main__":
    run_e2e_benchmark()
