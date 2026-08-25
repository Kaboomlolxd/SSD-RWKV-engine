import os

with open("compression_path_comparison_bench.py", "r") as f:
    content = f.read()

# Add E_DPU_Inline to COMPRESSION_PATHS
target = """    'D_VcLLM': {"""
insert = """    'E_DPU_Inline': {
        'label': 'E: SmartNIC/DPU Inline Decompression',
        'description': 'Hardware Decompression on NIC before PCIe',
        'compression_ratio': 10.0,
        'decompress_gbs': 200.0,  # e.g., BlueField-3 hardware decompression engines
        'gpu_compute_for_decompress': False,
        'ppl_70b_estimate': '~5.0',
        'source': 'DPU Inline (Thesis Chapter 10ss)',
        'arxiv': 'N/A',
    },
"""
content = content.replace(target, insert + target)

with open("compression_path_comparison_bench.py", "w") as f:
    f.write(content)
