"""
Legacy I/O ceiling scenarios (June 2026).

These combinations are **not** on the RAM→tok/s frontier: they are either
subsumed by a frontier preset, proven slower at the same or higher RAM, or
diagnostic-only. Kept for regression via ``bench/bench_io_ceiling.py --legacy``.

Do not add new scenarios here — extend ``bench/_ram_frontier.py`` frontier tiers.
"""

from __future__ import annotations

# Each entry: (label, kwargs dict for _run_scenario in bench_io_ceiling.py)

LEGACY_SCENARIOS: list[tuple[str, dict]] = [
    (
        "strict (mmap warm)",
        {
            "io_backend": "mmap",
            "stream_layer_cache": False,
            "fused_gemm": False,
            "mmap_dontneed": False,
            "decode_disk_cache": "0",
        },
    ),
    (
        "strict + fused att+ffn",
        {
            "io_backend": "mmap",
            "stream_layer_cache": False,
            "fused_gemm": True,
            "mmap_dontneed": False,
            "decode_disk_cache": "0",
        },
    ),
    (
        "partial hot3 + fused (~260MB)",
        {
            "io_backend": "mmap",
            "stream_layer_cache": True,
            "fused_gemm": True,
            "mmap_dontneed": False,
            "decode_disk_cache": "auto",
            "apply_partial_fused": True,
        },
    ),
    (
        "stream+cache (auto unfused)",
        {
            "io_backend": "mmap",
            "stream_layer_cache": True,
            "fused_gemm": False,
            "mmap_dontneed": False,
            "decode_disk_cache": "auto",
            "apply_stream_defaults": True,
        },
    ),
    (
        "bounded stream+cache",
        {
            "io_backend": "mmap",
            "stream_layer_cache": True,
            "fused_gemm": False,
            "mmap_dontneed": False,
            "decode_disk_cache": "auto",
            "apply_bounded": True,
        },
    ),
    (
        "strict + disk cache zlib",
        {
            "io_backend": "mmap",
            "stream_layer_cache": False,
            "fused_gemm": False,
            "mmap_dontneed": False,
            "decode_disk_cache": "auto",
            "decode_cache_compress": "1",
        },
    ),
    (
        "stacked strict (shadow+fused)",
        {
            "io_backend": "mmap",
            "stream_layer_cache": False,
            "fused_gemm": True,
            "mmap_dontneed": False,
            "decode_disk_cache": "auto",
            "apply_stacked_strict": True,
            "pack_override_key": "shadow_sel",
        },
    ),
    (
        "shadow_sel + fused (no promote)",
        {
            "io_backend": "mmap",
            "stream_layer_cache": False,
            "fused_gemm": True,
            "mmap_dontneed": False,
            "decode_disk_cache": "0",
            "pack_override_key": "shadow_sel",
        },
    ),
]

SUBSUMED_BY = {
    "strict (mmap warm)": "F1 ssd-tier-min (fused + disk)",
    "strict + fused att+ffn": "F1 ssd-tier-min (+ disk cache)",
    "partial hot3 + fused (~260MB)": "F3 partial-hot3 (strict partial wins on NVMe)",
    "stream+cache (auto unfused)": "F2 bounded-fused or F5 promote-max",
    "bounded stream+cache": "F2 bounded-fused (+ fused)",
    "strict + disk cache zlib": "F1 with optional RWKV_DECODE_CACHE_COMPRESS=1",
    "stacked strict (shadow+fused)": "F1 / F5s depending on goal",
    "shadow_sel + fused (no promote)": "F5s promote-shadow",
}
