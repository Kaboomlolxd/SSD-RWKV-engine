"""Best-effort process working-set telemetry for RAM-budget diagnostics."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
try:
    import resource
except ImportError:  # Windows
    resource = None


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows ``PROCESS_MEMORY_COUNTERS`` with pointer-width fields."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("quota_peak_paged_pool", ctypes.c_size_t),
        ("quota_paged_pool", ctypes.c_size_t),
        ("quota_peak_non_paged_pool", ctypes.c_size_t),
        ("quota_non_paged_pool", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _process_rss_windows_bytes() -> int:
    """Read the Windows working set through the correctly typed PSAPI ABI."""
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        # Without an explicit signature ctypes uses a 32-bit default
        # return/argument ABI. On 64-bit Windows that can truncate the process
        # handle and silently return zero, which made every benchmark and
        # worker report RSS=0.
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if get_info(process, ctypes.byref(counters), counters.cb):
            return int(counters.working_set)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 0


def _process_rss_posix_bytes() -> int:
    """Read the best available RSS value on POSIX-like systems."""
    if resource is None:
        try:
            import psutil  # type: ignore[import-not-found]

            return int(psutil.Process().memory_info().rss)
        except ImportError:
            return 0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(usage * (1024 if usage < 1024 * 1024 * 1024 else 1))


def process_rss_bytes() -> int:
    """Return current process RSS without adding a hard psutil dependency."""
    if os.name == "nt":
        return _process_rss_windows_bytes()
    return _process_rss_posix_bytes()


__all__ = ["process_rss_bytes"]
