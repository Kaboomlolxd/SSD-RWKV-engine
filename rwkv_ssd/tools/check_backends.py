#!/usr/bin/env python3
"""Report availability of optional inference backends and tools."""

from __future__ import annotations

import json
import sys

import torch

from rwkv_ssd.backends.albatross import AlbatrossBackend, find_albatross_root
from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.backends.rwkvcpp import find_rwkvcpp_dll, find_rwkvcpp_root, is_rwkvcpp_available
from rwkv_ssd.runtime.io_mmap_advise import MADV_WILLNEED
from rwkv_ssd.runtime.io_posix_fadvise import _POSIX_FADV_WILLNEED


def main() -> None:
    rwkv_root = find_rwkvcpp_root()
    rwkv_dll = find_rwkvcpp_dll(rwkv_root)
    report = {
        "primary": {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
        },
        "chatrwkv": {
            "available": find_chatrwkv_root() is not None,
            "root": str(find_chatrwkv_root() or ""),
        },
        "rwkvcpp": {
            "available": is_rwkvcpp_available(),
            "root": str(rwkv_root or ""),
            "dll": str(rwkv_dll or ""),
            "engine_wired": is_rwkvcpp_available(),
        },
        "albatross": {
            "available": AlbatrossBackend.availability_error() is None,
            "root": str(find_albatross_root() or ""),
            "cuda": torch.cuda.is_available(),
            "engine_wired": True,
        },
        "lightning_proxy": {
            "note": "rwkv_lightning is HTTP-only; use app/lightning_proxy.py with RWKV_LIGHTNING_URL",
        },
        "madvise_mmap": MADV_WILLNEED is not None,
        "posix_fadvise": _POSIX_FADV_WILLNEED is not None,
        "io_uring_engine": False,
    }
    print(json.dumps(report, indent=2))
    if not report["chatrwkv"]["available"]:
        print(
            "ChatRWKV: clone https://github.com/BlinkDL/ChatRWKV and set CHATRWKV_ROOT",
            file=sys.stderr,
        )
    if rwkv_root and not report["rwkvcpp"]["available"]:
        print(
            "rwkv.cpp: repo found but DLL missing — build with "
            "cmake -S backends/rwkvcpp_ref -B backends/rwkvcpp_ref/build && "
            "cmake --build backends/rwkvcpp_ref/build --config Release",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
