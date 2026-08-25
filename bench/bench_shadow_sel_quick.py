#!/usr/bin/env python3
"""Quick shadow_sel vs lut2 comparison after decode fixes."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
PACK_SHADOW = ROOT / "test_model/trinity_eval/trinity_safe_0.1b"
PACK_LUT2 = ROOT / "test_model/trinity_eval/trinity_grouped_0.1b"


def run(pack: Path, label: str) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.throughput_defaults import apply_ssd_tier_fused_defaults

    os.environ["RWKV_PRESET"] = "f1"
    os.environ["RWKV_WARM_DISK_CACHE"] = "0"
    os.environ["RWKV_SSD_TIER"] = "1"
    manifest = Manifest.load(pack)
    cfg = EngineConfig(
        pack_dir=pack,
        checkpoint_path=str(CKPT),
        backend="chatrwkv",
        mode="streaming",
        device="cpu",
        max_tokens=24,
        greedy=True,
        skeleton_load=True,
        stream_layer_cache=False,
        max_layers_in_z=0,
    )
    apply_ssd_tier_fused_defaults(cfg, manifest)
    eng = InferenceEngine(cfg)
    eng.load()
    t0 = time.perf_counter()
    eng.generate("Shadow sel quick bench")
    wall = time.perf_counter() - t0
    m = eng.metrics
    n = 24
    staging = sum(L.staging_ms for L in m.layers) / n
    print(
        f"{label}: {round(n / wall, 2)} tok/s | staging {round(staging, 1)} ms/tok | "
        f"provider {round(m.provider_cache_bytes / 1e6, 1)} MB"
    )
    eng.close()


def main() -> None:
    if not CKPT.is_file():
        print("checkpoint missing, skip")
        return
    for k in list(os.environ):
        if k.startswith("RWKV_"):
            os.environ.pop(k, None)
    run(PACK_SHADOW, "shadow_sel F1")
    for k in list(os.environ):
        if k.startswith("RWKV_"):
            os.environ.pop(k, None)
    run(PACK_LUT2, "lut2 F1")


if __name__ == "__main__":
    main()
