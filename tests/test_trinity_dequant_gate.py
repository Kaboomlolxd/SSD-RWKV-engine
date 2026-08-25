"""Trinity decode-only gate on synthetic pack (fast)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store


def _decode_only_wall_s(pack: Path, trials: int = 3) -> float:
    """Median wall time over ``trials`` decode runs. First run is the cold
    one; median filters that out so the test is stable on tiny synthetic packs
    where cold-start noise can dominate a single decode (tens of µs)."""
    manifest = Manifest.load(pack)
    entries = manifest.streamed_tensors() or manifest.tensors
    device = torch.device("cpu")
    times: list[float] = []
    for _ in range(max(trials, 1)):
        with open_weight_store(manifest.weights_path) as store:
            blobs = [(e, store.read_bytes(e)) for e in entries]
        t0 = time.perf_counter()
        for entry, raw in blobs:
            decode_weight_to_tensor(raw, entry, device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def test_trinity_lut2_decode_not_orders_of_magnitude_slower(
    synthetic_pack: Path, synthetic_pack_trinity_lut2: Path
) -> None:
    """Smoke: LUT decode on tiny synthetic pack (IDEAS 10% gate is checked on real packs)."""
    base_s = _decode_only_wall_s(synthetic_pack)
    lut_s = _decode_only_wall_s(synthetic_pack_trinity_lut2)
    regression = (lut_s - base_s) / max(base_s, 1e-9)
    # This tiny-pack smoke test only rejects an order-of-magnitude regression;
    # its tens-of-microseconds dense baseline makes tighter ratios unstable
    # under full-suite scheduler load. Real-pack performance owns the 10% gate.
    assert regression <= 9.0, f"trinity_lut2 decode regression {regression:.0%} > 900%"
