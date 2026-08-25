"""M5 P2.b storage promotion gate on synthetic packs (fast)."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.pack_bench import pack_read_stats
from rwkv_ssd.tools.eval_m5_codec import STORAGE_GATE


def _ratio_vs(pack_a: Path, pack_b: Path) -> float:
    """How many times smaller is pack_b than pack_a (weights.bin)."""
    a = pack_read_stats(pack_a)["weights_mb"]
    b = pack_read_stats(pack_b)["weights_mb"]
    assert b > 0
    return a / b


def test_scale_u8_meets_storage_gate(
    synthetic_pack: Path, synthetic_pack_u8: Path
) -> None:
    assert _ratio_vs(synthetic_pack, synthetic_pack_u8) >= STORAGE_GATE


def test_scale_u4_meets_storage_gate(
    synthetic_pack: Path, synthetic_pack_u4: Path
) -> None:
    assert _ratio_vs(synthetic_pack, synthetic_pack_u4) >= STORAGE_GATE


def test_trinity_lut2_meets_storage_gate(
    synthetic_pack: Path, synthetic_pack_trinity_lut2: Path
) -> None:
    assert _ratio_vs(synthetic_pack, synthetic_pack_trinity_lut2) >= STORAGE_GATE
