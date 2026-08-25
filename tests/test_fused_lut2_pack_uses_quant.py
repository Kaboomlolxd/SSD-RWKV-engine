"""Tests that fused_lut2_enabled is gated by pack codec (regression for fp16_grouped streaming)."""

from __future__ import annotations

import os

import pytest

from rwkv_ssd.runtime.lut_gemm_fused import fused_lut2_enabled


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RWKV_LUT_GEMM_FUSED", raising=False)


@pytest.mark.parametrize(
    "mode,stream_layer_cache,pack_uses_quant,expected",
    [
        ("resident", False, True, False),
        ("resident", False, False, False),
        ("streaming", False, True, True),
        ("streaming", False, False, False),
        ("streaming", True, True, True),
        ("streaming", True, False, False),
        ("partial", False, False, False),
        ("partial", False, True, True),
    ],
)
def test_fused_lut2_enabled_pack_uses_quant_gating(
    mode: str, stream_layer_cache: bool, pack_uses_quant: bool, expected: bool
) -> None:
    assert (
        fused_lut2_enabled(mode, stream_layer_cache, pack_uses_quant=pack_uses_quant)
        is expected
    )


@pytest.mark.parametrize(
    "pack_uses_quant",
    [True, False],
)
def test_fused_lut2_explicit_true_is_overridden_by_pack_uses_quant_false(
    pack_uses_quant: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RWKV_LUT_GEMM_FUSED=1 is hard-required only if the pack actually uses a quant codec."""
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    expected = pack_uses_quant is True
    assert (
        fused_lut2_enabled("streaming", False, pack_uses_quant=pack_uses_quant)
        is expected
    )
