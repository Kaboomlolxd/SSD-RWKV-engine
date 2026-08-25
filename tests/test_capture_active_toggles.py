"""Tests for capture_active_toggles (A2 — bench stacked bitmask)."""

from __future__ import annotations

import os

import pytest

from rwkv_ssd.runtime.throughput_defaults import capture_active_toggles


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "RWKV_DECODE_SHADOW",
        "RWKV_WARM_DISK_CACHE",
        "RWKV_LUT_GEMM_FUSED",
        "RWKV_STREAM_LAYER_CACHE",
        "RWKV_WARM_PROVIDER_CACHE",
    ):
        monkeypatch.delenv(k, raising=False)


def test_all_off_when_env_unset() -> None:
    snap = capture_active_toggles()
    assert snap["shadow_active"] is False
    assert snap["decode_disk_cache_active"] is False
    assert snap["fused_gemm_active"] is False
    assert snap["stream_layer_cache_active"] is False
    assert snap["warm_z_active"] is False
    assert snap["stacked_count"] == 0
    assert snap["stacked"] is False


def test_individual_on_off() -> None:
    os.environ["RWKV_DECODE_SHADOW"] = "1"
    os.environ["RWKV_WARM_DISK_CACHE"] = "0"
    snap = capture_active_toggles()
    assert snap["shadow_active"] is True
    assert snap["decode_disk_cache_active"] is False
    assert snap["stacked_count"] == 1
    assert snap["stacked"] is False


def test_stacked_flag_turns_on_at_two() -> None:
    os.environ["RWKV_DECODE_SHADOW"] = "1"
    snap = capture_active_toggles()
    assert snap["stacked_count"] == 1
    assert snap["stacked"] is False

    os.environ["RWKV_LUT_GEMM_FUSED"] = "1"
    snap = capture_active_toggles()
    assert snap["stacked_count"] == 2
    assert snap["stacked"] is True


def test_auto_default_for_fused() -> None:
    os.environ["RWKV_LUT_GEMM_FUSED"] = "auto"
    snap = capture_active_toggles()
    assert snap["fused_gemm_active"] is False


def test_explicit_off_strings() -> None:
    for off in ("0", "false", "off", "no", "FALSE", "Off"):
        os.environ["RWKV_DECODE_SHADOW"] = off
        snap = capture_active_toggles()
        assert snap["shadow_active"] is False, f"failed for {off!r}"


def test_explicit_on_strings() -> None:
    for on in ("1", "true", "on", "yes", "TRUE", "On"):
        os.environ["RWKV_DECODE_SHADOW"] = on
        snap = capture_active_toggles()
        assert snap["shadow_active"] is True, f"failed for {on!r}"
