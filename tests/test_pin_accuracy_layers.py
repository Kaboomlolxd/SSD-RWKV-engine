"""Tests for RWKV_PIN_ACCURACY_LAYERS (A3) and snapshot_pinned_layers.

Verifies:
- Bounded-z streaming mode adds block layer 0 and n_layer-1 to the
  pinned set when RWKV_PIN_ACCURACY_LAYERS=auto (default).
- warm_z / max_z >= n_layer does NOT trigger pinning (nothing to pin).
- Explicit off (RWKV_PIN_ACCURACY_LAYERS=0) disables the auto-pin.
- LRU eviction skips pinned layers.
- ``extra_layer_ids`` from caller is always honored.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RWKV_PIN_ACCURACY_LAYERS", raising=False)


def _make_z(*layer_ids: int) -> dict:
    return {f"blocks.{lid}.att.x_r": object() for lid in layer_ids}


def test_snapshot_pinned_layers_default_auto_pins_first_last() -> None:
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    # max_z>=3 leaves room for pins + at least one LRU slot.
    pinned = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=3)
    assert 0 in pinned
    assert 11 in pinned


def test_snapshot_pinned_layers_bounded_f2_skips_auto_pins() -> None:
    """F2 max_z=2: auto accuracy pins would consume the whole window."""
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    pinned = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=2)
    assert pinned == set()
    pinned1 = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=1)
    assert pinned1 == set()


def test_snapshot_pinned_layers_warm_z_no_pin() -> None:
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    pinned = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=12)
    assert 0 not in pinned
    assert 11 not in pinned


def test_snapshot_pinned_layers_explicit_off_disables() -> None:
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "0"
    pinned = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=3)
    assert pinned == set()


def test_snapshot_pinned_layers_explicit_on() -> None:
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "1"
    # Explicit on forces pins even under a tight z cap.
    pinned = snapshot_pinned_layers(None, n_block_layers=12, max_layers_in_z=1)
    assert 0 in pinned
    assert 11 in pinned


def test_snapshot_pinned_layers_includes_z_resident() -> None:
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    z = _make_z(0, 1, 2, 3)
    pinned = snapshot_pinned_layers(
        z, n_block_layers=12, max_layers_in_z=3, extra_layer_ids={5, 6}
    )
    assert {0, 1, 2, 3, 5, 6} <= pinned
    assert 11 in pinned


def test_lru_touch_skips_pinned_layer() -> None:
    from rwkv_ssd.runtime.z_layer_retention import ZLayerRetention

    z = _make_z(0, 1, 5, 11)
    ret = ZLayerRetention(max_layers_in_z=1, pinned_layer_ids={11})
    ret.touch(z, 0)
    ret.touch(z, 1)
    ret.touch(z, 11)
    ret.touch(z, 5)
    assert "blocks.11.att.x_r" in z, "pinned layer must not be evicted"
    assert "blocks.5.att.x_r" in z
    assert 5 in ret._lru


def test_resolve_accuracy_pin_active() -> None:
    from rwkv_ssd.runtime.z_layer_retention import _resolve_accuracy_pin_active

    assert _resolve_accuracy_pin_active() is True
    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "0"
    assert _resolve_accuracy_pin_active() is False
    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "1"
    assert _resolve_accuracy_pin_active() is True
    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "off"
    assert _resolve_accuracy_pin_active() is False
    os.environ["RWKV_PIN_ACCURACY_LAYERS"] = "yes"
    assert _resolve_accuracy_pin_active() is True
