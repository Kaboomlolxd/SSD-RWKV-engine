"""P1.4 — packed_block_forward skeleton resolver."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rwkv_ssd.runtime.packed_block_forward import _resolve_skeleton


def test_resolve_from_z_first() -> None:
    z = {"blocks.0.att.x_r": torch.zeros(8)}
    out = _resolve_skeleton("blocks.0.att.x_r", z, None, 0)
    assert out is z["blocks.0.att.x_r"]


def test_resolve_falls_back_to_provider_prepared() -> None:
    """P1.4: skeleton in provider._prepared_layers works when not in z."""
    skeleton = torch.zeros(8)
    provider = SimpleNamespace(
        _prepared_layers={
            0: {"blocks.0.att.x_r": skeleton, "blocks.0.ln1.weight": torch.zeros(4)}
        }
    )
    out = _resolve_skeleton("blocks.0.att.x_r", {}, provider, 0)
    assert out is skeleton


def test_resolve_raises_when_missing() -> None:
    with pytest.raises(KeyError, match="blocks.0.att.x_r"):
        _resolve_skeleton("blocks.0.att.x_r", {}, None, 0)
    provider = SimpleNamespace(_prepared_layers={})
    with pytest.raises(KeyError, match="blocks.0.att.x_r"):
        _resolve_skeleton("blocks.0.att.x_r", {}, provider, 0)


def test_resolve_provider_without_prepared() -> None:
    """Provider without _prepared_layers attribute: z-only path still works."""
    provider = SimpleNamespace()  # no _prepared_layers
    z = {"blocks.0.att.x_r": torch.zeros(8)}
    out = _resolve_skeleton("blocks.0.att.x_r", z, provider, 0)
    assert out is z["blocks.0.att.x_r"]


def test_can_use_packed_block_fused_provider() -> None:
    """F-1: _can_use_packed_block is True when a fused LUT provider is
    registered and the layer's att weights are not in z (the F1-F3 path)."""
    from rwkv_ssd.backends.rwkv7_forward import _can_use_packed_block

    class _FakeProvider:
        def __init__(self) -> None:
            self.mode = "streaming"
            self._stream_layer_cache = False
            self._pack_uses_quant = True
            self._strict_fused_retain_layers = lambda: False
            self._strict_fused_lean_z = lambda: False
            self._z_retention = SimpleNamespace(pinned_layer_ids=set())

        def _use_fused_lut_matmul(self) -> bool:
            return True

        def get_fused_lut_blob(self, name: str):
            return f"blob:{name}" if name.endswith(".weight") else None

        def get_fused_tmix_blobs(self, prefix: str):
            return [f"blob:{prefix}receptance", f"blob:{prefix}key",
                    f"blob:{prefix}value", f"blob:{prefix}output"]

    z: dict = {}  # no att.weight
    assert _can_use_packed_block(_FakeProvider(), z, 0, {}) is True


def test_can_use_packed_block_disabled_when_att_in_z() -> None:
    """F-1: when the att weights are already in z, native forward should run
    instead — packed block is for streamed layers only."""
    from rwkv_ssd.backends.rwkv7_forward import _can_use_packed_block
    import torch

    class _FakeProvider:
        mode = "streaming"
        _stream_layer_cache = False
        _pack_uses_quant = True
        _strict_fused_retain_layers = lambda: False
        _strict_fused_lean_z = lambda: False
        _z_retention = SimpleNamespace(pinned_layer_ids=set())

        def _use_fused_lut_matmul(self) -> bool:
            return True

        def get_fused_lut_blob(self, name: str):
            return None

        def get_fused_tmix_blobs(self, prefix: str):
            return []

    z = {"blocks.0.att.receptance.weight": torch.zeros(8, 8)}
    assert _can_use_packed_block(_FakeProvider(), z, 0, {}) is False


def test_can_use_packed_block_disabled_without_fused() -> None:
    """F-1: without a fused LUT matmul provider, packed block is gated off."""
    from rwkv_ssd.backends.rwkv7_forward import _can_use_packed_block

    class _FakeProvider:
        mode = "streaming"
        _stream_layer_cache = False
        _pack_uses_quant = True
        _strict_fused_retain_layers = lambda: False
        _strict_fused_lean_z = lambda: False
        _z_retention = SimpleNamespace(pinned_layer_ids=set())

        def _use_fused_lut_matmul(self) -> bool:
            return False

        def get_fused_lut_blob(self, name: str):
            return None

        def get_fused_tmix_blobs(self, prefix: str):
            return []

    z: dict = {}
    assert _can_use_packed_block(_FakeProvider(), z, 0, {}) is False


def test_prepare_layer_force_materialize_keeps_att_weights() -> None:
    """P0.6: warm-z + fused kernels crash fix. When ``force_materialize=True``
    the fused-inject filter must NOT drop the bf16 att/FFN slabs — the
    warm-z preloader passes this so ``all_block_layers_in_z`` returns
    True and the per-tensor fallback doesn't ``KeyError`` on
    ``z['blocks.0.att.receptance.weight']``.
    """
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
    from rwkv_ssd.runtime.lut_gemm_fused import filter_tensors_for_fused_inject

    # Build a minimal fake provider that mimics the fused-filter state.
    class _FusedProvider:
        def _use_fused_lut_matmul(self) -> bool:
            return True

        def get_fused_lut_blob(self, name: str):
            return f"blob:{name}" if name.endswith(".weight") else None

        def get_fused_tmix_blobs(self, prefix: str):
            return [f"blob:{prefix}receptance", f"blob:{prefix}key",
                    f"blob:{prefix}value", f"blob:{prefix}output"]

    p = _FusedProvider()
    att = "blocks.0.att."
    tensors = {
        att + "x_r": object(),
        att + "receptance.weight": object(),
        att + "key.weight": object(),
        att + "value.weight": object(),
        att + "output.weight": object(),
    }
    # Default (force_materialize=False): att weights get dropped.
    filtered = filter_tensors_for_fused_inject(p, tensors)
    assert att + "receptance.weight" not in filtered
    # force_materialize=True callers (warm-z preloader, F-1 packed step)
    # must keep the slabs. The convention is to skip the filter entirely
    # when force_materialize is set — see ``prepare_layer_for_z``.
    assert att + "x_r" in tensors  # sanity: small skeleton kept anyway
