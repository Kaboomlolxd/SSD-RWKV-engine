"""The native layer packed-head option is explicit and shape-safe."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from rwkv_ssd.backends.rwkvcpp import (
    RWKVCppBackend,
    _native_layer_packed_head_enabled,
)
from rwkv_ssd.runtime.manifest import TensorEntry


def test_native_layer_packed_head_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKVCPP_NATIVE_LAYER_PACKED_HEAD", raising=False)
    assert _native_layer_packed_head_enabled() is False
    monkeypatch.setenv("RWKVCPP_NATIVE_LAYER_PACKED_HEAD", "1")
    assert _native_layer_packed_head_enabled() is True
    monkeypatch.setenv("RWKVCPP_NATIVE_LAYER_PACKED_HEAD", "auto")
    assert _native_layer_packed_head_enabled() is False


def test_native_layer_packed_head_uses_fused_projection_without_dense_head() -> None:
    backend = RWKVCppBackend()
    blob = b"SG8\x01" + b"packed"
    backend._layer_packed_head = (blob, 5, 3)
    hidden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = torch.arange(5, dtype=torch.float32)

    with patch(
        "rwkv_ssd.runtime.lut_gemm_fused.lut2_gemv",
        return_value=expected,
    ) as gemv:
        logits = backend._layer_logits_global(hidden)

    gemv.assert_called_once()
    assert np.array_equal(logits, expected.numpy())
    assert "head.weight" not in backend._layer_global_weights


def test_native_layer_packed_head_skips_materialization_and_provider_alias(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RWKVCPP_NATIVE_LAYER_PACKED_HEAD", "1")
    backend = RWKVCppBackend()
    names = {
        "emb.weight": [4, 3],
        "blocks.0.ln0.weight": [3],
        "blocks.0.ln0.bias": [3],
        "ln_out.weight": [3],
        "ln_out.bias": [3],
        "head.weight": [5, 3],
    }
    tensors = {
        name: torch.ones(shape, dtype=torch.float32)
        for name, shape in names.items()
        if name != "head.weight"
    }
    entries = [
        TensorEntry(
            name,
            9999,
            "float32",
            shape,
            0,
            8,
            4096,
            "streamed",
            dequant="scale_u8_grouped" if name == "head.weight" else "none",
        )
        for name, shape in names.items()
    ]

    class Provider:
        def __init__(self) -> None:
            self.released: list[set[str]] = []
            self.primed = False

        def _prime_fused_global_blobs(self) -> None:
            self.primed = True

        def get_fused_lut_blob(self, name: str):
            assert self.primed
            if name == "head.weight":
                return b"SG8\x01head", 5, 3
            return None

        def load_layer_tensors_materialized(self, requested):
            return {entry.name: tensors[entry.name] for entry in requested}

        def release_native_global_cache(self, requested) -> None:
            self.released.append(set(requested))

    provider = Provider()
    backend._load_layer_global_weights(provider, {9999: entries})

    assert provider.primed is True
    assert set(backend._layer_global_weights) == backend._LAYER_GLOBAL_NAMES - {
        "head.weight"
    }
    assert backend._layer_packed_head == (b"SG8\x01head", 5, 3)
    assert provider.released == [
        backend._LAYER_GLOBAL_NAMES - {"head.weight"}
    ]
