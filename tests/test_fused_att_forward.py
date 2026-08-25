"""Fused LUT2 registration for att matrices + tmix_one_fused parity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.lut_gemm_fused import is_fused_lut_tensor_name
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_store import open_weight_store

pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not installed"),
]


def test_att_tensor_names_eligible() -> None:
    assert is_fused_lut_tensor_name("blocks.0.att.receptance.weight")
    assert not is_fused_lut_tensor_name("blocks.0.att.w1")
    assert not is_fused_lut_tensor_name("blocks.0.att.x_r")


def test_strict_streaming_registers_fused_att_blobs() -> None:
    pack = Path("test_model/trinity_eval/trinity_lut2_0.1b")
    if not pack.is_dir() or not (pack / "weights.bin").is_file():
        pytest.skip(f"real LUT2 payload missing: {pack}")
    manifest = Manifest.load(pack)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
        import os

        os.environ["RWKV_LUT_GEMM_FUSED"] = "1"
        cfg = EngineConfig(
            pack_dir=pack,
            mode="streaming",
            device="cpu",
            stream_layer_cache=False,
            max_layers_in_z=0,
        )
        provider = create_weight_provider(
            cfg,
            store,
            manifest.tensors,
            torch.device("cpu"),
            metrics,
        )
        by_layer = manifest.by_layer()
        layer0 = by_layer.get(0, [])
        if layer0:
            provider.load_layer_tensors(layer0)
        att_keys = [
            k
            for k in provider._fused_lut_blobs
            if ".att." in k and k.endswith(
                (
                    "receptance.weight",
                    "key.weight",
                    "value.weight",
                    "output.weight",
                )
            )
        ]
        assert len(att_keys) >= 4
    finally:
        store.close()


def test_tmix_one_fused_matches_native() -> None:
    pack = Path("test_model/trinity_eval/trinity_lut2_0.1b")
    if not pack.is_dir() or not (pack / "weights.bin").is_file():
        pytest.skip(f"real LUT2 payload missing: {pack}")
    root = find_chatrwkv_root()
    assert root is not None
    sys.path.insert(0, str(root / "rwkv_pip_package" / "src"))
    from rwkv.model import RWKV_x070_TMix_one

    manifest = Manifest.load(pack)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
        import os

        os.environ["RWKV_LUT_GEMM_FUSED"] = "0"
        cfg = EngineConfig(
            pack_dir=pack,
            mode="streaming",
            device="cpu",
            stream_layer_cache=False,
            max_layers_in_z=0,
        )
        provider = create_weight_provider(
            cfg,
            store,
            manifest.tensors,
            torch.device("cpu"),
            metrics,
        )
        by_layer = manifest.by_layer()
        layer_tensors = provider.load_layer_tensors(by_layer[0])
        prepared = provider.prepare_layer_for_z(0, layer_tensors)
        z = dict(prepared)

        os.environ["RWKV_LUT_GEMM_FUSED"] = "1"
        fused_provider = create_weight_provider(
            cfg,
            store,
            manifest.tensors,
            torch.device("cpu"),
            MetricsCollector(),
        )
        fused_provider.load_layer_tensors(by_layer[0])
        att = "blocks.0.att."
        H, N = 12, 64
        act = torch.bfloat16
        x = torch.randn(768, dtype=act) * 0.05
        x_prev = torch.randn(768, dtype=act) * 0.05
        v_first = torch.randn(768, dtype=act) * 0.05
        state = torch.zeros(H, N, N, dtype=torch.float32)
        from rwkv_ssd.runtime.rwkv7_linear import tmix_one_fused

        native_out, _, native_state, native_vf = RWKV_x070_TMix_one(
            0,
            H,
            N,
            x,
            x_prev,
            v_first,
            state,
            z[att + "x_r"],
            z[att + "x_w"],
            z[att + "x_k"],
            z[att + "x_v"],
            z[att + "x_a"],
            z[att + "x_g"],
            z[att + "w0"],
            z[att + "w1"],
            z[att + "w2"],
            z[att + "a0"],
            z[att + "a1"],
            z[att + "a2"],
            z[att + "v0"],
            z[att + "v1"],
            z[att + "v2"],
            z[att + "g1"],
            z[att + "g2"],
            z[att + "k_k"],
            z[att + "k_a"],
            z[att + "r_k"],
            z[att + "receptance.weight"],
            z[att + "key.weight"],
            z[att + "value.weight"],
            z[att + "output.weight"],
            z[att + "ln_x.weight"],
            z[att + "ln_x.bias"],
        )
        fused_out, _, fused_state, fused_vf = tmix_one_fused(
            0,
            H,
            N,
            x,
            x_prev,
            v_first,
            state.clone(),
            z[att + "x_r"],
            z[att + "x_w"],
            z[att + "x_k"],
            z[att + "x_v"],
            z[att + "x_a"],
            z[att + "x_g"],
            z[att + "w0"],
            z[att + "w1"],
            z[att + "w2"],
            z[att + "a0"],
            z[att + "a1"],
            z[att + "a2"],
            z[att + "v0"],
            z[att + "v1"],
            z[att + "v2"],
            z[att + "g1"],
            z[att + "g2"],
            z[att + "k_k"],
            z[att + "k_a"],
            z[att + "r_k"],
            att,
            z,
            fused_provider,
            act,
        )
        assert torch.allclose(
            native_out.float(), fused_out.float(), rtol=0.02, atol=2.0
        )
        assert torch.allclose(native_state, fused_state, rtol=0.02, atol=2.0)
        assert torch.allclose(
            native_vf.float(), fused_vf.float(), rtol=0.02, atol=2.0
        )
    finally:
        store.close()
