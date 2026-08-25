"""Fused LUT blob registration on strict streaming."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider


def test_grouped_u8_transposed_adapter_is_fused(monkeypatch) -> None:
    """Packed TMix adapters must not be decoded into redundant dense slabs."""
    from rwkv_ssd.runtime.lut_gemm_fused import filter_tensors_for_fused_inject

    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_LUT_SMALL_TRANSPOSED", "1")
    provider = object.__new__(ManifestWeightProvider)
    provider._use_fused_lut_matmul = lambda: True
    provider.get_fused_lut_blob = lambda name: (
        (b"SG8\x01", 8, 8) if name.endswith(".att.w1") else None
    )
    entry = TensorEntry(
        name="blocks.0.att.w1",
        layer_id=0,
        dtype="bfloat16",
        shape=[8, 8],
        offset=0,
        length=16,
        alignment=1,
        residency="streamed",
        dequant="scale_u8_grouped",
    )
    assert provider._is_fused_lut_entry(entry) is True
    tensors = {entry.name: torch.zeros((8, 8), dtype=torch.bfloat16)}
    assert filter_tensors_for_fused_inject(provider, tensors) == {}


def test_short_fused_prefill_uses_token_kernel(monkeypatch) -> None:
    """Do not send a one-token CPU quantized prompt through dense BF16 prefill."""
    import rwkv_ssd.backends.rwkv7_forward as forward_module

    calls: list[tuple[int, object]] = []

    def fake_forward_one(model, token_id, state, provider, by_layer, **kwargs):
        del model, provider, by_layer, kwargs
        calls.append((int(token_id), state))
        return None, state

    def unexpected_sequence(*args, **kwargs):
        del args, kwargs
        raise AssertionError("short fused prefill selected dense sequence kernel")

    provider = SimpleNamespace(_use_fused_lut_matmul=lambda: True)
    monkeypatch.delenv("RWKV_STREAM_FUSED_PREFILL_MIN_TOKENS", raising=False)
    monkeypatch.setattr(forward_module, "forward_one", fake_forward_one)
    monkeypatch.setattr(
        "rwkv_ssd.backends.rwkv7_batch.forward_sequence_one_dense",
        unexpected_sequence,
    )

    state = object()
    result = forward_module._prefill_streaming_tokens(
        SimpleNamespace(z={}),
        [7],
        state,
        provider,
        {},
        [0],
        None,
    )

    assert result is state
    assert calls == [(7, state)]


def test_strict_streaming_registers_fused_ffn_blobs() -> None:
    pack = Path("test_model/trinity_eval/trinity_lut2_0.1b")
    if not (pack / "weights.bin").is_file():
        pytest.skip("archived Trinity manifest has no weights.bin payload")
    manifest = Manifest.load(pack)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
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
        import os

        os.environ["RWKV_LUT_GEMM_FUSED"] = "1"
        by_layer = manifest.by_layer()
        for layer_id in sorted(by_layer.keys()):
            if layer_id < 0:
                continue
            provider.load_layer_tensors(by_layer[layer_id])
        ffn_keys = [k for k in provider._fused_lut_blobs if ".ffn." in k]
        assert ffn_keys or not provider._use_fused_lut_matmul()
    finally:
        store.close()


def test_mixed_codec_layer_registers_batched_tmix_span(monkeypatch) -> None:
    """Grouped-U8 + BF16 control entries must still expose the QKV batch."""
    import rwkv_ssd.runtime.weight_provider as provider_module

    provider = object.__new__(ManifestWeightProvider)
    provider._fused_lut_blobs = {}
    provider._fused_tmix_blobs = {}
    provider._max_packed_cache_bytes = 0
    provider._device = torch.device("cpu")
    provider._decode_device = torch.device("cpu")
    provider._trinity_layer_cache = None
    fused_names = {
        "blocks.0.att.receptance.weight",
        "blocks.0.att.key.weight",
        "blocks.0.att.value.weight",
        "blocks.0.att.output.weight",
    }
    provider._use_fused_lut_matmul = lambda: True
    provider._is_fused_lut_entry = lambda entry: entry.name in fused_names
    provider._skip_fused_materialize = lambda: False
    provider._register_fused_att_tmix_span = (
        ManifestWeightProvider._register_fused_att_tmix_span.__get__(provider)
    )
    provider._register_fused_transposed_lut_blobs = (
        ManifestWeightProvider._register_fused_transposed_lut_blobs.__get__(provider)
    )

    entries = []
    for index, name in enumerate(
        (
            "blocks.0.att.receptance.weight",
            "blocks.0.att.key.weight",
            "blocks.0.att.value.weight",
            "blocks.0.att.output.weight",
        )
    ):
        entries.append(
            TensorEntry(
                name=name,
                layer_id=0,
                dtype="bfloat16",
                shape=[4, 4],
                offset=index * 16,
                length=16,
                alignment=1,
                residency="streamed",
                dequant="scale_u8_grouped",
            )
        )
    entries.append(
        TensorEntry(
            name="blocks.0.att.x_r",
            layer_id=0,
            dtype="bfloat16",
            shape=[4],
            offset=80,
            length=4,
            alignment=1,
            residency="streamed",
            dequant="none",
        )
    )
    monkeypatch.setattr(
        provider_module,
        "decode_weight_to_tensor",
        lambda _raw, entry, _device, **_kwargs: torch.zeros(entry.shape),
    )
    ManifestWeightProvider._decode_from_layer_raw(
        provider, bytes(128), 0, entries, None
    )
    batch = provider._fused_tmix_blobs.get("blocks.0.att.")
    assert batch is not None
    assert len(batch[0]) == 4
