"""Fused LUT2 GEMV prototype (Track D)."""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.lut_gemm_fused import (
    _lut2_arrays,
    activation_fp32_enabled,
    clear_lut2_packed_cache,
    grouped_u8_cmix_gemv,
    grouped_u8_transposed_gemv,
    lut2_gemv_cpu,
    lut2_linear_forward,
    lut2_packed_cache_stats,
    set_lut2_packed_cache_limit,
)
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_u8_grouped_to_tensor,
    encode_scale_u8_grouped,
)
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.trinity_codec import encode_trinity_lut2


def test_lut2_gemv_matches_materialized() -> None:
    w = torch.randn(32, 16, dtype=torch.bfloat16)
    x = torch.randn(16, dtype=torch.bfloat16)
    blob = encode_trinity_lut2(w)
    y_fused = lut2_gemv_cpu(blob, x, out_features=32, in_features=16)
    y_ref = lut2_linear_forward(blob, x, (32, 16))
    torch.testing.assert_close(y_fused.float(), y_ref.float(), rtol=0, atol=0.07)


def test_lut2_linear_forward_disabled() -> None:
    w = torch.randn(8, 8, dtype=torch.bfloat16)
    x = torch.randn(8, dtype=torch.bfloat16)
    blob = encode_trinity_lut2(w)
    y = lut2_linear_forward(blob, x, (8, 8))
    assert y.shape == (8,)


def test_fused_cpu_activation_fp32_override(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_ACTIVATION_FP32", "0")
    assert activation_fp32_enabled() is False
    monkeypatch.setenv("RWKV_LUT_ACTIVATION_FP32", "1")
    assert activation_fp32_enabled() is True


def test_lut2_indices_use_source_view_without_unconditional_copy() -> None:
    clear_lut2_packed_cache()
    set_lut2_packed_cache_limit(0)
    blob = encode_trinity_lut2(torch.randn(64, 64, dtype=torch.bfloat16))
    _cb, packed = _lut2_arrays(
        memoryview(blob), out_features=64, in_features=64
    )
    assert packed.flags.owndata is False
    assert lut2_packed_cache_stats()["bytes"] >= packed.nbytes


def test_lut2_index_cache_is_byte_bounded() -> None:
    clear_lut2_packed_cache()
    blobs = [
        encode_trinity_lut2(torch.randn(64, 64, dtype=torch.bfloat16))
        for _ in range(4)
    ]
    # Keep at most one packed index payload.  The four-byte codebook overhead
    # is included in the reported cache size, so use the first payload's size
    # as the cap after it has been materialized.
    _cb, packed = _lut2_arrays(blobs[0], out_features=64, in_features=64)
    cap = int(packed.nbytes + 16)
    set_lut2_packed_cache_limit(cap)
    for blob in blobs[1:]:
        _lut2_arrays(blob, out_features=64, in_features=64)
    stats = lut2_packed_cache_stats()
    assert stats["bytes"] <= cap
    assert stats["entries"] <= 1
    set_lut2_packed_cache_limit(0)
    clear_lut2_packed_cache()


def test_grouped_u8_transposed_gemv_matches_decoded_weight(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_KERNEL", "auto")
    weight = torch.randn(32, 8, dtype=torch.bfloat16)
    x = torch.randn(32, dtype=torch.bfloat16)
    blob = encode_scale_u8_grouped(weight, group_size=8)
    entry = TensorEntry(
        name="blocks.0.att.w1",
        layer_id=0,
        dtype="bfloat16",
        shape=[32, 8],
        offset=0,
        length=len(blob),
        alignment=4096,
        residency="streamed",
        dequant="scale_u8_grouped",
    )
    decoded = decode_scale_u8_grouped_to_tensor(
        blob, entry, torch.device("cpu")
    )
    actual = grouped_u8_transposed_gemv(
        blob, x, out_features=8, in_features=32
    )
    expected = x.float() @ decoded.float()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.5)


def test_grouped_u8_cmix_gemv_matches_two_stage_reference(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_KERNEL", "auto")
    key = torch.randn(32, 8, dtype=torch.float32)
    value = torch.randn(8, 32, dtype=torch.float32)
    x = torch.randn(8, dtype=torch.float32)
    key_blob = encode_scale_u8_grouped(key, group_size=8)
    value_blob = encode_scale_u8_grouped(value, group_size=8)
    key_entry = TensorEntry(
        name="blocks.0.ffn.key.weight",
        layer_id=0,
        dtype="float32",
        shape=[32, 8],
        offset=0,
        length=len(key_blob),
        alignment=4096,
        residency="streamed",
        dequant="scale_u8_grouped",
    )
    value_entry = TensorEntry(
        name="blocks.0.ffn.value.weight",
        layer_id=0,
        dtype="float32",
        shape=[8, 32],
        offset=0,
        length=len(value_blob),
        alignment=4096,
        residency="streamed",
        dequant="scale_u8_grouped",
    )
    key_dense = decode_scale_u8_grouped_to_tensor(
        key_blob, key_entry, torch.device("cpu")
    ).float()
    value_dense = decode_scale_u8_grouped_to_tensor(
        value_blob, value_entry, torch.device("cpu")
    ).float()
    actual = grouped_u8_cmix_gemv(
        key_blob,
        value_blob,
        x,
        key_out_features=32,
        key_in_features=8,
        value_out_features=8,
        value_in_features=32,
    )
    expected = torch.relu(key_dense @ x) ** 2
    expected = value_dense @ expected
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-4)
