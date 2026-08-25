"""Compression Trinity LUT2 + zlib engine codec tests."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_bytes,
    decode_trinity_to_bytes,
    encode_trinity,
    encode_trinity_lut2,
    packed_length_trinity_lut2,
    storage_ratio_estimate,
)
from rwkv_ssd.tools.quant_quality import compare_tensors
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes


def _entry(name: str, shape: list[int], length: int) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype="bfloat16",
        shape=shape,
        offset=0,
        length=length,
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )


def test_trinity_lut2_roundtrip_bfloat16() -> None:
    t = torch.randn(32, 32, dtype=torch.bfloat16)
    entry = _entry("blocks.0.weight", [32, 32], packed_length_trinity_lut2(32 * 32))
    raw = encode_trinity_lut2(t)
    assert len(raw) == entry.length
    out = tensor_from_bytes(decode_trinity_lut2_to_bytes(raw, entry), entry, torch.device("cpu"))
    assert out.shape == t.shape
    assert out.dtype == t.dtype
    # Lossy — should be close, not bitwise identical.
    diff = (out.float() - t.float()).abs().mean()
    assert diff < 0.75


def test_trinity_zlib_roundtrip() -> None:
    t = torch.randn(16, 16, dtype=torch.bfloat16)
    numel = 16 * 16
    entry_lut = _entry("w", [16, 16], 0)
    entry_lut = TensorEntry(
        name=entry_lut.name,
        layer_id=entry_lut.layer_id,
        dtype=entry_lut.dtype,
        shape=entry_lut.shape,
        offset=0,
        length=len(encode_trinity(t)),
        alignment=4096,
        residency="streamed",
        dequant="trinity",
    )
    raw = encode_trinity(t)
    out = tensor_from_bytes(decode_trinity_to_bytes(raw, entry_lut), entry_lut, torch.device("cpu"))
    assert out.shape == t.shape


def test_trinity_storage_ratio_estimate() -> None:
    n = 65536
    ratio = storage_ratio_estimate(n)
    assert ratio >= 7.0


def test_groupwise_kmeans_improves_error_at_bounded_size() -> None:
    torch.manual_seed(7)
    scales = torch.linspace(0.01, 1.0, 16).repeat_interleave(64)
    t = (torch.randn(1024) * scales).reshape(32, 32).to(torch.bfloat16)
    global_raw = encode_trinity_lut2(t, codebook="kmeans")
    grouped_raw = encode_trinity_lut2(t, codebook="groupwise_kmeans", group_size=64)
    entry = _entry("blocks.0.weight", [32, 32], len(global_raw))
    global_out = tensor_from_bytes(
        decode_trinity_lut2_to_bytes(global_raw, entry), entry, torch.device("cpu")
    )
    grouped_entry = _entry("blocks.0.weight", [32, 32], len(grouped_raw))
    grouped_out = tensor_from_bytes(
        decode_trinity_lut2_to_bytes(grouped_raw, grouped_entry),
        grouped_entry,
        torch.device("cpu"),
    )
    assert compare_tensors(t, grouped_out)["rmse"] < compare_tensors(t, global_out)["rmse"]
    assert len(grouped_raw) == packed_length_trinity_lut2(
        t.numel(), codebook="groupwise_kmeans", group_size=64
    )
    assert (t.numel() * 2) / len(grouped_raw) >= 3.9


def test_same_size_grouped_variants_reduce_error() -> None:
    torch.manual_seed(17)
    scales = torch.linspace(0.01, 2.0, 32).repeat_interleave(128)
    t = (torch.randn(4096) * scales).reshape(64, 64).to(torch.bfloat16)
    old = encode_trinity_lut2(t, codebook="groupwise_kmeans", group_size=128)
    finer = encode_trinity_lut2(
        t, codebook="groupwise_kmeans_fp16", group_size=64
    )
    repaired = encode_trinity_lut2(
        t, codebook="groupwise_kmeans_residual", group_size=128
    )
    assert len(finer) == len(old)
    assert len(repaired) == len(old)

    def decode(blob: bytes) -> torch.Tensor:
        entry = _entry("blocks.0.weight", [64, 64], len(blob))
        return tensor_from_bytes(
            decode_trinity_lut2_to_bytes(blob, entry), entry, torch.device("cpu")
        )

    old_rmse = float(compare_tensors(t, decode(old))["rmse"])
    assert float(compare_tensors(t, decode(finer))["rmse"]) < old_rmse
    assert float(compare_tensors(t, decode(repaired))["rmse"]) < old_rmse


def test_activation_weighted_lut2_spends_error_budget_on_hot_features() -> None:
    torch.manual_seed(23)
    t = torch.randn(16, 128, dtype=torch.float32).to(torch.bfloat16)
    importance = torch.ones(128)
    importance[:8] = 1000.0
    baseline = encode_trinity_lut2(
        t, codebook="groupwise_kmeans_residual", group_size=128
    )
    calibrated = encode_trinity_lut2(
        t,
        codebook="groupwise_kmeans_activation",
        group_size=128,
        importance=importance,
    )
    assert len(calibrated) == len(baseline)

    def decode(blob: bytes) -> torch.Tensor:
        entry = _entry("blocks.0.weight", [16, 128], len(blob))
        return tensor_from_bytes(
            decode_trinity_lut2_to_bytes(blob, entry), entry, torch.device("cpu")
        ).float()

    baseline_error = ((decode(baseline) - t.float())[:, :8].square()).mean()
    calibrated_error = ((decode(calibrated) - t.float())[:, :8].square()).mean()
    assert calibrated_error < baseline_error


def test_trinity_streaming_golden(synthetic_pack_trinity_lut2: Path) -> None:
    from tests.helpers import greedy_token_ids

    resident = greedy_token_ids(
        synthetic_pack_trinity_lut2, "trinity", mode="resident", max_tokens=8
    )
    streaming = greedy_token_ids(
        synthetic_pack_trinity_lut2, "trinity", mode="streaming", max_tokens=8
    )
    assert resident == streaming


def test_trinity_full_streaming_golden(synthetic_pack_trinity: Path, monkeypatch) -> None:
    monkeypatch.setenv("RWKV_ALLOW_TRINITY_LAYER", "1")
    from tests.helpers import greedy_token_ids

    resident = greedy_token_ids(
        synthetic_pack_trinity, "trinity", mode="resident", max_tokens=8
    )
    streaming = greedy_token_ids(
        synthetic_pack_trinity, "trinity", mode="streaming", max_tokens=8
    )
    assert resident == streaming
