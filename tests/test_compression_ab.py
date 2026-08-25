from __future__ import annotations

import torch

from bench.bench_compression_ab import evaluate_compression_ab
from rwkv_ssd.tools.pack_runtime import _native_safe_codec, _quality_first_codec


def test_compression_ab_schema_and_quality_gate() -> None:
    torch.manual_seed(9)
    scales = torch.linspace(0.01, 1.0, 32).repeat_interleave(64)
    tensor = (torch.randn(2048) * scales).reshape(64, 32).to(torch.bfloat16)
    report = evaluate_compression_ab(
        [("blocks.0.att.key.weight", tensor)],
        group_sizes=(64, 256),
        activation_samples=4,
        min_compression_ratio=3.5,
    )
    assert report["schema_version"] == 1
    assert report["tensor_count"] == 1
    assert len(report["variants"]) == 15
    baseline, grouped64 = report["variants"][:2]
    assert grouped64["weighted_rmse"] < baseline["weighted_rmse"]
    assert grouped64["activation_output_rmse"] < baseline["activation_output_rmse"]
    assert grouped64["passed"] is True
    assert report["recommended_variant"] is not None


def test_balanced_quality_policy_protects_sensitive_tensors() -> None:
    matrix = torch.empty(128, 128)
    vector = torch.empty(128)
    assert _quality_first_codec("emb.weight", matrix, "trinity_lut2") == "scale_u8"
    assert _quality_first_codec("head.weight", matrix, "trinity_lut2") == "scale_u8"
    assert _quality_first_codec("blocks.0.ln1.weight", vector, "trinity_lut2") == "none"
    assert (
        _quality_first_codec("blocks.0.att.key.weight", matrix, "trinity_lut2")
        == "trinity_lut2"
    )


def test_native_safe_policy_protects_matrices_and_controls() -> None:
    assert _native_safe_codec("blocks.0.att.key.weight", "trinity_lut2") == "scale_u8_grouped"
    assert _native_safe_codec(
        "blocks.0.att.x_k",
        "trinity_lut2",
        tensor=torch.empty(1, 1, 768),
    ) == "none"
    assert _native_safe_codec(
        "blocks.0.ln1.weight",
        "trinity_lut2",
        tensor=torch.empty(768),
    ) == "none"
    assert _native_safe_codec("emb.weight", "trinity_lut2") == "scale_u8_grouped"
    assert _native_safe_codec(
        "head.weight", "trinity_lut2", tensor=torch.empty(65536, 2560)
    ) == "scale_u8_grouped"
    assert _native_safe_codec(
        "blocks.0.att.x_k", "scale_u8_grouped", tensor=torch.empty(1, 1, 2560)
    ) == "none"
    assert _native_safe_codec("blocks.0.att.x_k", "none") == "none"
    assert (
        _native_safe_codec(
            "blocks.0.att.x_k", "trinity_lut2", model_family="synthetic_rwkv"
        )
        == "trinity_lut2"
    )
