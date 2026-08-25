from __future__ import annotations

import torch

from rwkv_ssd.tools.plan_mixed_precision import rank_codec_promotions


def test_mixed_precision_planner_respects_budget_and_ranks_hot_tensor() -> None:
    torch.manual_seed(31)
    tensors = {
        "blocks.0.att.key.weight": torch.randn(64, 64).to(torch.bfloat16),
        "blocks.0.att.value.weight": torch.randn(64, 64).to(torch.bfloat16),
    }
    stats = {
        "blocks.0.att.key.weight": torch.full((64,), 10.0),
        "blocks.0.att.value.weight": torch.full((64,), 0.1),
    }
    report = rank_codec_promotions(
        tensors,
        stats,
        lut_group_size=128,
        u8_group_size=64,
        extra_budget_bytes=4_000,
    )
    assert report["selected_extra_bytes"] <= 4_000
    assert report["selected_tensors"] == 1
    assert next(iter(report["codec_map"])) == "blocks.0.att.key.weight"


def test_mixed_precision_planner_prioritizes_configured_recurrent_family() -> None:
    torch.manual_seed(32)
    tensors = {
        "blocks.0.att.key.weight": torch.randn(64, 64).to(torch.bfloat16),
        "blocks.0.att.value.weight": torch.randn(64, 64).to(torch.bfloat16),
    }
    stats = {
        "blocks.0.att.key.weight": torch.full((64,), 10.0),
        "blocks.0.att.value.weight": torch.full((64,), 0.1),
    }
    report = rank_codec_promotions(
        tensors,
        stats,
        lut_group_size=128,
        u8_group_size=64,
        extra_budget_bytes=4_000,
        mandatory_suffixes=(".att.value.weight",),
    )
    assert report["mandatory_suffixes"] == [".att.value.weight"]
    assert report["mandatory_selected"] == 1
    assert next(iter(report["codec_map"])) == "blocks.0.att.value.weight"


def test_mixed_precision_planner_includes_small_recurrent_controls() -> None:
    tensor = torch.linspace(-1.0, 1.0, 64).to(torch.bfloat16)
    report = rank_codec_promotions(
        {"blocks.0.att.x_k": tensor},
        {},
        extra_budget_bytes=10_000,
    )
    assert report["candidate_tensors"] == 1
    assert report["selected_tensors"] == 1
    assert report["codec_map"] == {"blocks.0.att.x_k": "scale_u8_grouped"}
