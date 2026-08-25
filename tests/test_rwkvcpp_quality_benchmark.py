from __future__ import annotations

import torch

from bench.bench_rwkvcpp_quality import summarize_quality_pairs


def test_rwkvcpp_quality_summary_has_strict_gates() -> None:
    ref_logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    good_logits = ref_logits + 0.001
    ref_state = torch.ones(8)
    good_state = ref_state + 0.001
    result = summarize_quality_pairs(
        [("hello", ref_logits, good_logits, ref_state, good_state)],
        top_k=2,
        min_top_k_overlap=1.0,
        max_kl=0.001,
        max_state_relative_l2=0.01,
    )
    assert result["passed"] is True
    assert all(result["gates"].values())


def test_rwkvcpp_quality_summary_rejects_drift() -> None:
    result = summarize_quality_pairs(
        [
            (
                "hello",
                torch.tensor([[5.0, 4.0, 0.0, -1.0]]),
                torch.tensor([[-1.0, 0.0, 4.0, 5.0]]),
                torch.ones(8),
                torch.zeros(8),
            )
        ],
        top_k=2,
        min_top_k_overlap=0.8,
        max_kl=0.05,
        max_state_relative_l2=0.1,
    )
    assert result["passed"] is False
    assert result["gates"]["minimum_top_k_overlap"] is False
    assert result["gates"]["maximum_state_relative_l2"] is False
