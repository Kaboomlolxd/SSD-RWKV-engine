"""Adaptive gate prefetch policy."""

from __future__ import annotations

from pathlib import Path

from tests.helpers import greedy_token_ids


def test_gate_prefetch_matches_layer_policy_greedy(synthetic_pack: Path) -> None:
    layer_ids = greedy_token_ids(
        synthetic_pack, "gate parity", mode="streaming", max_tokens=8
    )
    gate_ids = greedy_token_ids(
        synthetic_pack,
        "gate parity",
        mode="streaming",
        max_tokens=8,
        prefetch_policy="gate",
    )
    assert layer_ids == gate_ids
