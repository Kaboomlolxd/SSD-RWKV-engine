"""Streaming golden tests on quantized synthetic packs (M5)."""

from __future__ import annotations

from pathlib import Path

from tests.helpers import greedy_token_ids


def test_scale_u8_streaming_matches_resident(synthetic_pack_u8: Path) -> None:
    resident = greedy_token_ids(
        synthetic_pack_u8, "codec", mode="resident", max_tokens=8
    )
    streaming = greedy_token_ids(
        synthetic_pack_u8, "codec", mode="streaming", max_tokens=8
    )
    assert resident == streaming


def test_scale_u4_streaming_matches_resident(synthetic_pack_u4: Path) -> None:
    resident = greedy_token_ids(
        synthetic_pack_u4, "codec", mode="resident", max_tokens=8
    )
    streaming = greedy_token_ids(
        synthetic_pack_u4, "codec", mode="streaming", max_tokens=8
    )
    assert resident == streaming
