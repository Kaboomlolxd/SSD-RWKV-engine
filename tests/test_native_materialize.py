"""Native forward after provider cache materializes into z."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.rwkv7_weights import all_block_layers_in_z
from tests.helpers import greedy_token_ids


def test_stream_cache_second_token_uses_native_path(synthetic_pack_deep: Path) -> None:
    """After cache warms, decode should match resident greedy tokens."""
    base = greedy_token_ids(
        synthetic_pack_deep, "native mat", mode="resident", max_tokens=12
    )
    cached = greedy_token_ids(
        synthetic_pack_deep,
        "native mat",
        mode="streaming",
        max_tokens=12,
        stream_layer_cache=True,
        max_layers_in_z=1,
    )
    assert base == cached

