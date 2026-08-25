"""V1: ChatRWKV pack-driven streaming golden tests (RWKV-7 0.1B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from tests.chatrwkv_greedy import greedy_token_ids, make_chatrwkv_engine

pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not available"),
]


@pytest.fixture(scope="module")
def resident_engine(real_pack: Path, test_checkpoint: Path):
    engine = make_chatrwkv_engine(
        real_pack, test_checkpoint, mode="resident", max_tokens=8
    )
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def streaming_engine(real_pack: Path, test_checkpoint: Path):
    engine = make_chatrwkv_engine(
        real_pack, test_checkpoint, mode="streaming", max_tokens=8
    )
    yield engine
    engine.close()


def test_chatrwkv_streaming_matches_resident_8_tokens(
    real_pack: Path,
    test_checkpoint: Path,
    resident_engine,
    streaming_engine,
) -> None:
    resident = greedy_token_ids(
        real_pack,
        test_checkpoint,
        mode="resident",
        max_tokens=8,
        engine=resident_engine,
    )
    streaming = greedy_token_ids(
        real_pack,
        test_checkpoint,
        mode="streaming",
        max_tokens=8,
        engine=streaming_engine,
    )
    assert resident == streaming


def test_chatrwkv_real_weight_stationary_batch_matches_independent(
    real_pack: Path, test_checkpoint: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RWKV_PROMOTE_FULL_Z", "0")
    prompts = ["Hello", "A different prompt"]
    independent_engine = make_chatrwkv_engine(
        real_pack, test_checkpoint, mode="streaming", max_tokens=3
    )
    batch_engine = make_chatrwkv_engine(
        real_pack, test_checkpoint, mode="streaming", max_tokens=3
    )
    try:
        expected = [independent_engine.generate(prompt) for prompt in prompts]
        actual = batch_engine.generate_batch(prompts, max_tokens=3)
        assert actual == expected
        streamed_layers = sum(
            any(entry.residency == "streamed" for entry in entries)
            for layer_id, entries in batch_engine.manifest.by_layer().items()
            if layer_id >= 0
        )
        assert batch_engine.metrics.batch_size == len(prompts)
        # The final prompt forward supplies the first next-token logits, so
        # only the remaining decode transitions need a shared layer sweep.
        assert batch_engine.metrics.weight_sweeps == 2
        assert batch_engine.metrics.weight_layer_loads == 2 * streamed_layers
    finally:
        independent_engine.close()
        batch_engine.close()


@pytest.mark.slow
def test_chatrwkv_streaming_matches_resident_32_tokens(
    real_pack: Path, test_checkpoint: Path
) -> None:
    resident = greedy_token_ids(
        real_pack, test_checkpoint, mode="resident", max_tokens=32
    )
    streaming = greedy_token_ids(
        real_pack, test_checkpoint, mode="streaming", max_tokens=32
    )
    assert resident == streaming
