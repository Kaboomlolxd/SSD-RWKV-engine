"""Skeleton load: globals only in model.z for streaming."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.rwkv7_skeleton import estimate_z_bytes
from tests.chatrwkv_greedy import greedy_token_ids, make_chatrwkv_engine

pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not available"),
]


def test_skeleton_z_much_smaller_than_full_load(
    real_pack: Path, test_checkpoint: Path
) -> None:
    full_cfg = EngineConfig(
        pack_dir=real_pack,
        backend="chatrwkv",
        mode="streaming",
        device="cpu",
        max_tokens=1,
        checkpoint_path=str(test_checkpoint),
        strategy="cpu bf16",
        skeleton_load=False,
        verify_hash=False,
    )
    sk_cfg = EngineConfig(
        pack_dir=real_pack,
        backend="chatrwkv",
        mode="streaming",
        device="cpu",
        max_tokens=1,
        checkpoint_path=str(test_checkpoint),
        strategy="cpu bf16",
        skeleton_load=True,
        verify_hash=False,
    )
    full_engine = InferenceEngine(full_cfg)
    full_engine.load()
    full_bytes = estimate_z_bytes(full_engine.backend._model.z)
    full_engine.close()

    sk_engine = InferenceEngine(sk_cfg)
    sk_engine.load()
    sk_bytes = estimate_z_bytes(sk_engine.backend._model.z)
    z = sk_engine.backend._model.z
    sk_engine.close()

    block_keys = [k for k in z if k.startswith("blocks.")]
    assert block_keys == [], f"expected no block weights in z, got {block_keys[:5]}..."
    assert sk_bytes < full_bytes, (
        f"skeleton z={sk_bytes/1e6:.1f}MB should be smaller than full z={full_bytes/1e6:.1f}MB"
    )
    saved = full_bytes - sk_bytes
    assert saved > 50e6, f"expected >50MB block weights evicted, saved {saved/1e6:.1f}MB"


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


def test_skeleton_streaming_matches_resident(
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
