"""Opt-in CPU integration tests for the downloaded RWKV7a DeepEmbed-v1 model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.checkpoint_meta import load_checkpoint_tensors
from rwkv_ssd.tools.pack_runtime import detect_model_family_from_state
from tests.chatrwkv_greedy import greedy_token_ids, make_chatrwkv_engine


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "test_model" / "rwkv7a-g1d-0.1b-20260212-ctx8192.pth"
PACK = REPO / "test_model" / "runtime_pack_rwkv7a_v1"
CHATRWKV = REPO / "test_model" / "ChatRWKV"

pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.integration,
    pytest.mark.skipif(
        not CHECKPOINT.is_file(), reason="RWKV7a DeepEmbed-v1 checkpoint not downloaded"
    ),
    pytest.mark.skipif(
        not (PACK / "manifest.json").is_file(), reason="RWKV7a runtime pack not generated"
    ),
    pytest.mark.skipif(
        find_chatrwkv_root() is None or not CHATRWKV.exists(),
        reason="ChatRWKV not available",
    ),
]


def _manifest() -> dict:
    return json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))


def test_rwkv7a_checkpoint_and_pack_metadata() -> None:
    state = load_checkpoint_tensors(CHECKPOINT)
    assert detect_model_family_from_state(state) == "rwkv7_deepembed"

    manifest = _manifest()
    meta = manifest["meta"]
    assert manifest["model_family"] == "rwkv7_deepembed"
    assert meta["deepembed"] is True
    assert meta["deepembed_variant"] == "rwkv7a_v1"
    assert meta["deepembed_format"] == "rwkv7a_deepembed_v1"
    assert meta["deepembed_sidecar_required"] is False
    assert meta["deepembed_streaming_supported"] is True
    assert not (PACK / "DeepEmbed.bin").exists()


def test_rwkv7a_resident_and_streaming_greedy_parity() -> None:
    resident = make_chatrwkv_engine(
        PACK, CHECKPOINT, mode="resident", max_tokens=4, skeleton_load=False
    )
    streaming = make_chatrwkv_engine(
        PACK, CHECKPOINT, mode="streaming", max_tokens=4, skeleton_load=True
    )
    try:
        expected = greedy_token_ids(
            PACK,
            CHECKPOINT,
            mode="resident",
            prompt="Hi",
            max_tokens=4,
            engine=resident,
        )
        actual = greedy_token_ids(
            PACK,
            CHECKPOINT,
            mode="streaming",
            prompt="Hi",
            max_tokens=4,
            engine=streaming,
        )
        assert expected == actual
        # The shared engine contract samples the logits produced by the final
        # prompt token, then advances with the sampled token.  The previous
        # fixture re-fed the final prompt token and asserted a stale sequence.
        assert actual == [2046, 1851, 45, 21265]
    finally:
        resident.close()
        streaming.close()

def test_rwkv7a_sequence_prefill_matches_resident_token_loop() -> None:
    resident = make_chatrwkv_engine(
        PACK, CHECKPOINT, mode="resident", max_tokens=1, skeleton_load=False
    )
    streaming = make_chatrwkv_engine(
        PACK, CHECKPOINT, mode="streaming", max_tokens=1, skeleton_load=True
    )
    try:
        expected = greedy_token_ids(
            PACK,
            CHECKPOINT,
            mode="resident",
            prompt="Hi",
            max_tokens=1,
            engine=resident,
        )
        actual = greedy_token_ids(
            PACK,
            CHECKPOINT,
            mode="streaming",
            prompt="Hi",
            max_tokens=1,
            engine=streaming,
        )
        assert expected == actual
    finally:
        resident.close()
        streaming.close()
