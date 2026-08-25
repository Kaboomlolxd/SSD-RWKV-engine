"""Tests for the engine state snapshot API (P1 #13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rwkv_ssd.runtime.snapshot import (
    SnapshotMeta,
    load_snapshot,
    save_snapshot,
)
from rwkv_ssd.runtime.state_cache import RecurrentState


def test_save_load_round_trip_synthetic_state(tmp_path: Path) -> None:
    state = RecurrentState(
        h=torch.randn(64, dtype=torch.float32),
        rwkv7_state=None,
        last_token_id=42,
    )
    meta = SnapshotMeta(
        backend="synthetic",
        mode="streaming",
        model_family="rwkv7",
        max_layers_in_z=1,
        decouple_provider_cache=True,
        max_provider_cache_layers=0,
        prompt="hello",
        last_token_id=42,
    )
    path = tmp_path / "snap.bin"
    save_snapshot(path, state, meta)

    loaded_state, loaded_meta = load_snapshot(path)
    assert loaded_state.h is not None
    assert torch.allclose(loaded_state.h.float(), state.h.float())
    assert loaded_state.rwkv7_state is None
    assert loaded_state.last_token_id == 42
    assert loaded_meta.backend == "synthetic"
    assert loaded_meta.mode == "streaming"
    assert loaded_meta.prompt == "hello"


def test_save_load_round_trip_rwkv7_state(tmp_path: Path) -> None:
    rwkv7 = [
        torch.randn(2, 4, dtype=torch.bfloat16),
        torch.randn(3, 5, dtype=torch.float32),
    ]
    state = RecurrentState(
        h=None,
        rwkv7_state=rwkv7,
        last_token_id=99,
    )
    meta = SnapshotMeta(
        backend="chatrwkv",
        mode="streaming",
        model_family="rwkv7",
        prompt="rwkv7-prompt",
        last_token_id=99,
    )
    path = tmp_path / "snap.bin"
    save_snapshot(path, state, meta)

    loaded_state, loaded_meta = load_snapshot(path)
    assert loaded_state.rwkv7_state is not None
    assert len(loaded_state.rwkv7_state) == 2
    assert torch.equal(loaded_state.rwkv7_state[0], rwkv7[0])
    assert torch.equal(loaded_state.rwkv7_state[1], rwkv7[1])
    assert loaded_state.last_token_id == 99


def test_save_load_round_trip_single_rwkv7_tensor(tmp_path: Path) -> None:
    state = RecurrentState(
        rwkv7_state=[torch.ones(2, 3, dtype=torch.float32)], last_token_id=5
    )
    path = tmp_path / "single-rwkv7.bin"
    save_snapshot(
        path,
        state,
        SnapshotMeta(backend="chatrwkv", mode="streaming", model_family="rwkv7"),
    )

    loaded, _meta = load_snapshot(path)
    assert loaded.h is None
    assert loaded.rwkv7_state is not None
    assert len(loaded.rwkv7_state) == 1
    assert torch.equal(loaded.rwkv7_state[0], state.rwkv7_state[0])


def test_snapshot_meta_to_from_dict() -> None:
    meta = SnapshotMeta(
        backend="chatrwkv",
        mode="partial",
        model_family="rwkv7",
        max_layers_in_z=3,
        decouple_provider_cache=False,
        max_provider_cache_layers=2,
        prompt="p",
        last_token_id=7,
        extras={"bundle_version": "v0.6.15"},
    )
    d = meta.to_dict()
    back = SnapshotMeta.from_dict(d)
    assert back.backend == meta.backend
    assert back.mode == meta.mode
    assert back.max_layers_in_z == 3
    assert back.decouple_provider_cache is False
    assert back.extras["bundle_version"] == "v0.6.15"


def test_snapshot_meta_parses_string_false() -> None:
    meta = SnapshotMeta.from_dict({"decouple_provider_cache": "false"})
    assert meta.decouple_provider_cache is False


def test_engine_save_load_snapshot_synthetic(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
    )
    eng = InferenceEngine(cfg)
    eng.load()
    try:
        eng.generate("hi")
        snap_path = tmp_path / "snap.bin"
        eng.save_snapshot(snap_path, prompt="hi")
        assert snap_path.is_file()
        assert snap_path.stat().st_size > 0
        eng.load_snapshot(snap_path)
        state = eng.backend.get_recurrent_state()
        assert state is not None
    finally:
        eng.close()


def test_chatrwkv_backend_state_hooks_without_checkpoint() -> None:
    from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend

    class DummyModel:
        pass

    backend = ChatRWKVBackend()
    backend._rwkv7 = True
    backend._model = DummyModel()
    state = RecurrentState(
        rwkv7_state=[torch.ones(2, 3), torch.zeros(1, dtype=torch.bfloat16)],
        last_token_id=123,
    )
    backend.set_recurrent_state(state)
    loaded = backend.get_recurrent_state()
    assert loaded is not None
    assert loaded.last_token_id == 123
    assert loaded.rwkv7_state is not None
    assert torch.equal(loaded.rwkv7_state[0], state.rwkv7_state[0])
    assert torch.equal(loaded.rwkv7_state[1], state.rwkv7_state[1])
    loaded.rwkv7_state[0].zero_()
    loaded2 = backend.get_recurrent_state()
    assert loaded2 is not None
    assert loaded2.rwkv7_state is not None
    assert torch.equal(loaded2.rwkv7_state[0], state.rwkv7_state[0])


def test_engine_from_snapshot_round_trip(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    snap_path = tmp_path / "snap.bin"

    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
    )
    eng1 = InferenceEngine(cfg)
    eng1.load()
    try:
        eng1.generate("hello")
        eng1.save_snapshot(snap_path, prompt="hello")
    finally:
        eng1.close()

    cfg2 = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=4,
    )
    eng2 = InferenceEngine(cfg2)
    eng2.load()
    try:
        eng2.load_snapshot(snap_path)
        state = eng2.backend.get_recurrent_state()
        assert state is not None
        assert state.last_token_id == eng1.backend.get_recurrent_state().last_token_id
    finally:
        eng2.close()


def test_snapshot_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "snap.bin"
    bad.write_bytes(b"XXXX" + b"\x00" * 1024)
    with pytest.raises(ValueError, match="bad snapshot magic"):
        load_snapshot(bad)


def test_snapshot_truncated_payload(tmp_path: Path) -> None:
    state = RecurrentState(
        h=torch.randn(8, dtype=torch.float32), rwkv7_state=None, last_token_id=1
    )
    meta = SnapshotMeta(backend="synthetic", mode="streaming", model_family="rwkv7")
    path = tmp_path / "snap.bin"
    save_snapshot(path, state, meta)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises((ValueError, Exception)):
        load_snapshot(path)
