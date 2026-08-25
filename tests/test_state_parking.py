from __future__ import annotations

import numpy as np
import torch
import pytest

from rwkv_ssd.runtime.state_cache import RecurrentState
from rwkv_ssd.runtime.state_parking import (
    HierarchicalStateStore,
    StateParkingCompatibilityError,
    state_nbytes,
)


def test_state_parking_ram_hit_returns_independent_clone(tmp_path) -> None:
    state = RecurrentState(last_token_id=7, h=torch.arange(8, dtype=torch.float32))
    store = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=1024)
    store.put("session-a", state, model_family="synthetic")
    loaded = store.get("session-a")
    assert loaded is not None and loaded.h is not None
    loaded.h.zero_()
    again = store.get("session-a")
    assert again is not None and again.h is not None
    assert torch.equal(again.h, state.h)
    assert store.stats.ram_hits == 2


def test_state_parking_spills_lru_to_disk_and_promotes_on_read(tmp_path) -> None:
    one = RecurrentState(last_token_id=1, h=torch.ones(8, dtype=torch.float32))
    two = RecurrentState(last_token_id=2, h=torch.full((8,), 2.0))
    cap = state_nbytes(one)
    store = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=cap)
    store.put("one", one)
    store.put("two", two)
    assert store.ram_keys == ("two",)
    assert store.ram_bytes <= cap
    assert store.stats.ram_evictions == 1
    restored = store.get("one")
    assert restored is not None and restored.h is not None
    assert torch.equal(restored.h, one.h)
    assert store.ram_keys == ("one",)
    assert store.stats.disk_hits == 1


def test_state_parking_survives_restart_for_rwkv_and_external_states(tmp_path) -> None:
    root = tmp_path / "parking"
    first = HierarchicalStateStore(root, max_ram_bytes=0)
    rwkv = RecurrentState(
        last_token_id=3,
        rwkv7_state=[torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)],
    )
    external = RecurrentState(
        last_token_id=4,
        external_state=np.arange(10, dtype=np.float32),
    )
    first.put("rwkv", rwkv)
    first.put("external", external)
    assert first.ram_bytes == 0

    restarted = HierarchicalStateStore(root, max_ram_bytes=1024)
    got_rwkv = restarted.get("rwkv")
    got_external = restarted.get("external")
    assert got_rwkv is not None and got_rwkv.rwkv7_state is not None
    assert torch.equal(got_rwkv.rwkv7_state[0], rwkv.rwkv7_state[0])
    assert got_external is not None
    assert np.array_equal(got_external.external_state, external.external_state)
    assert restarted.stats.disk_hits == 2


def test_state_parking_corruption_is_fail_soft(tmp_path) -> None:
    store = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=0)
    path = store.put("broken", RecurrentState(h=torch.ones(4)))
    path.write_bytes(b"broken")
    assert store.get("broken") is None
    assert store.stats.disk_load_failures == 1


def test_state_parking_oversized_state_stays_disk_only(tmp_path) -> None:
    state = RecurrentState(h=torch.ones(1024, dtype=torch.float32))
    store = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=16)
    store.put("large", state)
    assert store.ram_bytes == 0
    assert store.contains("large")
    restored = store.get("large")
    assert restored is not None and restored.h is not None
    assert store.ram_bytes == 0


def test_state_parking_rejects_incompatible_identity_in_ram_and_on_disk(tmp_path) -> None:
    state = RecurrentState(last_token_id=9, h=torch.ones(4))
    store = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=1024)
    store.put(
        "session",
        state,
        backend="synthetic",
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        serialization_version=1,
    )
    with pytest.raises(StateParkingCompatibilityError):
        store.get(
            "session",
            backend="synthetic",
            model_fingerprint="model-b",
            tokenizer_fingerprint="tokenizer-a",
            serialization_version=1,
        )

    restarted = HierarchicalStateStore(tmp_path / "parking", max_ram_bytes=0)
    with pytest.raises(StateParkingCompatibilityError):
        restarted.get(
            "session",
            backend="synthetic",
            model_fingerprint="model-a",
            tokenizer_fingerprint="tokenizer-b",
            serialization_version=1,
        )
