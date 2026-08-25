from __future__ import annotations

import numpy as np
import pytest
import torch

from rwkv_ssd.runtime.state_cache import RecurrentState
from rwkv_ssd.runtime.state_delta_dag import StateDeltaDAG


def test_state_delta_dag_round_trips_exact_synthetic_state(tmp_path) -> None:
    dag = StateDeltaDAG(tmp_path / "dag")
    root = RecurrentState(h=torch.arange(16, dtype=torch.float32), last_token_id=3)
    child = RecurrentState(h=root.h.clone(), last_token_id=4)
    child.h[5] += 1
    root_id = dag.store(root)
    child_id = dag.store(child, parent_id=root_id)
    restored = dag.load(child_id)
    assert restored.last_token_id == 4
    assert torch.equal(restored.h, child.h)
    assert dag.node_info(child_id)["storage"] == "xor"


def test_state_delta_dag_round_trips_rwkv7_and_external_states(tmp_path) -> None:
    dag = StateDeltaDAG(tmp_path / "dag")
    rwkv = RecurrentState(
        rwkv7_state=[
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, dtype=torch.bfloat16),
        ],
        last_token_id=7,
    )
    rwkv_id = dag.store(rwkv)
    restored = dag.load(rwkv_id)
    assert restored.last_token_id == 7
    assert all(torch.equal(a, b) for a, b in zip(restored.rwkv7_state, rwkv.rwkv7_state))

    external = RecurrentState(
        external_state=np.arange(12, dtype=np.float32).reshape(3, 4),
        last_token_id=9,
    )
    external_id = dag.store(external)
    restored_external = dag.load(external_id)
    np.testing.assert_array_equal(restored_external.external_state, external.external_state)


def test_state_delta_dag_deduplicates_identical_nodes(tmp_path) -> None:
    dag = StateDeltaDAG(tmp_path / "dag")
    state = RecurrentState(h=torch.ones(32), last_token_id=1)
    first = dag.store(state)
    before = dag.storage_bytes()
    second = dag.store(state)
    assert second == first
    assert dag.storage_bytes() == before


def test_state_delta_dag_checkpoint_interval_bounds_chain(tmp_path) -> None:
    dag = StateDeltaDAG(tmp_path / "dag", checkpoint_interval=2)
    base = torch.arange(32, dtype=torch.float32)
    first = dag.store(RecurrentState(h=base.clone(), last_token_id=0))
    second_state = RecurrentState(h=base.clone(), last_token_id=1)
    second_state.h[0] += 1
    second = dag.store(second_state, parent_id=first)
    third_state = RecurrentState(h=second_state.h.clone(), last_token_id=2)
    third_state.h[1] += 1
    third = dag.store(third_state, parent_id=second)
    assert dag.node_info(second)["depth"] == 1
    assert dag.node_info(third)["storage"] == "full"
    assert dag.node_info(third)["depth"] == 0


def test_state_delta_dag_detects_corruption(tmp_path) -> None:
    dag = StateDeltaDAG(tmp_path / "dag")
    node = dag.store(RecurrentState(h=torch.ones(8), last_token_id=1))
    path = dag._path(node)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="corrupt state DAG payload"):
        dag.load(node)
