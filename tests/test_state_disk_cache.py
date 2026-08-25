"""SSD-backed prefix state cache (``.state_cache/``)."""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState
from rwkv_ssd.runtime.state_disk_cache import StateDiskCache


def test_state_disk_roundtrip(tmp_path) -> None:
    tensors = [torch.zeros(2, 3, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16)]
    disk = StateDiskCache(tmp_path)
    disk.store("sys", RecurrentState(last_token_id=42, rwkv7_state=tensors))
    loaded = disk.load("sys")
    assert loaded is not None
    assert loaded.last_token_id == 42
    assert loaded.rwkv7_state is not None
    assert len(loaded.rwkv7_state) == 2
    assert torch.equal(loaded.rwkv7_state[0], tensors[0])
    assert torch.equal(loaded.rwkv7_state[1], tensors[1])


def test_prefix_cache_loads_from_disk_after_restart(tmp_path) -> None:
    tensors = [torch.arange(6, dtype=torch.float32).reshape(2, 3)]
    cache1 = PrefixStateCache(max_entries=4, disk_dir=tmp_path)
    cache1.put("prefix", RecurrentState(last_token_id=9, rwkv7_state=tensors))
    cache2 = PrefixStateCache(max_entries=4, disk_dir=tmp_path)
    hit = cache2.get("prefix")
    assert hit is not None
    assert hit.rwkv7_state is not None
    assert hit.last_token_id == 9
    assert torch.equal(hit.rwkv7_state[0], tensors[0])
    assert cache2.stats.hits >= 1


def test_prefix_cache_contains_external_state(tmp_path) -> None:
    cache = PrefixStateCache(max_entries=4, disk_dir=tmp_path)
    state = RecurrentState(
        last_token_id=3,
        external_state=torch.arange(6, dtype=torch.float32).numpy(),
    )
    cache.put("external", state)

    restarted = PrefixStateCache(max_entries=4, disk_dir=tmp_path)
    assert restarted.contains("external")
    hit = restarted.get("external")
    assert hit is not None
    assert hit.external_state is not None


def test_prefix_cache_disk_write_unsupported_dtype_is_fail_soft(tmp_path) -> None:
    cache = PrefixStateCache(max_entries=4, disk_dir=tmp_path)
    cache.put(
        "f16",
        RecurrentState(
            last_token_id=1,
            rwkv7_state=[torch.ones(2, dtype=torch.float16)],
        ),
    )
    hit = cache.get("f16")
    assert hit is not None
    assert hit.rwkv7_state is not None
    assert hit.rwkv7_state[0].dtype == torch.float16
