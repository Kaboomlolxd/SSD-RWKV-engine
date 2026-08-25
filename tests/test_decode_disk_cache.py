"""Async decode disk cache writes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rwkv_ssd.runtime.decode_disk_cache import DecodeDiskCache
from rwkv_ssd.runtime.decode_disk_cache import _weights_key
from rwkv_ssd.runtime.device import xpu_compute_available
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack


def _entry(name: str, numel: int) -> TensorEntry:
    side = int(numel**0.5)
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype="bfloat16",
        shape=[side, side],
        offset=0,
        length=100,
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )


def test_async_store_layer(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    meta = {"weights_sha256": "test"}
    cache = DecodeDiskCache(pack, meta)
    entries = [_entry("blocks.0.weight", 64)]
    tensors = {
        "blocks.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    }
    cache.store_layer(0, entries, tensors, async_write=True)
    cache.flush()
    loaded = cache.try_load_layer(0, entries, torch.device("cpu"))
    assert loaded is not None
    assert loaded["blocks.0.weight"].shape == (8, 8)
    cache.close()


@pytest.mark.skipif(
    not xpu_compute_available(), reason="XPU matrix compute is unavailable"
)
def test_store_layer_copies_accelerator_tensor_to_host(
    tmp_path: Path,
) -> None:
    pack = create_synthetic_pack(tmp_path / "xpu-pack", quiet=True)
    cache = DecodeDiskCache(pack, {"weights_sha256": "xpu"})
    entries = [_entry("blocks.0.weight", 64)]
    tensors = {
        "blocks.0.weight": torch.ones(
            8, 8, dtype=torch.bfloat16, device=torch.device("xpu")
        )
    }
    cache.store_layer(0, entries, tensors, async_write=False)
    loaded = cache.try_load_layer(0, entries, torch.device("cpu"))
    assert loaded is not None
    assert torch.equal(loaded["blocks.0.weight"], tensors["blocks.0.weight"].cpu())
    cache.close()


def test_compressed_store_layer(tmp_path: Path) -> None:
    import os

    os.environ["RWKV_DECODE_CACHE_COMPRESS"] = "1"
    pack = create_synthetic_pack(tmp_path / "pack2", quiet=True)
    meta = {"weights_sha256": "test-compress"}
    cache = DecodeDiskCache(pack, meta)
    entries = [_entry("blocks.0.weight", 64)]
    tensors = {
        "blocks.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    }
    cache.store_layer(0, entries, tensors, async_write=False)
    path = cache._path_for(0)
    raw = path.read_bytes()
    assert raw[:4] == b"RDC\x02"
    loaded = cache.try_load_layer(0, entries, torch.device("cpu"))
    assert loaded is not None
    assert torch.allclose(loaded["blocks.0.weight"], tensors["blocks.0.weight"])
    cache.close()


def test_sharded_cache_key_changes_when_any_shard_changes() -> None:
    first = _weights_key(
        {"weights_sha256_by_file": {"ssd0/weights.bin": "aaa", "ssd1/weights.bin": "bbb"}}
    )
    reordered = _weights_key(
        {"weights_sha256_by_file": {"ssd1/weights.bin": "bbb", "ssd0/weights.bin": "aaa"}}
    )
    changed = _weights_key(
        {"weights_sha256_by_file": {"ssd0/weights.bin": "aaa", "ssd1/weights.bin": "ccc"}}
    )
    assert first == reordered
    assert first != changed


def test_rewriting_layer_invalidates_cached_mmap(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "rewrite-pack", quiet=True)
    cache = DecodeDiskCache(pack, {"weights_sha256": "rewrite"})
    entries = [_entry("blocks.0.weight", 64)]
    first = {"blocks.0.weight": torch.zeros(8, 8, dtype=torch.bfloat16)}
    second = {"blocks.0.weight": torch.ones(8, 8, dtype=torch.bfloat16)}
    cache.store_layer(0, entries, first, async_write=False)
    assert torch.equal(cache.try_load_layer(0, entries, torch.device("cpu"))["blocks.0.weight"], first["blocks.0.weight"])
    cache.store_layer(0, entries, second, async_write=False)
    loaded = cache.try_load_layer(0, entries, torch.device("cpu"))
    assert loaded is not None
    assert torch.equal(loaded["blocks.0.weight"], second["blocks.0.weight"])
    cache.close()
