"""Decode-disk-cache write stats + DNV-2 overlap telemetry."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.decode_disk_cache import (
    DecodeDiskCache,
    cache_write_stats,
    reset_cache_write_stats,
)


def test_cache_write_stats_initial_zero() -> None:
    reset_cache_write_stats()
    submits, sync_ms = cache_write_stats()
    assert submits == 0
    assert sync_ms == 0.0


def test_cache_write_stats_records_submits(tmp_path: Path) -> None:
    import torch

    from rwkv_ssd.runtime.manifest import TensorEntry

    reset_cache_write_stats()
    cache = DecodeDiskCache(tmp_path, {"weights_sha256": "deadbeef"})
    entries = [
        TensorEntry(
            "blocks.0.att.receptance.weight",
            0,
            "bfloat16",
            [4, 4],
            0,
            32,
            4096,
            "streamed",
        )
    ]
    tensors = {
        "blocks.0.att.receptance.weight": torch.zeros(4, 4, dtype=torch.bfloat16)
    }
    cache.store_layer(0, entries, tensors, async_write=True)
    cache.store_layer(0, entries, tensors, async_write=True)
    submits, sync_ms = cache_write_stats()
    cache.close()
    assert submits == 2
    assert sync_ms >= 0.0
    # Async writes: most of the work should be offloaded
    # (sync portion is just the bf16 encode, not the file write)
    assert sync_ms < 5000.0


def test_cache_write_stats_sync_path(tmp_path: Path) -> None:
    """Sync path records the encode cost in sync_ms too."""
    import torch

    from rwkv_ssd.runtime.manifest import TensorEntry

    reset_cache_write_stats()
    cache = DecodeDiskCache(tmp_path, {"weights_sha256": "beef0001"})
    entries = [
        TensorEntry(
            "blocks.0.att.receptance.weight",
            0,
            "bfloat16",
            [2, 2],
            0,
            8,
            4096,
            "streamed",
        )
    ]
    tensors = {
        "blocks.0.att.receptance.weight": torch.zeros(2, 2, dtype=torch.bfloat16)
    }
    cache.store_layer(0, entries, tensors, async_write=False)
    cache.close()
    submits, sync_ms = cache_write_stats()
    assert submits == 1
    assert sync_ms >= 0.0


def test_dnv3_mmap_reused_across_loads(tmp_path: Path) -> None:
    """DNV-3: subsequent try_load_layer reuses the same mmap (one open per file)."""
    import torch

    from rwkv_ssd.runtime.manifest import TensorEntry

    cache = DecodeDiskCache(tmp_path, {"weights_sha256": "dnv3"})
    entries = [
        TensorEntry(
            "blocks.0.att.r",
            0,
            "bfloat16",
            [4, 4],
            0,
            32,
            4096,
            "streamed",
        )
    ]
    tensors = {"blocks.0.att.r": torch.zeros(4, 4, dtype=torch.bfloat16)}
    cache.store_layer(0, entries, tensors, async_write=False)
    cache.flush()

    out1 = cache.try_load_layer(0, entries, torch.device("cpu"))
    cached_after_first = len(cache._mmap_cache)
    out2 = cache.try_load_layer(0, entries, torch.device("cpu"))
    cached_after_second = len(cache._mmap_cache)
    assert out1 is not None and out2 is not None
    assert cached_after_first == 1
    assert cached_after_second == 1, "mmap should be reused, not re-opened"
    cache.close()
    assert cache._mmap_cache == {}
