"""Bandwidth-throttled weight store for multi-SSD simulation.

Wraps one or more underlying stores and adds artificial per-store
latency to simulate the aggregate bandwidth of N physical SSDs.
Used for bench scripts that want to measure how the engine performs
under multi-SSD sharding without actually having multiple SSDs.

Each throttled store is assigned a fixed bandwidth cap (MB/s). The
read returns the actual bytes but sleeps for ``len_bytes / bandwidth``
before returning, simulating a slower SSD.
"""

from __future__ import annotations

import time
import threading
from pathlib import Path

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


class ThrottledWeightStore(WeightStore):
    """Wrap a store and throttle its read bandwidth."""

    def __init__(
        self,
        inner: WeightStore,
        *,
        bandwidth_mbps: float = 500.0,
        store_id: int = 0,
    ) -> None:
        self._inner = inner
        self._bandwidth_mbps = max(1.0, float(bandwidth_mbps))
        self._store_id = store_id
        self._total_bytes_read = 0
        self._total_sleep_s = 0.0
        self._stats_lock = threading.Lock()

    @property
    def stable_memoryviews(self) -> bool:
        return bool(getattr(self._inner, "stable_memoryviews", False))

    @property
    def store_id(self) -> int:
        return self._store_id

    @property
    def bandwidth_mbps(self) -> float:
        return self._bandwidth_mbps

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "store_id": self._store_id,
                "bandwidth_mbps": self._bandwidth_mbps,
                "total_bytes_read": self._total_bytes_read,
                "total_sleep_s": self._total_sleep_s,
            }

    def _throttle(self, nbytes: int) -> None:
        sleep_s = (nbytes / 1e6) / self._bandwidth_mbps
        with self._stats_lock:
            # Serialize the delay per simulated device.  Sleeping outside
            # the lock lets concurrent reads multiply the effective
            # bandwidth beyond the configured per-shard cap.
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._total_sleep_s += sleep_s
            self._total_bytes_read += nbytes

    def read_bytes(self, entry: TensorEntry) -> bytes:
        data = self._inner.read_bytes(entry)
        self._throttle(len(data))
        return data

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        self._inner.read_range(entry, byte_offset, length, dest)
        self._throttle(length)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        data = self._inner.read_bytes_span(offset, length)
        self._throttle(len(data))
        return data

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        data = self._inner.read_bytearray_span(offset, length)
        self._throttle(len(data))
        return data

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        data = self._inner.read_memoryview_span(offset, length)
        self._throttle(len(data))
        return data

    def read_bytes_for_shard(
        self, shard_name: str, offset: int, length: int
    ) -> bytes:
        """Delegate sharded span reads while accounting for their bytes."""
        fn = getattr(self._inner, "read_bytes_for_shard", None)
        if fn is None:
            raise NotImplementedError("wrapped store does not expose shard reads")
        data = fn(shard_name, offset, length)
        self._throttle(len(data))
        return data

    def read_layer_spans_parallel(
        self, layer_specs: list[tuple[str, int, int]]
    ) -> list[tuple[bytes, int]]:
        fn = getattr(self._inner, "read_layer_spans_parallel", None)
        if fn is None:
            raise NotImplementedError(
                "wrapped store does not expose parallel shard span reads"
            )
        results = fn(layer_specs)
        for raw, _base in results:
            self._throttle(len(raw))
        return results

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_prefetch(entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_release(entries)

    def close(self) -> None:
        self._inner.close()


def throttled_sharded_store(
    manifest: Path,
    *,
    per_shard_bandwidth_mbps: float = 500.0,
    n_shards: int = 1,
) -> "ThrottledShardedStore":
    """Build a sharded weight store where each shard is bandwidth-throttled.

    If ``n_shards == 1``, returns a single throttled store (simulates
    a single SSD at ``per_shard_bandwidth_mbps``).

    If ``n_shards > 1``, splits the pack logically into ``n_shards``
    shard groups (by layer_id %% n_shards) and wraps each shard's
    store with a throttled store at the same bandwidth. The aggregate
    bandwidth is then ``n_shards * per_shard_bandwidth_mbps``.

    This simulates a multi-SSD setup on a single SSD: the per-shard
    store reads the same file but sleeps as if it were slower.
    """
    from rwkv_ssd.runtime.io_mmap import MmapWeightStore
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store_sharded import ShardedWeightStore

    pack_dir = Path(manifest)
    m = Manifest.load(pack_dir)
    if n_shards == 1:
        inner = MmapWeightStore(m.weights_path)
        return ThrottledShardedStore(
            ThrottledWeightStore(inner, bandwidth_mbps=per_shard_bandwidth_mbps, store_id=0),
            n_shards=1,
        )
    inner = ShardedWeightStore(m, parallel_workers=n_shards)
    # Preserve every normalized-path key and compatibility alias.  Looking
    # up only ``shard_path.name`` breaks packs containing e.g. ``ssd0/`` and
    # ``ssd1/`` directories when their basenames happen to match.
    throttled_stores: dict[str, WeightStore] = {}
    wrapped_by_id: dict[int, ThrottledWeightStore] = {}
    for key, store in list(inner._stores.items()):
        store_id = id(store)
        wrapped = wrapped_by_id.get(store_id)
        if wrapped is None:
            wrapped = ThrottledWeightStore(
                store,
                bandwidth_mbps=per_shard_bandwidth_mbps,
                store_id=len(wrapped_by_id),
            )
            wrapped_by_id[store_id] = wrapped
        throttled_stores[key] = wrapped
    inner._stores = throttled_stores
    return ThrottledShardedStore(inner, n_shards=n_shards)


class ThrottledShardedStore(WeightStore):
    """Wrapper around a sharded store that aggregates throttle stats."""

    def __init__(self, inner: WeightStore, *, n_shards: int) -> None:
        self._inner = inner
        self._n_shards = n_shards

    @property
    def n_shards(self) -> int:
        return self._n_shards

    def _throttled_stores(self) -> list[ThrottledWeightStore]:
        result: list[ThrottledWeightStore] = []

        def walk(s: WeightStore) -> None:
            if isinstance(s, ThrottledWeightStore):
                result.append(s)
            elif isinstance(s, ThrottledShardedStore):
                walk(s._inner)
            elif hasattr(s, "_stores"):
                for sub in s._stores.values():
                    walk(sub)

        walk(self._inner)
        return result

    def aggregate_stats(self) -> dict:
        stores = self._throttled_stores()
        total_bytes = sum(s.stats["total_bytes_read"] for s in stores)
        total_sleep = sum(s.stats["total_sleep_s"] for s in stores)
        wall_s = max((s.stats["total_sleep_s"] for s in stores), default=0.0)
        return {
            "n_shards": self._n_shards,
            "total_bytes_read": total_bytes,
            "max_per_shard_sleep_s": wall_s,
            "sum_per_shard_sleep_s": total_sleep,
            "effective_bandwidth_mbps": (total_bytes / 1e6) / wall_s if wall_s > 0 else 0.0,
            "per_shard": [s.stats for s in stores],
        }

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._inner.read_bytes(entry)

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        self._inner.read_range(entry, byte_offset, length, dest)

    def read_bytes_for_shard(
        self, shard_name: str, offset: int, length: int
    ) -> bytes:
        fn = getattr(self._inner, "read_bytes_for_shard", None)
        if fn is None:
            raise NotImplementedError("wrapped store does not expose shard reads")
        return fn(shard_name, offset, length)

    def read_layer_spans_parallel(
        self, layer_specs: list[tuple[str, int, int]]
    ) -> list[tuple[bytes, int]]:
        fn = getattr(self._inner, "read_layer_spans_parallel", None)
        if fn is None:
            raise NotImplementedError(
                "wrapped store does not expose parallel shard span reads"
            )
        return fn(layer_specs)

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_prefetch(entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_release(entries)

    def close(self) -> None:
        self._inner.close()
