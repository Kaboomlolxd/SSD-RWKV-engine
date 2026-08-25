"""Sharded weight store — one TensorEntry per shard file (M-class).

When the pack was built with ``pack_runtime --shard N``, the manifest
records a per-tensor ``shard_file`` (the relative path of the file the
tensor lives in). The sharded store opens every shard once and routes
``read_bytes`` to the correct file. When the shards are placed on
different physical SSDs (e.g. via symlinks or distinct mount points),
this gives Kx parallel I/O for K shards.

The store is intentionally simple — same interface as the single-file
stores, just a thin routing layer on top. Parallel reads across shards
are supported via a small thread pool (set ``parallel_workers > 1``);
the prefetch thread in the provider is the natural place to issue
shard-spanning reads, so the single-worker case is the default.

For 0.1B-class packs the entire pack fits in RAM and the SSD is not
the bottleneck — sharding is mostly a deployment-time knob (move
shards to different mount points) and a research path toward
multi-SSD aggregate bandwidth for 7B+ packs where the SSD is the
bottleneck.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from rwkv_ssd.runtime.io_pread import PreadWeightStore
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore

__all__ = ["ShardedWeightStore", "open_sharded_weight_store"]


class ShardedWeightStore(WeightStore):
    """Read-only store that routes each tensor to its shard file.

    All shard files are opened with ``PreadWeightStore`` (thread-safe
    ``pread``). ``read_bytes`` / ``read_range`` look up the shard from
    ``entry.shard_file``; tensors with an empty ``shard_file`` fall
    through to the primary ``weights_path`` (backward-compat with
    legacy single-file packs).

    ``parallel_workers`` controls the size of the read thread pool
    used by :meth:`read_bytes_many` (concurrent reads across shards).
    The default of 1 keeps reads strictly sequential — safe and
    predictable, but doesn't exploit the multi-SSD aggregate
    bandwidth. Set to ``len(shards)`` to fully overlap.
    """

    def __init__(
        self,
        manifest: Manifest,
        *,
        parallel_workers: int = 1,
        files: Iterable[Path] | None = None,
    ) -> None:
        self._manifest = manifest
        self._root = manifest.pack_dir or manifest.weights_path.parent
        # Cache: normalized relative path → PreadWeightStore.
        self._stores: dict[str, PreadWeightStore] = {}
        paths = list(files) if files is not None else list(manifest.shard_files)
        for shard_path in paths:
            key = self._path_key(shard_path)
            store = self._stores.get(key)
            if store is None:
                store = PreadWeightStore(shard_path)
                self._stores[key] = store
            # Keep the basename as a compatibility alias for older callers.
            # close() deduplicates aliases by object id.
            self._stores.setdefault(shard_path.name, store)
        # Fallback for legacy single-file packs.
        if files is None and not self._stores and manifest.weights_path.is_file():
            key = self._path_key(manifest.weights_path)
            store = PreadWeightStore(manifest.weights_path)
            self._stores[key] = store
            self._stores.setdefault(manifest.weights_path.name, store)
        self._executor: ThreadPoolExecutor | None = None
        unique_store_count = len({id(store) for store in self._stores.values()})
        if parallel_workers > 1 and unique_store_count > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=parallel_workers,
                thread_name_prefix="shard-read",
            )

    def _path_key(self, path: Path) -> str:
        try:
            return path.relative_to(self._root).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _entry_key(value: str) -> str:
        return value.replace("\\", "/").lstrip("./")

    def _store_for(self, entry: TensorEntry) -> PreadWeightStore:
        if entry.shard_file:
            key = self._entry_key(entry.shard_file)
            store = self._stores.get(key)
            if store is None:
                # Shard path wasn't preloaded; fall back to the primary
                # weights_path (the manifest records the relative name
                # — we resolve to an absolute path and re-cache).
                abs_path = self._manifest.shard_path_for(entry)
                if abs_path != self._manifest.weights_path:
                    store = PreadWeightStore(abs_path)
                    self._stores[key] = store
                    self._stores.setdefault(abs_path.name, store)
            if store is not None:
                return store
        # Legacy / fallback: use the primary weights store.
        primary_key = self._path_key(self._manifest.weights_path)
        store = self._stores.get(primary_key)
        if store is None:
            store = self._stores[self._manifest.weights_path.name]
        return store

    def read_bytes(self, entry: TensorEntry) -> bytes:
        if entry.stripes:
            # Manifest-v2 striped tensor: gather logical extents in order.
            pieces = sorted(entry.stripes, key=lambda s: int(s.get("logical_offset", 0)))
            out = bytearray(entry.length)
            requests: list[tuple[dict, PreadWeightStore]] = []
            for stripe in pieces:
                store = self._stores.get(self._entry_key(stripe["shard_file"]))
                if store is None:
                    raise FileNotFoundError(f"shard not found: {stripe['shard_file']}")
                requests.append((stripe, store))
            if self._executor is not None and len(requests) > 1:
                futures = [
                    self._executor.submit(
                        store.read_bytes_span,
                        int(stripe["offset"]),
                        int(stripe["length"]),
                    )
                    for stripe, store in requests
                ]
                payloads = [future.result() for future in futures]
            else:
                payloads = [
                    store.read_bytes_span(
                        int(stripe["offset"]), int(stripe["length"])
                    )
                    for stripe, store in requests
                ]
            cursor = 0
            for (stripe, _store), payload in zip(requests, payloads):
                length = int(stripe["length"])
                logical = int(stripe.get("logical_offset", cursor))
                if logical + length > len(out):
                    raise ValueError(f"striped tensor {entry.name!r} exceeds logical length")
                out[logical : logical + length] = payload
                cursor = logical + length
            if cursor < entry.length:
                raise ValueError(f"striped tensor {entry.name!r} has a short extent list")
            return bytes(out)
        return self._store_for(entry).read_bytes(entry)

    def read_range(
        self,
        entry: TensorEntry,
        byte_offset: int,
        length: int,
        dest: memoryview,
    ) -> None:
        if entry.stripes:
            data = self.read_bytes(entry)
            if byte_offset < 0 or length < 0 or byte_offset + length > len(data):
                raise ValueError(f"read past end of tensor {entry.name}")
            dest[:length] = data[byte_offset : byte_offset + length]
            return
        self._store_for(entry).read_range(entry, byte_offset, length, dest)

    def read_entries_coalesced(
        self, entries: Iterable[TensorEntry]
    ) -> dict[str, bytes]:
        """Read a layer's entries with per-shard range coalescing.

        Layer-affinity manifests are gathered by reading the smallest range
        covering all entries on a shard.  Manifest-v2 entries contribute their
        physical stripe extents instead; adjacent extents on the same shard
        are merged and copied back into each tensor's logical byte buffer.
        This is deliberately CPU-side and uses the same ``pread`` stores as
        the ordinary path, so Windows remains supported and legacy manifests
        keep their existing routing behavior.
        """
        ordered_entries = list(entries)
        if not ordered_entries:
            return {}

        # ``(entry_name, logical_offset, payload_length, shard_store,
        # physical_offset, physical_length)``.  A normal layer-affinity entry
        # is just one physical extent; a striped entry contributes one per
        # stripe.
        requests: list[tuple[str, int, int, PreadWeightStore, int, int]] = []
        for entry in ordered_entries:
            if entry.stripes:
                for stripe in sorted(
                    entry.stripes,
                    key=lambda item: int(item.get("logical_offset", 0)),
                ):
                    shard_name = self._entry_key(stripe["shard_file"])
                    store = self._stores.get(shard_name)
                    if store is None:
                        raise FileNotFoundError(f"shard not found: {shard_name}")
                    logical = int(stripe.get("logical_offset", 0))
                    payload_length = int(stripe["length"])
                    physical_offset = int(stripe["offset"])
                    physical_length = int(
                        stripe.get("physical_length", payload_length)
                    )
                    requests.append(
                        (
                            entry.name,
                            logical,
                            payload_length,
                            store,
                            physical_offset,
                            physical_length,
                        )
                    )
            else:
                store = self._store_for(entry)
                requests.append(
                    (
                        entry.name,
                        0,
                        entry.length,
                        store,
                        entry.offset,
                        entry.length,
                    )
                )

        # The manifest validator enforces these invariants for on-disk packs;
        # retain a defensive check for callers constructing TensorEntry values
        # directly in tests or embedding the store in another tool.
        by_name = {entry.name: entry for entry in ordered_entries}
        cursors: dict[str, int] = {entry.name: 0 for entry in ordered_entries}
        for name, logical, payload_length, _store, _off, physical_length in requests:
            entry = by_name[name]
            if payload_length <= 0 or physical_length < payload_length:
                raise ValueError(f"invalid physical extent for tensor {name!r}")
            if logical != cursors[name]:
                raise ValueError(
                    f"logical extents for tensor {name!r} are not contiguous"
                )
            if logical + payload_length > entry.length:
                raise ValueError(f"logical extent exceeds tensor {name!r}")
            cursors[name] = logical + payload_length
        for entry in ordered_entries:
            if cursors[entry.name] != entry.length:
                raise ValueError(
                    f"logical extents for tensor {entry.name!r} cover "
                    f"{cursors[entry.name]} bytes, expected {entry.length}"
                )

        # Group by the actual open store.  The same basename can be an alias
        # for multiple nested shard paths, so object identity is intentional.
        grouped: dict[int, tuple[PreadWeightStore, list[tuple]]] = {}
        for request in requests:
            store_id = id(request[3])
            if store_id not in grouped:
                grouped[store_id] = (request[3], [])
            grouped[store_id][1].append(request)

        read_ranges: list[tuple[PreadWeightStore, int, int, list[tuple]]] = []
        for store, store_requests in grouped.values():
            for request in sorted(store_requests, key=lambda item: item[4]):
                physical_offset = request[4]
                physical_end = physical_offset + request[5]
                if not read_ranges or read_ranges[-1][0] is not store:
                    read_ranges.append((store, physical_offset, physical_end, [request]))
                    continue
                current_store, start, end, members = read_ranges[-1]
                if physical_offset > end:
                    read_ranges.append((store, physical_offset, physical_end, [request]))
                else:
                    read_ranges[-1] = (
                        current_store,
                        start,
                        max(end, physical_end),
                        members + [request],
                    )

        def _read(item: tuple[PreadWeightStore, int, int, list[tuple]]) -> bytes:
            store, start, end, _members = item
            return store.read_bytes_span(start, end - start)

        if self._executor is not None and len(read_ranges) > 1:
            futures = [self._executor.submit(_read, item) for item in read_ranges]
            payloads = [future.result() for future in futures]
        else:
            payloads = [_read(item) for item in read_ranges]

        out = {entry.name: bytearray(entry.length) for entry in ordered_entries}
        for (store, start, _end, members), payload in zip(read_ranges, payloads):
            del store  # the store is retained by the range tuple for clarity
            for name, logical, payload_length, _member_store, physical_offset, _physical_length in members:
                rel = physical_offset - start
                chunk = payload[rel : rel + payload_length]
                if len(chunk) != payload_length:
                    raise OSError(f"short coalesced read for tensor {name!r}")
                out[name][logical : logical + payload_length] = chunk
        return {entry.name: bytes(out[entry.name]) for entry in ordered_entries}

    def read_layer(self, entries: Iterable[TensorEntry]) -> dict[str, bytes]:
        """Compatibility alias for the CPU coalesced layer-gather helper."""
        return self.read_entries_coalesced(entries)

    def read_layer_coalesced(
        self, entries: Iterable[TensorEntry]
    ) -> dict[str, bytes]:
        """Explicitly named alias used by diagnostics and benchmark tools."""
        return self.read_entries_coalesced(entries)

    # Per-shard span read: in a sharded pack, a layer's tensors all live
    # in the same shard (round-robin by layer_id). The ``offset`` and
    # ``length`` passed in are the layer's span within that shard, not
    # the global weights.bin offset. This enables the parallel
    # cross-shard prefetch path: the prefetch job groups layers by shard
    # and issues one span read per shard simultaneously.
    def read_bytes_span(self, offset: int, length: int) -> bytes:
        raise NotImplementedError(
            "ShardedWeightStore uses per-shard reads; use "
            "read_bytes_for_shard(shard_path, offset, length) instead"
        )

    def read_bytes_for_shard(
        self, shard_name: str, offset: int, length: int
    ) -> bytes:
        """Read a span from a specific shard file (offset is local to the shard)."""
        store = self._stores.get(self._entry_key(shard_name))
        if store is None:
            raise FileNotFoundError(f"shard not found: {shard_name}")
        return store.read_bytes_span(offset, length)

    def read_memoryview_span(
        self, offset: int, length: int
    ) -> memoryview:
        raise NotImplementedError(
            "ShardedWeightStore uses per-shard reads; use "
            "read_memoryview_for_shard(shard_name, offset, length) instead"
        )

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        raise NotImplementedError(
            "ShardedWeightStore uses per-shard reads; use "
            "read_bytearray_for_shard(shard_name, offset, length) instead"
        )

    def read_bytes_many(
        self, entries: Iterable[TensorEntry]
    ) -> list[bytes]:
        """Read multiple tensors in parallel across shards.

        When ``parallel_workers == 1`` (default), this is equivalent
        to a sequential ``[read_bytes(e) for e in entries]``. When
        ``parallel_workers > 1``, the reads are issued via the
        thread pool — a single ``pread`` on each shard can be in
        flight simultaneously, multiplying the aggregate SSD
        bandwidth by up to ``min(parallel_workers, len(shards))``.
        """
        entries = list(entries)
        if any(entry.stripes for entry in entries):
            # Avoid submitting an outer ``read_bytes`` task that in turn
            # submits stripe-span tasks to this same executor and waits for
            # them.  With one outer task per worker that pattern can starve
            # the pool indefinitely.  The coalesced gather submits only the
            # leaf span reads, and preserves duplicate entries in the result.
            unique = {entry.name: entry for entry in entries}
            gathered = self.read_entries_coalesced(unique.values())
            return [gathered[entry.name] for entry in entries]
        if self._executor is None or len(entries) <= 1:
            return [self.read_bytes(e) for e in entries]
        futures = [self._executor.submit(self.read_bytes, e) for e in entries]
        return [f.result() for f in futures]

    def read_layer_spans_parallel(
        self, layer_specs: list[tuple[str, int, int]]
    ) -> list[tuple[bytes, int]]:
        """Read layer spans from multiple shards in parallel.

        ``layer_specs`` is a list of ``(shard_name, offset, length)``
        tuples. Each spec is read from its named shard. When
        ``parallel_workers > 1`` and specs span multiple shards, the
        reads execute in the thread pool — one pread per shard can
        be in flight simultaneously, exploiting multi-SSD aggregate
        bandwidth.

        Returns a list of ``(raw_bytes, base_offset)`` tuples in the
        same order as ``layer_specs``.
        """
        if self._executor is None or len(layer_specs) <= 1:
            return [
                (self.read_bytes_for_shard(name, off, length), off)
                for name, off, length in layer_specs
            ]
        futures = [
            self._executor.submit(self.read_bytes_for_shard, name, off, length)
            for name, off, length in layer_specs
        ]
        return [
            (f.result(), off) for f, (_, off, _) in zip(futures, layer_specs)
        ]

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        # Group by shard so each file gets a single advise call.
        any_ok = False
        groups: dict[int, tuple[PreadWeightStore, list[TensorEntry]]] = {}
        for entry in entries:
            store = self._store_for(entry)
            store_id = id(store)
            if store_id not in groups:
                groups[store_id] = (store, [])
            groups[store_id][1].append(entry)
        for store, shard_entries in groups.values():
            if store.advise_prefetch(shard_entries):
                any_ok = True
        return any_ok

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        any_ok = False
        groups: dict[int, tuple[PreadWeightStore, list[TensorEntry]]] = {}
        for entry in entries:
            store = self._store_for(entry)
            store_id = id(store)
            if store_id not in groups:
                groups[store_id] = (store, [])
            groups[store_id][1].append(entry)
        for store, shard_entries in groups.values():
            if store.advise_release(shard_entries):
                any_ok = True
        return any_ok

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        closed: set[int] = set()
        for store in self._stores.values():
            if id(store) in closed:
                continue
            closed.add(id(store))
            store.close()

    def __getstate__(self) -> dict:
        # ThreadPoolExecutor isn't picklable; strip it for safety.
        state = self.__dict__.copy()
        state["_executor"] = None
        return state


def open_sharded_weight_store(
    manifest: Manifest,
    *,
    parallel_workers: int = 1,
    files: Iterable[Path] | None = None,
) -> WeightStore:
    """Open a sharded weight store for a sharded manifest.

    Falls through to a single-file ``PreadWeightStore`` when the
    manifest is legacy (no shards). Use this from the engine config
    when ``manifest.is_sharded()`` is True.
    """
    from rwkv_ssd.runtime.io_paced import maybe_wrap_io_cap

    if files is None and not manifest.is_sharded():
        return maybe_wrap_io_cap(PreadWeightStore(manifest.weights_path))
    return maybe_wrap_io_cap(
        ShardedWeightStore(
            manifest,
            parallel_workers=parallel_workers,
            files=files,
        )
    )
