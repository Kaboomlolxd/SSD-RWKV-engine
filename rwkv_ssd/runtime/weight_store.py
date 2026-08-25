"""Weight store adapters and factory."""

from __future__ import annotations

from pathlib import Path
import os

from rwkv_ssd.runtime.io_mmap import MmapWeightStore
from rwkv_ssd.runtime.io_pread import PreadWeightStore
from rwkv_ssd.runtime.io_cold import ColdPreadWeightStore
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import IO_BACKENDS, WeightStore  # noqa: F401 — re-export

__all__ = [
    "IO_BACKENDS",
    "WeightStore",
    "MmapWeightStoreAdapter",
    "PreadWeightStoreAdapter",
    "open_weight_store",
]


class MmapWeightStoreAdapter(WeightStore):
    stable_memoryviews = True

    def __init__(
        self,
        weights_path: Path,
        *,
        mmap_sequential: bool = False,
        mmap_huge_pages: bool = True,
    ) -> None:
        self._inner = MmapWeightStore(
            weights_path,
            sequential_advise=mmap_sequential,
            huge_page_advise=mmap_huge_pages,
        )

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._inner.read_bytes(entry)

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        self._inner.read_range(entry, byte_offset, length, dest)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self._inner.read_bytes_span(offset, length)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        fn = getattr(self._inner, "read_bytearray_span", None)
        if fn is not None:
            return fn(offset, length)
        return bytearray(self._inner.read_bytes_span(offset, length))

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        fn = getattr(self._inner, "read_memoryview_span", None)
        if fn is not None:
            return fn(offset, length)
        return memoryview(self.read_bytes_span(offset, length))

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        fn = getattr(self._inner, "advise_prefetch", None)
        return bool(fn(entries)) if fn else False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        fn = getattr(self._inner, "advise_release", None)
        return bool(fn(entries)) if fn else False

    def close(self) -> None:
        self._inner.close()


class PreadWeightStoreAdapter(WeightStore):
    stable_memoryviews = False

    def __init__(self, weights_path: Path) -> None:
        self._inner = PreadWeightStore(weights_path)

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._inner.read_bytes(entry)

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        self._inner.read_range(entry, byte_offset, length, dest)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self._inner.read_bytes_span(offset, length)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        fn = getattr(self._inner, "read_bytearray_span", None)
        if fn is not None:
            return fn(offset, length)
        return bytearray(self._inner.read_bytes_span(offset, length))

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        fn = getattr(self._inner, "read_memoryview_span", None)
        if fn is not None:
            return fn(offset, length)
        return memoryview(self.read_bytes_span(offset, length))

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_prefetch(entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return self._inner.advise_release(entries)

    def close(self) -> None:
        self._inner.close()


class ColdPreadWeightStoreAdapter(WeightStore):
    def __init__(self, weights_path: Path) -> None:
        self._inner = ColdPreadWeightStore(weights_path)

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._inner.read_bytes(entry)

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        data = self._inner.read_bytes_span(entry.offset + byte_offset, length)
        dest[:length] = data

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self._inner.read_bytes_span(offset, length)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        return self._inner.read_bytearray_span(offset, length)

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        return memoryview(self.read_bytes_span(offset, length))

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return False

    def close(self) -> None:
        self._inner.close()


def open_weight_store(
    weights_path: Path,
    backend: str = "mmap",
    *,
    threaded_workers: int = 2,
    mmap_sequential: bool = False,
    hedged: bool = False,
) -> WeightStore:
    key = backend.strip().lower()
    if key == "threaded":
        from rwkv_ssd.runtime.io_threaded import ThreadedWeightStore

        inner = open_weight_store(
            weights_path,
            backend="mmap",
            threaded_workers=0,
            mmap_sequential=mmap_sequential,
            hedged=False,
        )
        store: WeightStore = ThreadedWeightStore(inner, workers=threaded_workers)
    elif key == "mmap":
        store = MmapWeightStoreAdapter(weights_path, mmap_sequential=mmap_sequential)
    elif key == "pread":
        store = PreadWeightStoreAdapter(weights_path)
    elif key in ("cold", "cold_pread", "cold-read"):
        store = ColdPreadWeightStoreAdapter(weights_path)
    else:
        raise ValueError(
            f"unknown weight store backend: {backend!r} "
            f"(use: {', '.join(sorted(IO_BACKENDS))})"
        )
    if hedged and key != "threaded":
        from rwkv_ssd.runtime.io_hedged import HedgedWeightStore

        mirror = open_weight_store(
            weights_path,
            backend=key,
            threaded_workers=0,
            mmap_sequential=mmap_sequential,
            hedged=False,
        )
        store = HedgedWeightStore(store, mirror)
    elif hedged and key == "threaded":
        from rwkv_ssd.runtime.io_hedged import HedgedWeightStore

        inner2 = open_weight_store(
            weights_path, backend="mmap", mmap_sequential=mmap_sequential
        )
        store = HedgedWeightStore(store, inner2)
    from rwkv_ssd.runtime.io_paced import maybe_wrap_io_cap

    store = maybe_wrap_io_cap(store)
    if os.environ.get("RWKV_PAGE_RESIDENCY", "0").strip().lower() in (
        "1", "true", "on", "yes"
    ):
        from rwkv_ssd.runtime.page_residency import ObservedResidencyWeightStore

        try:
            ttl_s = float(os.environ.get("RWKV_PAGE_RESIDENCY_TTL_S", "30"))
        except ValueError:
            ttl_s = 30.0
        store = ObservedResidencyWeightStore(store, ttl_s=max(0.001, ttl_s))
    return store
