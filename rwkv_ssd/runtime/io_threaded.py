"""Thread-offloaded weight reads (P0 #7 — overlap disk with caller thread)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore
from rwkv_ssd.runtime.weight_store import open_weight_store


class ThreadedWeightStore(WeightStore):
    """
    Wraps a synchronous store and runs ``read_bytes`` on a worker thread.

    Useful to overlap OS read submission with Python compute on CPU hosts.
    """

    def __init__(self, inner: WeightStore, workers: int = 2) -> None:
        self._inner = inner
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))

    @property
    def stable_memoryviews(self) -> bool:
        return bool(getattr(self._inner, "stable_memoryviews", False))

    def read_bytes(self, entry: TensorEntry) -> bytes:
        fut = self._executor.submit(self._inner.read_bytes, entry)
        return fut.result()

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        if hasattr(self._inner, "read_range"):
            fut = self._executor.submit(
                self._inner.read_range, entry, byte_offset, length, dest
            )
            fut.result()
            return
        data = self.read_bytes(entry)
        dest[:length] = data[byte_offset : byte_offset + length]

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        """Delegate contiguous reads through the same worker pool.

        Layer-span streaming calls this method directly; inheriting the base
        ``WeightStore`` implementation would raise ``NotImplementedError``
        even though the wrapped mmap/pread store supports spans.
        """
        return self._executor.submit(
            self._inner.read_bytes_span, offset, length
        ).result()

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        return self._executor.submit(
            self._inner.read_bytearray_span, offset, length
        ).result()

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        return self._executor.submit(
            self._inner.read_memoryview_span, offset, length
        ).result()

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        fn = getattr(self._inner, "advise_prefetch", None)
        return bool(fn(entries)) if fn else False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        fn = getattr(self._inner, "advise_release", None)
        return bool(fn(entries)) if fn else False

    def probably_resident(self, entries: list[TensorEntry]) -> bool:
        """Preserve residency estimates through the async I/O wrapper.

        ``RWKV_PAGE_RESIDENCY`` is commonly enabled around the selected I/O
        backend.  Without this delegation, wrapping an observed mmap/pread
        store in ``threaded`` silently turns every estimate into the base
        class's conservative ``False`` and causes redundant OS prefetch hints.
        """
        fn = getattr(self._inner, "probably_resident", None)
        return bool(fn(entries)) if fn else False

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._inner.close()


def open_threaded_store(
    weights_path: Path, base_backend: str = "mmap", workers: int = 2
) -> WeightStore:
    inner = open_weight_store(weights_path, backend=base_backend)
    return ThreadedWeightStore(inner, workers=workers)
