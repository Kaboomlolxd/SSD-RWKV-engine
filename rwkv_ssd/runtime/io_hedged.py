"""Hedged parallel reads for tail latency (P2.d)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


class HedgedWeightStore(WeightStore):
    """Race two identical ``read_bytes`` calls; return first completion."""

    def __init__(self, primary: WeightStore, secondary: WeightStore | None = None) -> None:
        self._primary = primary
        self._secondary = secondary or primary
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._owns_secondary = secondary is not None and secondary is not primary
        self._lock = threading.Lock()

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        self._primary.read_range(entry, byte_offset, length, dest)

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self._race("read_bytes", entry)

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self._race("read_bytes_span", offset, length)

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        return self._race("read_bytearray_span", offset, length)

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        return self._race("read_memoryview_span", offset, length)

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        return self._primary.advise_prefetch(entries)

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return self._primary.advise_release(entries)

    def _race(self, method: str, *args):
        """Return the first successful result, tolerating one failed leg.

        A transient error on the first completed leg should not mask a
        successful mirrored read on the other leg.
        """
        futures = [
            self._pool.submit(getattr(self._primary, method), *args),
            self._pool.submit(getattr(self._secondary, method), *args),
        ]
        first_error: Exception | None = None
        try:
            for future in as_completed(futures):
                try:
                    return future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            for future in futures:
                future.cancel()
        if first_error is not None:
            raise first_error
        raise RuntimeError(f"hedged {method} completed without a result")

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._primary.close()
        if self._owns_secondary:
            self._secondary.close()
