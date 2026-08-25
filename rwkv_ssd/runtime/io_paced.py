"""Optional read bandwidth cap for SSD thermal headroom.

When ``RWKV_SSD_IO_CAP_MBPS`` is set, wraps a :class:`WeightStore` and
sleeps after each read so sustained throughput stays at or below the cap.
This trades tok/s for lower NVMe controller duty cycle — useful when a
drive thermally throttles under full-speed streaming.

For multi-SSD simulation throttling, see :mod:`io_throttled`.
"""

from __future__ import annotations

import os
import time

from rwkv_ssd.runtime.io_throttled import ThrottledWeightStore
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


def resolve_io_cap_mbps() -> float | None:
    raw = os.environ.get("RWKV_SSD_IO_CAP_MBPS", "").strip()
    if not raw:
        return None
    try:
        cap = float(raw)
    except ValueError:
        return None
    return cap if cap > 0 else None


def maybe_wrap_io_cap(store: WeightStore) -> WeightStore:
    """Wrap ``store`` when ``RWKV_SSD_IO_CAP_MBPS`` is set."""
    cap = resolve_io_cap_mbps()
    if cap is None:
        return store
    return PacedWeightStore(store, bandwidth_mbps=cap)


class PacedWeightStore(ThrottledWeightStore):
    """Production IO cap — same throttle math as bench simulation."""

    def __init__(self, inner: WeightStore, *, bandwidth_mbps: float) -> None:
        super().__init__(inner, bandwidth_mbps=bandwidth_mbps, store_id=0)

    @property
    def pace_stats(self) -> dict:
        return self.stats


def advise_release_after_read(store: WeightStore, entries: list[TensorEntry]) -> None:
    """No-op helper reserved for future paced-release policies."""
    fn = getattr(store, "advise_release", None)
    if fn:
        fn(entries)
