"""Small cooperative throttling helper for long local inference runs."""

from __future__ import annotations

import time


def normalize_power_percent(value: int | float | None) -> int:
    """Clamp a user power target to the supported 1..100 range."""
    if value is None:
        return 100
    return max(1, min(100, int(value)))


def throttle_after_work(start_time: float, power_percent: int | float | None) -> None:
    """Sleep after a unit of work to approximate a target duty cycle.

    ``power_percent=50`` sleeps about as long as the measured work took.
    The helper is intentionally cooperative: it only runs at token/layer
    boundaries where the engine already has control.
    """
    pct = normalize_power_percent(power_percent)
    if pct >= 100:
        return
    work_s = max(0.0, time.perf_counter() - start_time)
    sleep_s = work_s * ((100.0 / pct) - 1.0)
    if sleep_s > 0:
        time.sleep(sleep_s)
