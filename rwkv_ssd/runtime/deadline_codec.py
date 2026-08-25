"""Deadline-aware hedging between preferred and fallback representations."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class FallbackBudget:
    max_fallbacks: int
    max_quality_cost: float
    used_fallbacks: int = 0
    used_quality_cost: float = 0.0

    def consume(self, quality_cost: float) -> bool:
        cost = max(0.0, float(quality_cost))
        if self.used_fallbacks >= max(0, self.max_fallbacks):
            return False
        if self.used_quality_cost + cost > max(0.0, self.max_quality_cost):
            return False
        self.used_fallbacks += 1
        self.used_quality_cost += cost
        return True


@dataclass(frozen=True)
class DeadlineRouteResult(Generic[T]):
    value: T
    representation: str
    timed_out: bool
    fallback_used: bool
    elapsed_ms: float


class DeadlineCodecRouter:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codec-deadline")
        self.timeouts = 0
        self.fallbacks = 0

    def route(
        self,
        preferred: Callable[[], T],
        fallback: Callable[[], T],
        *,
        deadline_ms: float,
        budget: FallbackBudget,
        fallback_quality_cost: float = 1.0,
        preferred_name: str = "preferred",
        fallback_name: str = "fallback",
    ) -> DeadlineRouteResult[T]:
        started = time.perf_counter()
        future = self._executor.submit(preferred)
        try:
            value = future.result(timeout=max(0.0, float(deadline_ms)) / 1000.0)
            return DeadlineRouteResult(
                value, preferred_name, False, False, (time.perf_counter() - started) * 1000
            )
        except TimeoutError:
            self.timeouts += 1
            if budget.consume(fallback_quality_cost):
                self.fallbacks += 1
                future.cancel()
                value = fallback()
                return DeadlineRouteResult(
                    value, fallback_name, True, True, (time.perf_counter() - started) * 1000
                )
            value = future.result()
            return DeadlineRouteResult(
                value, preferred_name, True, False, (time.perf_counter() - started) * 1000
            )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = ["DeadlineCodecRouter", "DeadlineRouteResult", "FallbackBudget"]
