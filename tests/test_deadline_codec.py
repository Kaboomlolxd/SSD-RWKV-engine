from __future__ import annotations

import threading
import time

from rwkv_ssd.runtime.deadline_codec import DeadlineCodecRouter, FallbackBudget


def test_deadline_router_returns_fast_preferred_representation() -> None:
    router = DeadlineCodecRouter()
    try:
        result = router.route(
            lambda: b"shadow",
            lambda: b"packed",
            deadline_ms=100,
            budget=FallbackBudget(1, 1.0),
            preferred_name="shadow",
            fallback_name="packed",
        )
        assert result.value == b"shadow"
        assert result.representation == "shadow"
        assert not result.fallback_used
    finally:
        router.close()


def test_deadline_router_uses_budgeted_fallback_on_timeout() -> None:
    release = threading.Event()
    router = DeadlineCodecRouter()

    def slow_preferred() -> bytes:
        release.wait(timeout=1)
        return b"shadow"

    budget = FallbackBudget(1, 1.0)
    try:
        result = router.route(
            slow_preferred,
            lambda: b"packed",
            deadline_ms=1,
            budget=budget,
            fallback_quality_cost=0.5,
            preferred_name="shadow",
            fallback_name="packed",
        )
        assert result.value == b"packed"
        assert result.timed_out and result.fallback_used
        assert budget.used_fallbacks == 1
        assert budget.used_quality_cost == 0.5
        assert router.timeouts == 1
        assert router.fallbacks == 1
    finally:
        release.set()
        router.close()


def test_deadline_router_waits_when_fallback_budget_is_exhausted() -> None:
    router = DeadlineCodecRouter()
    fallback_called = {"value": False}
    release = threading.Event()

    def preferred() -> str:
        release.wait(timeout=1)
        return "exact"

    def fallback() -> str:
        fallback_called["value"] = True
        return "approx"

    try:
        timer = threading.Timer(0.05, release.set)
        timer.start()
        result = router.route(
            preferred,
            fallback,
            deadline_ms=1,
            budget=FallbackBudget(0, 0.0),
        )
        assert result.value == "exact"
        assert result.timed_out
        assert not result.fallback_used
        assert not fallback_called["value"]
    finally:
        release.set()
        router.close()


def test_fallback_budget_enforces_quality_cost() -> None:
    budget = FallbackBudget(max_fallbacks=3, max_quality_cost=0.5)
    assert budget.consume(0.25)
    assert budget.consume(0.25)
    assert not budget.consume(0.01)
    assert budget.used_fallbacks == 2
