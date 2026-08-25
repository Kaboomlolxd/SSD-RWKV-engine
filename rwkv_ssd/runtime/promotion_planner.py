"""Request-lifetime and amortization-aware dense layer promotion planning.

This module is deliberately independent from ``AdaptiveResidencyController``.
That controller reacts to observed cache-format cost.  This planner answers a
different question before or during a request: will the expected remaining
decode tokens repay the one-time cost of promoting a particular layer?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PromotionCandidate:
    layer_id: int
    resident_bytes: int
    promotion_ms: float
    staging_ms_saved_per_token: float
    last_used_token: int = 0

    def future_savings_ms(self, expected_remaining_tokens: int) -> float:
        return max(0, int(expected_remaining_tokens)) * max(
            0.0, float(self.staging_ms_saved_per_token)
        )

    def net_benefit_ms(self, expected_remaining_tokens: int) -> float:
        return self.future_savings_ms(expected_remaining_tokens) - max(
            0.0, float(self.promotion_ms)
        )

    def benefit_per_byte(self, expected_remaining_tokens: int) -> float:
        if self.resident_bytes <= 0:
            return 0.0
        return self.net_benefit_ms(expected_remaining_tokens) / self.resident_bytes


@dataclass(frozen=True)
class PromotionPlan:
    policy: str
    expected_remaining_tokens: int
    ram_cap_bytes: int
    selected: tuple[PromotionCandidate, ...]
    resident_bytes: int
    promotion_cost_ms: float
    future_savings_ms: float

    @property
    def net_benefit_ms(self) -> float:
        return self.future_savings_ms - self.promotion_cost_ms


def plan_promotions(
    candidates: Iterable[PromotionCandidate],
    *,
    expected_remaining_tokens: int,
    ram_cap_bytes: int,
    policy: str = "benefit_per_byte",
) -> PromotionPlan:
    """Choose profitable promotions within one hard resident-byte budget.

    ``benefit_per_byte`` is the proposed policy. ``highest_stall`` and ``lru``
    exist as explicit A/B baselines; all three reject promotions whose one-time
    cost is not repaid within the expected remaining request lifetime.
    """
    if ram_cap_bytes < 0:
        raise ValueError("ram_cap_bytes must be non-negative")
    remaining = max(0, int(expected_remaining_tokens))
    key = str(policy).strip().lower()
    if key not in {"benefit_per_byte", "highest_stall", "lru"}:
        raise ValueError(f"unsupported promotion policy: {policy!r}")

    profitable = [
        item
        for item in candidates
        if item.resident_bytes > 0 and item.net_benefit_ms(remaining) > 0
    ]
    if key == "benefit_per_byte":
        profitable.sort(
            key=lambda item: (
                item.benefit_per_byte(remaining),
                item.net_benefit_ms(remaining),
                -item.layer_id,
            ),
            reverse=True,
        )
    elif key == "highest_stall":
        profitable.sort(
            key=lambda item: (
                item.staging_ms_saved_per_token,
                item.net_benefit_ms(remaining),
                -item.layer_id,
            ),
            reverse=True,
        )
    else:
        profitable.sort(key=lambda item: (item.last_used_token, item.layer_id))

    selected: list[PromotionCandidate] = []
    used = 0
    for item in profitable:
        if used + item.resident_bytes > ram_cap_bytes:
            continue
        selected.append(item)
        used += item.resident_bytes
    return PromotionPlan(
        policy=key,
        expected_remaining_tokens=remaining,
        ram_cap_bytes=int(ram_cap_bytes),
        selected=tuple(selected),
        resident_bytes=used,
        promotion_cost_ms=sum(max(0.0, item.promotion_ms) for item in selected),
        future_savings_ms=sum(item.future_savings_ms(remaining) for item in selected),
    )


__all__ = ["PromotionCandidate", "PromotionPlan", "plan_promotions"]
