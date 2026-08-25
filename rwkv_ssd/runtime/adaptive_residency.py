"""Windowed, hysteretic residency retiering for CPU streaming runs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


FORMATS = ("none", "packed", "prepared", "dense")


@dataclass(frozen=True)
class ResidencyDecision:
    old_format: str
    new_format: str
    reason: str
    observed_costs: dict[str, float]
    change_number: int


class AdaptiveResidencyController:
    """Choose among observed cache formats at request/token boundaries.

    The controller never invents a cost for an unobserved format.  This keeps
    adaptation deterministic and prevents a format switch based on a model
    that was never actually measured.  Callers apply a returned decision only
    after the current request has drained.
    """

    def __init__(
        self,
        *,
        initial_format: str,
        window: int = 8,
        min_dwell_tokens: int = 32,
        hysteresis: float = 0.15,
        max_changes: int = 2,
    ) -> None:
        initial = str(initial_format).strip().lower()
        if initial not in FORMATS:
            raise ValueError(f"unsupported residency format: {initial_format!r}")
        if window <= 0 or min_dwell_tokens < 0 or max_changes < 0:
            raise ValueError("adaptive residency limits must be non-negative/positive")
        if not 0.0 <= hysteresis < 1.0:
            raise ValueError("adaptive residency hysteresis must be in [0, 1)")
        self.current_format = initial
        self.window = int(window)
        self.min_dwell_tokens = int(min_dwell_tokens)
        self.hysteresis = float(hysteresis)
        self.max_changes = int(max_changes)
        self._costs: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._components: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._dwell_tokens = 0
        self._changes = 0

    @property
    def changes(self) -> int:
        return self._changes

    @property
    def dwell_tokens(self) -> int:
        return self._dwell_tokens

    def observe(
        self,
        cache_format: str,
        *,
        read_ms: float,
        staging_ms: float = 0.0,
        compute_ms: float = 0.0,
        tokens: int = 1,
    ) -> None:
        fmt = str(cache_format).strip().lower()
        if fmt not in FORMATS:
            raise ValueError(f"unsupported residency format: {cache_format!r}")
        io_cost = max(0.0, float(read_ms)) + max(0.0, float(staging_ms))
        compute_cost = max(0.0, float(compute_ms))
        cost = io_cost + compute_cost
        self._costs[fmt].append(cost)
        self._components[fmt].append((io_cost, compute_cost))
        self._dwell_tokens += max(0, int(tokens))

    def maybe_retier(
        self,
        *,
        explicit_cache_format: str = "auto",
        current_format: str | None = None,
    ) -> ResidencyDecision | None:
        """Return a sustained improvement decision, if one is justified."""
        explicit = str(explicit_cache_format or "auto").strip().lower()
        if explicit != "auto":
            return None
        current = (current_format or self.current_format).strip().lower()
        if current not in FORMATS:
            raise ValueError(f"unsupported current residency format: {current!r}")
        self.current_format = current
        if self._dwell_tokens < self.min_dwell_tokens:
            return None
        if self._changes >= self.max_changes:
            return None
        current_samples = self._costs.get(current)
        if not current_samples:
            return None
        observed = {
            fmt: sum(samples) / len(samples)
            for fmt, samples in self._costs.items()
            if samples
        }
        if len(observed) < 2:
            # During a normal engine run only the active format has direct
            # samples.  Use the measured I/O-vs-compute balance to probe one
            # adjacent tier after the dwell period.  Later windows then have
            # direct old/new measurements and use the comparison below.
            components = self._components.get(current)
            if not components:
                return None
            io_cost = sum(item[0] for item in components) / len(components)
            compute_cost = sum(item[1] for item in components) / len(components)
            index = FORMATS.index(current)
            new_format = current
            if io_cost > compute_cost * (1.0 + self.hysteresis) and index < len(FORMATS) - 1:
                new_format = FORMATS[index + 1]
            elif compute_cost > 0 and io_cost < compute_cost * (1.0 - self.hysteresis) and index > 0:
                new_format = FORMATS[index - 1]
            if new_format == current:
                return None
            old = current
            self.current_format = new_format
            self._changes += 1
            self._dwell_tokens = 0
            return ResidencyDecision(
                old_format=old,
                new_format=new_format,
                reason=(
                    f"measured io/compute balance {io_cost:.3f}/{compute_cost:.3f}ms "
                    f"crossed {self.hysteresis:.1%} hysteresis"
                ),
                observed_costs={current: observed[current]},
                change_number=self._changes,
            )
        best = min(observed, key=observed.get)
        if best == current:
            return None
        current_cost = observed[current]
        best_cost = observed[best]
        required = current_cost * (1.0 - self.hysteresis)
        if best_cost >= required:
            return None
        old = current
        self.current_format = best
        self._changes += 1
        self._dwell_tokens = 0
        return ResidencyDecision(
            old_format=old,
            new_format=best,
            reason=(
                f"windowed cost {best_cost:.3f}ms < {required:.3f}ms "
                f"({self.hysteresis:.1%} hysteresis)"
            ),
            observed_costs=observed,
            change_number=self._changes,
        )


__all__ = ["AdaptiveResidencyController", "FORMATS", "ResidencyDecision"]
