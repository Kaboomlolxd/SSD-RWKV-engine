"""Prefetch planning — layer sequence and adaptive (gate-style) lookahead."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rwkv_ssd.runtime.metrics import LayerTiming


@dataclass(frozen=True)
class PrefetchPlan:
    """Layer indices to prefetch ahead of the current layer (in order)."""

    layer_ids: tuple[int, ...]


class PrefetchPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        layer_ids: list[int],
        current_index: int,
        recent_timings: list[LayerTiming],
    ) -> PrefetchPlan:
        ...


class NextLayerPlanner(PrefetchPlanner):
    """Default: prefetch layer N+1 only (P0 #3)."""

    def plan(
        self,
        layer_ids: list[int],
        current_index: int,
        recent_timings: list[LayerTiming],
    ) -> PrefetchPlan:
        del recent_timings
        nxt = current_index + 1
        if nxt < len(layer_ids):
            return PrefetchPlan(layer_ids=(layer_ids[nxt],))
        return PrefetchPlan(layer_ids=())


class AdaptiveGatePlanner(PrefetchPlanner):
    """
    Gate-style prefetch (P2.a): when the previous layer was I/O-bound, also
    prefetch layer N+2 so the SSD queue stays full.
    """

    def __init__(self, io_bound_ratio: float = 1.0) -> None:
        self._ratio = io_bound_ratio

    def plan(
        self,
        layer_ids: list[int],
        current_index: int,
        recent_timings: list[LayerTiming],
    ) -> PrefetchPlan:
        targets: list[int] = []
        if current_index + 1 < len(layer_ids):
            targets.append(layer_ids[current_index + 1])
        if recent_timings:
            last = recent_timings[-1]
            io_ms = last.read_ms + last.prefetch_wait_ms
            compute_ms = max(last.compute_ms, 1e-6)
            if io_ms >= self._ratio * compute_ms and current_index + 2 < len(layer_ids):
                ahead = layer_ids[current_index + 2]
                if ahead not in targets:
                    targets.append(ahead)
        return PrefetchPlan(layer_ids=tuple(targets))


class LayerAwarePlanner(PrefetchPlanner):
    """
    Layer-aware lookahead (P2.a): always N+1; add N+2/N+3 when recent layers are I/O-bound.
    """

    def __init__(self, io_bound_ratio: float = 1.0, max_depth: int = 3) -> None:
        self._ratio = io_bound_ratio
        self._max_depth = max(1, max_depth)

    def plan(
        self,
        layer_ids: list[int],
        current_index: int,
        recent_timings: list[LayerTiming],
    ) -> PrefetchPlan:
        targets: list[int] = []
        for depth in range(1, self._max_depth + 1):
            idx = current_index + depth
            if idx >= len(layer_ids):
                break
            if depth == 1:
                targets.append(layer_ids[idx])
                continue
            if not recent_timings:
                break
            io_ms = sum(
                t.read_ms + t.prefetch_wait_ms for t in recent_timings[-2:]
            ) / min(2, len(recent_timings))
            compute_ms = max(
                sum(t.compute_ms for t in recent_timings[-2:])
                / min(2, len(recent_timings)),
                1e-6,
            )
            threshold = self._ratio * (depth - 1)
            if io_ms >= threshold * compute_ms:
                targets.append(layer_ids[idx])
            else:
                break
        return PrefetchPlan(layer_ids=tuple(targets))


def make_prefetch_planner(policy: str) -> PrefetchPlanner:
    key = policy.strip().lower()
    if key in ("layer", "next", "default"):
        return NextLayerPlanner()
    if key in ("gate", "adaptive", "adaptive_gate"):
        return AdaptiveGatePlanner()
    if key in ("layer_aware", "lookahead", "deep"):
        raw_depth = os.environ.get("RWKV_PREFETCH_MAX_DEPTH", "3").strip()
        try:
            max_depth = max(1, int(raw_depth))
        except ValueError:
            max_depth = 3
        return LayerAwarePlanner(max_depth=max_depth)
    raise ValueError(
        f"unknown prefetch policy {policy!r} (use: layer, gate, layer_aware)"
    )
