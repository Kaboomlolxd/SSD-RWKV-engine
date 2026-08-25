"""Shared bench duration profiles — lighter defaults, ``--quick``, ``--full``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BenchProfile:
    name: str
    max_tokens: int
    samples: int
    warmup: int
    frontier_ids: tuple[str, ...] | None = None  # None = all frontier scenarios
    skip_raw_gbs: bool = False
    skip_warm_disk_cache: bool = False


QUICK = BenchProfile(
    name="quick",
    max_tokens=12,
    samples=1,
    warmup=0,
    frontier_ids=("F1", "F3", "F5"),
    skip_raw_gbs=True,
    skip_warm_disk_cache=True,
)

FULL = BenchProfile(
    name="full",
    max_tokens=48,
    samples=3,
    warmup=1,
    skip_raw_gbs=False,
    skip_warm_disk_cache=False,
)

DEFAULT = BenchProfile(
    name="default",
    max_tokens=24,
    samples=1,
    warmup=1,
)

SYNTH_QUICK = BenchProfile(name="synth_quick", max_tokens=8, samples=1, warmup=0)
SYNTH_FULL = BenchProfile(name="synth_full", max_tokens=32, samples=2, warmup=0)


def resolve_profile(*, quick: bool, full: bool, heavy: bool) -> BenchProfile:
    if full:
        return FULL
    if quick or heavy:
        return QUICK
    return DEFAULT


def frontier_filter(profile: BenchProfile, scenario_id: str) -> bool:
    if profile.frontier_ids is None:
        return True
    return scenario_id in profile.frontier_ids
