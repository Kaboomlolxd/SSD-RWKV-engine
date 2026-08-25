"""Deterministic request-boundary tuning across several runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rwkv_ssd.runtime.config import EngineConfig


@dataclass(frozen=True)
class TuningContext:
    prompt_tokens: int
    expected_decode_tokens: int
    batch_size: int = 1
    cmix_active_fraction: float = 1.0
    page_cache_hot: bool = False
    available_ram_bytes: int = 0

    def bucket(self) -> tuple[int, int, int, int, bool]:
        return (
            0 if self.prompt_tokens < 64 else 1 if self.prompt_tokens < 512 else 2,
            0 if self.expected_decode_tokens < 16 else 1 if self.expected_decode_tokens < 128 else 2,
            0 if self.batch_size <= 1 else 1 if self.batch_size <= 4 else 2,
            0 if self.cmix_active_fraction < 0.25 else 1 if self.cmix_active_fraction < 0.75 else 2,
            bool(self.page_cache_hot),
        )


@dataclass(frozen=True)
class KnobProfile:
    name: str
    cache_format: str | None = None
    prefetch_enabled: bool | None = None
    io_chunk_bytes: int | None = None
    io_chunk_policy: str | None = None
    packed_cache_bytes: int | None = None
    prepared_cache_bytes: int | None = None
    estimated_ram_bytes: int = 0

    def apply(self, config: EngineConfig) -> None:
        for field in (
            "cache_format",
            "prefetch_enabled",
            "io_chunk_bytes",
            "io_chunk_policy",
            "packed_cache_bytes",
            "prepared_cache_bytes",
        ):
            value = getattr(self, field)
            if value is not None:
                setattr(config, field, value)


@dataclass(frozen=True)
class ObjectiveWeights:
    throughput: float = 1.0
    p95_latency_ms: float = 0.01
    peak_ram_gb: float = 0.2
    ssd_read_gb: float = 0.1
    energy_j: float = 0.05


@dataclass(frozen=True)
class TuningObservation:
    tok_s: float
    p95_latency_ms: float = 0.0
    peak_ram_bytes: int = 0
    ssd_read_bytes: int = 0
    energy_j: float = 0.0

    def score(self, weights: ObjectiveWeights) -> float:
        return (
            weights.throughput * max(0.0, self.tok_s)
            - weights.p95_latency_ms * max(0.0, self.p95_latency_ms)
            - weights.peak_ram_gb * max(0, self.peak_ram_bytes) / 1e9
            - weights.ssd_read_gb * max(0, self.ssd_read_bytes) / 1e9
            - weights.energy_j * max(0.0, self.energy_j)
        )


@dataclass(frozen=True)
class TuningDecision:
    old_profile: str | None
    new_profile: str
    exploratory: bool
    reason: str


class ContextualAutotuner:
    """Small bounded tuner; every context bucket learns independently."""

    def __init__(
        self,
        profiles: list[KnobProfile],
        *,
        weights: ObjectiveWeights | None = None,
        max_explorations_per_context: int | None = None,
        hysteresis: float = 0.05,
    ) -> None:
        if not profiles or len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError("profiles must be non-empty and uniquely named")
        if hysteresis < 0:
            raise ValueError("hysteresis must be non-negative")
        self.profiles = {profile.name: profile for profile in profiles}
        self._order = [profile.name for profile in profiles]
        self.weights = weights or ObjectiveWeights()
        self.max_explorations = (
            len(profiles)
            if max_explorations_per_context is None
            else max(0, int(max_explorations_per_context))
        )
        self.hysteresis = float(hysteresis)
        self._scores: dict[tuple, dict[str, list[float]]] = {}
        self._current: dict[tuple, str] = {}
        self._explorations: dict[tuple, int] = {}

    def _eligible(self, context: TuningContext) -> list[str]:
        return [
            name
            for name in self._order
            if context.available_ram_bytes <= 0
            or self.profiles[name].estimated_ram_bytes <= context.available_ram_bytes
        ]

    def choose(self, context: TuningContext) -> TuningDecision:
        bucket = context.bucket()
        eligible = self._eligible(context)
        if not eligible:
            raise RuntimeError("no tuning profile fits the available RAM budget")
        scores = self._scores.setdefault(bucket, {})
        explorations = self._explorations.get(bucket, 0)
        untried = [name for name in eligible if not scores.get(name)]
        old = self._current.get(bucket)
        if untried and explorations < self.max_explorations:
            selected = untried[0]
            self._current[bucket] = selected
            self._explorations[bucket] = explorations + 1
            return TuningDecision(old, selected, True, "bounded exploration")
        means = {
            name: sum(scores[name]) / len(scores[name])
            for name in eligible
            if scores.get(name)
        }
        if not means:
            selected = eligible[0]
            self._current[bucket] = selected
            return TuningDecision(old, selected, False, "no observations; baseline")
        best = max(means, key=means.get)
        if old in means:
            improvement = means[best] - means[old]
            if improvement <= abs(means[old]) * self.hysteresis:
                best = old
        self._current[bucket] = best
        return TuningDecision(old, best, False, "best Pareto score after hysteresis")

    def record(
        self,
        context: TuningContext,
        profile_name: str,
        observation: TuningObservation,
    ) -> float:
        if profile_name not in self.profiles:
            raise KeyError(profile_name)
        score = observation.score(self.weights)
        bucket_scores = self._scores.setdefault(context.bucket(), {})
        bucket_scores.setdefault(profile_name, []).append(score)
        return score

    def apply(self, config: EngineConfig, decision: TuningDecision) -> None:
        self.profiles[decision.new_profile].apply(config)

    def snapshot(self) -> dict[str, object]:
        return {
            "contexts": len(self._scores),
            "profiles": list(self._order),
            "scores": {
                repr(bucket): {
                    name: list(values) for name, values in profile_scores.items()
                }
                for bucket, profile_scores in self._scores.items()
            },
        }

    def save(self, path: str | Path, *, machine_fingerprint: str) -> Path:
        """Atomically persist observations for exactly one machine identity."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "machine_fingerprint": str(machine_fingerprint),
            "profiles": list(self._order),
            "scores": [
                {"bucket": list(bucket), "profiles": profile_scores}
                for bucket, profile_scores in self._scores.items()
            ],
            "current": [
                {"bucket": list(bucket), "profile": profile}
                for bucket, profile in self._current.items()
            ],
            "explorations": [
                {"bucket": list(bucket), "count": count}
                for bucket, count in self._explorations.items()
            ],
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, path: str | Path, *, machine_fingerprint: str) -> bool:
        target = Path(path)
        if not target.is_file():
            return False
        raw = json.loads(target.read_text(encoding="utf-8"))
        if int(raw.get("version", 0)) != 1:
            raise ValueError("unsupported contextual autotune profile version")
        if raw.get("machine_fingerprint") != str(machine_fingerprint):
            return False
        if raw.get("profiles") != self._order:
            return False
        def bucket(item: dict) -> tuple:
            values = list(item["bucket"])
            if values:
                values[-1] = bool(values[-1])
            return tuple(values)
        self._scores = {
            bucket(item): {
                str(name): [float(value) for value in values]
                for name, values in item["profiles"].items()
            }
            for item in raw.get("scores", [])
        }
        self._current = {
            bucket(item): str(item["profile"]) for item in raw.get("current", [])
        }
        self._explorations = {
            bucket(item): int(item["count"])
            for item in raw.get("explorations", [])
        }
        return True


def machine_fingerprint(*, backend: str = "", pack_identity: str = "") -> str:
    """Stable local-profile identity; invalidates on runtime/backend/pack change."""
    import torch

    payload = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "backend": backend,
        "pack_identity": pack_identity,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ContextualAutotuner",
    "KnobProfile",
    "ObjectiveWeights",
    "TuningContext",
    "TuningDecision",
    "TuningObservation",
    "machine_fingerprint",
]
