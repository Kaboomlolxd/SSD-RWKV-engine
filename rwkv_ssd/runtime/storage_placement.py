"""Cost-based placement of layer payloads across heterogeneous drives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DriveTier:
    name: str
    bandwidth_mbps: float
    latency_ms: float = 0.0
    capacity_bytes: int = 0

    def read_ms(self, byte_count: int) -> float:
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"drive {self.name!r} bandwidth must be positive")
        return max(0.0, self.latency_ms) + (
            max(0, int(byte_count)) / (self.bandwidth_mbps * 1_000_000.0) * 1000.0
        )


@dataclass(frozen=True)
class LayerPayload:
    layer_id: int
    byte_count: int
    reads_per_forward: float = 1.0


@dataclass(frozen=True)
class LayerPlacement:
    layer_id: int
    drive: str
    byte_count: int
    expected_stall_ms: float


def _fits(drive: DriveTier, used: int, byte_count: int) -> bool:
    return drive.capacity_bytes <= 0 or used + byte_count <= drive.capacity_bytes


def place_layers(
    layers: Iterable[LayerPayload],
    drives: Iterable[DriveTier],
    *,
    policy: str = "optimized",
) -> tuple[LayerPlacement, ...]:
    """Return a deterministic placement under per-drive capacity limits.

    The optimized policy puts the payload with the largest fast-vs-slow stall
    penalty first, then chooses the feasible drive with the lowest measured
    read cost. Round-robin is retained as the A/B baseline.
    """
    layer_list = list(layers)
    drive_list = list(drives)
    if not drive_list:
        raise ValueError("at least one drive tier is required")
    if any(layer.byte_count < 0 or layer.reads_per_forward < 0 for layer in layer_list):
        raise ValueError("layer bytes and read frequency must be non-negative")
    key = str(policy).strip().lower()
    if key not in {"optimized", "round_robin"}:
        raise ValueError(f"unsupported placement policy: {policy!r}")

    used = {drive.name: 0 for drive in drive_list}
    if len(used) != len(drive_list):
        raise ValueError("drive names must be unique")
    ordered = list(layer_list)
    if key == "optimized":
        def penalty(layer: LayerPayload) -> tuple[float, int]:
            costs = sorted(
                drive.read_ms(layer.byte_count) * layer.reads_per_forward
                for drive in drive_list
            )
            spread = costs[-1] - costs[0] if len(costs) > 1 else costs[0]
            return spread, layer.byte_count
        ordered.sort(key=penalty, reverse=True)

    placements: list[LayerPlacement] = []
    rr = 0
    for layer in ordered:
        feasible = [
            drive
            for drive in drive_list
            if _fits(drive, used[drive.name], layer.byte_count)
        ]
        if not feasible:
            raise ValueError(f"drive capacity exhausted at layer {layer.layer_id}")
        if key == "round_robin":
            drive = next(
                candidate
                for offset in range(len(drive_list))
                if (candidate := drive_list[(rr + offset) % len(drive_list)]) in feasible
            )
            rr = (drive_list.index(drive) + 1) % len(drive_list)
        else:
            drive = min(
                feasible,
                key=lambda item: (
                    item.read_ms(layer.byte_count) * layer.reads_per_forward,
                    used[item.name],
                    item.name,
                ),
            )
        used[drive.name] += layer.byte_count
        placements.append(
            LayerPlacement(
                layer_id=layer.layer_id,
                drive=drive.name,
                byte_count=layer.byte_count,
                expected_stall_ms=(
                    drive.read_ms(layer.byte_count) * layer.reads_per_forward
                ),
            )
        )
    return tuple(sorted(placements, key=lambda item: item.layer_id))


def total_expected_stall_ms(placements: Iterable[LayerPlacement]) -> float:
    return sum(item.expected_stall_ms for item in placements)


__all__ = [
    "DriveTier",
    "LayerPayload",
    "LayerPlacement",
    "place_layers",
    "total_expected_stall_ms",
]
