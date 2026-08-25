"""Explainable bottleneck diagnosis and conservative request admission."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.metrics import MetricsCollector


def diagnose_metrics(metrics: "MetricsCollector") -> dict[str, object]:
    read = sum(row.read_ms for row in metrics.layers)
    staging = sum(row.staging_ms + row.h2d_ms for row in metrics.layers)
    compute = sum(row.compute_ms for row in metrics.layers)
    components = {"read": read, "staging": staging, "compute": compute}
    bottleneck = max(components, key=components.get) if any(components.values()) else "unmeasured"
    recommendations = {
        "read": "increase useful residency/prefetch or improve storage placement",
        "staging": "promote profitable layers or use prepared/native slots",
        "compute": "use a native backend or larger weight-stationary batch",
        "unmeasured": "capture per-layer timings before changing configuration",
    }
    return {
        "bottleneck": bottleneck,
        "components_ms": {key: round(value, 3) for key, value in components.items()},
        "recommendation": recommendations[bottleneck],
        "evidence": "measured_request" if metrics.layers else "none",
    }


def layer_decision_trace(metrics: "MetricsCollector") -> list[dict[str, object]]:
    events = []
    for row in metrics.layers:
        if row.layer_cache_hits > 0:
            decision = "cache_hit"
            reason = f"{row.layer_cache_hits} tensors reused"
        elif row.read_ms > 0:
            decision = "streamed"
            reason = f"read_ms={row.read_ms:.3f}"
        else:
            decision = "resident_or_prepared"
            reason = "no measured storage read"
        events.append({"layer_id": row.layer_id, "decision": decision, "reason": reason})
    return events


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    predicted_stream_bytes: int
    limit_bytes: int
    reason: str


def predict_stream_bytes(engine: "InferenceEngine", max_tokens: int) -> int:
    manifest = engine.manifest
    if manifest is None or engine.config.mode == "resident":
        return 0
    streamed = sum(
        entry.length for entry in manifest.tensors if entry.residency == "streamed"
    )
    if engine.config.cache_format in {"dense", "prepared"} or engine.config.warm_z:
        return streamed
    return streamed * max(1, int(max_tokens))


def admit_request(
    engine: "InferenceEngine",
    max_tokens: int,
    *,
    limit_bytes: int | None = None,
) -> AdmissionDecision:
    if limit_bytes is None:
        try:
            limit_bytes = int(os.environ.get("RWKV_ADMISSION_MAX_STREAM_BYTES", "0"))
        except ValueError:
            limit_bytes = 0
    predicted = predict_stream_bytes(engine, max_tokens)
    if limit_bytes <= 0:
        return AdmissionDecision(True, predicted, 0, "admission byte limit disabled")
    admitted = predicted <= limit_bytes
    return AdmissionDecision(
        admitted,
        predicted,
        int(limit_bytes),
        "within predicted SSD byte budget" if admitted else "predicted SSD bytes exceed request budget",
    )


def append_decision_trace(
    path: str | Path,
    metrics: "MetricsCollector",
    *,
    request: dict[str, object] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": dict(request or {}),
        "diagnosis": diagnose_metrics(metrics),
        "promotion": {
            "layers": list(metrics.session_promoted_layers),
            "bytes": metrics.session_promotion_bytes,
            "estimated_net_ms": metrics.session_promotion_estimated_net_ms,
        },
        "layers": layer_decision_trace(metrics),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


__all__ = [
    "AdmissionDecision",
    "admit_request",
    "append_decision_trace",
    "diagnose_metrics",
    "layer_decision_trace",
    "predict_stream_bytes",
]
