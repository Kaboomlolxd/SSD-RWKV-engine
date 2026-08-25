from __future__ import annotations

import json

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.metrics import LayerTiming, MetricsCollector
from rwkv_ssd.runtime.runtime_intelligence import (
    admit_request,
    append_decision_trace,
    diagnose_metrics,
    layer_decision_trace,
)


def test_bottleneck_diagnosis_uses_measured_largest_component() -> None:
    metrics = MetricsCollector(
        layers=[
            LayerTiming(0, read_ms=5, staging_ms=2, compute_ms=1),
            LayerTiming(1, read_ms=4, staging_ms=1, compute_ms=2),
        ]
    )
    report = diagnose_metrics(metrics)
    assert report["bottleneck"] == "read"
    assert report["components_ms"]["read"] == 9
    assert "storage" in report["recommendation"]


def test_layer_trace_explains_stream_cache_and_resident_paths() -> None:
    metrics = MetricsCollector(
        layers=[
            LayerTiming(0, read_ms=1),
            LayerTiming(1, layer_cache_hits=3),
            LayerTiming(2),
        ]
    )
    assert [event["decision"] for event in layer_decision_trace(metrics)] == [
        "streamed", "cache_hit", "resident_or_prepared"
    ]


def test_admission_uses_one_storage_budget(synthetic_pack) -> None:
    engine = InferenceEngine(
        EngineConfig(pack_dir=synthetic_pack, backend="synthetic", mode="streaming")
    )
    engine.load()
    try:
        predicted = admit_request(engine, 4, limit_bytes=10**12)
        assert predicted.admitted
        rejected = admit_request(engine, 4, limit_bytes=1)
        assert not rejected.admitted
        assert rejected.predicted_stream_bytes > rejected.limit_bytes
    finally:
        engine.close()


def test_decision_trace_is_jsonl_and_engine_writes_it(synthetic_pack, tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="streaming",
            max_tokens=2,
            trace_path=trace,
        )
    )
    engine.load()
    try:
        engine.generate("trace me")
    finally:
        engine.close()
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["request"]["max_tokens"] == 2
    assert payload["diagnosis"]["evidence"] == "measured_request"
    assert payload["layers"]
