"""Layer-aware prefetch and mmap advise helpers."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.io_mmap_advise import layer_file_span
from rwkv_ssd.runtime.layer_io import merge_layer_entries
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.prefetch import LayerAwarePlanner, make_prefetch_planner
from rwkv_ssd.runtime.metrics import LayerTiming
from tests.helpers import greedy_token_ids


def test_layer_file_span_covers_entries() -> None:
    entries = [
        TensorEntry("a", 0, "float32", [8], 1000, 100, 4096, "streamed"),
        TensorEntry("b", 0, "float32", [8], 1000, 200, 4096, "streamed"),
    ]
    assert layer_file_span(entries) == (1000, 200)


def test_merge_layer_entries() -> None:
    by_layer = {
        0: [TensorEntry("a", 0, "float32", [1], 0, 4, 4096, "streamed")],
        1: [TensorEntry("b", 1, "float32", [1], 0, 4, 4096, "streamed")],
    }
    merged = merge_layer_entries(by_layer, [0, 1])
    assert len(merged) == 2
    assert merged[0].name == "a"


def test_layer_aware_planner_depth() -> None:
    planner = LayerAwarePlanner()
    layers = [0, 1, 2, 3]
    plan = planner.plan(layers, 0, [])
    assert plan.layer_ids == (1,)
    io_bound = [
        LayerTiming(0, read_ms=10.0, compute_ms=1.0),
        LayerTiming(1, read_ms=12.0, compute_ms=1.0),
    ]
    plan2 = planner.plan(layers, 0, io_bound)
    assert plan2.layer_ids == (1, 2, 3)


def test_layer_aware_greedy_parity(synthetic_pack: Path) -> None:
    base = greedy_token_ids(
        synthetic_pack, "parity", mode="streaming", max_tokens=8
    )
    aware = greedy_token_ids(
        synthetic_pack,
        "parity",
        mode="streaming",
        max_tokens=8,
        prefetch_policy="layer_aware",
    )
    assert base == aware


def test_layer_size_auto_chunk(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=8,
        io_chunk_policy="layer_size",
        io_chunk_bytes=0,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("chunk policy")
        assert sum(L.chunk_reads for L in engine.metrics.layers) >= 1
    finally:
        engine.close()


def test_make_prefetch_planner_layer_aware() -> None:
    assert isinstance(make_prefetch_planner("layer_aware"), LayerAwarePlanner)
