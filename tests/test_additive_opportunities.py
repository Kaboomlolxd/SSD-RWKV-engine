from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bench.bench_additive_opportunities import run_planner_ab
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_overlay import PackChunkStore
from rwkv_ssd.runtime.promotion_planner import PromotionCandidate, plan_promotions
from rwkv_ssd.runtime.snapshot import (
    SnapshotMeta,
    pack_identity,
    verify_snapshot_compatibility,
)
from rwkv_ssd.runtime.storage_placement import (
    DriveTier,
    LayerPayload,
    place_layers,
    total_expected_stall_ms,
)
from rwkv_ssd.tools.plan_storage_placement import plan_pack_placement
from rwkv_ssd.tools.plan_storage_placement import resolve_drive_specs
from rwkv_ssd.tools.repack_storage_placement import repack_storage_placement


def test_short_session_does_not_pay_promotion_cost() -> None:
    candidate = PromotionCandidate(0, 100, promotion_ms=10, staging_ms_saved_per_token=2)
    short = plan_promotions(
        [candidate], expected_remaining_tokens=4, ram_cap_bytes=100
    )
    long = plan_promotions(
        [candidate], expected_remaining_tokens=8, ram_cap_bytes=100
    )
    assert short.selected == ()
    assert [item.layer_id for item in long.selected] == [0]


def test_benefit_per_byte_beats_highest_stall_under_cap() -> None:
    candidates = [
        PromotionCandidate(0, 100, 1, 10),
        PromotionCandidate(1, 40, 1, 6),
        PromotionCandidate(2, 40, 1, 6),
    ]
    highest = plan_promotions(
        candidates,
        expected_remaining_tokens=10,
        ram_cap_bytes=100,
        policy="highest_stall",
    )
    benefit = plan_promotions(
        candidates,
        expected_remaining_tokens=10,
        ram_cap_bytes=100,
        policy="benefit_per_byte",
    )
    assert benefit.resident_bytes <= 100
    assert benefit.net_benefit_ms > highest.net_benefit_ms
    assert [item.layer_id for item in benefit.selected] == [1, 2]


def test_heterogeneous_optimizer_improves_round_robin() -> None:
    layers = [
        LayerPayload(0, 100_000_000, 1),
        LayerPayload(1, 800_000_000, 4),
        LayerPayload(2, 200_000_000, 1),
        LayerPayload(3, 700_000_000, 2),
    ]
    drives = [
        DriveTier("fast", 6000, capacity_bytes=1_000_000_000),
        DriveTier("slow", 700, capacity_bytes=1_000_000_000),
    ]
    rr = place_layers(layers, drives, policy="round_robin")
    optimized = place_layers(layers, drives, policy="optimized")
    assert total_expected_stall_ms(optimized) < total_expected_stall_ms(rr)
    assert sum(item.byte_count for item in optimized if item.drive == "fast") <= 1_000_000_000


def test_pack_placement_tool_uses_manifest_layers(synthetic_pack: Path) -> None:
    result = plan_pack_placement(
        synthetic_pack,
        [
            {"name": "fast", "bandwidth_mbps": 5000},
            {"name": "slow", "bandwidth_mbps": 500},
        ],
    )
    assert result["placements"]
    assert result["optimized_stall_ms"] <= result["round_robin_stall_ms"]
    assert result["physical_hardware_validated"] is False


def test_drive_specs_can_be_derived_from_storage_diagnostics(tmp_path: Path) -> None:
    diagnostic = tmp_path / "fast.json"
    diagnostic.write_text(
        '{"sequential_tensor_read":{"median_mbps":1234},'
        '"per_layer":[{"read_ms":0.4},{"read_ms":0.2}]}',
        encoding="utf-8",
    )
    resolved = resolve_drive_specs(
        [{"name": "fast", "diagnostic_json": "fast.json"}], base_dir=tmp_path
    )
    assert resolved[0]["bandwidth_mbps"] == 1234
    assert resolved[0]["latency_ms"] == 0.2


def test_reviewed_storage_plan_repacks_and_verifies_every_tensor(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    plan = plan_pack_placement(
        synthetic_pack,
        [
            {"name": "fast", "bandwidth_mbps": 5000},
            {"name": "slow", "bandwidth_mbps": 500},
        ],
    )
    output = tmp_path / "placed"
    result = repack_storage_placement(synthetic_pack, output, plan)
    source = Manifest.load(synthetic_pack)
    placed = Manifest.load(output)
    assert placed.is_sharded()
    assert result["tensors_verified"] == len(source.tensors)
    assert len(result["shard_files"]) >= 1
    assert placed.meta["shard_strategy"] == "heterogeneous_cost_optimized"


def test_cross_model_store_reports_unique_bytes_and_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tuned = tmp_path / "tuned"
    base.mkdir()
    tuned.mkdir()
    (base / "weights.bin").write_bytes(b"A" * 16 + b"B" * 16)
    (tuned / "weights.bin").write_bytes(b"A" * 16 + b"C" * 16)
    store = PackChunkStore(tmp_path / "store", chunk_bytes=16)
    store.ingest_directory(base, "base")
    second = store.ingest_directory(tuned, "tuned")
    stats = store.store_stats()
    assert second["unique_bytes_added"] == 16
    assert second["metadata_bytes"] > 0
    assert stats["logical_bytes"] == 64
    assert stats["unique_bytes"] == 48
    assert stats["dedup_ratio"] == pytest.approx(64 / 48)


def test_snapshot_pack_identity_rejects_wrong_local_pack(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    meta = SnapshotMeta(
        backend="synthetic",
        mode="streaming",
        model_family="synthetic_rwkv",
        extras={"pack_identity": pack_identity(synthetic_pack)},
    )
    verify_snapshot_compatibility(meta, synthetic_pack, backend="synthetic")
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text('{"different":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_snapshot_compatibility(meta, other, backend="synthetic")


def test_engine_snapshot_carries_sampling_and_identity(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.snapshot import load_snapshot

    engine = InferenceEngine(
        EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="streaming",
            max_tokens=2,
            greedy=False,
            temperature=0.7,
        )
    )
    engine.load()
    try:
        torch.manual_seed(1)
        engine.generate("handoff")
        path = engine.save_snapshot(tmp_path / "handoff.bin", prompt="handoff")
        _state, meta = load_snapshot(path)
        assert meta.extras["pack_identity"] == pack_identity(synthetic_pack)
        assert meta.extras["sampling"] == {"greedy": False, "temperature": 0.7}
    finally:
        engine.close()


def test_planner_ab_schema_and_claim_scope() -> None:
    result = run_planner_ab()
    assert result["promotion"]["short"]["benefit_per_byte"]["resident_bytes"] <= 70
    assert result["placement"]["modeled_speedup"] > 1.0
    assert result["placement"]["physical_hardware_validated"] is False
