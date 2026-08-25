"""Sharded weight store — M-class (multi-SSD aggregate bandwidth).

The sharded pack format (see ``manifest.is_sharded()``) splits the
weights across multiple files, each potentially on a separate
physical SSD. These tests verify the routing layer returns the same
bytes regardless of which shard the tensor lives in, and that the
parallel-read path issues concurrent reads.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rwkv_ssd.runtime.io_pread import PreadWeightStore
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.weight_store_sharded import (
    ShardedWeightStore,
    open_sharded_weight_store,
)


def _build_manifest_with_shards(pack_dir: Path, n_shards: int) -> Manifest:
    """Create a fake manifest that references ``n_shards`` shard files.

    Each shard file is a real file with a small payload. The tensors
    are split round-robin across the shards so the routing layer
    has to dispatch to multiple files.
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []
    tensors: list[dict] = []
    payload = b"\x00" * 64 + b"\xff" * 64
    for shard_id in range(n_shards):
        shard_name = f"weights.shard.{shard_id}.bin"
        shard_path = pack_dir / shard_name
        shard_path.write_bytes(payload * 4, encoding=None) if False else shard_path.write_bytes(payload * 4)
        shard_files.append(shard_name)
    for i in range(n_shards * 3):
        shard_id = i % n_shards
        tensors.append(
            {
                "name": f"blocks.{i}.att.key.weight",
                "layer_id": i,
                "dtype": "bf16",
                "shape": [16],
                "offset": (i % 4) * 128,
                "length": 128,
                "alignment": 4096,
                "residency": "streamed",
                "shard_file": shard_files[shard_id],
            }
        )
    import json
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": shard_files[0],
                "weights_files": shard_files,
                "tensors": tensors,
                "meta": {"n_layer": n_shards * 3},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Manifest.load(pack_dir)


def test_sharded_store_routes_by_shard_file(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _build_manifest_with_shards(pack_dir, n_shards=3)
    assert manifest.is_sharded()
    assert len(manifest.shard_files) == 3

    store = ShardedWeightStore(manifest)
    try:
        # Tensors 0, 3, 6, ... go to shard 0; 1, 4, 7, ... to shard 1; etc.
        for entry in manifest.tensors[:3]:
            payload = store.read_bytes(entry)
            assert len(payload) == entry.length
    finally:
        store.close()


def test_manifest_v2_striped_tensor_is_gathered(tmp_path: Path) -> None:
    import json

    (tmp_path / "weights.shard.0.bin").write_bytes(b"abcd" + b"\0" * 8)
    (tmp_path / "weights.shard.1.bin").write_bytes(b"efgh" + b"\0" * 8)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "model_family": "rwkv",
                "weights_file": "weights.shard.0.bin",
                "weights_files": ["weights.shard.0.bin", "weights.shard.1.bin"],
                "tensors": [
                    {
                        "name": "blocks.0.weight",
                        "layer_id": 0,
                        "dtype": "u8",
                        "shape": [8],
                        "offset": 0,
                        "length": 8,
                        "alignment": 1,
                        "residency": "streamed",
                        "stripes": [
                            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 4, "logical_offset": 0},
                            {"shard_file": "weights.shard.1.bin", "offset": 0, "length": 4, "logical_offset": 4},
                        ],
                    }
                ],
                "meta": {"n_shards": 2, "shard_strategy": "striped_round_robin"},
            }
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(tmp_path)
    store = ShardedWeightStore(manifest, parallel_workers=2)
    try:
        assert store.read_bytes(manifest.tensors[0]) == b"abcdefgh"
        assert store.read_bytes_many([manifest.tensors[0], manifest.tensors[0]]) == [
            b"abcdefgh",
            b"abcdefgh",
        ]
    finally:
        store.close()


def test_shard_pack_striped_round_trip(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    from rwkv_ssd.runtime.weight_store import open_weight_store
    from rwkv_ssd.tools.shard_pack import shard_pack

    output = tmp_path / "striped"
    shard_pack(
        synthetic_pack,
        output,
        n_shards=2,
        strategy="stripe",
        stripe_bytes=64,
    )
    source_manifest = Manifest.load(synthetic_pack)
    striped_manifest = Manifest.load(output)
    assert striped_manifest.meta["shard_strategy"] == "striped_round_robin"
    assert any(entry.stripes for entry in striped_manifest.tensors)
    source_store = open_weight_store(source_manifest.weights_path, backend="pread")
    striped_store = ShardedWeightStore(striped_manifest, parallel_workers=2)
    try:
        expected = {
            entry.name: source_store.read_bytes(entry)
            for entry in source_manifest.tensors
        }
        for entry in striped_manifest.tensors:
            assert striped_store.read_bytes(entry) == expected[entry.name]
    finally:
        source_store.close()
        striped_store.close()


def test_sharded_store_matches_single_file_legacy(tmp_path: Path) -> None:
    """Tensors with an empty ``shard_file`` fall through to the primary
    weights path (backward-compat with legacy single-file packs)."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    weights_path = pack_dir / "weights.bin"
    weights_path.write_bytes(b"\xab" * 4096)
    import json
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": "weights.bin",
                "tensors": [
                    {
                        "name": "blocks.0.att.key.weight",
                        "layer_id": 0,
                        "dtype": "bf16",
                        "shape": [16],
                        "offset": 0,
                        "length": 128,
                        "alignment": 4096,
                        "residency": "streamed",
                    }
                ],
                "meta": {"n_layer": 1},
            }
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(pack_dir)
    assert not manifest.is_sharded()

    store = open_sharded_weight_store(manifest)
    try:
        assert isinstance(store, PreadWeightStore)
        payload = store.read_bytes(manifest.tensors[0])
        assert payload == b"\xab" * 128
    finally:
        store.close()


def test_sharded_store_parallel_reads_actually_overlap(
    tmp_path: Path,
) -> None:
    """Verify the parallel-read path issues concurrent reads.

    We instrument each shard's ``read_bytes`` to sleep for the
    payload length, then issue ``read_bytes_many`` over the shards
    and confirm the wall time is well under the sequential bound.
    """
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    n_shards = 4
    shard_files: list[str] = []
    sleep_us = 20_000  # 20 ms per read
    for shard_id in range(n_shards):
        shard_name = f"weights.shard.{shard_id}.bin"
        (pack_dir / shard_name).write_bytes(b"\x00" * 4096)
        shard_files.append(shard_name)
    import json
    tensors: list[dict] = []
    for i in range(n_shards):
        tensors.append(
            {
                "name": f"blocks.{i}.att.key.weight",
                "layer_id": i,
                "dtype": "bf16",
                "shape": [16],
                "offset": 0,
                "length": 128,
                "alignment": 4096,
                "residency": "streamed",
                "shard_file": shard_files[i],
            }
        )
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": shard_files[0],
                "weights_files": shard_files,
                "tensors": tensors,
                "meta": {"n_layer": n_shards},
            }
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(pack_dir)
    store = ShardedWeightStore(manifest, parallel_workers=n_shards)
    try:
        # Patch each shard's inner store to sleep — this simulates
        # SSD I/O latency without actually doing I/O.
        # Directly prove simultaneous worker entry. A wall-clock-only check is
        # flaky after CPU-heavy suites on Windows because ready threads can be
        # descheduled for longer than this synthetic 20 ms delay.
        entered = threading.Barrier(n_shards, timeout=2.0)
        for shard_store in store._stores.values():
            orig = shard_store.read_bytes

            def slow_read(entry, _orig=orig):
                entered.wait()
                time.sleep(sleep_us / 1_000_000)
                return _orig(entry)

            shard_store.read_bytes = slow_read  # type: ignore[method-assign]

        entries = manifest.tensors
        t0 = time.perf_counter()
        results = store.read_bytes_many(entries)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        assert len(results) == n_shards
        assert all(len(r) == 128 for r in results)
        # The barrier would time out and fail if reads were issued serially.
        # Keep only a broad hang guard; scheduler latency is not correctness.
        assert wall_ms < 2_000.0, (
            f"parallel read took {wall_ms:.1f} ms — sequential bound is "
            f"{n_shards * sleep_us / 1000:.1f} ms"
        )
    finally:
        store.close()


def test_sharded_manifest_is_sharded_detection(tmp_path: Path) -> None:
    """``Manifest.is_sharded()`` returns True only when the pack was
    built with multiple shard files (or any tensor carries a
    ``shard_file``)."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "weights.bin").write_bytes(b"\x00" * 128)
    import json
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": "weights.bin",
                "tensors": [
                    {
                        "name": "blocks.0.att.key.weight",
                        "layer_id": 0,
                        "dtype": "bf16",
                        "shape": [16],
                        "offset": 0,
                        "length": 128,
                        "alignment": 4096,
                        "residency": "streamed",
                    }
                ],
                "meta": {"n_layer": 1},
            }
        ),
        encoding="utf-8",
    )
    assert not Manifest.load(pack_dir).is_sharded()


def test_shard_pack_tool_round_trip(tmp_path: Path) -> None:
    """``rwkv_ssd.tools.shard_pack`` splits a single-file pack into
    multiple shards and updates the manifest with ``shard_file`` per
    tensor. Verifying here that the round-trip preserves all tensor
    bytes (read from the sharded store equals the source).
    """
    from rwkv_ssd.tools.shard_pack import shard_pack

    pack_dir = tmp_path / "src"
    pack_dir.mkdir(parents=True, exist_ok=True)
    # Build the source file from the tensor payloads in order so the
    # offsets in the manifest are real byte ranges into the file.
    n_layer = 3
    n_tensors_per_layer = 2
    tensor_size = 1024
    tensor_payloads: list[bytes] = []
    for i in range(n_layer):
        for j in range(n_tensors_per_layer):
            # Use a unique per-tensor byte pattern so we can verify the
            # bytes round-trip correctly.
            tensor_payloads.append(bytes([0xA0 + i * 16 + j]) * tensor_size)
    source_bytes = b"".join(tensor_payloads)
    (pack_dir / "weights.bin").write_bytes(source_bytes)
    import json
    tensors = []
    offset = 0
    for i in range(n_layer):
        for j in range(n_tensors_per_layer):
            tensors.append(
                {
                    "name": f"blocks.{i}.att.j{j}.weight",
                    "layer_id": i,
                    "dtype": "bf16",
                    "shape": [16],
                    "offset": offset,
                    "length": tensor_size,
                    "alignment": 4096,
                    "residency": "streamed",
                }
            )
            offset += tensor_size
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": "weights.bin",
                "tensors": tensors,
                "meta": {"n_layer": n_layer},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "sharded"
    stats = shard_pack(pack_dir, out_dir, n_shards=2)
    assert stats["n_shards"] == 2
    assert (out_dir / "weights.shard.0.bin").is_file()
    assert (out_dir / "weights.shard.1.bin").is_file()
    assert (out_dir / "manifest.json").is_file()

    # Verify the sharded store returns the same bytes as the source.
    src_m = Manifest.load(pack_dir)
    sh_m = Manifest.load(out_dir)
    assert sh_m.is_sharded()
    # Build a name → source bytes map from the source manifest.
    src_bytes_by_name: dict[str, bytes] = {}
    with open(src_m.weights_path, "rb") as f:
        for entry in src_m.tensors:
            f.seek(entry.offset)
            src_bytes_by_name[entry.name] = f.read(entry.length)
    sh_store = ShardedWeightStore(sh_m)
    try:
        for entry in sh_m.tensors:
            sh_data = sh_store.read_bytes(entry)
            src_data = src_bytes_by_name[entry.name]
            assert sh_data == src_data, (
                f"data mismatch for {entry.name!r}: "
                f"src={src_data[:16].hex()} sh={sh_data[:16].hex()}"
            )
    finally:
        sh_store.close()

    from rwkv_ssd.tools.verify_pack import verify

    assert verify(out_dir, quiet=True)


def test_sharded_store_supports_nested_relative_paths(tmp_path: Path) -> None:
    """Relative shard paths must work when files are stored in subdirectories."""
    pack_dir = tmp_path / "nested"
    (pack_dir / "ssd0").mkdir(parents=True)
    (pack_dir / "ssd1").mkdir()
    names = ["ssd0/weights.bin", "ssd1/weights.bin"]
    for i, name in enumerate(names):
        (pack_dir / name).write_bytes(bytes([0x20 + i]) * 4096)

    import json

    tensors = [
        {
            "name": f"blocks.{i}.weight",
            "layer_id": i,
            "dtype": "bf16",
            "shape": [16],
            "offset": 0,
            "length": 128,
            "alignment": 4096,
            "residency": "streamed",
            "shard_file": name,
        }
        for i, name in enumerate(names)
    ]
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": names[0],
                "weights_files": names,
                "tensors": tensors,
                "meta": {"n_layer": 2},
            }
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(pack_dir)
    store = ShardedWeightStore(manifest, parallel_workers=2)
    try:
        assert store.read_bytes(manifest.tensors[0]) == b"\x20" * 128
        assert store.read_bytes(manifest.tensors[1]) == b"!" * 128
        assert store.read_bytes_for_shard(names[1], 0, 16) == b"!" * 16
    finally:
        store.close()


def _write_v2_manifest(
    pack_dir: Path,
    stripes: list[dict],
    *,
    entry_length: int = 8,
    shard_bytes: bytes = b"0123456789abcdef",
    alignment: int = 1,
) -> None:
    import json

    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "weights.shard.0.bin").write_bytes(shard_bytes)
    (pack_dir / "weights.shard.1.bin").write_bytes(shard_bytes)
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "model_family": "rwkv",
                "weights_file": "weights.shard.0.bin",
                "weights_files": [
                    "weights.shard.0.bin",
                    "weights.shard.1.bin",
                ],
                "tensors": [
                    {
                        "name": "blocks.0.weight",
                        "layer_id": 0,
                        "dtype": "u8",
                        "shape": [entry_length],
                        "offset": 0,
                        "length": entry_length,
                        "alignment": alignment,
                        "residency": "streamed",
                        "stripes": stripes,
                    }
                ],
                "meta": {"n_shards": 2, "shard_strategy": "striped_round_robin"},
            }
        ),
        encoding="utf-8",
    )


def test_manifest_v2_rejects_missing_logical_stripe_range(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 2, "logical_offset": 0, "physical_length": 2},
            {"shard_file": "weights.shard.1.bin", "offset": 0, "length": 2, "logical_offset": 4, "physical_length": 2},
        ],
    )
    with pytest.raises(ValueError, match="missing logical stripe"):
        Manifest.load(tmp_path)


def test_manifest_v2_rejects_overlapping_logical_stripes(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 4, "logical_offset": 0, "physical_length": 4},
            {"shard_file": "weights.shard.1.bin", "offset": 0, "length": 4, "logical_offset": 2, "physical_length": 4},
        ],
    )
    with pytest.raises(ValueError, match="overlapping logical stripe"):
        Manifest.load(tmp_path)


def test_manifest_v2_rejects_invalid_physical_length(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 8, "logical_offset": 0, "physical_length": 4},
        ],
    )
    with pytest.raises(ValueError, match="invalid stripe extent"):
        Manifest.load(tmp_path)


def test_manifest_v2_rejects_invalid_physical_alignment(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 1, "length": 4, "logical_offset": 0, "physical_length": 4},
            {"shard_file": "weights.shard.1.bin", "offset": 0, "length": 4, "logical_offset": 4, "physical_length": 4},
        ],
        alignment=4,
    )
    with pytest.raises(ValueError, match="not aligned"):
        Manifest.load(tmp_path)


def test_manifest_v2_rejects_truncated_shard_file(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 8, "logical_offset": 0, "physical_length": 32},
        ],
    )
    with pytest.raises(ValueError, match="exceeds shard"):
        Manifest.load(tmp_path)


def test_manifest_v2_rejects_undeclared_stripe_shard(tmp_path: Path) -> None:
    _write_v2_manifest(
        tmp_path,
        [
            {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 4, "logical_offset": 0, "physical_length": 4},
            {"shard_file": "missing.bin", "offset": 0, "length": 4, "logical_offset": 4, "physical_length": 4},
        ],
    )
    with pytest.raises(ValueError, match="not listed in weights_files"):
        Manifest.load(tmp_path)


def test_sharded_store_coalesces_adjacent_layer_extents(tmp_path: Path) -> None:
    import json

    pack_dir = tmp_path / "coalesced"
    pack_dir.mkdir()
    (pack_dir / "weights.shard.0.bin").write_bytes(b"abcdefgh" + b"\0" * 8)
    (pack_dir / "weights.shard.1.bin").write_bytes(b"12345678" + b"\0" * 8)
    tensors = [
        {
            "name": "blocks.0.a",
            "layer_id": 0,
            "dtype": "u8",
            "shape": [4],
            "offset": 0,
            "length": 4,
            "alignment": 1,
            "residency": "streamed",
            "stripes": [
                {"shard_file": "weights.shard.0.bin", "offset": 0, "length": 4, "logical_offset": 0, "physical_length": 4}
            ],
        },
        {
            "name": "blocks.0.b",
            "layer_id": 0,
            "dtype": "u8",
            "shape": [4],
            "offset": 0,
            "length": 4,
            "alignment": 1,
            "residency": "streamed",
            "stripes": [
                {"shard_file": "weights.shard.0.bin", "offset": 4, "length": 4, "logical_offset": 0, "physical_length": 4}
            ],
        },
    ]
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "model_family": "rwkv",
                "weights_file": "weights.shard.0.bin",
                "weights_files": ["weights.shard.0.bin", "weights.shard.1.bin"],
                "tensors": tensors,
                "meta": {"n_layer": 1},
            }
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(pack_dir)
    store = ShardedWeightStore(manifest)
    calls = {id(inner): 0 for inner in set(store._stores.values())}
    try:
        for inner in set(store._stores.values()):
            original = inner.read_bytes_span

            def counted(offset, length, *, _original=original, _inner=inner):
                calls[id(_inner)] += 1
                return _original(offset, length)

            inner.read_bytes_span = counted  # type: ignore[method-assign]
        gathered = store.read_layer(manifest.tensors)
        assert gathered["blocks.0.a"] == b"abcd"
        assert gathered["blocks.0.b"] == b"efgh"
        assert sum(calls.values()) == 1
    finally:
        store.close()


def test_provider_uses_coalesced_gather_for_striped_layer(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    import torch

    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
    from rwkv_ssd.tools.shard_pack import shard_pack

    output = tmp_path / "provider-striped"
    shard_pack(
        synthetic_pack,
        output,
        n_shards=2,
        strategy="stripe",
        stripe_bytes=64,
    )
    manifest = Manifest.load(output)
    store = ShardedWeightStore(manifest, parallel_workers=2)
    calls = {"coalesced": 0}
    original = store.read_layer_coalesced

    def counted(entries):
        calls["coalesced"] += 1
        return original(entries)

    store.read_layer_coalesced = counted  # type: ignore[method-assign]
    provider = ManifestWeightProvider(
        "streaming",
        store,
        manifest.tensors,
        torch.device("cpu"),
        MetricsCollector(),
        prefetch=False,
    )
    try:
        layer_id = next(
            layer_id
            for layer_id, entries in manifest.by_layer().items()
            if layer_id >= 0 and any(entry.stripes for entry in entries)
        )
        loaded = provider.load_layer_tensors(manifest.by_layer()[layer_id])
        assert loaded
        assert calls["coalesced"] == 1
    finally:
        provider.close()
        store.close()


def test_shard_pack_round_trips_bf16_shadow_striping(tmp_path: Path) -> None:
    import json

    from rwkv_ssd.tools.shard_pack import shard_pack

    source = tmp_path / "shadow-source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"abcdefgh")
    (source / "shadow.bin").write_bytes(b"ABCDEFGH")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "rwkv",
                "weights_file": "weights.bin",
                "tensors": [
                    {
                        "name": "blocks.0.weight",
                        "layer_id": 0,
                        "dtype": "u8",
                        "shape": [8],
                        "offset": 0,
                        "length": 8,
                        "alignment": 1,
                        "residency": "streamed",
                        "fast_offset": 0,
                        "fast_length": 8,
                    }
                ],
                "meta": {"shadow_file": "shadow.bin"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "striped-shadow"
    shard_pack(source, output, 2, strategy="stripe", stripe_bytes=3)
    manifest = Manifest.load(output)
    entry = manifest.tensors[0]
    assert manifest.has_bf16_shadow()
    assert len(manifest.shadow_paths()) == 2
    assert len(entry.fast_stripes) == 3
    assert entry.fast_shard_file == ""

    from rwkv_ssd.runtime.decode_shadow import read_shadow_layer

    shadow_store = ShardedWeightStore(
        manifest, files=manifest.shadow_paths(), parallel_workers=2
    )
    try:
        raw, base = read_shadow_layer(shadow_store, [entry])
        assert base == 0
        assert raw == b"ABCDEFGH"
    finally:
        shadow_store.close()
