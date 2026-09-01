from __future__ import annotations

import shutil
import json
from pathlib import Path

import pytest
import torch

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_verify import verify_pack
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.tools.pack_runtime import _finalize_weight_compression, pack


zstd = pytest.importorskip("zstandard")


def test_zstd_store_preserves_logical_offsets(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "compressed"
    shutil.copytree(synthetic_pack, compressed)
    original = Manifest.load(compressed)
    expected = {
        entry.name: original.weights_path.read_bytes()[
            entry.offset : entry.offset + entry.length
        ]
        for entry in original.tensors[:4]
    }

    _finalize_weight_compression(compressed, "zstd", quiet=True)
    manifest = Manifest.load(compressed)
    ok, messages = verify_pack(compressed, check_hash=True)
    assert ok, messages
    assert manifest.weights_path.name == "weights.bin.zst"
    assert manifest.meta["weights_uncompressed_bytes"] > manifest.weights_path.stat().st_size

    with open_weight_store(manifest.weights_path, backend="mmap") as store:
        for entry in manifest.tensors[:4]:
            assert store.read_bytes(entry) == expected[entry.name]
        entry = manifest.tensors[0]
        assert bytes(store.read_memoryview_span(entry.offset, entry.length)) == expected[entry.name]


def test_pack_emits_verified_zstd_pack(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save(
        {"blocks.0.weight": torch.arange(4096, dtype=torch.float32).reshape(64, 64)},
        checkpoint,
    )
    output = tmp_path / "pack"
    pack(
        checkpoint,
        output,
        model_family="synthetic",
        compression="zstd",
        quiet=True,
    )

    manifest = Manifest.load(output)
    assert manifest.weights_path.name == "weights.bin.zst"
    assert not (output / "weights.bin").exists()
    ok, messages = verify_pack(output, check_hash=True)
    assert ok, messages
    with open_weight_store(manifest.weights_path, backend="pread") as store:
        entry = manifest.tensors[0]
        assert len(store.read_bytes(entry)) == entry.length


def test_zstd_store_accepts_unknown_frame_content_size(tmp_path: Path) -> None:
    """Streaming zstd frames must not be mistaken for huge known sizes."""
    compressor = zstd.ZstdCompressor()
    logical = b"unknown-size frame payload" * 257
    # Keep the compressor object alive for both calls: ``compressobj`` carries
    # the frame state and deliberately omits the content-size field.
    stream = compressor.compressobj()
    frame = stream.compress(logical) + stream.flush()
    assert zstd.frame_content_size(frame) in (-1, zstd.CONTENTSIZE_UNKNOWN)

    packed = tmp_path / "unknown"
    packed.mkdir()
    (packed / "weights.bin.zst").write_bytes(frame)
    (packed / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_family": "synthetic",
                "weights_file": "weights.bin.zst",
                "tensors": [
                    {
                        "name": "blocks.0.weight",
                        "layer_id": 0,
                        "dtype": "u8",
                        "shape": [len(logical)],
                        "offset": 0,
                        "length": len(logical),
                        "alignment": 1,
                        "residency": "streamed",
                    }
                ],
                "meta": {
                    "weights_compression": "zstd",
                    "weights_uncompressed_bytes": len(logical),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = Manifest.load(packed)
    with open_weight_store(manifest.weights_path, backend="mmap") as store:
        assert store.read_bytes(manifest.tensors[0]) == logical


def test_pack_transformation_tools_reject_zstd_source(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "compressed-source"
    shutil.copytree(synthetic_pack, compressed)
    _finalize_weight_compression(compressed, "zstd", quiet=True)

    from rwkv_ssd.tools.repack_storage_placement import repack_storage_placement
    from rwkv_ssd.tools.shard_pack import shard_pack

    with pytest.raises(ValueError, match="decompress/repack first"):
        shard_pack(compressed, tmp_path / "sharded", n_shards=2)
    with pytest.raises(ValueError, match="decompress/repack first"):
        repack_storage_placement(
            compressed,
            tmp_path / "placed",
            {"placements": [{"layer_id": 0, "drive": "ssd0"}]},
        )


def test_synthetic_engine_runs_from_zstd_pack(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "engine-pack"
    shutil.copytree(synthetic_pack, compressed)
    _finalize_weight_compression(compressed, "zstd", quiet=True)

    baseline = InferenceEngine(
        EngineConfig(
            pack_dir=synthetic_pack,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            max_tokens=4,
        )
    )
    from_zstd = InferenceEngine(
        EngineConfig(
            pack_dir=compressed,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            max_tokens=4,
        )
    )
    baseline.load()
    from_zstd.load()
    try:
        assert from_zstd.generate("compressed path") == baseline.generate("compressed path")
    finally:
        baseline.close()
        from_zstd.close()
