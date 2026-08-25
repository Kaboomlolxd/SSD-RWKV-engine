from __future__ import annotations

import pytest

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_overlay import ChunkStoreWeightStore, PackChunkStore


def test_pack_overlay_deduplicates_profiles_and_materializes_exactly(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    common = b"abcdefgh" * 4
    (first / "weights.bin").write_bytes(common + b"11111111")
    (second / "weights.bin").write_bytes(common + b"22222222")
    (first / "manifest.json").write_text('{"profile":1}', encoding="utf-8")
    (second / "manifest.json").write_text('{"profile":2}', encoding="utf-8")
    store = PackChunkStore(tmp_path / "store", chunk_bytes=8)
    first_stats = store.ingest_directory(first, "first")
    second_stats = store.ingest_directory(second, "second")
    assert first_stats["new_chunks"] > 0
    assert second_stats["reused_chunks"] >= 4
    assert second_stats["metadata_bytes"] > 0
    assert len(second_stats["profile_sha256"]) == 64
    output = store.materialize("second", tmp_path / "materialized")
    assert (output / "weights.bin").read_bytes() == (second / "weights.bin").read_bytes()
    assert (output / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_overlay_file_reads_across_chunk_boundaries(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = bytes(range(40))
    (source / "weights.bin").write_bytes(payload)
    store = PackChunkStore(tmp_path / "store", chunk_bytes=8)
    store.ingest_directory(source, "test")
    overlay_file = store.read_file("test", "weights.bin")
    assert overlay_file.read(6, 20) == payload[6:26]
    with pytest.raises(ValueError, match="bounds"):
        overlay_file.read(39, 2)


def test_pack_overlay_detects_chunk_corruption(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"abcdefgh")
    store = PackChunkStore(tmp_path / "store", chunk_bytes=8)
    store.ingest_directory(source, "test")
    profile = store.load_profile("test")
    digest = profile["files"]["weights.bin"]["chunks"][0]
    store._chunk_path(digest).write_bytes(b"corrupt!")
    with pytest.raises(ValueError, match="checksum"):
        store.materialize("test", tmp_path / "out")


def test_pack_overlay_rejects_unsafe_paths(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"x")
    store = PackChunkStore(tmp_path / "store")
    with pytest.raises(ValueError, match="unsafe"):
        store.ingest_directory(source, "test", include=["../outside"])


def test_chunk_store_weight_store_reads_manifest_entries_without_materialization(
    synthetic_pack, tmp_path
) -> None:
    store = PackChunkStore(tmp_path / "store", chunk_bytes=127)
    store.ingest_directory(synthetic_pack, "model")
    manifest = Manifest.load(synthetic_pack)
    direct = ChunkStoreWeightStore(store, "model")
    try:
        with manifest.weights_path.open("rb") as handle:
            for entry in manifest.tensors[:8]:
                handle.seek(entry.offset)
                assert direct.read_bytes(entry) == handle.read(entry.length)
        first = manifest.tensors[0]
        assert direct.read_bytes_span(first.offset + 3, 11) == (
            manifest.weights_path.read_bytes()[first.offset + 3 : first.offset + 14]
        )
        assert direct.read_calls == min(8, len(manifest.tensors)) + 1
        assert direct.bytes_read > 0
    finally:
        direct.close()


def test_pack_overlay_remove_and_gc_only_unreferenced_chunks(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "weights.bin").write_bytes(b"A" * 8 + b"B" * 8)
    (second / "weights.bin").write_bytes(b"A" * 8 + b"C" * 8)
    store = PackChunkStore(tmp_path / "store", chunk_bytes=8)
    store.ingest_directory(first, "first")
    store.ingest_directory(second, "second")
    assert store.remove_profile("first") is True
    result = store.garbage_collect()
    assert result["removed_chunks"] == 1
    assert result["removed_bytes"] == 8
    assert store.materialize("second", tmp_path / "out").joinpath(
        "weights.bin"
    ).read_bytes() == b"A" * 8 + b"C" * 8


def test_pack_overlay_profile_checksum_detects_metadata_corruption(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"abcdefgh")
    store = PackChunkStore(tmp_path / "store", chunk_bytes=4)
    store.ingest_directory(source, "profile")
    path = store.profiles_dir / "profile.json"
    raw = path.read_text(encoding="utf-8").replace('"size": 8', '"size": 7')
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="profile checksum"):
        store.load_profile("profile")
