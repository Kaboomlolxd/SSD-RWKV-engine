from __future__ import annotations

import torch

from rwkv_ssd.runtime.ggml_weight_bridge import GgmlWeightBridge
from rwkv_ssd.runtime.prepared_artifact_cache import PreparedArtifactCache


def test_prepared_artifact_cache_get_or_build_and_checksum(tmp_path) -> None:
    cache = PreparedArtifactCache(tmp_path / "cache")
    key = cache.make_key("test", b"source", {"abi": 1})
    builds = {"count": 0}

    def build() -> bytes:
        builds["count"] += 1
        return b"prepared"

    assert cache.get_or_build(key, build) == b"prepared"
    assert cache.get_or_build(key, build) == b"prepared"
    assert builds["count"] == 1
    assert cache.stats.hits == 1
    assert cache.stats.writes == 1

    path = cache._path(key)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(raw)
    assert cache.get(key) is None
    assert cache.stats.corruptions == 1


def test_prepared_artifact_cache_enforces_disk_cap(tmp_path) -> None:
    cache = PreparedArtifactCache(tmp_path / "cache", max_bytes=350)
    for index in range(4):
        source = bytes([index])
        key = cache.make_key("test", source, {})
        cache.put(key, bytes([index]) * 128)
    assert cache.disk_bytes() <= 350
    assert cache.stats.evictions > 0


class _CopyModel:
    supports_tensor_upload = True
    supports_tensor_slot_upload = False

    def __init__(self) -> None:
        self.uploads = []

    def tensor_nbytes(self, name: str) -> int:
        return 8 if name.endswith(".weight") else 0

    def set_tensor(self, name: str, payload: bytes) -> None:
        self.uploads.append((name, payload))


def test_ggml_bridge_reuses_persistent_prepared_payload(tmp_path) -> None:
    cache = PreparedArtifactCache(tmp_path / "cache")
    tensors = {"blocks.0.weight": torch.tensor([1.0, 2.0])}
    first_model = _CopyModel()
    first = GgmlWeightBridge(first_model, artifact_cache=cache)
    assert first.upload_layer(0, tensors).uploaded_tensors == 1
    assert first.artifact_stats()["writes"] == 1

    second_model = _CopyModel()
    second = GgmlWeightBridge(second_model, artifact_cache=cache)
    assert second.upload_layer(0, tensors).uploaded_tensors == 1
    assert second.artifact_stats()["hits"] == 1
    assert first_model.uploads == second_model.uploads
