"""Provider cache RAM: no duplicate raw+prepared tensors."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.pack_codec import encode_scale_u8_grouped
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.ggml_weight_bridge import GgmlWeightBridge
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend, _resolve_layer_cache_bytes


class _MemoryWeightStore:
    """Small deterministic store used to exercise provider residency logic."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read_bytes(self, entry: TensorEntry) -> bytes:
        return self.payload[entry.offset : entry.offset + entry.length]

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        return self.payload[offset : offset + length]

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        return memoryview(self.payload)[offset : offset + length]

    def close(self) -> None:
        return None


class _StableMemoryWeightStore(_MemoryWeightStore):
    stable_memoryviews = True

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        return memoryview(self.payload)[offset : offset + length]


def _grouped_entry(*, residency: str = "streamed") -> tuple[TensorEntry, bytes]:
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    blob = encode_scale_u8_grouped(tensor, group_size=4)
    return (
        TensorEntry(
            name="blocks.0.att.key.weight",
            layer_id=0,
            dtype="float32",
            shape=[4, 4],
            offset=0,
            length=len(blob),
            alignment=1,
            residency=residency,
            dequant="scale_u8_grouped",
        ),
        blob,
    )


def test_prepare_drops_raw_cache_duplicate() -> None:
    pack = Path("test_model/trinity_eval/trinity_lut2_0.1b")
    if not (pack / "weights.bin").is_file():
        pytest.skip("archived Trinity manifest has no weights.bin payload")
    manifest = Manifest.load(pack)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    metrics = MetricsCollector()
    try:
        cfg = EngineConfig(
            pack_dir=pack,
            mode="streaming",
            device="cpu",
            stream_layer_cache=True,
            decouple_provider_cache=True,
            max_layers_in_z=2,
        )
        provider = create_weight_provider(
            cfg,
            store,
            manifest.tensors,
            torch.device("cpu"),
            metrics,
        )
        by_layer = manifest.by_layer()
        layer0 = by_layer.get(0, [])
        if not layer0:
            return
        layer_tensors = provider.load_layer_tensors(layer0)
        provider.prepare_layer_for_z(0, layer_tensors)
        raw_keys = [k for k in provider._cache if k.startswith("blocks.0.")]
        assert raw_keys == [], f"raw cache should be dropped after prepare: {raw_keys}"
        assert 0 in provider._prepared_layers
    finally:
        store.close()


def test_native_resident_grouped_u8_preserves_raw_record_without_dense_duplicate() -> None:
    entry, blob = _grouped_entry(residency="resident")
    metrics = MetricsCollector()
    provider = ManifestWeightProvider(
        mode="resident",
        store=_MemoryWeightStore(blob),
        entries=[entry],
        device=torch.device("cpu"),
        metrics=metrics,
        stream_layer_cache=False,
        decouple_provider_cache=False,
        max_layers_in_z=0,
        max_provider_cache_layers=0,
    )
    try:
        loaded = provider.load_layer_tensors_native([entry])
        assert isinstance(loaded[entry.name], memoryview)
        assert bytes(loaded[entry.name]) == blob
        assert entry.name in provider._native_resident_blobs
        assert entry.name not in provider._cache
        assert metrics.layers[-1].bytes_read == len(blob)
    finally:
        provider.close()


def test_native_resident_alias_is_counted_once() -> None:
    entry, blob = _grouped_entry(residency="resident")
    provider = ManifestWeightProvider(
        mode="partial",
        store=_MemoryWeightStore(blob),
        entries=[entry],
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        native_layer_streaming=True,
        decouple_provider_cache=True,
        max_layers_in_z=0,
        max_provider_cache_layers=0,
        max_provider_cache_bytes=1024 * 1024,
    )
    try:
        provider.load_layer_tensors_native([entry])
        stats = provider.cache_stats()
        # The resident bytes are also exposed through the native layer view;
        # provider accounting must not charge that alias twice.
        assert stats["packed_cache_bytes"] == len(blob)
        assert stats["provider_cache_bytes"] == len(blob)
        assert stats["provider_resident_bytes"] == len(blob)
    finally:
        provider.close()


def test_native_stable_mmap_payload_is_reported_separately() -> None:
    entry, blob = _grouped_entry(residency="resident")
    provider = ManifestWeightProvider(
        mode="partial",
        store=_StableMemoryWeightStore(blob),
        entries=[entry],
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        native_layer_streaming=True,
        decouple_provider_cache=True,
        max_layers_in_z=0,
        max_provider_cache_layers=0,
        max_provider_cache_bytes=1024 * 1024,
    )
    try:
        loaded = provider.load_layer_tensors_native([entry])
        assert isinstance(loaded[entry.name], memoryview)
        stats = provider.cache_stats()
        assert stats["packed_cache_bytes"] == len(blob)
        assert stats["provider_mmap_bytes"] == len(blob)
        assert stats["provider_cache_bytes"] == 0
    finally:
        provider.close()


def test_native_layer_view_cache_hits_and_eviction_releases_raw_view() -> None:
    entry, blob = _grouped_entry(residency="streamed")
    metrics = MetricsCollector()
    provider = ManifestWeightProvider(
        mode="streaming",
        store=_MemoryWeightStore(blob),
        entries=[entry],
        device=torch.device("cpu"),
        metrics=metrics,
        stream_layer_cache=True,
        decouple_provider_cache=True,
        max_layers_in_z=0,
        max_provider_cache_layers=1,
    )
    try:
        first = provider.load_layer_tensors_native([entry])
        first_read = metrics.layers[-1].bytes_read
        second = provider.load_layer_tensors_native([entry])
        second_timing = metrics.layers[-1]
        assert first_read == len(blob)
        assert second is first
        assert second_timing.bytes_read == 0
        assert second_timing.layer_cache_hits >= 1
        assert 0 in provider.cached_layer_ids()

        provider.evict_streamed_layer(0, force=True)
        assert 0 not in provider.cached_layer_ids()
        assert 0 not in provider._native_layer_views
        assert provider.packed_weight_bytes() == 0
    finally:
        provider.close()


class _RecordingNativeLayerModel:
    supports_layer_streaming = True
    supports_native_u8 = True

    def __init__(self, blob_length: int) -> None:
        self.blob_length = blob_length
        self.uploads: list[tuple[str, int]] = []

    def layer_begin(self, layer_id: int) -> None:
        self.active_layer = layer_id

    def tensor_nbytes(self, name: str) -> int:
        return self.blob_length

    def layer_set_tensor_scale_u8_grouped(self, name: str, data: memoryview) -> None:
        self.uploads.append((name, len(data)))


def test_provider_read_bytes_and_native_upload_bytes_are_separate() -> None:
    entry, blob = _grouped_entry(residency="streamed")
    metrics = MetricsCollector()
    provider = ManifestWeightProvider(
        mode="streaming",
        store=_MemoryWeightStore(blob),
        entries=[entry],
        device=torch.device("cpu"),
        metrics=metrics,
        stream_layer_cache=True,
        decouple_provider_cache=True,
        max_layers_in_z=0,
        max_provider_cache_layers=1,
    )
    model = _RecordingNativeLayerModel(len(blob))
    bridge = GgmlWeightBridge(model)
    try:
        tensors = provider.load_layer_tensors_native([entry])
        read_before_upload = metrics.layers[-1].bytes_read
        stats = bridge.upload_layer(0, tensors)
        assert read_before_upload == len(blob)
        assert stats.uploaded_bytes == len(blob)
        assert stats.packed_bytes == len(blob)
        assert stats.active_bytes == len(blob)
        assert model.uploads == [(entry.name, len(blob))]
        # Native upload accounting must not mutate the provider's source-read
        # counter; the benchmark reports the two byte domains separately.
        assert metrics.layers[-1].bytes_read == read_before_upload
    finally:
        provider.close()


def test_native_layer_cache_limit_is_bounded_and_configurable(monkeypatch) -> None:
    monkeypatch.delenv("RWKVCPP_LAYER_CACHE_BYTES", raising=False)
    assert _resolve_layer_cache_bytes(4096) == 4096

    monkeypatch.setenv("RWKVCPP_LAYER_CACHE_BYTES", "0x2000")
    assert _resolve_layer_cache_bytes(4096) == 8192

    monkeypatch.setenv("RWKVCPP_LAYER_CACHE_BYTES", "auto")
    assert _resolve_layer_cache_bytes(4096) == 4096

    monkeypatch.setenv("RWKVCPP_LAYER_CACHE_BYTES", "not-a-size")
    assert _resolve_layer_cache_bytes(4096) == 4096

    backend = RWKVCppBackend()
    backend.set_layer_cache_bytes_hint(12345)
    assert backend._layer_cache_bytes_hint == 12345
    backend.set_layer_cache_bytes_hint(-1)
    assert backend._layer_cache_bytes_hint == 0


def test_native_cache_telemetry_is_present_in_structured_metrics() -> None:
    metrics = MetricsCollector()
    metrics.native_upload_bytes = 100
    metrics.native_packed_bytes = 60
    metrics.native_active_bytes = 160
    metrics.native_layer_cache_bytes = 256
    metrics.native_layer_cache_hits = 3
    metrics.native_layer_cache_misses = 2
    metrics.native_layer_cache_evictions = 1
    metrics.native_decoded_cache_hits = 2

    report = metrics.to_dict()
    assert report["native_upload_bytes"] == 100
    assert report["native_packed_bytes"] == 60
    assert report["native_active_bytes"] == 160
    assert report["native_layer_cache_bytes"] == 256
    assert report["native_layer_cache_hits"] == 3
    assert report["native_layer_cache_misses"] == 2
    assert report["native_layer_cache_evictions"] == 1
    assert report["native_decoded_cache_hits"] == 2
