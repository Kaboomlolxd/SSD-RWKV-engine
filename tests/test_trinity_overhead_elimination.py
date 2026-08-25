"""P0/P1 Trinity overhead elimination regression tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.throughput_defaults import (
    _warm_disk_cache_auto,
    apply_low_ram_defaults,
    apply_stacked_strict_defaults,
    warm_disk_cache_active,
)
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_tensor,
    encode_trinity_lut2,
    packed_length_trinity_lut2,
)
from rwkv_ssd.runtime.lut_gemm_fused import (
    lut2_gemv_cpu,
    lut2_tmix_gemv_batched,
)
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
from rwkv_ssd.runtime.metrics import MetricsCollector


def _trinity_manifest(tmp_path: Path, n_layer: int = 12) -> Manifest:
    return Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "weights.bin",
        tensors=[
            TensorEntry(
                name="blocks.0.att.key.weight",
                layer_id=0,
                dtype="bf16",
                shape=[768, 768],
                offset=0,
                length=8,
                alignment=4096,
                residency="streamed",
                dequant="trinity_lut2",
            )
        ],
        meta={"n_layer": n_layer},
    )


def test_warm_disk_cache_auto_on_trinity_without_promote(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("RWKV_WARM_DISK_CACHE", raising=False)
    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    manifest = _trinity_manifest(tmp_path)
    assert _warm_disk_cache_auto(manifest) is True
    assert warm_disk_cache_active(manifest) is True


def test_warm_disk_cache_auto_off_when_promote_on(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RWKV_WARM_DISK_CACHE", raising=False)
    monkeypatch.setenv("RWKV_PROMOTE_FULL_Z", "1")
    manifest = _trinity_manifest(tmp_path)
    assert _warm_disk_cache_auto(manifest) is False


def test_low_ram_defaults_use_partial_ssd_tier(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RWKV_PROMOTE_FULL_Z", raising=False)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming", low_ram=True)
    manifest = _trinity_manifest(tmp_path)
    apply_low_ram_defaults(cfg, manifest)
    assert cfg.mode == "partial"
    assert cfg.stream_layer_cache is False
    assert cfg.max_layers_in_z == 0
    assert cfg.residency_profile is not None
    assert "hot3" in cfg.residency_profile.name
    assert os.environ.get("RWKV_LUT_GEMM_FUSED") == "1"


def test_stacked_strict_enables_warm_disk_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RWKV_WARM_DISK_CACHE", raising=False)
    cfg = EngineConfig(pack_dir=tmp_path, mode="streaming")
    apply_stacked_strict_defaults(cfg, _trinity_manifest(tmp_path))
    assert os.environ.get("RWKV_WARM_DISK_CACHE") == "auto"


def test_batched_tmix_gemv_matches_reference_matmul() -> None:
    in_f, out_f = 16, 32
    xs = [torch.randn(in_f, dtype=torch.bfloat16) for _ in range(4)]
    blobs = []
    refs = []
    for x in xs:
        w = torch.randn(out_f, in_f, dtype=torch.bfloat16)
        blob = encode_trinity_lut2(w)
        blobs.append(blob)
        refs.append(lut2_gemv_cpu(blob, x, out_features=out_f, in_features=in_f))
    y_r, y_k, y_v, y_o = lut2_tmix_gemv_batched(
        tuple(blobs),
        tuple(xs),
        out_features=out_f,
        in_features=in_f,
    )
    for y, ref in zip((y_r, y_k, y_v, y_o), refs):
        torch.testing.assert_close(y.float(), ref.float(), rtol=0, atol=0.01)


def test_single_gemv_still_matches_materialized() -> None:
    w = torch.randn(32, 16, dtype=torch.bfloat16)
    x = torch.randn(16, dtype=torch.bfloat16)
    blob = encode_trinity_lut2(w)
    y_fused = lut2_gemv_cpu(blob, x, out_features=32, in_features=16)
    y_ref = lut2_gemv_cpu(blob, x, out_features=32, in_features=16)
    torch.testing.assert_close(y_fused.float(), y_ref.float(), rtol=0, atol=0.01)


def test_flute_layout_roundtrip() -> None:
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    entry = TensorEntry(
        name="blocks.0.att.key.weight",
        layer_id=0,
        dtype="bfloat16",
        shape=[64, 32],
        offset=0,
        length=packed_length_trinity_lut2(64 * 32) + 64,
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    blob = encode_trinity_lut2(w, layout="flute_row")
    assert blob[:4] == b"TR2\x02"
    out = decode_trinity_lut2_to_tensor(blob, entry, torch.device("cpu"))
    diff = (out.float() - w.float()).abs().mean()
    assert diff < 0.75


def test_engine_blocks_trinity_layer_codec(tmp_path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"\x00" * 4096)
    from rwkv_ssd.runtime.weight_store import open_weight_store

    entries = [
        TensorEntry(
            name="blocks.0.att.key.weight",
            layer_id=0,
            dtype="bf16",
            shape=[4, 4],
            offset=0,
            length=64,
            alignment=4096,
            residency="streamed",
            dequant="trinity_layer",
        )
    ]
    store = open_weight_store(weights, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=entries,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
    )
    raw = b"\x00" * 64
    with pytest.raises(RuntimeError, match="trinity_layer.*blocked"):
        provider._decode_from_layer_raw(raw, 0, entries, None)
    store.close()


def test_strict_streaming_prefers_fused_over_disk_cache_when_warm(
    synthetic_pack_trinity_lut2: Path, monkeypatch, tmp_path
) -> None:
    """Warm disk cache at load must not force bf16 re-read every token on strict fused."""
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_WARM_DISK_CACHE", "1")
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_id = next(i for i in sorted(by_layer.keys()) if i >= 0)
    entries = by_layer[layer_id]
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
        decode_disk_cache="1",
        pack_dir=synthetic_pack_trinity_lut2,
        manifest_meta=manifest.meta,
    )
    assert provider._prefer_disk_cache_over_fused_lut(entries) is False
    provider.close()


def test_stream_cache_prefers_disk_cache_for_shadow_layers(
    synthetic_pack_trinity_lut2: Path, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_WARM_DISK_CACHE", "1")
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store
    from rwkv_ssd.runtime.decode_disk_cache import DecodeDiskCache

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_id = next(i for i in sorted(by_layer.keys()) if i >= 0)
    entries = by_layer[layer_id]
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=True,
        decode_disk_cache="1",
        pack_dir=synthetic_pack_trinity_lut2,
        manifest_meta=manifest.meta,
    )
    # LUT-only layer: fused preferred unless shadow hybrid
    if not provider._layer_uses_shadow(entries):
        assert provider._prefer_disk_cache_over_fused_lut(entries) is False
    materialized = provider.load_layer_tensors_materialized(entries)
    disk = provider._disk_cache
    assert disk is not None
    disk.store_layer(layer_id, entries, materialized, async_write=False)
    provider.evict_streamed_layer(layer_id)
    timing = provider.begin_layer(layer_id)
    if provider._prefer_disk_cache_over_fused_lut(entries):
        hit = provider._try_cached_layer(layer_id, entries, timing)
        assert hit is not None
        assert timing.disk_cache_hits > 0
    provider.close()


def _fake_fused_provider(*, lean_z: bool) -> Any:
    """Empty-entry providers have ``_pack_uses_quant=False``; force fused retain."""
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    provider = ManifestWeightProvider(
        mode="streaming",
        store=object(),
        entries=[],
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
    )
    provider._pack_uses_quant = True
    # The real provider knows its codec set during construction.  This test
    # deliberately overrides the empty fixture after construction, so clear
    # the cached fused decision before exercising strict-retain policy.
    provider._fused_lut_enabled = None
    if lean_z:
        # max_layers_in_z<=0 is the auto lean-z trigger when env is auto.
        provider._z_retention.max_layers_in_z = 0
    return provider


def test_strict_fused_skeleton_vectors_survive_z_evict(monkeypatch) -> None:
    """Non-lean retain keeps skeleton in ``z``; lean-z drops it (provider holds it)."""
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_RETAIN", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_LEAN_Z", "0")
    from rwkv_ssd.backends.rwkv7_forward import _evict_layer_from_z

    z = {
        "blocks.0.att.x_r": torch.zeros(8, dtype=torch.bfloat16),
        "blocks.0.att.receptance.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    provider = _fake_fused_provider(lean_z=False)
    assert provider._strict_fused_retain_layers()
    assert not provider._strict_fused_lean_z()
    _evict_layer_from_z(z, 0, provider)
    assert "blocks.0.att.x_r" in z
    assert "blocks.0.att.receptance.weight" not in z
    provider.close()


def test_strict_fused_lean_z_evicts_skeleton_from_z(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_RETAIN", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_LEAN_Z", "1")
    from rwkv_ssd.backends.rwkv7_forward import _evict_layer_from_z

    z = {
        "blocks.0.att.x_r": torch.zeros(8, dtype=torch.bfloat16),
        "blocks.0.att.receptance.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    provider = _fake_fused_provider(lean_z=True)
    assert provider._strict_fused_lean_z()
    _evict_layer_from_z(z, 0, provider)
    assert "blocks.0.att.x_r" not in z
    assert "blocks.0.att.receptance.weight" not in z
    provider.close()


def test_strict_fused_retain_keeps_prepared_across_layers(
    synthetic_pack_trinity_lut2: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_RETAIN", "1")
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_ids = sorted(i for i in by_layer.keys() if i >= 0)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
        max_provider_cache_layers=2,
    )
    for layer_id in layer_ids[:3]:
        entries = by_layer[layer_id]
        tensors = provider.load_layer_tensors(entries)
        prepared = provider.prepare_layer_for_z(layer_id, tensors)
        provider._retain_provider_layer(layer_id)
        assert prepared
    assert len(provider._prepared_layers) >= 2
    first_id = layer_ids[0]
    assert first_id in provider._prepared_layers
    provider.close()


def test_layer_span_registers_tmix_batch_not_per_tensor_dict(
    synthetic_pack_trinity_lut2: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_id = next(i for i in sorted(by_layer.keys()) if i >= 0)
    entries = by_layer[layer_id]
    att_entries = [e for e in entries if ".att." in e.name and e.name.endswith(".weight")]
    if len(att_entries) < 4:
        pytest.skip("synthetic pack missing att weights")
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
    )
    out = provider.load_layer_tensors(entries)
    att_prefix = f"blocks.{layer_id}.att."
    assert provider.get_fused_tmix_blobs(att_prefix) is not None
    for suffix in (
        "receptance.weight",
        "key.weight",
        "value.weight",
        "output.weight",
    ):
        assert att_prefix + suffix not in out
    provider.close()


def test_strict_fused_prepared_omits_fused_weight_slabs(
    synthetic_pack_trinity_lut2: Path, monkeypatch
) -> None:
    """Strict fused retain must not keep bf16 att/ffn/head slabs in ``_prepared_layers``."""
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    monkeypatch.setenv("RWKV_STRICT_FUSED_RETAIN", "1")
    from rwkv_ssd.runtime.lut_gemm_fused import is_fused_lut_tensor_name
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.weight_store import open_weight_store

    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    by_layer = manifest.by_layer()
    layer_ids = sorted(i for i in by_layer.keys() if i >= 0)
    store = open_weight_store(manifest.weights_path, backend="mmap")
    provider = ManifestWeightProvider(
        mode="streaming",
        store=store,
        entries=manifest.tensors,
        device=torch.device("cpu"),
        metrics=MetricsCollector(),
        stream_layer_cache=False,
    )
    try:
        for layer_id in layer_ids[:4]:
            entries = by_layer[layer_id]
            tensors = provider.load_layer_tensors(entries)
            prepared = provider.prepare_layer_for_z(layer_id, tensors)
            provider._retain_provider_layer(layer_id)
            fused = [k for k in prepared if is_fused_lut_tensor_name(k)]
            assert fused == [], f"layer {layer_id} prepared fused keys: {fused}"
        slab_bytes = sum(
            t.numel() * t.element_size()
            for prep in provider._prepared_layers.values()
            for k, t in prep.items()
            if is_fused_lut_tensor_name(k)
        )
        assert slab_bytes == 0
    finally:
        provider.close()


def test_packed_block_forward_module_imports() -> None:
    from rwkv_ssd.runtime.packed_block_forward import (
        forward_block_packed,
        packed_block_forward_enabled,
    )

    assert callable(forward_block_packed)
    assert packed_block_forward_enabled(None, "blocks.0.att.") is False
