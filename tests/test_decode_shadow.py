"""Selective bf16 shadow and hybrid decode."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from rwkv_ssd.runtime.decode_shadow import (
    decode_shadow_layer_from_span,
    entry_has_shadow,
    layer_any_shadow,
    layer_has_shadow,
    split_shadow_lut_entries,
)
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack


def test_shadow_contiguous_and_decode(tmp_path: Path) -> None:
    pack = create_synthetic_pack(
        tmp_path / "pack",
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        bf16_shadow=True,
        quiet=True,
    )
    manifest = Manifest.load(pack)
    assert manifest.has_bf16_shadow()
    entries = [e for e in manifest.tensors if e.layer_id == 0]
    assert layer_has_shadow(entries)
    from rwkv_ssd.runtime.decode_shadow import (
        entries_shadow_contiguous_span,
        entries_shadow_layer_read_span,
    )

    span = entries_shadow_contiguous_span(entries)
    assert span is not None
    layer_span = entries_shadow_layer_read_span(entries)
    assert layer_span is not None
    assert layer_span[0] <= span[0]
    assert layer_span[1] >= span[1]
    base, total = layer_span
    from rwkv_ssd.runtime.weight_store import open_weight_store

    with open_weight_store(manifest.shadow_path()) as store:
        raw = store.read_bytearray_span(base, total)
    out = decode_shadow_layer_from_span(raw, entries, base, torch.device("cpu"))
    assert len(out) == len(entries)


def test_selective_shadow_hybrid_decode(tmp_path: Path) -> None:
    """Large tensors shadowed; small tensors stay LUT-only."""
    from rwkv_ssd.runtime.bf16_shadow import write_bf16_shadow
    from rwkv_ssd.tools.make_synthetic_pack import build_synthetic_state
    from rwkv_ssd.tools.pack_runtime import _layer_id_from_name, pack

    pack_dir = tmp_path / "selective"
    ckpt = pack_dir / "src.pt"
    pack_dir.mkdir(parents=True, exist_ok=True)
    state = build_synthetic_state(n_layer=2, n_embd=32)
    torch.save(state, ckpt)
    pack(
        ckpt,
        pack_dir,
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        quiet=True,
    )
    import json

    meta_path = pack_dir / "manifest.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors_meta: list[dict] = raw["tensors"]
    from rwkv_ssd.runtime.pack_layout import sort_tensor_names

    names = sort_tensor_names(list(state.keys()), "layer_grouped", _layer_id_from_name)
    write_bf16_shadow(
        state,
        names,
        pack_dir,
        tensors_meta,
        layer_id_fn=_layer_id_from_name,
        shadow_min_numel=4096,
        quiet=True,
    )
    raw["tensors"] = tensors_meta
    from rwkv_ssd.runtime.bf16_shadow import SHADOW_FILENAME

    raw["meta"]["shadow_file"] = SHADOW_FILENAME
    meta_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    manifest = Manifest.load(pack_dir)
    layer0 = [e for e in manifest.tensors if e.layer_id == 0]
    shadow, lut = split_shadow_lut_entries(layer0)
    assert layer_any_shadow(layer0)
    assert not layer_has_shadow(layer0)
    assert shadow and lut
    for e in shadow:
        assert entry_has_shadow(e)
        assert e.numel >= 4096
    for e in lut:
        assert not entry_has_shadow(e)

    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
    from rwkv_ssd.runtime.weight_store import open_weight_store

    with open_weight_store(manifest.weights_path) as store, open_weight_store(
        manifest.shadow_path()
    ) as shadow_store:
        provider = ManifestWeightProvider(
            mode="streaming",
            store=store,
            entries=manifest.tensors,
            device=torch.device("cpu"),
            metrics=MetricsCollector(),
            shadow_store=shadow_store,
            pack_dir=pack_dir,
            manifest_meta=manifest.meta,
            prefetch=False,
            stream_layer_cache=False,
        )
        try:
            decoded = provider.load_layer_tensors(layer0)
            assert len(decoded) == len(layer0)
            for name, tensor in decoded.items():
                assert tensor.numel() > 0
        finally:
            provider.close()


def test_shadow_weight_retain_skips_second_decode(tmp_path: Path) -> None:
    """Strict fused retain keeps shadow bf16 weights for matvec without re-decode."""
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider
    from rwkv_ssd.runtime.weight_store import open_weight_store
    from rwkv_ssd.tools.make_synthetic_pack import build_synthetic_state
    from rwkv_ssd.tools.pack_runtime import _layer_id_from_name, pack

    pack_dir = tmp_path / "retain"
    ckpt = pack_dir / "src.pt"
    pack_dir.mkdir(parents=True, exist_ok=True)
    state = build_synthetic_state(n_layer=2, n_embd=64)
    torch.save(state, ckpt)
    pack(
        ckpt,
        pack_dir,
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        bf16_shadow=True,
        shadow_min_numel=256,
        quiet=True,
    )
    manifest = Manifest.load(pack_dir)
    layer0 = [e for e in manifest.tensors if e.layer_id == 0]
    metrics = MetricsCollector()
    with open_weight_store(manifest.weights_path) as store, open_weight_store(
        manifest.shadow_path()
    ) as shadow_store:
        provider = ManifestWeightProvider(
            mode="streaming",
            store=store,
            entries=manifest.tensors,
            device=torch.device("cpu"),
            metrics=metrics,
            shadow_store=shadow_store,
            pack_dir=pack_dir,
            manifest_meta=manifest.meta,
            prefetch=False,
            stream_layer_cache=False,
        )
        try:
            os.environ["RWKV_LUT_GEMM_FUSED"] = "1"
            os.environ["RWKV_STRICT_FUSED_RETAIN"] = "1"
            os.environ["RWKV_STRICT_FUSED_RETAIN_SHADOW"] = "1"
            first = provider.load_layer_tensors(layer0)
            prepared = provider.prepare_layer_for_z(0, first, metrics.start_layer(0))
            provider.release_prepared_tensors(0)
            kept = provider._prepared_layers[0]
            assert any(provider._retainable_shadow_weight_key(k) for k in kept.keys())
            metrics.layers.clear()
            second = provider.load_layer_tensors(layer0)
            assert metrics.layers[-1].layer_cache_hits == len(layer0)
            assert len(second) == len(layer0)
        finally:
            provider.close()
            os.environ.pop("RWKV_LUT_GEMM_FUSED", None)
            os.environ.pop("RWKV_STRICT_FUSED_RETAIN", None)
            os.environ.pop("RWKV_STRICT_FUSED_RETAIN_SHADOW", None)
