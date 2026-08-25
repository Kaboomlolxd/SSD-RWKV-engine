"""I/O-only prefetch for quant packs — overlap read without worker-thread decode."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_provider import (
    ManifestWeightProvider,
    resolve_prefetch_io_only,
    pack_uses_quant_codec,
)
from rwkv_ssd.runtime.weight_store import open_weight_store


def test_resolve_prefetch_io_only_auto_cpu_trinity(
    synthetic_pack_trinity_lut2: Path,
) -> None:
    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    assert pack_uses_quant_codec(manifest.tensors)
    assert resolve_prefetch_io_only(
        None,
        device=__import__("torch").device("cpu"),
        entries=manifest.tensors,
        mode="streaming",
    )


def test_resolve_prefetch_io_only_off_for_fp16(synthetic_pack: Path) -> None:
    manifest = Manifest.load(synthetic_pack)
    assert not pack_uses_quant_codec(manifest.tensors)
    assert not resolve_prefetch_io_only(
        None,
        device=__import__("torch").device("cpu"),
        entries=manifest.tensors,
        mode="streaming",
    )


def test_io_only_prefetch_stores_raw_not_decoded(
    synthetic_pack_trinity_lut2: Path,
) -> None:
    manifest = Manifest.load(synthetic_pack_trinity_lut2)
    import torch

    from rwkv_ssd.runtime.metrics import MetricsCollector

    device = torch.device("cpu")
    with open_weight_store(manifest.weights_path) as store:
        provider = ManifestWeightProvider(
            mode="streaming",
            store=store,
            entries=manifest.tensors,
            device=device,
            metrics=MetricsCollector(),
            prefetch=True,
            prefetch_io_only=True,
            stream_layer_cache=False,
        )
        try:
            layer_entries = [
                e for e in manifest.tensors if e.layer_id == 0 and e.name.startswith("blocks.")
            ]
            if not layer_entries:
                pytest.skip("no block tensors on layer 0")
            provider.prefetch_entries(layer_entries)
            fut = provider._prefetch_future
            assert fut is not None
            result = fut.result(timeout=30)
            assert result.tensors == {}
            assert 0 in result.raw_by_layer
            spans = result.raw_by_layer[0]
            assert spans.weights is not None
            raw, base = spans.weights
            assert len(raw) > 0
            decoded = provider._decode_stream_entries(layer_entries)
            assert decoded
        finally:
            provider.close()
