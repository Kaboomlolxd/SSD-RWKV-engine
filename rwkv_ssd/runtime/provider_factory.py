"""Construct weight providers with consistent defaults."""

from __future__ import annotations

from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.trinity_accel import resolve_trinity_decode_device
from rwkv_ssd.runtime.layer_io import merge_layer_entries
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.prefetch import PrefetchPlanner, make_prefetch_planner
from rwkv_ssd.runtime.staging import PingPongStaging
from rwkv_ssd.runtime.weight_provider import (
    ManifestWeightProvider,
    pack_uses_quant_codec,
)
from rwkv_ssd.runtime.weight_store_base import WeightStore


def create_weight_provider(
    config: EngineConfig,
    store: WeightStore,
    entries: list[TensorEntry],
    device: torch.device,
    metrics: MetricsCollector,
    staging: PingPongStaging | None = None,
    *,
    model_z: dict | None = None,
    shadow_store: WeightStore | None = None,
    pack_dir: Path | None = None,
    manifest_meta: dict | None = None,
    native_layer_streaming: bool = False,
) -> ManifestWeightProvider:
    from rwkv_ssd.runtime.layer_keys import layer_ids
    from rwkv_ssd.runtime.stream_cache_policy import (
        resolve_max_layers_in_z,
        resolve_max_provider_cache_layers,
    )
    from rwkv_ssd.runtime.z_layer_retention import snapshot_pinned_layers

    n_block = len(layer_ids(entries))
    max_z = resolve_max_layers_in_z(
        config.max_layers_in_z,
        n_block,
        stream_layer_cache=config.stream_layer_cache,
        warm_z=config.warm_z,
    )
    pinned = (
        snapshot_pinned_layers(
            model_z,
            n_block_layers=n_block,
            max_layers_in_z=max_z,
        )
        if model_z
        else set()
    )
    budget_limited = bool(
        (config.ram_budget_gb and config.ram_budget_gb > 0)
        or (config.low_ram and config.max_provider_cache_layers > 0)
    )
    max_provider = resolve_max_provider_cache_layers(
        config.max_provider_cache_layers,
        n_block,
        max_layers_in_z=max_z,
        stream_layer_cache=config.stream_layer_cache,
        decouple_provider_cache=config.decouple_provider_cache,
        warm_z=config.warm_z,
        budget_limited=budget_limited,
    )
    planner = make_prefetch_planner(config.prefetch_policy)
    decode_device = resolve_trinity_decode_device(
        config.trinity_decode_device, fallback=device
    )
    if decode_device.type != "cpu":
        import logging

        logging.getLogger(__name__).info(
            "Trinity LUT decode on %s (weights inject to %s)",
            decode_device,
            device,
        )
    return ManifestWeightProvider(
        mode=config.mode,
        store=store,
        entries=entries,
        device=device,
        decode_device=decode_device,
        metrics=metrics,
        staging=staging,
        prefetch=config.prefetch_enabled,
        chunk_bytes=config.io_chunk_bytes,
        chunk_policy=config.io_chunk_policy,
        planner=planner,
        stream_layer_cache=config.stream_layer_cache,
        warm_z=config.warm_z,
        max_layers_in_z=max_z,
        max_provider_cache_layers=max_provider,
        max_provider_cache_bytes=(
            config.prepared_cache_bytes
            if config.prepared_cache_bytes > 0
            else config.max_provider_cache_bytes
        ),
        cache_format=config.cache_format,
        max_packed_cache_bytes=config.packed_cache_bytes,
        decouple_provider_cache=config.decouple_provider_cache,
        decode_disk_cache=config.decode_disk_cache,
        pinned_layer_ids=pinned,
        model_z=model_z if isinstance(model_z, dict) else None,
        ngram_weight_cache=config.ngram_weight_cache,
        mmap_willneed=config.mmap_willneed,
        mmap_dontneed=config.mmap_dontneed,
        shadow_store=shadow_store,
        pack_dir=pack_dir,
        manifest_meta=manifest_meta,
        prefetch_io_only=config.prefetch_io_only,
        native_layer_streaming=native_layer_streaming,
    )


def prefetch_ahead(
    provider: ManifestWeightProvider,
    planner: PrefetchPlanner,
    layer_ids: list[int],
    current_index: int,
    by_layer: dict[int, list[TensorEntry]],
    *,
    z: dict | None = None,
) -> None:
    """Issue batched prefetch for all layers in the planner's plan."""
    from rwkv_ssd.runtime.rwkv7_weights import layer_weights_in_z

    recent = provider.recent_layer_timings(limit=4)
    plan = planner.plan(layer_ids, current_index, recent)
    if not plan.layer_ids:
        return
    target_ids: list[int] = []
    for layer_id in plan.layer_ids:
        if z is not None and layer_weights_in_z(z, layer_id):
            continue
        # Provider already holds prepared skeleton / LUT blobs — prefetch
        # would only contend with decode I/O and get abandoned.
        if provider.layer_prepared_for_forward(layer_id):
            continue
        target_ids.append(layer_id)
        provider.hint_prefetch_layer(by_layer.get(layer_id, []))
    if not target_ids:
        return
    entries = merge_layer_entries(by_layer, target_ids)
    provider.prefetch_entries(entries)


def release_layer(
    provider: ManifestWeightProvider,
    entries: list[TensorEntry],
) -> None:
    """Drop OS page-cache hold on a consumed layer (Linux madvise)."""
    provider.hint_release_layer(entries)
