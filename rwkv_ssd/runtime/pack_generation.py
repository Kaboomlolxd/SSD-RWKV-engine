"""Pack-driven greedy generation with optional prefix state cache."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rwkv_ssd.runtime.mtp_spec import MTPConfig
from rwkv_ssd.runtime.transcript_cache import longest_cached_prefix, split_transcript
from rwkv_ssd.runtime.generation_control import make_generation_control

if TYPE_CHECKING:
    from rwkv_ssd.backends.pack_backend import PackBackend
    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider


def _prefill_with_cache(
    backend: PackBackend,
    provider: ManifestWeightProvider,
    metrics: MetricsCollector,
    *,
    prefix_cache: PrefixStateCache,
    cache_key: str,
    stable_prefix: str,
    suffix: str,
    cancel_event=None,
    deadline: float | None = None,
) -> RecurrentState:
    control = make_generation_control(
        cancel_event=cancel_event,
        deadline=deadline,
    )
    if control is not None:
        control.check()
    cached = prefix_cache.get(cache_key) if cache_key else None
    t_prefill = time.perf_counter()
    if cached is not None:
        metrics.state_cache_hit = True
        if suffix:
            state = backend.prefill_text(
                suffix,
                provider,
                metrics,
                initial_state=cached,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        else:
            state = cached
        prefix_cache.stats.prefill_ms_saved += metrics.prefill_wall_s
    else:
        metrics.state_cache_hit = False
        if stable_prefix:
            state = backend.prefill_text(
                stable_prefix,
                provider,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            if cache_key:
                prefix_cache.put(cache_key, state)
            if suffix:
                state = backend.prefill_text(
                    suffix,
                    provider,
                    metrics,
                    initial_state=state,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
        else:
            state = backend.prefill_text(
                suffix or stable_prefix,
                provider,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
    if control is not None:
        control.check()
    metrics.prefill_wall_s = time.perf_counter() - t_prefill
    return state


def _resolve_transcript_split(
    prompt: str,
    prefix_cache: PrefixStateCache,
) -> tuple[str, str, str]:
    cache_key, stable, suffix = longest_cached_prefix(
        prompt,
        has_entry=prefix_cache.contains,
    )
    if cache_key:
        return cache_key, stable, suffix
    return split_transcript(prompt)


def generate_greedy_tokens(
    backend: PackBackend,
    provider: ManifestWeightProvider,
    prompt: str,
    max_tokens: int,
    metrics: MetricsCollector,
    *,
    config: EngineConfig,
    prefix_cache: PrefixStateCache | None,
    token_callback=None,
    cancel_event=None,
    deadline: float | None = None,
) -> list[int]:
    if prefix_cache is not None and config.prefix_cache_mode == "transcript":
        cache_key, stable, suffix = _resolve_transcript_split(prompt, prefix_cache)
        if cache_key or stable or suffix:
            state = _prefill_with_cache(
                backend,
                provider,
                metrics,
                prefix_cache=prefix_cache,
                cache_key=cache_key,
                stable_prefix=stable,
                suffix=suffix,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            token_ids = backend.decode_greedy(
                state,
                provider,
                max_tokens,
                metrics,
                temperature=config.temperature if not config.greedy else 0.0,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        else:
            token_ids = backend.generate(
                prompt,
                provider,
                max_tokens,
                metrics,
                temperature=config.temperature,
                greedy=config.greedy,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
    elif config.system_prefix and prefix_cache is not None:
        if not prompt.startswith(config.system_prefix):
            prompt = config.system_prefix + prompt
        user = prompt[len(config.system_prefix) :]
        cache_key = config.system_prefix
        state = _prefill_with_cache(
            backend,
            provider,
            metrics,
            prefix_cache=prefix_cache,
            cache_key=cache_key,
            stable_prefix=config.system_prefix,
            suffix=user,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        token_ids = backend.decode_greedy(
            state,
            provider,
            max_tokens,
            metrics,
            temperature=config.temperature if not config.greedy else 0.0,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
    else:
        token_ids = backend.generate(
            prompt,
            provider,
            max_tokens,
            metrics,
            temperature=config.temperature,
            greedy=config.greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    if config.mtp_speculative:
        mtp = MTPConfig(
            enabled=True,
            draft_tokens=config.mtp_draft_tokens,
            min_prefill_tokens=config.mtp_min_prefill_tokens,
        )
        prefill_tokens = len(prompt.encode("utf-8"))
        if mtp.should_speculate(
            prefill_tokens=prefill_tokens, decode_tokens=max_tokens
        ):
            metrics.mtp_gate_open = 1

    stats = provider.cache_stats()
    metrics.cache_format = str(stats["cache_format"])
    metrics.provider_cache_bytes = int(stats["provider_cache_bytes"])
    metrics.provider_layer_cache_bytes = int(
        stats.get("provider_layer_cache_bytes", 0)
    )
    metrics.provider_resident_bytes = int(stats.get("provider_resident_bytes", 0))
    metrics.provider_mmap_bytes = int(stats.get("provider_mmap_bytes", 0))
    metrics.packed_cache_bytes = int(stats["packed_cache_bytes"])
    metrics.prepared_cache_bytes = int(stats["prepared_cache_bytes"])
    metrics.lut2_index_cache_bytes = int(stats["lut2_index_cache_bytes"])
    metrics.packed_cache_evictions = int(stats["packed_cache_evictions"])
    metrics.provider_cache_evictions = int(stats.get("provider_cache_evictions", 0))
    total_cmix = int(stats.get("cmix_total_elements", 0))
    metrics.cmix_samples = int(stats.get("cmix_samples", 0))
    metrics.cmix_zero_fraction = (
        int(stats.get("cmix_zero_elements", 0)) / total_cmix if total_cmix else 0.0
    )
    metrics.cmix_active_fraction = (
        int(stats.get("cmix_active_elements", 0)) / total_cmix if total_cmix else 0.0
    )
    metrics.cmix_tile_size = int(stats.get("cmix_tile_size", 0))
    metrics.cmix_tile_samples = int(stats.get("cmix_tile_samples", 0))
    metrics.cmix_tile_active_fraction = float(
        stats.get("cmix_tile_active_fraction", 0.0)
    )
    metrics.cmix_tile_occupancy = dict(stats.get("cmix_tile_occupancy", {}))
    metrics.cmix_selective_tiles_read = int(stats.get("cmix_selective_tiles_read", 0))
    metrics.cmix_selective_tiles_skipped = int(stats.get("cmix_selective_tiles_skipped", 0))
    metrics.cmix_selective_bytes_read = int(stats.get("cmix_selective_bytes_read", 0))
    metrics.cmix_prefetch_hits = int(stats.get("cmix_prefetch_hits", 0))
    metrics.cmix_prefetch_misses = int(stats.get("cmix_prefetch_misses", 0))
    metrics.cmix_prefetch_wasted_tiles = int(stats.get("cmix_prefetch_wasted_tiles", 0))
    metrics.cmix_prefetch_bytes_submitted = int(stats.get("cmix_prefetch_bytes_submitted", 0))
    metrics.cmix_hot_cache_hits = int(stats.get("cmix_hot_cache_hits", 0))
    metrics.cmix_hot_cache_bytes = int(stats.get("cmix_hot_cache_bytes", 0))
    if not metrics.weight_cache_bytes:
        metrics.weight_cache_bytes = metrics.provider_cache_bytes
    return token_ids


def generate_greedy_token_stream(
    backend: PackBackend,
    provider: ManifestWeightProvider,
    prompt: str,
    max_tokens: int,
    metrics: MetricsCollector,
    *,
    config: EngineConfig,
    prefix_cache: PrefixStateCache | None,
    token_callback=None,
    cancel_event=None,
    deadline: float | None = None,
) -> Iterator[int]:
    token_ids = generate_greedy_tokens(
        backend,
        provider,
        prompt,
        max_tokens,
        metrics,
        config=config,
        prefix_cache=prefix_cache,
        token_callback=token_callback,
        cancel_event=cancel_event,
        deadline=deadline,
    )
    yield from token_ids
