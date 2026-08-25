"""Per-layer and per-step timing for the inference engine."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path


def _percentile(values: list[float], percentile: float) -> float:
    """Small dependency-free percentile helper for request telemetry."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


@dataclass
class LayerTiming:
    layer_id: int
    read_ms: float = 0.0
    staging_ms: float = 0.0
    h2d_ms: float = 0.0
    # Accelerator decode breakdowns.  These are informational subcomponents
    # of staging_ms and are intentionally not added to total_ms.
    decode_device_ms: float = 0.0
    d2h_ms: float = 0.0
    compute_ms: float = 0.0
    bubble_ms: float = 0.0
    prefetch_wait_ms: float = 0.0
    prefetch_hits: int = 0
    chunk_reads: int = 0
    layer_cache_hits: int = 0
    ngram_hits: int = 0
    shadow_hits: int = 0
    disk_cache_hits: int = 0
    bytes_read: int = 0

    @property
    def total_ms(self) -> float:
        return (
            self.read_ms
            + self.staging_ms
            + self.h2d_ms
            + self.compute_ms
            + self.bubble_ms
        )


@dataclass
class MetricsCollector:
    layers: list[LayerTiming] = field(default_factory=list)
    prompt_tokens: int = 0
    tokens_generated: int = 0
    total_wall_s: float = 0.0
    prefill_wall_s: float = 0.0
    decode_wall_s: float = 0.0
    ttft_s: float = 0.0
    token_latencies_ms: list[float] = field(default_factory=list)
    queue_wait_s: float = 0.0
    state_cache_hit: bool = False
    prefetch_overlaps: int = 0
    weight_cache_bytes: int = 0
    provider_cache_bytes: int = 0
    provider_layer_cache_bytes: int = 0
    provider_resident_bytes: int = 0
    provider_mmap_bytes: int = 0
    cache_format: str = "auto"
    packed_cache_bytes: int = 0
    prepared_cache_bytes: int = 0
    lut2_index_cache_bytes: int = 0
    packed_cache_evictions: int = 0
    provider_cache_evictions: int = 0
    cmix_zero_fraction: float = 0.0
    cmix_active_fraction: float = 0.0
    cmix_samples: int = 0
    cmix_tile_size: int = 0
    cmix_tile_samples: int = 0
    cmix_tile_active_fraction: float = 0.0
    cmix_tile_occupancy: dict[str, int] = field(default_factory=dict)
    cmix_selective_tiles_read: int = 0
    cmix_selective_tiles_skipped: int = 0
    cmix_selective_bytes_read: int = 0
    cmix_prefetch_hits: int = 0
    cmix_prefetch_misses: int = 0
    cmix_prefetch_wasted_tiles: int = 0
    cmix_prefetch_bytes_submitted: int = 0
    cmix_hot_cache_hits: int = 0
    cmix_hot_cache_bytes: int = 0
    z_bytes: int = 0
    mtp_gate_open: int = 0
    cache_write_submits: int = 0
    cache_write_sync_ms: float = 0.0
    power_percent: int = 100
    batch_size: int = 1
    weight_sweeps: int = 0
    weight_layer_loads: int = 0
    page_cache_prefetch_skips: int = 0
    codec_deadline_timeouts: int = 0
    codec_deadline_fallbacks: int = 0
    session_promoted_layers: list[int] = field(default_factory=list)
    session_promotion_bytes: int = 0
    session_promotion_ms: float = 0.0
    session_promotion_estimated_net_ms: float = 0.0
    session_expected_remaining_tokens: int = 0
    process_rss_bytes: int = 0
    process_rss_peak_bytes: int = 0
    memory_budget_bytes: int = 0
    memory_budget_exceeded: bool = False
    streamed_bytes: int = 0
    native_upload_bytes: int = 0
    # Native bridge byte domains are intentionally separate.  ``native_upload``
    # is the payload passed from Python into the ABI, ``native_packed`` is the
    # packed provider representation within that payload, and ``native_active``
    # is the decoded byte count copied into the active GGML layer tensor.
    native_packed_bytes: int = 0
    native_active_bytes: int = 0
    native_evictions: int = 0
    native_queue_wait_ms: float = 0.0
    native_layer_cache_bytes: int = 0
    native_layer_cache_hits: int = 0
    native_layer_cache_misses: int = 0
    native_layer_cache_evictions: int = 0
    # A ready hit means the complete active layer was restored before provider
    # decode/materialization, so it is the cache counter that represents work
    # avoided at the provider boundary.  Tensor-level cache hits remain in
    # ``native_layer_cache_hits``.
    native_decoded_cache_hits: int = 0
    native_layer_streaming: bool = False
    cold_measurement: bool = False
    batch_prefill_wall_s: float = 0.0
    batch_decode_wall_s: float = 0.0

    def start_layer(self, layer_id: int) -> LayerTiming:
        row = LayerTiming(layer_id=layer_id)
        self.layers.append(row)
        return row

    def reset_for_generate(self) -> None:
        """Clear per-call timing so repeated ``generate()`` does not accumulate."""
        self.layers.clear()
        self.prompt_tokens = 0
        self.tokens_generated = 0
        self.total_wall_s = 0.0
        self.prefill_wall_s = 0.0
        self.decode_wall_s = 0.0
        self.ttft_s = 0.0
        self.token_latencies_ms = []
        self.queue_wait_s = 0.0
        self.state_cache_hit = False
        self.prefetch_overlaps = 0
        self.mtp_gate_open = 0
        self.batch_size = 1
        self.weight_sweeps = 0
        self.weight_layer_loads = 0
        self.page_cache_prefetch_skips = 0
        self.codec_deadline_timeouts = 0
        self.codec_deadline_fallbacks = 0
        self.session_promoted_layers = []
        self.session_promotion_bytes = 0
        self.session_promotion_ms = 0.0
        self.session_promotion_estimated_net_ms = 0.0
        self.session_expected_remaining_tokens = 0
        self.process_rss_bytes = 0
        self.process_rss_peak_bytes = 0
        self.memory_budget_bytes = 0
        self.memory_budget_exceeded = False
        self.streamed_bytes = 0
        self.native_upload_bytes = 0
        self.native_packed_bytes = 0
        self.native_active_bytes = 0
        self.native_evictions = 0
        self.native_queue_wait_ms = 0.0
        self.native_layer_cache_bytes = 0
        self.native_layer_cache_hits = 0
        self.native_layer_cache_misses = 0
        self.native_layer_cache_evictions = 0
        self.native_decoded_cache_hits = 0
        self.native_layer_streaming = False
        self.cold_measurement = False
        self.batch_prefill_wall_s = 0.0
        self.batch_decode_wall_s = 0.0
        self.cache_format = "auto"
        self.provider_layer_cache_bytes = 0
        self.provider_resident_bytes = 0
        self.provider_mmap_bytes = 0
        self.packed_cache_bytes = 0
        self.prepared_cache_bytes = 0
        self.lut2_index_cache_bytes = 0
        self.packed_cache_evictions = 0
        self.provider_cache_evictions = 0
        self.cmix_zero_fraction = 0.0
        self.cmix_active_fraction = 0.0
        self.cmix_samples = 0
        self.cmix_tile_size = 0
        self.cmix_tile_samples = 0
        self.cmix_tile_active_fraction = 0.0
        self.cmix_tile_occupancy = {}
        self.cmix_selective_tiles_read = 0
        self.cmix_selective_tiles_skipped = 0
        self.cmix_selective_bytes_read = 0
        self.cmix_prefetch_hits = 0
        self.cmix_prefetch_misses = 0
        self.cmix_prefetch_wasted_tiles = 0
        self.cmix_prefetch_bytes_submitted = 0
        self.cmix_hot_cache_hits = 0
        self.cmix_hot_cache_bytes = 0
        # Keep cache_write_* cumulative across calls (process-lifetime writers).

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "layer_id",
                    "read_ms",
                    "bytes_read",
                    "staging_ms",
                    "h2d_ms",
                    "decode_device_ms",
                    "d2h_ms",
                    "compute_ms",
                    "bubble_ms",
                    "prefetch_wait_ms",
                    "prefetch_hits",
                    "chunk_reads",
                    "layer_cache_hits",
                    "ngram_hits",
                    "shadow_hits",
                    "disk_cache_hits",
                    "total_ms",
                ]
            )
            for L in self.layers:
                w.writerow(
                    [
                        L.layer_id,
                        f"{L.read_ms:.4f}",
                        L.bytes_read,
                        f"{L.staging_ms:.4f}",
                        f"{L.h2d_ms:.4f}",
                        f"{L.decode_device_ms:.4f}",
                        f"{L.d2h_ms:.4f}",
                        f"{L.compute_ms:.4f}",
                        f"{L.bubble_ms:.4f}",
                        f"{L.prefetch_wait_ms:.4f}",
                        L.prefetch_hits,
                        L.chunk_reads,
                        L.layer_cache_hits,
                        L.ngram_hits,
                        L.shadow_hits,
                        L.disk_cache_hits,
                        f"{L.total_ms:.4f}",
                    ]
                )

    def observe_memory(self, *, budget_bytes: int = 0) -> None:
        from rwkv_ssd.runtime.memory_usage import process_rss_bytes

        current = process_rss_bytes()
        self.process_rss_bytes = current
        self.process_rss_peak_bytes = max(self.process_rss_peak_bytes, current)
        self.memory_budget_bytes = max(0, int(budget_bytes))
        self.memory_budget_exceeded = bool(
            self.memory_budget_bytes and self.process_rss_peak_bytes > self.memory_budget_bytes
        )

    def summary(self) -> str:
        if not self.layers:
            return "no layer timings recorded"
        n = len(self.layers)
        read = sum(x.read_ms for x in self.layers) / n
        h2d = sum(x.h2d_ms for x in self.layers) / n
        decode_device = sum(x.decode_device_ms for x in self.layers) / n
        d2h = sum(x.d2h_ms for x in self.layers) / n
        compute = sum(x.compute_ms for x in self.layers) / n
        lookahead = sum(x.prefetch_wait_ms for x in self.layers) / n
        hits = sum(x.prefetch_hits for x in self.layers)
        chunks = sum(x.chunk_reads for x in self.layers)
        lcache = sum(x.layer_cache_hits for x in self.layers)
        ngram = sum(x.ngram_hits for x in self.layers)
        tok_s = (
            self.tokens_generated / self.total_wall_s if self.total_wall_s > 0 else 0.0
        )
        cache = " hit" if self.state_cache_hit else ""
        prefill = f" prefill={self.prefill_wall_s:.3f}s" if self.prefill_wall_s else ""
        return (
            f"layers={n} avg_read_ms={read:.2f} avg_h2d_ms={h2d:.2f} "
            f"avg_decode_device_ms={decode_device:.2f} avg_d2h_ms={d2h:.2f} "
            f"avg_compute_ms={compute:.2f} avg_prefetch_wait_ms={lookahead:.2f} "
            f"tok/s={tok_s:.2f} prefetch_overlaps={self.prefetch_overlaps} "
            f"prefetch_hits={hits} chunk_reads={chunks} "
            f"layer_cache_hits={lcache} ngram_hits={ngram} "
            f"mtp_gate_open={self.mtp_gate_open} "
            f"packed_cache={self.packed_cache_bytes}B "
            f"prepared_cache={self.prepared_cache_bytes}B "
            f"provider_layer_cache={self.provider_layer_cache_bytes}B "
            f"provider_resident={self.provider_resident_bytes}B "
            f"provider_mmap={self.provider_mmap_bytes}B "
            f"cache_format={self.cache_format} "
            f"cmix_zero={self.cmix_zero_fraction:.3f} "
            f"cmix_active={self.cmix_active_fraction:.3f} "
            f"cmix_samples={self.cmix_samples} "
            f"weight_cache={self.weight_cache_bytes}B{cache} "
            f"cache_writes={self.cache_write_submits} "
            f"cache_sync_ms={self.cache_write_sync_ms:.1f}"
            f"{prefill}"
        )

    def to_dict(self) -> dict:
        """Structured metrics for HTTP/CLI responses (A4).

        Mirrors the dataclass fields plus per-layer timings. ``cache_stats``
        is filled in by the engine after the call, since ``PrefixStateCache``
        is owned outside this collector.
        """
        d: dict = {
            "prompt_tokens": self.prompt_tokens,
            "tokens_generated": self.tokens_generated,
            "total_wall_s": round(self.total_wall_s, 4),
            "prefill_wall_s": round(self.prefill_wall_s, 4)
            if self.prefill_wall_s
            else 0.0,
            "decode_wall_s": round(self.decode_wall_s, 4),
            "ttft_s": round(self.ttft_s, 4),
            "queue_wait_s": round(self.queue_wait_s, 4),
            "median_token_latency_ms": round(
                _percentile(self.token_latencies_ms, 50.0), 3
            ),
            "p95_token_latency_ms": round(
                _percentile(self.token_latencies_ms, 95.0), 3
            ),
            "state_cache_hit": self.state_cache_hit,
            "prefetch_overlaps": self.prefetch_overlaps,
            "weight_cache_bytes": self.weight_cache_bytes,
            "provider_cache_bytes": self.provider_cache_bytes,
            "provider_layer_cache_bytes": self.provider_layer_cache_bytes,
            "provider_resident_bytes": self.provider_resident_bytes,
            "provider_mmap_bytes": self.provider_mmap_bytes,
            "cache_format": self.cache_format,
            "packed_cache_bytes": self.packed_cache_bytes,
            "prepared_cache_bytes": self.prepared_cache_bytes,
            "lut2_index_cache_bytes": self.lut2_index_cache_bytes,
            "packed_cache_evictions": self.packed_cache_evictions,
            "provider_cache_evictions": self.provider_cache_evictions,
            "cmix_zero_fraction": round(self.cmix_zero_fraction, 6),
            "cmix_active_fraction": round(self.cmix_active_fraction, 6),
            "cmix_samples": self.cmix_samples,
            "cmix_tile_size": self.cmix_tile_size,
            "cmix_tile_samples": self.cmix_tile_samples,
            "cmix_tile_active_fraction": round(
                self.cmix_tile_active_fraction, 6
            ),
            "cmix_tile_occupancy": dict(self.cmix_tile_occupancy),
            "cmix_selective_tiles_read": self.cmix_selective_tiles_read,
            "cmix_selective_tiles_skipped": self.cmix_selective_tiles_skipped,
            "cmix_selective_bytes_read": self.cmix_selective_bytes_read,
            "cmix_prefetch_hits": self.cmix_prefetch_hits,
            "cmix_prefetch_misses": self.cmix_prefetch_misses,
            "cmix_prefetch_wasted_tiles": self.cmix_prefetch_wasted_tiles,
            "cmix_prefetch_bytes_submitted": self.cmix_prefetch_bytes_submitted,
            "cmix_hot_cache_hits": self.cmix_hot_cache_hits,
            "cmix_hot_cache_bytes": self.cmix_hot_cache_bytes,
            "z_bytes": self.z_bytes,
            "mtp_gate_open": self.mtp_gate_open,
            "cache_write_submits": self.cache_write_submits,
            "cache_write_sync_ms": round(self.cache_write_sync_ms, 3),
            "power_percent": self.power_percent,
            "batch_size": self.batch_size,
            "weight_sweeps": self.weight_sweeps,
            "weight_layer_loads": self.weight_layer_loads,
            "page_cache_prefetch_skips": self.page_cache_prefetch_skips,
            "codec_deadline_timeouts": self.codec_deadline_timeouts,
            "codec_deadline_fallbacks": self.codec_deadline_fallbacks,
            "session_promoted_layers": list(self.session_promoted_layers),
            "session_promotion_bytes": self.session_promotion_bytes,
            "session_promotion_ms": round(self.session_promotion_ms, 3),
            "session_promotion_estimated_net_ms": round(
                self.session_promotion_estimated_net_ms, 3
            ),
            "session_expected_remaining_tokens": self.session_expected_remaining_tokens,
            "process_rss_bytes": self.process_rss_bytes,
            "process_rss_peak_bytes": self.process_rss_peak_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "memory_budget_exceeded": self.memory_budget_exceeded,
            "streamed_bytes": max(
                self.streamed_bytes, sum(int(layer.bytes_read) for layer in self.layers)
            ),
            "cache_hits": sum(
                int(layer.prefetch_hits)
                + int(layer.layer_cache_hits)
                + int(layer.disk_cache_hits)
                for layer in self.layers
            ),
            "native_upload_bytes": self.native_upload_bytes,
            "native_packed_bytes": self.native_packed_bytes,
            "native_active_bytes": self.native_active_bytes,
            "native_evictions": self.native_evictions,
            "native_queue_wait_ms": round(self.native_queue_wait_ms, 3),
            "native_layer_cache_bytes": self.native_layer_cache_bytes,
            "native_layer_cache_hits": self.native_layer_cache_hits,
            "native_layer_cache_misses": self.native_layer_cache_misses,
            "native_layer_cache_evictions": self.native_layer_cache_evictions,
            "native_decoded_cache_hits": self.native_decoded_cache_hits,
            "native_layer_streaming": self.native_layer_streaming,
            "cold_measurement": self.cold_measurement,
            "batch_prefill_wall_s": round(self.batch_prefill_wall_s, 4),
            "batch_decode_wall_s": round(self.batch_decode_wall_s, 4),
            "tok_s": (
                round(self.tokens_generated / self.total_wall_s, 2)
                if self.total_wall_s > 0 and self.tokens_generated > 0
                else 0.0
            ),
            "layers": [
                {
                    "layer_id": L.layer_id,
                    "read_ms": round(L.read_ms, 3),
                    "bytes_read": L.bytes_read,
                    "staging_ms": round(L.staging_ms, 3),
                    "h2d_ms": round(L.h2d_ms, 3),
                    "decode_device_ms": round(L.decode_device_ms, 3),
                    "d2h_ms": round(L.d2h_ms, 3),
                    "compute_ms": round(L.compute_ms, 3),
                    "bubble_ms": round(L.bubble_ms, 3),
                    "prefetch_wait_ms": round(L.prefetch_wait_ms, 3),
                    "prefetch_hits": L.prefetch_hits,
                    "chunk_reads": L.chunk_reads,
                    "layer_cache_hits": L.layer_cache_hits,
                    "ngram_hits": L.ngram_hits,
                    "shadow_hits": L.shadow_hits,
                    "disk_cache_hits": L.disk_cache_hits,
                }
                for L in self.layers
            ],
        }
        return d


class Timer:
    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
