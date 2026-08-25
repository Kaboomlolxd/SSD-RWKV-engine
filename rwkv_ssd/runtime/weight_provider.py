"""Resident, partial, and streaming weight access."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rwkv_ssd.runtime.chunk_schedule import chunk_bytes_for_entry
from rwkv_ssd.runtime.layer_io import entries_contiguous_span, entries_layer_read_span
from rwkv_ssd.runtime.dequant import decode_weight_blob, decode_weight_to_tensor
from rwkv_ssd.runtime.trinity_codec import LayerZlibCache, decode_trinity_layer_span
from rwkv_ssd.runtime.io_chunked import DEFAULT_CHUNK_BYTES, read_tensor_chunked
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.metrics import LayerTiming, Timer
from rwkv_ssd.runtime.prefetch import NextLayerPlanner, PrefetchPlanner
from rwkv_ssd.runtime.staging import PingPongStaging
from rwkv_ssd.runtime.rwkv7_weights import prepare_rwkv7_tensor_for_z
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes
from rwkv_ssd.runtime.weight_store_base import WeightStore

if TYPE_CHECKING:
    from rwkv_ssd.runtime.metrics import MetricsCollector

logger = logging.getLogger(__name__)

_QUANT_CODECS = frozenset(
    {"trinity_lut2", "trinity", "trinity_layer", "scale_u8", "scale_u8_grouped", "scale_u4"}
)


@dataclass
class _LayerSpanPrefetch:
    """Prefetched raw spans for one layer (shadow and/or weights.bin)."""

    shadow: tuple[bytes | memoryview | bytearray, int] | None = None
    weights: tuple[bytes | memoryview | bytearray, int] | None = None


@dataclass
class _PrefetchJobResult:
    """Background prefetch: raw layer spans and/or decoded tensors."""

    raw_by_layer: dict[int, _LayerSpanPrefetch]
    tensors: dict[str, torch.Tensor]


def _materialize_span_buffer(raw: bytes | memoryview | bytearray) -> bytes | bytearray:
    """Detach mmap views so weight stores can close safely."""
    if isinstance(raw, memoryview):
        return bytearray(raw)
    return raw


_CODEC_POLICY_SENSITIVE_LAYERS = frozenset(
    {
        "head.weight",
        "lm_head.weight",
        "output.weight",
    }
)


def _codec_policy_entry_overrides(
    entry: TensorEntry,
    codec_policy: str,
) -> str | None:
    """Return a shadow-vs-LUT routing decision based on the runtime codec policy.

    The engine cannot swap the on-disk codec at runtime — ``trinity_lut2`` blobs
    are not interchangeable with ``scale_u8`` blobs in the same pack. The runtime
    policy therefore **routes** sensitive tensors to the bf16 shadow sidecar when
    available; otherwise the manifest ``dequant`` is used as-is.

    * ``strict`` / ``auto``: no override (use manifest ``dequant``).
    * ``accuracy``: prefer shadow for ``head``/``lm_head``/``output``.
    * ``hybrid``: prefer shadow for large tensors; LUT2 for small (<256K elems).

    Returns ``"shadow"`` when the policy routes to the shadow sidecar, else ``None``.
    """
    policy = (codec_policy or "").strip().lower()
    if policy in ("", "auto", "strict"):
        return None
    if policy == "accuracy" and entry.name in _CODEC_POLICY_SENSITIVE_LAYERS:
        return "shadow"
    if policy == "hybrid" and entry.numel >= 262144:
        return "shadow"
    return None


def pack_uses_quant_codec(entries: list[TensorEntry]) -> bool:
    for entry in entries:
        codec = (entry.dequant or "none").strip().lower()
        if codec in _QUANT_CODECS:
            return True
    return False


def resolve_prefetch_io_only(
    pref: bool | None,
    *,
    device: torch.device,
    entries: list[TensorEntry],
    mode: str,
) -> bool:
    """
    I/O-only prefetch overlaps SSD read with compute without decode on the worker.

    Auto-on for CPU streaming packs with Trinity / M5 quant codecs (decode-bound).
    """
    if pref is not None:
        return pref
    env = os.environ.get("RWKV_PREFETCH_IO_ONLY", "auto").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    return (
        mode in ("streaming", "partial")
        and device.type == "cpu"
        and pack_uses_quant_codec(entries)
    )


class WeightProvider:
    mode: str

    def load_layer_tensors(self, entries: list[TensorEntry]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_layer_tensors_dense(
        self, entries: list[TensorEntry]
    ) -> dict[str, torch.Tensor]:
        """Load a layer as ordinary dense tensors.

        RWKV providers may omit tensors that are consumed by a fused LUT
        injector.  Sequence backends need every projection explicitly, so
        this separate contract makes the consumer intent unambiguous.
        """
        return self.load_layer_tensors(entries)

    def begin_layer(self, layer_id: int) -> LayerTiming:
        raise NotImplementedError

    def end_layer(self, timing: LayerTiming) -> None:
        pass

    def prefetch_layer(self, entries: list[TensorEntry]) -> None:
        pass

    def close(self) -> None:
        pass


class ManifestWeightProvider(WeightProvider):
    """
    Single provider for resident | partial | streaming.

    - resident: preload all tensors
    - partial: preload resident-flag tensors only
    - streaming: read streamed tensors from disk; optional RAM cache (P2.c hot-layer)
    """

    def __init__(
        self,
        mode: str,
        store: WeightStore,
        entries: list[TensorEntry],
        device: torch.device,
        metrics: MetricsCollector,
        staging: PingPongStaging | None = None,
        prefetch: bool = True,
        chunk_bytes: int = 0,
        chunk_policy: str = "uniform",
        planner: PrefetchPlanner | None = None,
        stream_layer_cache: bool = False,
        warm_z: bool = False,
        max_layers_in_z: int = 1,
        pinned_layer_ids: set[int] | None = None,
        model_z: dict[str, torch.Tensor] | None = None,
        ngram_weight_cache: bool = False,
        mmap_willneed: bool = True,
        mmap_dontneed: bool = False,
        decode_device: torch.device | None = None,
        shadow_store: WeightStore | None = None,
        pack_dir: Path | None = None,
        manifest_meta: dict | None = None,
        decouple_provider_cache: bool = True,
        max_provider_cache_layers: int | None = None,
        max_provider_cache_bytes: int = 0,
        cache_format: str = "auto",
        max_packed_cache_bytes: int = 0,
        decode_disk_cache: str | None = None,
        prefetch_io_only: bool | None = None,
        native_layer_streaming: bool = False,
    ) -> None:
        self.mode = mode
        self._store = store
        self._device = device
        self._decode_device = decode_device or device
        self._metrics = metrics
        self._staging = staging
        chunk_bytes = max(0, int(chunk_bytes))
        chunk_policy = chunk_policy
        if (
            chunk_policy.strip().lower() not in ("", "uniform", "off", "none")
            and chunk_bytes <= 0
        ):
            chunk_bytes = DEFAULT_CHUNK_BYTES
        self._chunk_bytes = chunk_bytes
        self._chunk_policy = chunk_policy
        self._planner = planner or NextLayerPlanner()
        self._stream_layer_cache = stream_layer_cache
        # The native rwkv.cpp layer-local path owns the global activation/head
        # view after it has been converted for the native backend.  Keep its
        # provider accounting separate from the ordinary ChatRWKV/sequence
        # providers so a global tensor is not charged twice.
        self._native_layer_streaming = bool(native_layer_streaming)
        self._warm_z = warm_z
        cache_format = (cache_format or "auto").strip().lower()
        if cache_format not in {"auto", "none", "packed", "prepared", "dense"}:
            raise ValueError(f"unsupported cache_format: {cache_format!r}")
        self._cache_format = cache_format
        from rwkv_ssd.runtime.z_layer_retention import ZLayerRetention

        self._max_provider_cache_bytes = max(0, int(max_provider_cache_bytes))
        # An explicit byte budget also bounds partial-tier decoded layers.
        # Without this, pinned partial layers bypassed the provider LRU and a
        # configured cap could be exceeded silently.
        bound_provider = (
            (stream_layer_cache or self._max_provider_cache_bytes > 0)
            and not warm_z
        )
        self._decouple_provider_cache = decouple_provider_cache and bound_provider
        self._max_provider_cache_layers = (
            max_provider_cache_layers
            if max_provider_cache_layers is not None
            else max_layers_in_z
        )
        self._max_packed_cache_bytes = max(0, int(max_packed_cache_bytes))

        def _on_z_evict(layer_id: int) -> None:
            if bound_provider and not self._decouple_provider_cache:
                self.evict_streamed_layer(layer_id)

        self._z_retention = ZLayerRetention(
            max_layers_in_z=max_layers_in_z,
            pinned_layer_ids=pinned_layer_ids,
            warm_z=warm_z,
            on_evict=_on_z_evict,
        )
        self._bound_provider_cache = bound_provider
        self._provider_lru: list[int] = []
        self._provider_cache_evictions = 0
        self._ngram_weight_cache = ngram_weight_cache
        self._mmap_willneed = mmap_willneed
        self._mmap_dontneed = mmap_dontneed
        self._ngram_blobs: dict[tuple[str, int, int], bytes] = {}
        self._pack_uses_quant = pack_uses_quant_codec(entries)
        self._all_entries = entries
        self._trinity_layer_cache = LayerZlibCache()
        self._shadow_store = shadow_store
        self._use_shadow = self._resolve_shadow_pref(manifest_meta or {})
        self._codec_policy = (
            manifest_meta.get("codec_policy", "auto") if manifest_meta else "auto"
        )
        if self._codec_policy == "auto":
            env_policy = os.environ.get("RWKV_CODEC_POLICY", "auto").strip().lower()
            if env_policy in ("strict", "accuracy", "hybrid"):
                self._codec_policy = env_policy
        self._codec_map = manifest_meta.get("codec_map", {}) if manifest_meta else {}
        self._grouped_lut2_pack = bool(
            manifest_meta
            and str(manifest_meta.get("trinity_codebook", "")).startswith("groupwise_")
        )
        if self._codec_map:
            logger.info(
                "manifest has codec_map with %d entries; routing is build-time, "
                "runtime honors RWKV_CODEC_POLICY for shadow vs LUT only",
                len(self._codec_map),
            )
        self._disk_cache = None
        self._use_disk_cache = False
        if pack_dir is not None and manifest_meta is not None:
            from rwkv_ssd.runtime.decode_disk_cache import (
                DecodeDiskCache,
                cache_enabled,
            )

            disk_pref = (
                decode_disk_cache
                if decode_disk_cache is not None
                else os.environ.get("RWKV_DECODE_DISK_CACHE", "auto")
            )
            self._use_disk_cache = cache_enabled(disk_pref)
            if self._use_disk_cache:
                self._disk_cache = DecodeDiskCache(pack_dir, manifest_meta)
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_lock = threading.Lock()
        # Native rwkv.cpp layer uploads can use the packed SG8 record directly
        # even when a block is marked resident by a partial profile.  Keep one
        # bounded raw copy for those resident matrices so repeated
        # autoregressive steps do not re-read the same payload from the pack.
        # Streamed entries deliberately do not use this cache: their residency
        # budget is governed by the normal provider/z policies.
        self._native_resident_blobs: dict[
            str, bytes | memoryview | bytearray
        ] = {}
        # Native layer views retain packed matrix records and decoded vector
        # controls for a streamed layer.  This is separate from the ordinary
        # dense provider cache because the native ABI must not turn SG8
        # matrices into a second full BF16 copy.
        self._native_layer_views: dict[int, dict[str, Any]] = {}
        # The rwkv.cpp layer plan may borrow pointers into these views.  The
        # callback is installed only by that backend and is invoked before a
        # provider eviction releases a view.
        self._native_layer_invalidator: Any = None
        # Sequence backends consume a complete dense layer mapping.  When the
        # bounded provider cache is enabled, retain that mapping alongside the
        # per-tensor cache so repeated decode tokens do not rebuild a dict.
        # Views are removed together with their layer tensors on eviction.
        self._dense_layer_views: dict[int, dict[str, torch.Tensor]] = {}
        self._prepared_layers: dict[int, dict[str, torch.Tensor]] = {}
        self._prefetch_io_only = resolve_prefetch_io_only(
            prefetch_io_only,
            device=device,
            entries=entries,
            mode=mode,
        )
        self._prefetch_raw: dict[int, _LayerSpanPrefetch] = {}
        self._executor = (
            ThreadPoolExecutor(max_workers=1)
            if prefetch and mode != "resident"
            else None
        )
        self._prefetch_future: Future[_PrefetchJobResult] | None = None
        self._prefetch_lock = threading.Lock()
        self._pending: dict[str, torch.Tensor] = {}
        self._model_z: dict[str, torch.Tensor] | None = model_z
        # The fused decision depends only on immutable provider configuration
        # plus environment toggles that are resolved before a provider is
        # used.  Cache it: this method is queried from every packed adapter
        # and FFN projection on every token.
        self._fused_lut_enabled: bool | None = None
        self._fused_lut_blobs: dict[
            str, tuple[bytes | memoryview | bytearray, int, int]
        ] = {}
        self._fused_tmix_blobs: dict[
            str,
            tuple[
                tuple[
                    bytes | memoryview | bytearray,
                    bytes | memoryview | bytearray,
                    bytes | memoryview | bytearray,
                    bytes | memoryview | bytearray,
                ],
                int,
                int,
            ],
        ] = {}
        self._packed_layer_lru: list[int] = []
        self._packed_cache_evictions = 0
        # Optional CMix activation telemetry for evaluating selective FFN
        # reads. Counting zeros on CUDA synchronizes, so this is opt-in.
        self._cmix_sparsity_enabled = os.environ.get(
            "RWKV_CMIX_SPARSITY", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._cmix_zero_elements = 0
        self._cmix_active_elements = 0
        self._cmix_total_elements = 0
        self._cmix_samples = 0
        tile_stats_raw = os.environ.get(
            "RWKV_CMIX_TILE_STATS",
            os.environ.get("RWKV_CMIX_TILE", "0"),
        )
        self._cmix_tile_stats_enabled = tile_stats_raw.strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        try:
            self._cmix_tile_size = max(
                1, int(os.environ.get("RWKV_CMIX_TILE_SIZE", "32"))
            )
        except ValueError:
            self._cmix_tile_size = 32
        self._cmix_tile_total = 0
        self._cmix_tile_active = 0
        self._cmix_tile_occupancy: dict[str, int] = {}
        self._cmix_selective_reads_enabled = os.environ.get(
            "RWKV_CMIX_SELECTIVE_READS", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._cmix_tiled_values: dict[str, object] = {}
        self._cmix_selective_tiles_read = 0
        self._cmix_selective_tiles_skipped = 0
        self._cmix_selective_bytes_read = 0
        self._cmix_speculative_prefetch_enabled = os.environ.get(
            "RWKV_CMIX_SPECULATIVE_PREFETCH", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._cmix_prefetch_hits = 0
        self._cmix_prefetch_misses = 0
        self._cmix_prefetch_wasted_tiles = 0
        self._cmix_prefetch_bytes_submitted = 0
        try:
            self._cmix_hot_cache_limit_bytes = max(
                0, int(os.environ.get("RWKV_CMIX_HOT_CACHE_BYTES", "0"))
            )
        except ValueError:
            self._cmix_hot_cache_limit_bytes = 0
        self._cmix_hot_cache_hits = 0
        try:
            self._codec_deadline_ms = max(
                0.0, float(os.environ.get("RWKV_CODEC_DEADLINE_MS", "0"))
            )
        except ValueError:
            self._codec_deadline_ms = 0.0
        self._deadline_router = None
        self._deadline_budget = None
        if self._codec_deadline_ms > 0:
            from rwkv_ssd.runtime.deadline_codec import DeadlineCodecRouter, FallbackBudget

            self._deadline_router = DeadlineCodecRouter()
            self._deadline_budget = FallbackBudget(
                max_fallbacks=max(
                    0, int(os.environ.get("RWKV_CODEC_MAX_FALLBACKS", "1"))
                ),
                max_quality_cost=max(
                    0.0, float(os.environ.get("RWKV_CODEC_FALLBACK_QUALITY_BUDGET", "1"))
                ),
            )

        # The native packed-index cache is process-wide.  A configured packed
        # cap is also applied there; the provider-level cap below accounts for
        # the retained raw LUT blobs themselves.
        from rwkv_ssd.runtime.lut_gemm_fused import set_lut2_packed_cache_limit

        set_lut2_packed_cache_limit(self._max_packed_cache_bytes)

        if mode == "resident":
            preload = entries
        elif mode == "partial":
            preload = [e for e in entries if e.residency == "resident"]
        else:
            preload = []

        for entry in preload:
            if self._tensor_resident_in_z(entry):
                continue
            with Timer() as t:
                raw = self._read_packed_bytes(entry)
                self._cache[entry.name] = decode_weight_to_tensor(
                    raw,
                    entry,
                    device,
                    trinity_layer_cache=self._trinity_layer_cache,
                    decode_device=self._decode_device,
                )
            row = metrics.start_layer(entry.layer_id)
            row.read_ms += t.elapsed_ms

        # The RWKV-7 skeleton keeps the vocabulary head in ``model.z`` because
        # the ordinary ChatRWKV forward path needs a dense matrix.  A fused CPU
        # stream already has a packed representation, though, and routing the
        # head through the native GEMV avoids a 65,536 x 2,560 BF16 ``mm`` on
        # every token.  Prime only the packed blob here; keep the dense tensor
        # as a compatibility fallback for resident, accelerator, and explicit
        # non-fused configurations.
        # Native rwkv.cpp performs the final projection from its own global
        # tensor view.  Priming the Python fused head here would retain a
        # second packed copy that the native layer ABI never consumes.
        if not self._native_layer_streaming:
            self._prime_fused_global_blobs()

    def _prime_fused_global_blobs(self) -> None:
        """Register packed global matrices used by the fused CPU hot path.

        Block tensors are registered as their layer spans are read.  Global
        matrices do not pass through that scheduler, so without this small
        one-time step ``head.weight`` silently falls back to dense Torch mm
        even when its packed blob is available.  The raw view is retained by
        ``_fused_lut_blobs`` and is not dequantized, preserving the streaming
        memory model.  Keep the gate narrow: native CPU GEMV is intentionally
        not used for XPU/CUDA or resident paths.
        """
        if self._device.type != "cpu" or self.mode not in ("streaming", "partial"):
            return
        if not self._use_fused_lut_matmul():
            return
        supported = {"trinity_lut2", "scale_u8_grouped"}
        for entry in self._all_entries:
            if entry.name != "head.weight" or len(entry.shape) != 2:
                continue
            codec = (entry.dequant or "none").strip().lower()
            if codec not in supported or self.get_fused_lut_blob(entry.name) is not None:
                return
            # ``fused_lut2_enabled`` is deliberately broader than the native
            # ABI: it also permits the Numba/NumPy fallback.  That fallback is
            # useful for small diagnostics but is a poor replacement for the
            # dense 65,536 x 2,560 head on every token.  Do not install the
            # packed head unless the exact native W@x entry point exists; the
            # dense tensor in ``model.z`` remains the safe fallback otherwise.
            kernel = os.environ.get("RWKV_LUT_KERNEL", "auto").strip().lower()
            if kernel in ("numpy", "numba", "python", "np"):
                return
            try:
                from rwkv_ssd.native.lut2_gather_loader import lib

                native = lib()
            except (ImportError, OSError):
                return
            symbol = (
                "trinity_grouped_lut2_gemv_f32_export"
                if codec == "trinity_lut2"
                else "scale_u8_grouped_gemv_f32_export"
            )
            if native is None or not hasattr(native, symbol):
                return
            raw = self._read_packed_bytes(entry)
            self._register_fused_lut_blob(entry, raw)
            logger.info(
                "fused CPU head: using packed %s GEMV for %s (%s)",
                codec,
                entry.name,
                "x".join(str(int(dim)) for dim in entry.shape),
            )
            return

    def set_model_z(self, z: dict[str, torch.Tensor] | None) -> None:
        """Update live ``model.z`` so I/O/decode skips layers already resident."""
        self._model_z = z

    def _layer_resident_in_z(self, layer_id: int) -> bool:
        if self._model_z is None or layer_id < 0:
            return False
        from rwkv_ssd.runtime.rwkv7_weights import layer_ready_in_z

        return layer_ready_in_z(self._model_z, layer_id)

    def layer_prepared_for_forward(self, layer_id: int) -> bool:
        """True when provider already holds usable prepared skeleton and/or LUT blobs.

        Used to skip redundant prefetch I/O on F2/F5 where decode reuses
        ``_prepared_layers`` / ``_fused_lut_blobs`` without weights in ``z``.
        """
        if layer_id < 0:
            return False
        prep = self._prepared_layers.get(layer_id)
        if prep:
            return True
        # Fused path may retain blobs with an empty prepared marker.
        if self._use_fused_lut_matmul():
            att = f"blocks.{layer_id}.att.receptance.weight"
            if self.get_fused_lut_blob(att) is not None:
                return True
        with self._cache_lock:
            if layer_id in self._native_layer_views:
                return True
        return False

    def _tensor_resident_in_z(self, entry: TensorEntry) -> bool:
        """True when a resident manifest entry already lives in ``model.z``."""
        if self._model_z is None:
            return False
        if entry.name in self._model_z:
            return True
        if entry.layer_id >= 0 and not self._should_stream(entry):
            return self._layer_resident_in_z(entry.layer_id)
        return False

    def _get_resident_tensor(
        self, entry: TensorEntry, timing: LayerTiming
    ) -> torch.Tensor:
        if entry.name in self._cache:
            return self._cache[entry.name]
        if self._model_z is not None and entry.name in self._model_z:
            return self._model_z[entry.name]
        return self._load_one(entry, timing)

    def get_prepared_tensor(self, name: str) -> torch.Tensor | None:
        """Prepared layer tensor (shadow bf16 weights retained under strict fused)."""
        if name in self._cache:
            return self._cache[name]
        # Packed RWKV-7 forwards ask for the compact TMix control matrices by
        # their canonical ``blocks.<layer>.`` name.  The old fallback walked
        # every prepared layer for every lookup; with a 32-layer 2.9B cache
        # that turned the eight adapter projections into thousands of Python
        # dictionary probes per token.  Resolve the owning layer directly and
        # keep the scan only for synthetic/legacy names without a layer id.
        if name.startswith("blocks."):
            try:
                layer_id = int(name.split(".", 2)[1])
            except (IndexError, ValueError):
                layer_id = -1
            if layer_id >= 0:
                prepared = self._prepared_layers.get(layer_id)
                if prepared is not None:
                    return prepared.get(name)
                return None
        for prepared in self._prepared_layers.values():
            if name in prepared:
                return prepared[name]
        return None

    def _retainable_shadow_weight_key(self, name: str) -> bool:
        if not name.endswith(".weight"):
            return False
        if name in ("emb.weight", "head.weight") or name.endswith("embed.weight"):
            return True
        if ".att." in name or ".ffn." in name:
            return True
        # blocks.N.att.*.weight / blocks.N.ffn.*.weight or synthetic blocks.N.weight
        if name.startswith("blocks.") and ".weight" in name:
            return True
        return False

    def _retain_shadow_weights_in_provider(self) -> bool:
        if not self._use_shadow:
            return False
        raw = os.environ.get("RWKV_STRICT_FUSED_RETAIN_SHADOW", "auto").strip().lower()
        if raw in ("0", "false", "off", "no"):
            return False
        if raw in ("1", "true", "on", "yes"):
            return True
        return self._strict_fused_retain_layers()

    def _layer_stream_resident_in_provider(
        self,
        layer_id: int,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None,
    ) -> dict[str, torch.Tensor] | None:
        """Skip re-decode when strict fused already holds shadow weights + LUT blobs."""
        if not self._strict_fused_retain_layers():
            return None
        prep = self._prepared_layers.get(layer_id)
        # Empty ``{}`` is not a usable prepared cache (need skeleton keys for
        # packed forward / inject). Fall through to pack read / decode.
        if not prep:
            return None
        out: dict[str, torch.Tensor] = {}
        for entry in stream_entries:
            if entry.name in prep:
                out[entry.name] = prep[entry.name]
            elif self._is_fused_lut_entry(entry) and self._skip_fused_materialize():
                if self.get_fused_lut_blob(entry.name) is None:
                    return None
            else:
                return None
        if timing is not None:
            timing.layer_cache_hits += len(stream_entries)
        return out

    def get_fused_lut_blob(self, name: str) -> tuple[bytes, int, int] | None:
        return self._fused_lut_blobs.get(name)

    def get_fused_tmix_blobs(
        self, att_prefix: str
    ) -> tuple[tuple[bytes, bytes, bytes, bytes], int, int] | None:
        return self._fused_tmix_blobs.get(att_prefix)

    def register_cmix_tiled_value_matrix(self, name: str, matrix: object) -> None:
        """Register an explicit tiled CMix value matrix for opt-in reads."""
        if not hasattr(matrix, "matmul"):
            raise TypeError("CMix tiled value matrix must provide matmul(activation)")
        set_limit = getattr(matrix, "set_hot_cache_limit", None)
        if callable(set_limit):
            set_limit(self._cmix_hot_cache_limit_bytes)
        self._cmix_tiled_values[str(name)] = matrix

    def get_cmix_tiled_value_matrix(self, name: str) -> object | None:
        if not self._cmix_selective_reads_enabled:
            return None
        return self._cmix_tiled_values.get(name)

    def begin_cmix_tile_prefetch(self, name: str) -> object | None:
        if not self._cmix_selective_reads_enabled or not self._cmix_speculative_prefetch_enabled:
            return None
        matrix = self._cmix_tiled_values.get(name)
        begin = getattr(matrix, "begin_temporal_prefetch", None)
        return begin() if callable(begin) else None

    def record_cmix_selective_stats(self, stats: object) -> None:
        self._cmix_selective_tiles_read += int(getattr(stats, "tiles_read", 0))
        self._cmix_selective_tiles_skipped += int(getattr(stats, "tiles_skipped", 0))
        self._cmix_selective_bytes_read += int(getattr(stats, "bytes_read", 0))
        self._cmix_prefetch_hits += int(getattr(stats, "prefetch_hits", 0))
        self._cmix_prefetch_misses += int(getattr(stats, "prefetch_misses", 0))
        self._cmix_prefetch_wasted_tiles += int(
            getattr(stats, "prefetch_wasted_tiles", 0)
        )
        self._cmix_prefetch_bytes_submitted += int(
            getattr(stats, "prefetch_bytes_submitted", 0)
        )
        self._cmix_hot_cache_hits += int(getattr(stats, "hot_cache_hits", 0))

    def _use_fused_lut_matmul(self) -> bool:
        # The fused LUT GEMV implementation intentionally converts the
        # activation to NumPy and runs a native/Numba/NumPy CPU kernel.  That
        # is useful for strict CPU streaming, but it would silently pull the
        # dominant Arc/iGPU matmuls back to the host.  XPU uses the dense Torch
        # path instead: LUT2 is gathered onto XPU and ordinary matmul stays on
        # the selected accelerator.
        if self._device.type == "xpu":
            return False
        cached = getattr(self, "_fused_lut_enabled", None)
        if cached is not None:
            return bool(cached)
        from rwkv_ssd.runtime.lut_gemm_fused import fused_lut2_enabled

        self._fused_lut_enabled = fused_lut2_enabled(
            self.mode,
            self._stream_layer_cache,
            pack_uses_quant=self._pack_uses_quant,
        )
        return self._fused_lut_enabled

    def _strict_fused_retain_layers(self) -> bool:
        """Keep prepared skeleton + LUT blobs across tokens on strict fused (F1)."""
        if self._cache_format in ("none", "dense"):
            return False
        if self._stream_layer_cache:
            return False
        if not self._use_fused_lut_matmul():
            return False
        raw = os.environ.get("RWKV_STRICT_FUSED_RETAIN", "auto").strip().lower()
        if raw in ("0", "false", "off", "no"):
            return False
        return True

    def _strict_fused_lean_z(self) -> bool:
        """
        Drop per-layer skeleton from ``model.z`` after each step; LUT blobs stay in provider.

        Auto when ``max_layers_in_z`` is 0 (F1/F3 strict paths) — saves ~15 MB/layer in ``z``
        with ~1 ms/layer skeleton re-decode on revisit (blobs retained).
        """
        if not self._strict_fused_retain_layers():
            return False
        raw = os.environ.get("RWKV_STRICT_FUSED_LEAN_Z", "auto").strip().lower()
        if raw in ("0", "false", "off", "no"):
            return False
        if raw in ("1", "true", "on", "yes"):
            return True
        return not self._stream_layer_cache and self._z_retention.max_layers_in_z <= 0

    def _is_fused_lut_entry(self, entry: TensorEntry) -> bool:
        if not self._use_fused_lut_matmul():
            return False
        codec = (entry.dequant or "none").strip().lower()
        if codec not in {"trinity_lut2", "scale_u8_grouped"} or len(entry.shape) != 2:
            return False
        from rwkv_ssd.runtime.lut_gemm_fused import (
            is_fused_lut_tensor_name,
            is_fused_lut_transposed_tensor_name,
            small_transposed_lut_enabled,
        )

        if is_fused_lut_tensor_name(entry.name):
            return True
        return (
            codec == "scale_u8_grouped"
            and small_transposed_lut_enabled()
            and is_fused_lut_transposed_tensor_name(entry.name)
        )

    def _skip_fused_materialize(self) -> bool:
        from rwkv_ssd.runtime.rwkv7_weights import promote_full_z_enabled

        if not self._use_fused_lut_matmul():
            return False
        return not self._stream_layer_cache or not promote_full_z_enabled()

    def _register_fused_lut_blob(
        self, entry: TensorEntry, blob: bytes | memoryview | bytearray
    ) -> None:
        out_f, in_f = int(entry.shape[0]), int(entry.shape[1])
        # Keep stable mmap/bytearray views instead of copying every packed
        # layer into a second Python ``bytes`` object.  The view owns a
        # reference to its exporter, so pread buffers remain alive as needed.
        stable: bytes | memoryview | bytearray
        if isinstance(blob, memoryview):
            stable = blob
        elif isinstance(blob, (bytes, bytearray)):
            stable = blob
        else:
            stable = bytes(blob)
        self._fused_lut_blobs[entry.name] = (stable, out_f, in_f)
        self._touch_packed_layer(entry.layer_id)

    def _register_fused_transposed_lut_blobs(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
    ) -> None:
        """Retain SG8 TMix ``x @ W`` blobs without dropping dense fallbacks."""
        from rwkv_ssd.runtime.lut_gemm_fused import (
            is_fused_lut_transposed_tensor_name,
            small_transposed_lut_enabled,
        )

        if not small_transposed_lut_enabled():
            return

        for entry in stream_entries:
            if (
                (entry.dequant or "none").strip().lower() != "scale_u8_grouped"
                or not is_fused_lut_transposed_tensor_name(entry.name)
                or entry.name in self._fused_lut_blobs
            ):
                continue
            rel = entry.offset - base
            self._register_fused_lut_blob(
                entry, raw[rel : rel + entry.length]
            )

    def _touch_packed_layer(self, layer_id: int) -> None:
        if self._max_packed_cache_bytes <= 0 or layer_id < 0:
            return
        if layer_id in self._packed_layer_lru:
            self._packed_layer_lru.remove(layer_id)
        self._packed_layer_lru.append(layer_id)
        while self._packed_layer_lru and self.packed_weight_bytes() > self._max_packed_cache_bytes:
            old = self._packed_layer_lru.pop(0)
            self._packed_cache_evictions += 1
            self.evict_streamed_layer(old, force=True)

    def _register_fused_att_tmix_span(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
    ) -> None:
        """Register 4 att LUT2 blobs for batched TMix GEMV (one native kernel)."""
        from rwkv_ssd.runtime.lut_gemm_fused import (
            att_fused_suffixes,
            att_prefix_from_key,
        )

        by_prefix: dict[
            str,
            dict[str, tuple[bytes | memoryview | bytearray, int, int]],
        ] = {}
        for entry in stream_entries:
            if not self._is_fused_lut_entry(entry):
                continue
            prefix = att_prefix_from_key(entry.name)
            if prefix is None:
                continue
            rel = entry.offset - base
            blob = raw[rel : rel + entry.length]
            out_f, in_f = int(entry.shape[0]), int(entry.shape[1])
            suffix = entry.name[len(prefix) :]
            by_prefix.setdefault(prefix, {})[suffix] = (blob, out_f, in_f)
        order = att_fused_suffixes()
        for prefix, items in by_prefix.items():
            if not all(s in items for s in order):
                continue
            shapes = {items[s][1:] for s in order}
            if len(shapes) != 1:
                continue
            out_f, in_f = next(iter(shapes))
            blobs = tuple(items[s][0] for s in order)
            self._fused_tmix_blobs[prefix] = (blobs, out_f, in_f)
            # Keep the per-blob entries in ``_fused_lut_blobs`` too: the
            # batched ``lut2_tmix_gemv_batched`` falls back to per-blob
            # GEMV when the native batched kernel is unavailable, and the
            # ``matvec_z_or_lut`` fallback path also looks them up by name.
            layer_id = next((e.layer_id for e in stream_entries if e.name.startswith(prefix)), -1)
            self._touch_packed_layer(layer_id)

    def _apply_fused_lut_split(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
        out: dict[str, torch.Tensor],
        *,
        force_materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not self._use_fused_lut_matmul():
            return out
        for entry in stream_entries:
            if not self._is_fused_lut_entry(entry):
                continue
            rel = entry.offset - base
            self._register_fused_lut_blob(entry, raw[rel : rel + entry.length])
            # Only drop the bf16 slab when both the fused kernel is in use AND we
            # are NOT being asked to materialize (e.g. the .decode_cache/ writer
            # needs the actual tensor to serialize).
            if self._skip_fused_materialize() and not force_materialize:
                out.pop(entry.name, None)
        return out

    @property
    def planner(self) -> PrefetchPlanner:
        return self._planner

    def recent_layer_timings(self, limit: int = 8) -> list[LayerTiming]:
        if limit <= 0:
            return []
        return self._metrics.layers[-limit:]

    def layer_has_streamed_tensors(self, entries: list[TensorEntry]) -> bool:
        return any(self._should_stream(e) for e in entries)

    def prepare_layer_for_z(
        self,
        layer_id: int,
        layer_tensors: dict[str, torch.Tensor],
        timing: LayerTiming | None = None,
        *,
        force_materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Apply RWKV-7 z-layout transforms; cache when ``stream_layer_cache``.

        ``force_materialize=True`` skips the fused-inject filter so the
        bf16 att/FFN weight slabs stay in the result. The warm-z preloader
        passes this so the native ``model.forward`` (which reads
        ``z[att + 'receptance.weight']`` etc.) finds the weights after
        ``inject_layer_into_z``. The F-1 packed-block step also passes it
        so the per-tensor fallback can find the att/FFN tensors in z when
        the packed path can't be used.
        """
        from rwkv_ssd.runtime.lut_gemm_fused import filter_tensors_for_fused_inject

        if not force_materialize:
            layer_tensors = filter_tensors_for_fused_inject(self, layer_tensors)
        stored = self._prepared_layers.get(layer_id)
        # Empty ``{}`` is not a usable prepared cache.
        if stored:
            # A normal streaming/decode pass may intentionally retain only
            # skeleton or fused tensors.  Sequence prefill asks for a dense
            # layer, so never let that partial cache satisfy a forced
            # materialization request.  Reuse of it would leave the native
            # ChatRWKV kernels without the dense projection slabs.
            required_dense = f"blocks.{layer_id}.att.receptance.weight"
            if force_materialize and required_dense not in stored:
                self._prepared_layers.pop(layer_id, None)
                stored = None
        if stored:
            if not force_materialize:
                filtered = filter_tensors_for_fused_inject(self, stored)
                if len(filtered) != len(stored):
                    self._prepared_layers[layer_id] = filtered
                    stored = filtered
            if timing is not None:
                timing.layer_cache_hits += len(stored)
            return stored
        with Timer() as t_prep:
            prepared = {
                name: prepare_rwkv7_tensor_for_z(name, tensor)
                for name, tensor in layer_tensors.items()
            }
        # The packed decoder naturally produces tensors in the manifest's
        # primary dtype (normally BF16), while the skeleton may have been
        # explicitly constructed with another activation dtype.  The fused
        # RWKV-7 path consumes the prepared control/skeleton tensors directly
        # rather than copying them through ``materialize_prepared_layers``;
        # keep that path dtype-consistent with the resident globals.  Without
        # this alignment, a valid ``cpu float32`` skeleton fails in
        # ``group_norm``/``layer_norm`` with a mixed-dtype error on AVX2 CPUs.
        model_z = self._model_z
        target_dtype = None
        if isinstance(model_z, dict):
            reference = model_z.get("emb.weight")
            if isinstance(reference, torch.Tensor):
                target_dtype = reference.dtype
        if target_dtype is not None:
            # An opt-in fused CPU FP32 activation path keeps its prepared
            # skeleton in FP32 as well.  The resident global tensors may stay
            # BF16; only the packed block's small controls need to match the
            # activation dtype consumed by group/layer norm.
            try:
                from rwkv_ssd.runtime.lut_gemm_fused import (
                    activation_fp32_enabled,
                )

                if (
                    activation_fp32_enabled()
                    and self._device.type == "cpu"
                    and self._use_fused_lut_matmul()
                    and not any(
                        key.startswith("blocks.")
                        and key.endswith(".att.receptance.weight")
                        for key in model_z
                    )
                ):
                    target_dtype = torch.float32
            except ImportError:
                pass
            prepared = {
                name: tensor.to(dtype=target_dtype)
                if tensor.dtype != target_dtype
                else tensor
                for name, tensor in prepared.items()
            }
        if not force_materialize:
            prepared = filter_tensors_for_fused_inject(self, prepared)
        if timing is not None:
            timing.staging_ms += t_prep.elapsed_ms
        retain_prepared = self._stream_layer_cache or self._strict_fused_retain_layers()
        if retain_prepared and layer_id not in self._z_retention.pinned_layer_ids:
            self._prepared_layers[layer_id] = prepared
        if self._stream_layer_cache:
            prefix = f"blocks.{layer_id}."
            with self._cache_lock:
                for key in list(self._cache.keys()):
                    if key.startswith(prefix):
                        del self._cache[key]
        return prepared

    def clear_prepared_pinned_layers(self) -> None:
        """Drop provider prepared cache for layers already resident in ``z``."""
        for layer_id in list(self._z_retention.pinned_layer_ids):
            self._prepared_layers.pop(layer_id, None)

    def release_prepared_tensors(self, layer_id: int) -> None:
        """
        Drop duplicate skeleton tensors from provider RAM after inject into ``z``.

        Keeps the layer id in ``_prepared_layers`` so strict fused retain skips re-decode.
        Selective-shadow packs also retain bf16 ``.weight`` tensors for matvec reuse.
        """
        if layer_id not in self._prepared_layers:
            return
        if self._stream_layer_cache:
            return
        prepared = self._prepared_layers[layer_id]
        if self._retain_shadow_weights_in_provider():
            kept: dict[str, torch.Tensor] = {}
            for k, v in prepared.items():
                if self._retainable_shadow_weight_key(k):
                    # Keep by reference — these tensors are not injected into
                    # ``z`` on the fused path, so a clone only doubled RAM.
                    kept[k] = v
                elif not k.endswith(".weight"):
                    # skeleton (ln*, x_*, w*, etc.) — cheap vs re-decoding shadow weights
                    kept[k] = v
            self._prepared_layers[layer_id] = kept
        elif self._use_fused_lut_matmul() and self._skip_fused_materialize():
            # F1/F3 fused path: the normal decode path has already filtered
            # att/FFN slabs out of ``prepared`` (they live in
            # ``_fused_lut_blobs``).  The layer-outer sequence prefill path is
            # different: it deliberately asks for a fully materialized layer
            # so ChatRWKV's sequence kernels can consume it.  Before this
            # cleanup was added, that full prepared mapping survived the
            # prefill eviction and kept several gigabytes of dead bf16 slabs;
            # worse, the prefill eviction cleared the packed blobs as well,
            # forcing the first decode token to read and stage every layer
            # again.  Preserve the small skeleton, drop only the large fused
            # matrices, and leave the packed provider entries available for
            # the fused decode path.
            from rwkv_ssd.runtime.lut_gemm_fused import is_fused_lut_tensor_name

            if any(is_fused_lut_tensor_name(key) for key in prepared):
                self._prepared_layers[layer_id] = {
                    key: value
                    for key, value in prepared.items()
                    if not is_fused_lut_tensor_name(key)
                }
            return
        else:
            self._prepared_layers[layer_id] = {}

    def _resolve_shadow_pref(self, meta: dict) -> bool:
        raw = os.environ.get("RWKV_DECODE_SHADOW", "auto").strip().lower()
        has = self._shadow_store is not None and meta.get("shadow_file")
        if raw in ("0", "false", "off", "no"):
            return False
        if raw in ("1", "true", "on", "yes"):
            return bool(has)
        return bool(has)

    def _shadow_preferred_for_entry(self, entry: TensorEntry) -> bool:
        """Should this tensor prefer shadow decode over LUT under the current codec_policy?"""
        if not self._use_shadow:
            return False
        from rwkv_ssd.runtime.decode_shadow import entry_has_shadow

        if not entry_has_shadow(entry):
            return False
        policy = self._codec_policy
        if policy in ("auto", "strict"):
            return True
        if policy == "hybrid":
            small = entry.numel < 262144  # <256K elems → LUT is fast enough
            return not small
        if policy == "accuracy":
            if entry.name in ("head.weight", "lm_head.weight", "output.weight"):
                return True
        return True

    def _shadow_policy_override(self, entry: TensorEntry) -> str | None:
        """Return ``"shadow"`` if ``codec_policy`` routes this tensor to shadow."""
        return _codec_policy_entry_overrides(entry, self._codec_policy)

    def _layer_uses_shadow(self, entries: list[TensorEntry]) -> bool:
        from rwkv_ssd.runtime.decode_shadow import layer_any_shadow

        if not (self._use_shadow and self._shadow_store is not None):
            return False
        policy = self._codec_policy
        if policy in ("auto", "strict"):
            return layer_any_shadow(entries)
        return any(self._shadow_preferred_for_entry(e) for e in entries)

    def _shadow_ready(self, entries: list[TensorEntry]) -> bool:
        """Full shadow: every streamed tensor has a shadow blob."""
        from rwkv_ssd.runtime.decode_shadow import layer_has_shadow

        return (
            self._use_shadow
            and self._shadow_store is not None
            and layer_has_shadow(entries)
        )

    def _should_stream(self, entry: TensorEntry) -> bool:
        if self.mode == "resident":
            return False
        if self.mode == "streaming":
            return True
        return entry.residency == "streamed"

    def _needs_stream_read(self, entry: TensorEntry) -> bool:
        """Whether this entry needs a pack read for the current model state.

        ``mode=streaming`` controls the default policy, but it must not
        overwrite globals (or a hot layer) already installed in ``model.z``.
        This matters for RWKV skeletons because ``emb.weight`` there is the
        layer-normalized embedding, not the raw checkpoint tensor.
        """
        return self._should_stream(entry) and not self._tensor_resident_in_z(entry)

    def _read_packed_bytes(
        self,
        entry: TensorEntry,
        timing: LayerTiming | None = None,
        *,
        prefer_memoryview: bool = False,
    ) -> bytes | memoryview | bytearray:
        """Read packed bytes from store (no dequant)."""
        nkey = (entry.name, entry.offset, entry.length)
        if self._ngram_weight_cache and nkey in self._ngram_blobs:
            if timing is not None:
                timing.ngram_hits += 1
            return self._ngram_blobs[nkey]

        chunk_b = self._chunk_bytes
        if self._chunk_policy not in ("", "uniform", "off", "none"):
            chunk_b = chunk_bytes_for_entry(entry, self._chunk_policy, chunk_b)
        elif chunk_b <= 0:
            chunk_b = 0

        if (
            prefer_memoryview
            and chunk_b <= 0
            and bool(getattr(self._store, "stable_memoryviews", False))
        ):
            read_view = getattr(self._store, "read_memoryview_span", None)
            if callable(read_view):
                with Timer() as t_read:
                    raw = read_view(entry.offset, entry.length)
                if timing is not None:
                    timing.read_ms += t_read.elapsed_ms
                    timing.chunk_reads += 1
                    timing.bytes_read += len(raw)
                self._metrics.streamed_bytes += len(raw)
                return raw

        if chunk_b > 0:

            def read_range(
                ent: TensorEntry, byte_offset: int, length: int, dest: memoryview
            ) -> None:
                self._store.read_range(ent, byte_offset, length, dest)

            raw, stats = read_tensor_chunked(read_range, entry, chunk_bytes=chunk_b)
            if timing is not None:
                timing.read_ms += stats.read_ms
                timing.chunk_reads += stats.chunks
                timing.bytes_read += len(raw)
        else:
            with Timer() as t_read:
                raw = self._store.read_bytes(entry)
            if timing is not None:
                timing.read_ms += t_read.elapsed_ms
                timing.chunk_reads += 1
                timing.bytes_read += len(raw)

        self._metrics.streamed_bytes += len(raw)
        if self._ngram_weight_cache:
            self._ngram_blobs[nkey] = raw
        return raw

    def _read_entry_bytes(
        self, entry: TensorEntry, timing: LayerTiming | None = None
    ) -> bytes:
        return decode_weight_blob(self._read_packed_bytes(entry, timing), entry)

    def _read_layer_span_from_ngram(
        self,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None,
    ) -> tuple[bytes, int] | None:
        """Assemble a layer span from the n-gram blob cache when every entry hits.

        Works with alignment padding between entries: the assembled buffer
        covers ``[base, end)`` where ``end`` is the last entry's
        ``offset + length``, and any gaps between entries are left as
        zero (matching what the disk read would return for a contiguous
        layer span that includes padding).
        """
        if not self._ngram_blobs:
            return None
        if not all(
            (e.name, e.offset, e.length) in self._ngram_blobs for e in stream_entries
        ):
            return None
        ordered = sorted(stream_entries, key=lambda e: (e.offset, e.name))
        base = ordered[0].offset
        end = ordered[-1].offset + ordered[-1].length
        raw = bytearray(end - base)
        for entry in ordered:
            blob = self._ngram_blobs[(entry.name, entry.offset, entry.length)]
            rel = entry.offset - base
            raw[rel : rel + entry.length] = blob
        if timing is not None:
            timing.ngram_hits += len(stream_entries)
        return bytes(raw), base

    def _store_layer_span_in_ngram(
        self,
        stream_entries: list[TensorEntry],
        raw: bytes | memoryview | bytearray,
        base: int,
    ) -> None:
        if not self._ngram_weight_cache:
            return
        for entry in stream_entries:
            rel = entry.offset - base
            blob = raw[rel : rel + entry.length]
            self._ngram_blobs[(entry.name, entry.offset, entry.length)] = bytes(blob)

    def _read_layer_span_bytes(
        self,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None = None,
        *,
        use_shadow: bool = False,
        prefer_memoryview: bool = False,
    ) -> tuple[bytes | memoryview | bytearray, int] | None:
        """One layer span read from weights.bin or shadow.bin."""
        if not stream_entries:
            return None
        if self._ngram_weight_cache and not use_shadow:
            cached = self._read_layer_span_from_ngram(stream_entries, timing)
            if cached is not None:
                return cached
        if use_shadow and any(
            entry.fast_stripes or entry.fast_shard_file for entry in stream_entries
        ):
            from rwkv_ssd.runtime.decode_shadow import read_shadow_layer

            with Timer() as t_read:
                raw, base = read_shadow_layer(self._shadow_store, stream_entries)
            if raw is None:
                return None
            if timing is not None:
                timing.read_ms += t_read.elapsed_ms
                timing.chunk_reads += 1
                timing.shadow_hits += len(stream_entries)
                timing.bytes_read += len(raw)
            self._metrics.streamed_bytes += len(raw)
            return raw, base
        if use_shadow:
            from rwkv_ssd.runtime.decode_shadow import entries_shadow_layer_read_span

            span = entries_shadow_layer_read_span(stream_entries)
            store = self._shadow_store
        else:
            span = entries_layer_read_span(stream_entries)
            store = self._store
        if span is None or store is None:
            return None
        base, total = span

        # A sharded pack has no single global span: offsets are local to the
        # shard named by each entry.  Normal layer packs keep all tensors for
        # a layer together, so use the shard-aware span API when available;
        # mixed-shard layers fall back to per-tensor reads below.
        if not use_shadow and any(entry.shard_file for entry in stream_entries):
            shard_names = {entry.shard_file.replace("\\", "/") for entry in stream_entries}
            if len(shard_names) != 1:
                return None
            read_shard = getattr(store, "read_bytes_for_shard", None)
            if callable(read_shard):
                shard_name = next(iter(shard_names))
                with Timer() as t_read:
                    raw = read_shard(shard_name, base, total)
                if timing is not None:
                    timing.read_ms += t_read.elapsed_ms
                    timing.chunk_reads += 1
                    timing.bytes_read += len(raw)
                self._metrics.streamed_bytes += len(raw)
                self._store_layer_span_in_ngram(stream_entries, raw, base)
                return raw, base

        use_mmap_views = prefer_memoryview or (
            os.environ.get("RWKV_USE_MMAP_VIEWS", "0").strip().lower()
            in ("1", "true", "on", "yes")
        )
        readers = []
        if use_mmap_views:
            readers.append(getattr(store, "read_memoryview_span", None))
        readers.extend(
            [
                getattr(store, "read_bytearray_span", None),
                getattr(store, "read_bytes_span", None),
            ]
        )
        raw = None
        elapsed_ms = 0.0
        for read_fn in readers:
            if not callable(read_fn):
                continue
            try:
                with Timer() as t_read:
                    candidate = read_fn(base, total)
            except NotImplementedError:
                # The base WeightStore exposes optional span methods that
                # deliberately raise when a backend has no span support.
                continue
            raw = candidate
            elapsed_ms = t_read.elapsed_ms
            break
        if raw is None:
            return None
        if timing is not None:
            timing.read_ms += elapsed_ms
            timing.chunk_reads += 1
            timing.bytes_read += len(raw)
            if use_shadow:
                timing.shadow_hits += len(stream_entries)
        self._metrics.streamed_bytes += len(raw)
        if not use_shadow:
            self._store_layer_span_in_ngram(stream_entries, raw, base)
        return raw, base

    def _read_hybrid_layer_spans_parallel(
        self,
        shadow_entries: list[TensorEntry],
        lut_entries: list[TensorEntry],
        timing: LayerTiming | None = None,
    ) -> tuple[
        tuple[bytes | memoryview | bytearray, int] | None,
        tuple[bytes | memoryview | bytearray, int] | None,
    ]:
        """Read shadow.bin and weights.bin layer spans concurrently (hybrid packs)."""
        shadow_out: tuple[bytes | memoryview | bytearray, int] | None = None
        lut_out: tuple[bytes | memoryview | bytearray, int] | None = None

        def _shadow() -> None:
            nonlocal shadow_out
            shadow_out = self._read_layer_span_bytes(
                shadow_entries, timing, use_shadow=True
            )

        def _lut() -> None:
            nonlocal lut_out
            lut_out = self._read_layer_span_bytes(lut_entries, timing)

        if shadow_entries and lut_entries:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_shadow = pool.submit(_shadow)
                f_lut = pool.submit(_lut)
                f_shadow.result()
                f_lut.result()
        elif shadow_entries:
            _shadow()
        elif lut_entries:
            _lut()
        return shadow_out, lut_out

    def _decode_from_layer_raw(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None,
        *,
        force_materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        codecs = {(e.dequant or "none").strip().lower() for e in stream_entries}
        # Register the four attention projection blobs before dispatching on
        # the layer's codec set.  Production grouped-U8 layers also contain
        # dense BF16 control/skeleton tensors, so their codec set is mixed and
        # does not enter the homogeneous ``scale_u8_grouped`` branch below.
        # Without this registration the batched TMix/QKV ABI is silently
        # disabled and every streamed token falls back to three independent
        # native GEMVs for receptance/key/value.
        if self._use_fused_lut_matmul():
            self._register_fused_att_tmix_span(raw, base, stream_entries)
            self._register_fused_transposed_lut_blobs(
                raw, base, stream_entries
            )
        with Timer() as t_stage:
            if codecs == {"trinity_layer"}:
                allow = os.environ.get("RWKV_ALLOW_TRINITY_LAYER", "0").strip().lower()
                if allow not in ("1", "true", "on", "yes"):
                    raise RuntimeError(
                        "trinity_layer (zlib layer bundle) is blocked on the engine hot path; "
                        "repack with trinity_lut2 (--pack-codec trinity_lut2) or set "
                        "RWKV_ALLOW_TRINITY_LAYER=1 for legacy tests only"
                    )
                rel0 = stream_entries[0].offset - base
                packed = raw[rel0 : rel0 + stream_entries[0].length]
                out = decode_trinity_layer_span(
                    packed,
                    stream_entries,
                    self._trinity_layer_cache,
                    self._device,
                    decode_device=self._decode_device,
                    timing=timing,
                )
            elif codecs == {"trinity_lut2"}:
                if self._use_fused_lut_matmul() and not force_materialize:
                    out = self._decode_lut2_fused_selective(
                        raw, base, stream_entries, timing=timing
                    )
                else:
                    from rwkv_ssd.runtime.trinity_codec import (
                        decode_lut2_layer_from_span,
                    )

                    out = decode_lut2_layer_from_span(
                        raw,
                        stream_entries,
                        base,
                        self._device,
                        decode_device=self._decode_device,
                        timing=timing,
                    )
            elif codecs == {"scale_u8"}:
                from rwkv_ssd.runtime.pack_codec import decode_scale_layer_from_span

                out = decode_scale_layer_from_span(
                    raw, stream_entries, base, self._device, bits=8
                )
            elif codecs == {"scale_u8_grouped"}:
                if self._use_fused_lut_matmul() and not force_materialize:
                    out = self._decode_grouped_u8_fused_selective(
                        raw, base, stream_entries, timing=timing
                    )
                else:
                    from rwkv_ssd.runtime.pack_codec import (
                        decode_scale_u8_grouped_layer_from_span,
                    )

                    out = decode_scale_u8_grouped_layer_from_span(
                        raw, stream_entries, base, self._device
                    )
            elif codecs == {"scale_u4"}:
                from rwkv_ssd.runtime.pack_codec import decode_scale_layer_from_span

                out = decode_scale_layer_from_span(
                    raw, stream_entries, base, self._device, bits=4
                )
            elif codecs <= {"trinity", "trinity_lut2"} and "none" not in codecs:
                from rwkv_ssd.runtime.trinity_codec import decode_entries_from_span

                out = decode_entries_from_span(
                    raw,
                    stream_entries,
                    base,
                    self._device,
                    decode_device=self._decode_device,
                    timing=timing,
                )
            else:
                out = {}
                for entry in stream_entries:
                    rel = entry.offset - base
                    blob = raw[rel : rel + entry.length]
                    if self._is_fused_lut_entry(entry):
                        self._register_fused_lut_blob(entry, blob)
                        if self._skip_fused_materialize() and not force_materialize:
                            continue
                    out[entry.name] = decode_weight_to_tensor(
                        blob,
                        entry,
                        self._device,
                        trinity_layer_cache=self._trinity_layer_cache,
                        decode_device=self._decode_device,
                    )
        if timing is not None:
            timing.staging_ms += t_stage.elapsed_ms
        return self._apply_fused_lut_split(
            raw, base, stream_entries, out, force_materialize=force_materialize
        )

    def _decode_lut2_fused_selective(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode only non-fused LUT2 tensors; register fused blobs without bf16 gather."""
        fused = [e for e in stream_entries if self._is_fused_lut_entry(e)]
        other = [e for e in stream_entries if e not in fused]
        for entry in fused:
            if entry.name in self._fused_lut_blobs:
                continue
            rel = entry.offset - base
            self._register_fused_lut_blob(entry, raw[rel : rel + entry.length])
        out: dict[str, torch.Tensor] = {}
        if other:
            from rwkv_ssd.runtime.trinity_codec import decode_lut2_layer_from_span

            out = decode_lut2_layer_from_span(
                raw,
                other,
                base,
                self._device,
                decode_device=self._decode_device,
                timing=timing,
            )
        return out

    def _decode_grouped_u8_fused_selective(
        self,
        raw: bytes | memoryview,
        base: int,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None = None,
    ) -> dict[str, torch.Tensor]:
        """Register eligible SG8 matrices and decode only the small tensors."""
        fused = [e for e in stream_entries if self._is_fused_lut_entry(e)]
        other = [e for e in stream_entries if e not in fused]
        for entry in fused:
            rel = entry.offset - base
            self._register_fused_lut_blob(entry, raw[rel : rel + entry.length])
        out: dict[str, torch.Tensor] = {}
        if other:
            with Timer() as t_stage:
                for entry in other:
                    rel = entry.offset - base
                    out[entry.name] = decode_weight_to_tensor(
                        raw[rel : rel + entry.length],
                        entry,
                        self._device,
                        trinity_layer_cache=self._trinity_layer_cache,
                        decode_device=self._decode_device,
                    )
            if timing is not None:
                timing.staging_ms += t_stage.elapsed_ms
        return out

    def _load_one(self, entry: TensorEntry, timing: LayerTiming) -> torch.Tensor:
        with self._cache_lock:
            if entry.name in self._cache:
                if self._stream_layer_cache and self._should_stream(entry):
                    timing.layer_cache_hits += 1
                return self._cache[entry.name]
        if entry.name in self._pending:
            timing.prefetch_hits += 1
            t = self._pending.pop(entry.name)
            if self._stream_layer_cache and self._should_stream(entry):
                self._cache[entry.name] = t
            return t

        raw = self._read_packed_bytes(entry, timing)
        if self._is_fused_lut_entry(entry):
            self._register_fused_lut_blob(entry, raw)
            if self._skip_fused_materialize():
                # The fused-LUT fast path expects a single
                # layer-span read (``_decode_stream_entries``) so the
                # blob is contiguous in memory. For stores that do
                # not implement ``read_bytes_span`` (the sharded
                # store, see ``weight_store_sharded``), the per-tensor
                # read still produces a valid single-entry blob — the
                # fused matmul just uses this entry's bytes directly
                # via ``provider.get_fused_lut_blob``. Fall through to
                # the per-tensor decode so the blob is registered
                # and the matmul works; the layer-span path is an
                # optimization, not a requirement.
                if getattr(self._store, "read_bytes_span", None) is None:
                    pass  # per-tensor fused blob is fine
                else:
                    raise RuntimeError(
                        f"fused LUT entry {entry.name!r} requires layer-span decode"
                    )

        h2d_only = 0.0
        with Timer() as t_stage:
            t = decode_weight_to_tensor(
                raw,
                entry,
                self._device,
                trinity_layer_cache=self._trinity_layer_cache,
                decode_device=self._decode_device,
            )
            # ``decode_weight_to_tensor`` already returns on ``self._device``.
            # The historical staging block copied CUDA -> CUDA into one shared
            # slot, synchronized immediately, and returned aliasing views for
            # every tensor in the layer. Besides preventing overlap, later
            # tensors could overwrite earlier ones. The event-owned ring is
            # reserved for slab/immediate-consumer transfers; per-tensor loads
            # keep their independently owned decoded allocation.
        timing.staging_ms += t_stage.elapsed_ms
        timing.h2d_ms += h2d_only

        if self._stream_layer_cache and self._should_stream(entry):
            with self._cache_lock:
                self._cache[entry.name] = t
        return t

    def _record_disk_cache_hits(
        self, layer_id: int, n_hits: int, timing: LayerTiming | None
    ) -> None:
        if n_hits <= 0:
            return
        if timing is not None:
            timing.disk_cache_hits += n_hits
            return
        row = self._metrics.start_layer(layer_id)
        row.disk_cache_hits += n_hits

    def _prefer_disk_cache_over_fused_lut(
        self, stream_entries: list[TensorEntry]
    ) -> bool:
        """When to load full bf16 layers from ``.decode_cache/`` instead of mmap+fused."""
        if not self._use_fused_lut_matmul():
            return True
        from rwkv_ssd.runtime.lut_gemm_fused import prefer_fused_lut_over_disk_cache

        raw = os.environ.get("RWKV_PREFER_DISK_CACHE", "auto").strip().lower()
        if raw in ("1", "true", "on", "yes"):
            return True
        if raw in ("0", "false", "off", "no"):
            return False

        # Strict fused: mmap LUT spans + fused GEMV beats re-reading ~5 MB bf16/layer every token.
        if not self._stream_layer_cache:
            return False

        if not prefer_fused_lut_over_disk_cache():
            return True
        # Stream+cache: disk cache helps hybrid shadow layers on revisit.
        return self._layer_uses_shadow(stream_entries)

    def _try_cached_layer(
        self,
        layer_id: int,
        stream_entries: list[TensorEntry],
        timing: LayerTiming,
    ) -> dict[str, torch.Tensor] | None:
        """RAM provider cache or persistent ``.decode_cache/`` before pack read."""
        if not stream_entries:
            return None
        if all(entry.name in self._pending for entry in stream_entries):
            # Per-layer hit: the prefetch job submitted for this layer
            # served every entry from the pre-decoded blob. Counter is
            # incremented once per layer (matches the _decode_stream_entries
            # path which also counts one hit per layer).
            timing.prefetch_hits += 1
            if self._stream_layer_cache:
                timing.layer_cache_hits += len(stream_entries)
            tensors = {
                entry.name: self._pending.pop(entry.name) for entry in stream_entries
            }
            if self._stream_layer_cache:
                with self._cache_lock:
                    for name, t in tensors.items():
                        self._cache[name] = t
            return tensors
        with self._cache_lock:
            cache_hit = all(entry.name in self._cache for entry in stream_entries)
            if cache_hit:
                timing.layer_cache_hits += len(stream_entries)
                return {entry.name: self._cache[entry.name] for entry in stream_entries}
        if self._disk_cache is not None and self._use_disk_cache:
            if self._prefer_disk_cache_over_fused_lut(stream_entries):
                cached = self._disk_cache.try_load_layer(
                    layer_id, stream_entries, self._device
                )
                if cached is not None:
                    self._record_disk_cache_hits(layer_id, len(stream_entries), timing)
                    if self._stream_layer_cache:
                        self._cache.update(cached)
                    return cached
        return None

    def _decode_stream_entries(
        self,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load streamed tensors; one read when blobs are packed contiguously."""
        layer_id = stream_entries[0].layer_id if stream_entries else -1
        resident = self._layer_stream_resident_in_provider(
            layer_id, stream_entries, timing
        )
        if resident is not None:
            if self._stream_layer_cache:
                for name, t in resident.items():
                    self._cache[name] = t
            return resident
        if stream_entries and all(e.name in self._cache for e in stream_entries):
            return {e.name: self._cache[e.name] for e in stream_entries}
        skip_disk = not self._prefer_disk_cache_over_fused_lut(stream_entries)
        if (
            not skip_disk
            and self._disk_cache is not None
            and self._use_disk_cache
            and stream_entries
        ):
            cached = self._disk_cache.try_load_layer(
                layer_id, stream_entries, self._device
            )
            if cached is not None:
                self._record_disk_cache_hits(layer_id, len(stream_entries), timing)
                if self._stream_layer_cache:
                    self._cache.update(cached)
                return cached

        if any(entry.stripes for entry in stream_entries):
            coalesced = self._decode_striped_entries_coalesced(
                stream_entries, timing
            )
            if coalesced is not None:
                if self._stream_layer_cache:
                    with self._cache_lock:
                        self._cache.update(coalesced)
                self._maybe_store_disk_cache(layer_id, stream_entries, coalesced)
                return coalesced

        from rwkv_ssd.runtime.decode_shadow import (
            decode_shadow_layer_from_span,
            split_shadow_lut_entries,
        )

        shadow_entries, lut_entries = (
            split_shadow_lut_entries(stream_entries)
            if self._layer_uses_shadow(stream_entries)
            else ([], stream_entries)
        )
        pref = self._prefetch_raw.pop(layer_id, None)
        if pref is not None and timing is not None:
            timing.prefetch_hits += 1

        if shadow_entries and not lut_entries:
            if self._deadline_router is not None and self._deadline_budget is not None:
                def _preferred_shadow() -> dict[str, torch.Tensor]:
                    read = self._read_layer_span_bytes(
                        shadow_entries, None, use_shadow=True
                    )
                    if read is None:
                        raise OSError("shadow representation unavailable")
                    raw_shadow, base_shadow = read
                    return decode_shadow_layer_from_span(
                        raw_shadow, shadow_entries, base_shadow, self._device
                    )

                def _fallback_lut() -> dict[str, torch.Tensor]:
                    read = self._read_layer_span_bytes(shadow_entries, None)
                    if read is None:
                        raise OSError("packed fallback representation unavailable")
                    raw_lut, base_lut = read
                    return self._decode_from_layer_raw(
                        raw_lut,
                        base_lut,
                        shadow_entries,
                        None,
                        force_materialize=True,
                    )

                route = self._deadline_router.route(
                    _preferred_shadow,
                    _fallback_lut,
                    deadline_ms=self._codec_deadline_ms,
                    budget=self._deadline_budget,
                    preferred_name="shadow",
                    fallback_name="packed",
                )
                if route.timed_out:
                    self._metrics.codec_deadline_timeouts += 1
                if route.fallback_used:
                    self._metrics.codec_deadline_fallbacks += 1
                out = route.value
                if timing is not None:
                    timing.read_ms += route.elapsed_ms
                    if route.representation == "shadow":
                        timing.shadow_hits += len(shadow_entries)
                if self._stream_layer_cache:
                    with self._cache_lock:
                        self._cache.update(out)
                self._maybe_store_disk_cache(layer_id, stream_entries, out)
                return out
            if pref is not None and pref.shadow is not None:
                raw, base = pref.shadow
            else:
                read = self._read_layer_span_bytes(
                    shadow_entries, timing, use_shadow=True
                )
                if read is None:
                    return self._decode_stream_entries_fallback(stream_entries, timing)
                raw, base = read
            with Timer() as t_stage:
                out = decode_shadow_layer_from_span(
                    raw, shadow_entries, base, self._device
                )
            if timing is not None:
                timing.staging_ms += t_stage.elapsed_ms
            if self._stream_layer_cache:
                with self._cache_lock:
                    for name, t in out.items():
                        self._cache[name] = t
            self._maybe_store_disk_cache(layer_id, stream_entries, out)
            return out

        if lut_entries and not shadow_entries:
            if pref is not None and pref.weights is not None:
                raw, base = pref.weights
                out = self._decode_from_layer_raw(raw, base, lut_entries, timing)
                if self._stream_layer_cache:
                    with self._cache_lock:
                        for name, t in out.items():
                            self._cache[name] = t
                self._maybe_store_disk_cache(layer_id, stream_entries, out)
                return out

            cached = self._read_layer_span_from_ngram(lut_entries, timing)
            if cached is not None:
                raw, base = cached
                out = self._decode_from_layer_raw(raw, base, lut_entries, timing)
                if self._stream_layer_cache:
                    with self._cache_lock:
                        for name, t in out.items():
                            self._cache[name] = t
                self._maybe_store_disk_cache(layer_id, stream_entries, out)
                return out

            read = self._read_layer_span_bytes(lut_entries, timing)
            if read is None:
                return self._decode_stream_entries_fallback(stream_entries, timing)
            raw, base = read
            out = self._decode_from_layer_raw(raw, base, lut_entries, timing)
            if self._stream_layer_cache:
                with self._cache_lock:
                    for name, t in out.items():
                        self._cache[name] = t
            self._maybe_store_disk_cache(layer_id, stream_entries, out)
            return out

        if shadow_entries and lut_entries:
            out: dict[str, torch.Tensor] = {}
            shadow_read = pref.shadow if pref is not None else None
            lut_read = pref.weights if pref is not None else None
            if shadow_read is None and lut_read is None:
                shadow_read, lut_read = self._read_hybrid_layer_spans_parallel(
                    shadow_entries, lut_entries, timing
                )
            elif shadow_read is None:
                shadow_read = self._read_layer_span_bytes(
                    shadow_entries, timing, use_shadow=True
                )
            elif lut_read is None:
                lut_read = self._read_layer_span_bytes(lut_entries, timing)
            if shadow_read is None or lut_read is None:
                return self._decode_stream_entries_fallback(stream_entries, timing)
            raw, base = shadow_read
            with Timer() as t_stage:
                out.update(
                    decode_shadow_layer_from_span(
                        raw, shadow_entries, base, self._device
                    )
                )
            if timing is not None:
                timing.staging_ms += t_stage.elapsed_ms
                timing.shadow_hits += len(shadow_entries)
            raw, base = lut_read
            out.update(self._decode_from_layer_raw(raw, base, lut_entries, timing))

            if self._stream_layer_cache:
                with self._cache_lock:
                    for name, t in out.items():
                        self._cache[name] = t
            self._maybe_store_disk_cache(layer_id, stream_entries, out)
            return out

        return self._decode_stream_entries_fallback(stream_entries, timing)

    def _decode_striped_entries_coalesced(
        self,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None,
        *,
        force_materialize: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        """Gather one striped layer by shard, then decode its logical tensors."""
        read_layer = getattr(self._store, "read_layer_coalesced", None)
        if not callable(read_layer):
            return None
        with Timer() as t_read:
            raw_by_name = read_layer(stream_entries)
        if timing is not None:
            timing.read_ms += t_read.elapsed_ms
            timing.chunk_reads += 1
            timing.bytes_read += sum(len(blob) for blob in raw_by_name.values())

        out: dict[str, torch.Tensor] = {}
        with Timer() as t_stage:
            for entry in stream_entries:
                blob = raw_by_name[entry.name]
                if self._is_fused_lut_entry(entry):
                    self._register_fused_lut_blob(entry, blob)
                    if self._skip_fused_materialize() and not force_materialize:
                        continue
                out[entry.name] = decode_weight_to_tensor(
                    blob,
                    entry,
                    self._device,
                    trinity_layer_cache=self._trinity_layer_cache,
                    decode_device=self._decode_device,
                )
        if timing is not None:
            timing.staging_ms += t_stage.elapsed_ms
        return out

    def _decode_stream_entries_fallback(
        self,
        stream_entries: list[TensorEntry],
        timing: LayerTiming | None,
    ) -> dict[str, torch.Tensor]:
        out = {}
        for entry in stream_entries:
            if self._is_fused_lut_entry(entry) and self._skip_fused_materialize():
                row = (
                    timing
                    if timing is not None
                    else self._metrics.start_layer(entry.layer_id)
                )
                raw = self._read_packed_bytes(entry, row)
                self._register_fused_lut_blob(entry, raw)
                continue
            row = (
                timing
                if timing is not None
                else self._metrics.start_layer(entry.layer_id)
            )
            out[entry.name] = self._load_one(entry, row)
        return out

    def _maybe_store_disk_cache(
        self,
        layer_id: int,
        entries: list[TensorEntry],
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if self._disk_cache is None or not self._use_disk_cache or not tensors:
            return
        codecs = {(e.dequant or "none").strip().lower() for e in entries}
        shadow = self._layer_uses_shadow(entries)
        if not shadow and not codecs & {
            "trinity_lut2",
            "trinity",
            "trinity_layer",
            "scale_u8",
            "scale_u8_grouped",
            "scale_u4",
        }:
            return
        if not all(entry.name in tensors for entry in entries):
            return
        try:
            self._disk_cache.store_layer(layer_id, entries, tensors)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug("decode disk cache write skipped: %s", exc)

    def _load_layer_contiguous(
        self,
        stream_entries: list[TensorEntry],
        all_entries: list[TensorEntry],
        timing: LayerTiming,
    ) -> dict[str, torch.Tensor] | None:
        out: dict[str, torch.Tensor] = {}
        for entry in all_entries:
            if not self._needs_stream_read(entry):
                out[entry.name] = self._get_resident_tensor(entry, timing)
        layer_id = stream_entries[0].layer_id if stream_entries else -1
        hit = self._try_cached_layer(layer_id, stream_entries, timing)
        if hit is not None:
            out.update(hit)
            return out
        if self._layer_uses_shadow(stream_entries):
            out.update(self._decode_stream_entries(stream_entries, timing))
            return out
        if entries_layer_read_span(stream_entries) is None:
            return None
        if getattr(self._store, "read_bytes_span", None) is None:
            return None
        streamed = self._decode_stream_entries(stream_entries, timing)
        out.update(streamed)
        return out

    def load_layer_tensors(self, entries: list[TensorEntry]) -> dict[str, torch.Tensor]:
        if not entries:
            return {}
        layer_id = entries[0].layer_id
        timing = self.begin_layer(layer_id)
        stream_entries = [e for e in entries if self._needs_stream_read(e)]
        if stream_entries:
            hit = self._try_cached_layer(layer_id, stream_entries, timing)
            if hit is not None:
                out = {}
                for entry in entries:
                    if not self._needs_stream_read(entry):
                        out[entry.name] = self._get_resident_tensor(entry, timing)
                    else:
                        out[entry.name] = hit[entry.name]
                if self._bound_provider_cache and layer_id >= 0:
                    self._retain_provider_layer(layer_id)
                from rwkv_ssd.runtime.lut_gemm_fused import (
                    filter_tensors_for_fused_inject,
                )

                return filter_tensors_for_fused_inject(self, out)
            batched = self._load_layer_contiguous(stream_entries, entries, timing)
            if batched is not None:
                if self._bound_provider_cache and layer_id >= 0:
                    self._retain_provider_layer(layer_id)
                from rwkv_ssd.runtime.lut_gemm_fused import (
                    filter_tensors_for_fused_inject,
                )

                return filter_tensors_for_fused_inject(self, batched)
        out = {}
        for entry in entries:
            if self._needs_stream_read(entry):
                out[entry.name] = self._load_one(entry, timing)
            else:
                out[entry.name] = self._get_resident_tensor(entry, timing)
        if self._bound_provider_cache and layer_id >= 0:
            self._retain_provider_layer(layer_id)
        from rwkv_ssd.runtime.lut_gemm_fused import filter_tensors_for_fused_inject

        return filter_tensors_for_fused_inject(self, out)

    def load_layer_tensors_materialized(
        self, entries: list[TensorEntry]
    ) -> dict[str, torch.Tensor]:
        """Decode every manifest entry to bf16 (for ``.decode_cache/`` warm)."""
        if not entries:
            return {}
        layer_id = entries[0].layer_id
        timing = self.begin_layer(layer_id)
        stream_entries = [e for e in entries if self._needs_stream_read(e)]
        out: dict[str, torch.Tensor] = {}
        for entry in entries:
            if not self._needs_stream_read(entry):
                out[entry.name] = self._get_resident_tensor(entry, timing)
        if stream_entries:
            cached = self._try_cached_layer(layer_id, stream_entries, timing)
            if cached is not None:
                out.update(cached)
                if self._bound_provider_cache and 0 <= layer_id < 9000:
                    # Dense sequence backends use this materialized contract
                    # directly.  Keep the same LRU accounting as the RWKV
                    # load path so a configured provider cache remains
                    # bounded on cache hits as well as cold loads.
                    self._retain_provider_layer(layer_id)
                return out
            striped = (
                self._decode_striped_entries_coalesced(
                    stream_entries,
                    timing,
                    force_materialize=True,
                )
                if any(entry.stripes for entry in stream_entries)
                else None
            )
            if striped is not None:
                out.update(striped)
            else:
                span_read = self._read_layer_span_bytes(stream_entries, timing)
                if span_read is not None:
                    raw, base = span_read
                    out.update(
                        self._decode_from_layer_raw(
                            raw, base, stream_entries, timing, force_materialize=True
                        )
                    )
                else:
                    for entry in stream_entries:
                        out[entry.name] = self._load_one(entry, timing)
        if self._stream_layer_cache:
            streamed_names = {entry.name for entry in stream_entries}
            with self._cache_lock:
                for name, tensor in out.items():
                    if name in streamed_names:
                        self._cache[name] = tensor
        if self._bound_provider_cache and 0 <= layer_id < 9000:
            self._retain_provider_layer(layer_id)
        return out

    def load_layer_tensors_native(
        self, entries: list[TensorEntry]
    ) -> dict[str, torch.Tensor | bytes | memoryview | bytearray]:
        """Load a layer for the native rwkv.cpp grouped-U8 bridge.

        Matrix entries in the SG8 grouped-U8 format are returned as packed
        byte views.  rwkv.cpp owns the copy and performs the dequantized GEMV
        (or the small elementwise decode) on the CPU.  Dense/vector controls
        still return normal PyTorch tensors because GGML's existing upload ABI
        is the correct representation for them.

        This is intentionally separate from ``load_layer_tensors_materialized``:
        calling the latter first would allocate a float32 staging slab and
        defeat the native-U8 residency path.
        """
        if not entries:
            return {}
        layer_id = entries[0].layer_id
        timing = self.begin_layer(layer_id)

        if self._bound_provider_cache and 0 <= layer_id < 9000:
            with self._cache_lock:
                cached_native = self._native_layer_views.get(layer_id)
            if cached_native is not None and all(
                entry.name in cached_native for entry in entries
            ):
                timing.layer_cache_hits += len(entries)
                self._retain_provider_layer(layer_id)
                return cached_native

        stream_entries = [entry for entry in entries if self._needs_stream_read(entry)]
        out: dict[str, torch.Tensor | bytes | memoryview | bytearray] = {}

        # A native rwkv.cpp layer plan must receive the original grouped-U8
        # record for matrix tensors.  In partial mode the residency planner
        # may preload a whole hot layer into ``_cache``; using that decoded
        # BF16 tensor here would take the dense upload path, lose the native
        # grouped-U8 orientation metadata, and can feed a packed-only GGML
        # plan an invalid matrix layout.  Keep globals materialized (the
        # backend's embedding/norm/head path needs them), but preserve SG8
        # records for every block matrix regardless of its residency flag.
        resident_native_matrices = [
            entry
            for entry in entries
            if not self._needs_stream_read(entry)
            and (entry.dequant or "none").strip().lower() == "scale_u8_grouped"
            and len(entry.shape) == 2
        ]
        resident_native_matrix_names = {
            entry.name for entry in resident_native_matrices
        }

        for entry in entries:
            if (
                not self._needs_stream_read(entry)
                and entry.name not in resident_native_matrix_names
            ):
                out[entry.name] = self._get_resident_tensor(entry, timing)

        for entry in resident_native_matrices:
            with self._cache_lock:
                blob = self._native_resident_blobs.get(entry.name)
            if blob is None:
                blob = self._read_packed_bytes(
                    entry,
                    timing,
                    prefer_memoryview=self._native_layer_streaming,
                )
                with self._cache_lock:
                    # Another caller cannot normally race a single engine, but
                    # setdefault keeps the cache deterministic for diagnostics
                    # that invoke the provider concurrently.
                    blob = self._native_resident_blobs.setdefault(entry.name, blob)
                    # The native upload no longer needs the eagerly decoded
                    # resident matrix.  Drop that duplicate when it came from
                    # the provider preload; globals are intentionally left in
                    # place because the backend's embedding/head path uses
                    # their dense values.
                    if entry.layer_id >= 0:
                        self._cache.pop(entry.name, None)
            # The native ABI copies the payload synchronously, so a short-lived
            # view is sufficient and avoids a second decoded matrix allocation.
            out[entry.name] = memoryview(blob)

        def add_raw_or_decode(
            entry: TensorEntry,
            blob: bytes | memoryview | bytearray,
        ) -> None:
            codec = (entry.dequant or "none").strip().lower()
            if codec == "scale_u8_grouped" and len(entry.shape) == 2:
                # memoryview keeps a coalesced span alive without a second
                # Python copy.  The native ABI copies synchronously into its
                # model-owned packed record.
                out[entry.name] = memoryview(blob)
            else:
                out[entry.name] = decode_weight_to_tensor(
                    blob,
                    entry,
                    self._device,
                    trinity_layer_cache=self._trinity_layer_cache,
                    decode_device=self._decode_device,
                )

        if stream_entries:
            span_read = self._read_layer_span_bytes(
                stream_entries,
                timing,
                prefer_memoryview=True,
            )
            if span_read is not None:
                raw, base = span_read
                raw_view = memoryview(raw)
                for entry in stream_entries:
                    rel = entry.offset - base
                    add_raw_or_decode(
                        entry,
                        raw_view[rel : rel + entry.length],
                    )
            else:
                for entry in stream_entries:
                    add_raw_or_decode(entry, self._read_packed_bytes(entry, timing))

        if self._bound_provider_cache and 0 <= layer_id < 9000:
            with self._cache_lock:
                self._native_layer_views[layer_id] = out
            self._retain_provider_layer(layer_id)

        return out

    def native_layer_payloads_persistent(self, entries: list[TensorEntry]) -> bool:
        """Whether native borrowed payloads may be retained across layer begins.

        The bounded native cache stores pointers, not another packed copy.  A
        pointer is safe to retain when the provider itself owns the backing
        object for the lifetime of the engine: either the packed layer cache
        is enabled, or a partial-tier layer is already resident and its raw
        grouped-U8 matrices live in ``_native_resident_blobs``.  Streamed
        partial-tier layers must remain transient so a pread/chunk buffer is
        not accidentally promoted into an unbounded native lifetime cache.
        """
        if not entries:
            return False
        if self._stream_layer_cache:
            return True
        if self.mode == "partial" and all(
            not self._needs_stream_read(entry) for entry in entries
        ):
            return True
        # A stable mmap alone is not enough for dense borrowing: the decoded
        # CPU tensor returned by ``load_layer_tensors_native`` is temporary
        # unless the provider retains the layer view.  Letting the native
        # wrapper retain it would bypass the provider's RAM budget.
        return False

    def native_packed_layer_payloads_persistent(
        self, entries: list[TensorEntry]
    ) -> bool:
        """Whether grouped-U8 payloads can be retained as mmap pointers.

        Grouped-U8 matrices are consumed directly from their packed bytes by
        rwkv.cpp. With a stable read-only mmap those bytes remain valid for
        the engine lifetime and do not need a second native copy. Dense
        controls in the same layer are intentionally excluded: they may be
        freshly decoded tensors and use the ordinary bounded dense-cache
        policy.
        """
        if not entries:
            return False
        if self._stream_layer_cache:
            return True
        if not bool(getattr(self._store, "stable_memoryviews", False)):
            return False
        packed = [
            entry
            for entry in entries
            if (entry.dequant or "none").strip().lower() == "scale_u8_grouped"
            and len(entry.shape) == 2
        ]
        return bool(packed)

    def native_dense_layer_payloads_persistent(
        self, entries: list[TensorEntry]
    ) -> bool:
        """Whether dense native pointers remain owned across layer switches."""
        if not entries:
            return False
        if self._stream_layer_cache:
            # A byte-capped provider cache may evict a decoded layer while the
            # native plan is still alive. Let rwkv.cpp own dense controls in
            # that case; an uncapped stream cache owns them for the engine
            # lifetime and can safely lend the pointers.
            return self._max_provider_cache_bytes <= 0
        return self.mode == "partial" and all(
            not self._needs_stream_read(entry) for entry in entries
        )

    def set_native_layer_invalidator(self, callback: Any | None) -> None:
        """Install the callback used to invalidate borrowed native views."""
        self._native_layer_invalidator = callback

    def release_native_global_cache(self, names: object) -> None:
        """Drop provider aliases after the native backend takes global ownership.

        The bounded rwkv.cpp path converts the embedding, norms, and output
        head once into its global host view.  Keeping the original tensors in
        ``_cache`` would make provider telemetry look like a second model copy
        and, for a float32 pack, would also keep a large duplicate alive.  The
        native backend remains the owner after this hand-off; ordinary
        ChatRWKV/sequence providers never call this method.
        """
        if not self._native_layer_streaming:
            return
        try:
            names_iter = list(names)  # type: ignore[arg-type]
        except TypeError:
            return
        with self._cache_lock:
            for name in names_iter:
                self._cache.pop(str(name), None)
        for name in names_iter:
            self._fused_lut_blobs.pop(str(name), None)
            self._fused_tmix_blobs.pop(str(name), None)

    def load_layer_tensors_dense(
        self, entries: list[TensorEntry]
    ) -> dict[str, torch.Tensor]:
        """Return every entry in ``entries`` as a dense tensor.

        ``load_layer_tensors_materialized`` already forces materialization of
        LUT-backed entries and bypasses the RWKV fused-inject filtering.  Keep
        this named adapter separate so sequence backends never need to reach
        into provider implementation details.
        """
        if not entries:
            return {}
        layer_id = entries[0].layer_id
        if self._stream_layer_cache and 0 <= layer_id < 9000:
            with self._cache_lock:
                view = self._dense_layer_views.get(layer_id)
            if view is not None and all(entry.name in view for entry in entries):
                timing = self.begin_layer(layer_id)
                timing.layer_cache_hits += len(entries)
                self._retain_provider_layer(layer_id)
                return view

        out = self.load_layer_tensors_materialized(entries)
        if self._stream_layer_cache and 0 <= layer_id < 9000:
            with self._cache_lock:
                self._dense_layer_views[layer_id] = out
        return out

    def hint_prefetch_layer(self, entries: list[TensorEntry]) -> None:
        if not self._mmap_willneed or not entries:
            return
        resident = getattr(self._store, "probably_resident", None)
        if callable(resident) and resident(entries):
            self._metrics.page_cache_prefetch_skips += 1
            return
        fn = getattr(self._store, "advise_prefetch", None)
        if fn:
            fn(entries)

    def hint_release_layer(self, entries: list[TensorEntry]) -> None:
        if not self._mmap_dontneed or not entries:
            return
        fn = getattr(self._store, "advise_release", None)
        if fn:
            fn(entries)

    def prefetch_layer(self, entries: list[TensorEntry]) -> None:
        self.prefetch_entries(entries)

    def prefetch_entries(self, entries: list[TensorEntry]) -> None:
        if not self._executor or not entries or self.mode == "resident":
            return
        stream_entries = [
            e
            for e in entries
            if self._should_stream(e)
            and e.name not in self._cache
            and not self._layer_resident_in_z(e.layer_id)
            and not self.layer_prepared_for_forward(e.layer_id)
        ]
        if not stream_entries:
            return
        resident = getattr(self._store, "probably_resident", None)
        if self._prefetch_io_only and callable(resident) and resident(stream_entries):
            self._metrics.page_cache_prefetch_skips += 1
            return

        # For sharded packs: group layers by shard and issue one parallel
        # read per shard via the store's read_bytes_many. This is the
        # path that exploits multi-SSD bandwidth — layers on different
        # shards are read simultaneously rather than sequentially.
        # For single-file packs this falls through to the sequential
        # path (one layer at a time).
        if (
            hasattr(self._store, "read_bytes_many")
            and getattr(self._store, "_executor", None) is not None
        ):
            self._prefetch_entries_sharded(stream_entries)
            return

        def job() -> _PrefetchJobResult:
            from rwkv_ssd.runtime.decode_shadow import split_shadow_lut_entries

            by_layer: dict[int, list[TensorEntry]] = {}
            for entry in stream_entries:
                by_layer.setdefault(entry.layer_id, []).append(entry)
            raw_by_layer: dict[int, _LayerSpanPrefetch] = {}
            decoded: dict[str, torch.Tensor] = {}
            for group in by_layer.values():
                layer_id = group[0].layer_id
                if self._prefetch_io_only:
                    spans = _LayerSpanPrefetch()
                    if self._layer_uses_shadow(group):
                        shadow_e, lut_e = split_shadow_lut_entries(group)
                        if shadow_e and lut_e:
                            s_read, l_read = self._read_hybrid_layer_spans_parallel(
                                shadow_e, lut_e
                            )
                            if s_read is not None:
                                raw, base = s_read
                                spans.shadow = (
                                    self._retain_span_buffer(raw),
                                    base,
                                )
                            if l_read is not None:
                                raw, base = l_read
                                spans.weights = (
                                    self._retain_span_buffer(raw),
                                    base,
                                )
                        elif shadow_e:
                            read = self._read_layer_span_bytes(
                                shadow_e, use_shadow=True
                            )
                            if read is not None:
                                raw, base = read
                                spans.shadow = (
                                    self._retain_span_buffer(raw),
                                    base,
                                )
                        elif lut_e:
                            read = self._read_layer_span_bytes(lut_e)
                            if read is not None:
                                raw, base = read
                                spans.weights = (
                                    self._retain_span_buffer(raw),
                                    base,
                                )
                    else:
                        read = self._read_layer_span_bytes(group)
                        if read is not None:
                            raw, base = read
                            spans.weights = (
                                self._retain_span_buffer(raw),
                                base,
                            )
                    if spans.shadow is not None or spans.weights is not None:
                        raw_by_layer[layer_id] = spans
                else:
                    decoded.update(self._decode_stream_entries(group))
            return _PrefetchJobResult(raw_by_layer=raw_by_layer, tensors=decoded)

        with self._prefetch_lock:
            prev = self._prefetch_future
            if prev is not None and not prev.done():
                # The executor has one worker. Submitting another job while
                # this one is running would queue unbounded stale reads after
                # begin_layer abandons the future. Keep one opportunistic read
                # in flight and let the next call drain it when ready.
                return
            if prev is not None:
                # Previous prefetch already finished (typical for compute-
                # bound tiers). Drain and use the data — the worker has
                # already done the I/O + decode so this is a cheap dict
                # copy. ``prefetch_overlaps`` is incremented here because
                # the data was ready before the next prefetch was issued
                # (true overlap).
                self._prefetch_future = None
                try:
                    prev_result = prev.result()
                    for lid, spans in prev_result.raw_by_layer.items():
                        self._prefetch_raw[lid] = _LayerSpanPrefetch(
                            shadow=(
                                self._retain_span_buffer(spans.shadow[0]),
                                spans.shadow[1],
                            )
                            if spans.shadow is not None
                            else None,
                            weights=(
                                self._retain_span_buffer(spans.weights[0]),
                                spans.weights[1],
                            )
                            if spans.weights is not None
                            else None,
                        )
                    self._pending.update(prev_result.tensors)
                    if prev_result.raw_by_layer or prev_result.tensors:
                        self._metrics.prefetch_overlaps += 1
                except Exception as exc:
                    logger.warning("prefetch drain failed: %s", exc)
            self._prefetch_future = self._executor.submit(job)

    def _prefetch_entries_sharded(self, stream_entries: list[TensorEntry]) -> None:
        """Parallel cross-shard prefetch: read layers on different shards simultaneously.

        Groups entries by shard and issues one parallel read per shard
        via the store's ``read_layer_spans_parallel``. This is the
        path that exploits multi-SSD aggregate bandwidth — layers N,
        N+1, N+2 (on different shards) are read in parallel rather
        than sequentially.

        For a 4-shard pack with ``parallel_workers=4``, the aggregate
        bandwidth is up to 4x the per-SSD bandwidth.
        """
        from rwkv_ssd.runtime.layer_io import entries_layer_read_span

        by_layer: dict[int, list[TensorEntry]] = {}
        for entry in stream_entries:
            by_layer.setdefault(entry.layer_id, []).append(entry)

        if not by_layer:
            return

        # Build (shard_name, offset, length) specs for each layer.
        # The offset is local to the shard (not global weights.bin).
        layer_specs: list[tuple[str, int, int]] = []
        layer_to_spec_idx: dict[int, int] = {}
        for layer_id, group in by_layer.items():
            span = entries_layer_read_span(group)
            if span is None:
                continue
            base, total = span
            # All tensors in this layer are in the same shard
            # (shard_pack assigns by layer_id % n_shards).
            shard_name = group[0].shard_file
            if not shard_name:
                # Legacy single-file pack — shouldn't reach this path
                # but fall through gracefully.
                continue
            layer_to_spec_idx[layer_id] = len(layer_specs)
            layer_specs.append((shard_name, base, total))

        if not layer_specs:
            return

        def job() -> _PrefetchJobResult:
            raw_by_layer: dict[int, _LayerSpanPrefetch] = {}
            decoded: dict[str, torch.Tensor] = {}

            if self._prefetch_io_only:
                # Parallel read across shards: one pread per shard in flight.
                if hasattr(self._store, "read_layer_spans_parallel"):
                    results = self._store.read_layer_spans_parallel(layer_specs)
                else:
                    results = [
                        (self._store.read_bytes_for_shard(name, off, length), off)
                        for name, off, length in layer_specs
                    ]
                # Map results back to layer_ids
                spec_idx_to_layer: dict[int, int] = {}
                idx = 0
                for layer_id in by_layer:
                    if layer_id in layer_to_spec_idx:
                        spec_idx_to_layer[layer_to_spec_idx[layer_id]] = layer_id
                        idx += 1
                for spec_idx, (raw, base) in enumerate(results):
                    layer_id = spec_idx_to_layer.get(spec_idx)
                    if layer_id is None:
                        continue
                    raw_by_layer[layer_id] = _LayerSpanPrefetch(
                        weights=(self._retain_span_buffer(raw), base),
                    )
            else:
                # Full decode: each layer's tensors are decoded.
                for layer_id, group in by_layer.items():
                    if layer_id not in layer_to_spec_idx:
                        continue
                    try:
                        decoded.update(self._decode_stream_entries(group))
                    except Exception as exc:
                        logger.debug(
                            "sharded prefetch decode failed for layer %d: %s",
                            layer_id,
                            exc,
                        )
            return _PrefetchJobResult(raw_by_layer=raw_by_layer, tensors=decoded)

        with self._prefetch_lock:
            prev = self._prefetch_future
            if prev is not None and not prev.done():
                # Keep the single-worker queue bounded; the current job is
                # still a valid best-effort prefetch for one of these layers.
                return
            if prev is not None:
                self._prefetch_future = None
                try:
                    prev_result = prev.result()
                    for lid, spans in prev_result.raw_by_layer.items():
                        self._prefetch_raw[lid] = _LayerSpanPrefetch(
                            shadow=(
                                self._retain_span_buffer(spans.shadow[0]),
                                spans.shadow[1],
                            )
                            if spans.shadow is not None
                            else None,
                            weights=(
                                self._retain_span_buffer(spans.weights[0]),
                                spans.weights[1],
                            )
                            if spans.weights is not None
                            else None,
                        )
                    self._pending.update(prev_result.tensors)
                    if prev_result.raw_by_layer or prev_result.tensors:
                        self._metrics.prefetch_overlaps += 1
                except Exception as exc:
                    logger.warning("prefetch drain failed: %s", exc)
            self._prefetch_future = self._executor.submit(job)

    def begin_layer(self, layer_id: int) -> LayerTiming:
        timing = self._metrics.start_layer(layer_id)
        with self._prefetch_lock:
            fut = self._prefetch_future
            # Keep an unfinished future registered so a subsequent
            # prefetch_entries call does not queue another stale job. A ready
            # future is consumed below and can be replaced on the next call.
            if fut is not None and fut.done():
                self._prefetch_future = None
        if fut is not None:
            # Non-blocking drain: if the prefetch worker is still mid-I/O
            # we abandon the future (the worker's bytes are wasted, but the
            # main thread doesn't block on the SSD). The downstream
            # ``_try_cached_layer`` / ``_decode_stream_entries`` will then
            # fall through to the disk-read path — same cost as the no-cache
            # F-1 path. If the worker is already done (compute-heavy tier
            # like F5 / F2 with light I/O) we still get the zero-wait fast
            # path. ``prefetch_entries`` no longer blocks on the previous
            # future either, so submission is also non-blocking.
            if fut.done():
                try:
                    with Timer() as t_wait:
                        ready = fut.result()
                    timing.prefetch_wait_ms += t_wait.elapsed_ms
                    for lid, spans in ready.raw_by_layer.items():
                        self._prefetch_raw[lid] = _LayerSpanPrefetch(
                            shadow=(
                                self._retain_span_buffer(spans.shadow[0]),
                                spans.shadow[1],
                            )
                            if spans.shadow is not None
                            else None,
                            weights=(
                                self._retain_span_buffer(spans.weights[0]),
                                spans.weights[1],
                            )
                            if spans.weights is not None
                            else None,
                        )
                    self._pending.update(ready.tensors)
                    if self._stream_layer_cache:
                        for name, t in ready.tensors.items():
                            self._cache[name] = t
                except Exception as exc:
                    logger.warning("prefetch failed: %s", exc)
            else:
                # Prefetch not done; record the abandoned wait so the
                # prefetch_overlaps counter still reflects what happened
                # (zero — the data was not available in time).
                timing.prefetch_wait_ms += 0.0
        return timing

    @staticmethod
    def _byte_owner(value: bytes | memoryview | bytearray) -> object:
        """Return the root exporter for a possibly nested memoryview."""
        owner: object = value.obj if isinstance(value, memoryview) else value
        while isinstance(owner, memoryview):
            owner = owner.obj
        return owner

    def _retain_span_buffer(
        self, raw: bytes | memoryview | bytearray
    ) -> bytes | memoryview | bytearray:
        """Keep stable mmap views zero-copy; detach transient read buffers."""
        if bool(getattr(self._store, "stable_memoryviews", False)):
            return raw
        return _materialize_span_buffer(raw)

    def _owned_byte_size(
        self,
        value: bytes | memoryview | bytearray,
        seen_owners: set[int],
        *,
        exclude_owners: set[int] | None = None,
    ) -> int:
        """Count process-owned bytes once, excluding stable file-backed views."""
        owner = self._byte_owner(value)
        owner_id = id(owner)
        if exclude_owners and owner_id in exclude_owners:
            return 0
        if isinstance(value, memoryview) and bool(
            getattr(self._store, "stable_memoryviews", False)
        ):
            # A stable mmap view is not a Python-owned copy.  Its logical
            # payload is still reported by ``provider_mmap_bytes``.
            return 0
        if owner_id in seen_owners:
            return 0
        seen_owners.add(owner_id)
        if isinstance(owner, (bytes, bytearray, memoryview)):
            return len(owner)
        return len(value)

    def _layer_cached_bytes(self, layer_id: int) -> int:
        """Return owned bytes for one evictable provider layer.

        Stable mmap-backed grouped-U8 views are deliberately excluded from
        the provider RAM cap; they are file-backed and exposed separately.
        Native resident blobs are also excluded here because their owner is
        accounted once in ``provider_resident_bytes``.
        """
        prefix = f"blocks.{layer_id}."
        total = 0
        tensor_ids: set[int] = set()
        for key, tensor in self._cache.items():
            if key.startswith(prefix):
                tensor_ids.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
        prepared = self._prepared_layers.get(layer_id)
        if prepared is not None:
            for tensor in prepared.values():
                if id(tensor) not in tensor_ids:
                    tensor_ids.add(id(tensor))
                    total += tensor.numel() * tensor.element_size()
        with self._cache_lock:
            native_view = self._native_layer_views.get(layer_id)
            resident_blobs = tuple(self._native_resident_blobs.values())
        resident_owners = {id(self._byte_owner(value)) for value in resident_blobs}
        if native_view is not None:
            seen_owners: set[int] = set()
            for value in native_view.values():
                if isinstance(value, torch.Tensor):
                    if id(value) not in tensor_ids:
                        tensor_ids.add(id(value))
                        total += value.numel() * value.element_size()
                elif isinstance(value, (bytes, bytearray, memoryview)):
                    total += self._owned_byte_size(
                        value,
                        seen_owners,
                        exclude_owners=resident_owners,
                    )
        return total

    def _provider_cached_bytes(self) -> int:
        return sum(
            self._layer_cached_bytes(layer_id) for layer_id in self._provider_lru
        )

    def _retain_provider_layer(self, layer_id: int) -> None:
        if not self._bound_provider_cache or layer_id < 0:
            return
        explicit_byte_cap = self._max_provider_cache_bytes > 0
        if layer_id in self._z_retention.pinned_layer_ids and not explicit_byte_cap:
            return
        if self._strict_fused_retain_layers() and not explicit_byte_cap:
            if layer_id not in self._provider_lru:
                self._provider_lru.append(layer_id)
            return
        if layer_id in self._provider_lru:
            self._provider_lru.remove(layer_id)
        self._provider_lru.append(layer_id)
        cap = self._max_provider_cache_layers
        if cap > 0:
            while len(self._provider_lru) > cap:
                old = self._provider_lru.pop(0)
                self._provider_cache_evictions += 1
                self.evict_streamed_layer(old)
        byte_cap = self._max_provider_cache_bytes
        if byte_cap > 0:
            while self._provider_lru and self._provider_cached_bytes() > byte_cap:
                old = self._provider_lru.pop(0)
                self._provider_cache_evictions += 1
                self.evict_streamed_layer(old, force=explicit_byte_cap)

    def evict_streamed_layer(self, layer_id: int, *, force: bool = False) -> None:
        """Drop decoded tensors for one block layer (paired with z eviction).

        ``force=True`` evicts even when the strict-fused-retain guard is on, so
        the warm-disk-cache writer can reclaim the provider RAM after serializing
        the bf16 layer to ``.decode_cache/``.
        """
        prefix = f"blocks.{layer_id}."
        with self._cache_lock:
            for key in list(self._cache.keys()):
                if key.startswith(prefix):
                    del self._cache[key]
            self._dense_layer_views.pop(layer_id, None)
            self._native_layer_views.pop(layer_id, None)
        invalidator = self._native_layer_invalidator
        if callable(invalidator):
            # Native borrowed records are non-owning.  Invalidate them before
            # the provider releases the last tensor/view for this layer.
            invalidator(int(layer_id))
        if self._strict_fused_retain_layers() and not force:
            return
        self._prepared_layers.pop(layer_id, None)
        prefix = f"blocks.{layer_id}."
        for key in list(self._fused_lut_blobs.keys()):
            if key.startswith(prefix):
                del self._fused_lut_blobs[key]
        for key in list(self._fused_tmix_blobs.keys()):
            if key.startswith(prefix):
                del self._fused_tmix_blobs[key]
        if layer_id in self._packed_layer_lru:
            self._packed_layer_lru.remove(layer_id)

    def cached_layer_ids(self) -> list[int]:
        found: set[int] = set()
        with self._cache_lock:
            cache_keys = list(self._cache.keys())
            native_layer_ids = list(self._native_layer_views.keys())
        found.update(native_layer_ids)
        for key in cache_keys:
            if not key.startswith("blocks."):
                continue
            try:
                found.add(int(key.split("blocks.")[1].split(".")[0]))
            except (IndexError, ValueError):
                continue
        return sorted(found)

    def _provider_resident_owned_bytes(self) -> int:
        """Bytes owned outside the evictable provider-layer LRU."""
        total = 0
        seen_tensors: set[int] = set()
        seen_owners: set[int] = set()
        z = self._model_z
        with self._cache_lock:
            cache_snapshot = dict(self._cache)
            native_views = dict(self._native_layer_views)
            resident_blobs = tuple(self._native_resident_blobs.values())
        lru_layers = set(self._provider_lru)

        for name, tensor in cache_snapshot.items():
            layer_id = -1
            if name.startswith("blocks."):
                try:
                    layer_id = int(name.split(".", 2)[1])
                except (IndexError, ValueError):
                    layer_id = -1
            if layer_id in lru_layers:
                continue
            if z is not None and name in z:
                continue
            if id(tensor) not in seen_tensors:
                seen_tensors.add(id(tensor))
                total += tensor.numel() * tensor.element_size()

        for layer_id, prepared in self._prepared_layers.items():
            if layer_id in lru_layers:
                continue
            for tensor in prepared.values():
                if id(tensor) not in seen_tensors:
                    seen_tensors.add(id(tensor))
                    total += tensor.numel() * tensor.element_size()

        resident_owners = {id(self._byte_owner(value)) for value in resident_blobs}
        for value in resident_blobs:
            total += self._owned_byte_size(value, seen_owners)

        # Count native views that are not part of the LRU.  Resident packed
        # owners are already accounted above; mapped views remain a separate
        # non-owned category and do not inflate the RAM total.
        for layer_id, native_view in native_views.items():
            if layer_id in lru_layers:
                continue
            for value in native_view.values():
                if isinstance(value, torch.Tensor):
                    if id(value) not in seen_tensors:
                        seen_tensors.add(id(value))
                        total += value.numel() * value.element_size()
                elif isinstance(value, (bytes, bytearray, memoryview)):
                    total += self._owned_byte_size(
                        value,
                        seen_owners,
                        exclude_owners=resident_owners,
                    )

        # Fused Python blobs are ordinary provider ownership unless they are
        # mmap views.  Dedup exact aliases (not distinct slices of one mmap).
        for blob, _out_f, _in_f in self._fused_lut_blobs.values():
            if isinstance(blob, (bytes, bytearray, memoryview)):
                total += self._owned_byte_size(
                    blob,
                    seen_owners,
                    exclude_owners=resident_owners,
                )
        for blobs, _out_f, _in_f in self._fused_tmix_blobs.values():
            for blob in blobs:
                if isinstance(blob, (bytes, bytearray, memoryview)):
                    total += self._owned_byte_size(
                        blob,
                        seen_owners,
                        exclude_owners=resident_owners,
                    )
        return total

    def provider_mmap_bytes(self) -> int:
        """Logical payload bytes retained through stable file-backed views."""
        total = 0
        seen_objects: set[int] = set()
        with self._cache_lock:
            resident_blobs = tuple(self._native_resident_blobs.values())
            native_views = tuple(self._native_layer_views.values())
        resident_owners = {id(self._byte_owner(value)) for value in resident_blobs}

        def add(value: object, *, skip_resident_owner: bool = False) -> None:
            nonlocal total
            if not isinstance(value, memoryview):
                return
            if not bool(getattr(self._store, "stable_memoryviews", False)):
                return
            owner_id = id(self._byte_owner(value))
            if skip_resident_owner and owner_id in resident_owners:
                return
            value_id = id(value)
            if value_id in seen_objects:
                return
            seen_objects.add(value_id)
            total += len(value)

        for value in resident_blobs:
            add(value)
        for native_view in native_views:
            for value in native_view.values():
                add(value, skip_resident_owner=True)
        for blob, _out_f, _in_f in self._fused_lut_blobs.values():
            add(blob)
        for blobs, _out_f, _in_f in self._fused_tmix_blobs.values():
            for blob in blobs:
                add(blob)
        return total

    def cached_weight_bytes(self) -> int:
        """Process-owned provider bytes, with mapped payloads reported separately."""
        return self._provider_cached_bytes() + self._provider_resident_owned_bytes()

    def packed_weight_bytes(self) -> int:
        """Logical bytes retained by packed provider views and fused blobs.

        This is a payload/ownership diagnostic rather than a process-RAM
        number.  Stable mmap slices are included here and in
        ``provider_mmap_bytes``; ``cached_weight_bytes`` uses owned bytes only.
        Resident native blobs are not counted a second time through their
        layer views.
        """
        seen: set[int] = set()
        total = 0
        for blob, _out_f, _in_f in self._fused_lut_blobs.values():
            if id(blob) not in seen:
                seen.add(id(blob))
                total += len(blob)
        for blobs, _out_f, _in_f in self._fused_tmix_blobs.values():
            for blob in blobs:
                if id(blob) not in seen:
                    seen.add(id(blob))
                    total += len(blob)
        with self._cache_lock:
            resident_blobs = tuple(self._native_resident_blobs.values())
            native_views = tuple(self._native_layer_views.values())
        resident_owners = {id(self._byte_owner(value)) for value in resident_blobs}
        for blob in resident_blobs:
            if id(blob) not in seen:
                seen.add(id(blob))
                total += len(blob)
        for view in native_views:
            for value in view.values():
                if not isinstance(value, (bytes, bytearray, memoryview)):
                    continue
                if id(self._byte_owner(value)) in resident_owners:
                    continue
                if id(value) not in seen:
                    seen.add(id(value))
                    total += len(value)
        return total

    def prepared_weight_bytes(self) -> int:
        """Bytes retained by decoded/prepared tensors, excluding packed blobs."""
        total = 0
        with self._cache_lock:
            cache_snapshot = dict(self._cache)
        z = self._model_z
        for name, tensor in cache_snapshot.items():
            if z is not None and name in z:
                continue
            total += tensor.numel() * tensor.element_size()
        for prepared in self._prepared_layers.values():
            for tensor in prepared.values():
                total += tensor.numel() * tensor.element_size()
        return total

    def cache_stats(self) -> dict[str, object]:
        from rwkv_ssd.runtime.lut_gemm_fused import lut2_packed_cache_stats

        packed_global = lut2_packed_cache_stats()
        provider_layer = self._provider_cached_bytes()
        provider_resident = self._provider_resident_owned_bytes()
        provider_total = provider_layer + provider_resident
        return {
            "cache_format": self._cache_format,
            "packed_cache_bytes": self.packed_weight_bytes(),
            "prepared_cache_bytes": self.prepared_weight_bytes(),
            "provider_cache_bytes": provider_total,
            "provider_layer_cache_bytes": provider_layer,
            "provider_resident_bytes": provider_resident,
            "provider_mmap_bytes": self.provider_mmap_bytes(),
            "provider_total_bytes": provider_total,
            "provider_cache_limit_bytes": self._max_provider_cache_bytes,
            "packed_cache_limit_bytes": self._max_packed_cache_bytes,
            "packed_cache_evictions": self._packed_cache_evictions,
            "provider_cache_evictions": self._provider_cache_evictions,
            "lut2_index_cache_bytes": packed_global["bytes"],
            "lut2_index_cache_entries": packed_global["entries"],
            "cmix_zero_elements": self._cmix_zero_elements,
            "cmix_active_elements": self._cmix_active_elements,
            "cmix_total_elements": self._cmix_total_elements,
            "cmix_samples": self._cmix_samples,
            "cmix_tile_size": (
                self._cmix_tile_size if self._cmix_tile_stats_enabled else 0
            ),
            "cmix_tile_samples": self._cmix_tile_total,
            "cmix_tile_active_fraction": (
                self._cmix_tile_active / self._cmix_tile_total
                if self._cmix_tile_total
                else 0.0
            ),
            "cmix_tile_occupancy": dict(self._cmix_tile_occupancy),
            "cmix_selective_tiles_read": self._cmix_selective_tiles_read,
            "cmix_selective_tiles_skipped": self._cmix_selective_tiles_skipped,
            "cmix_selective_bytes_read": self._cmix_selective_bytes_read,
            "cmix_prefetch_hits": self._cmix_prefetch_hits,
            "cmix_prefetch_misses": self._cmix_prefetch_misses,
            "cmix_prefetch_wasted_tiles": self._cmix_prefetch_wasted_tiles,
            "cmix_prefetch_bytes_submitted": self._cmix_prefetch_bytes_submitted,
            "cmix_hot_cache_hits": self._cmix_hot_cache_hits,
            "cmix_hot_cache_bytes": sum(
                int(getattr(matrix, "hot_cache_bytes", 0))
                for matrix in self._cmix_tiled_values.values()
            ),
        }

    def record_cmix_sparsity(self, activation: torch.Tensor) -> None:
        """Record post-ReLU CMix activation sparsity when enabled."""
        if not self._cmix_sparsity_enabled:
            return
        with torch.no_grad():
            flat = activation.detach().reshape(-1)
            total = int(flat.numel())
            zero = int(torch.count_nonzero(flat == 0).item()) if total else 0
            active = total - zero
            self._cmix_zero_elements += zero
            self._cmix_active_elements += active
            self._cmix_total_elements += total
            self._cmix_samples += 1
            if self._cmix_tile_stats_enabled and total:
                tile_size = self._cmix_tile_size
                for start in range(0, total, tile_size):
                    tile = flat[start : start + tile_size]
                    tile_count = int(tile.numel())
                    active_count = int(torch.count_nonzero(tile != 0).item())
                    self._cmix_tile_total += 1
                    self._cmix_tile_active += int(active_count > 0)
                    # Ten fixed buckets make the optional diagnostic compact
                    # and comparable across runs: 0.0 through 1.0 occupancy.
                    bucket = min(
                        10,
                        int((active_count * 10 + tile_count - 1) / tile_count),
                    )
                    key = f"{bucket / 10:.1f}"
                    self._cmix_tile_occupancy[key] = (
                        self._cmix_tile_occupancy.get(key, 0) + 1
                    )

    def release_all_streamed_layers(self) -> None:
        """Drop all decoded block layers from provider RAM (weights live in ``z``)."""
        block_ids: set[int] = set(self._provider_lru)
        block_ids.update(self._prepared_layers.keys())
        with self._cache_lock:
            cache_keys = list(self._cache.keys())
            block_ids.update(self._native_layer_views.keys())
        for key in cache_keys:
            if not key.startswith("blocks."):
                continue
            try:
                block_ids.add(int(key.split("blocks.")[1].split(".")[0]))
            except (IndexError, ValueError):
                continue
        for layer_id in sorted(block_ids):
            self.evict_streamed_layer(layer_id)
        self._provider_lru.clear()

    def end_layer(self, timing: LayerTiming) -> None:
        pass

    def close(self) -> None:
        from rwkv_ssd.runtime.lut_gemm_fused import clear_lut2_packed_cache

        self._fused_lut_blobs.clear()
        self._fused_tmix_blobs.clear()
        with self._cache_lock:
            self._dense_layer_views.clear()
            self._native_resident_blobs.clear()
            self._native_layer_views.clear()
        for matrix in set(id(value) for value in self._cmix_tiled_values.values()):
            for value in self._cmix_tiled_values.values():
                if id(value) == matrix and hasattr(value, "close"):
                    value.close()
                    break
        self._cmix_tiled_values.clear()
        if self._deadline_router is not None:
            self._deadline_router.close()
            self._deadline_router = None
        clear_lut2_packed_cache()
        if self._disk_cache is not None:
            self._disk_cache.close()
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
