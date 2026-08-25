"""Inference engine orchestration — V0 through M4."""

from __future__ import annotations

import json
import logging
import os
import queue
import time
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
from rwkv_ssd.backends.factory import create_backend
from rwkv_ssd.backends.pack_backend import PackBackend
from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.device import resolve_device
from rwkv_ssd.runtime.errors import CapabilityNotSupportedError
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.rwkv7_skeleton import estimate_z_bytes
from rwkv_ssd.runtime.pack_generation import generate_greedy_tokens
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from rwkv_ssd.runtime.sampling import sampling_context
from rwkv_ssd.runtime.pack_verify import load_manifest_checked
from rwkv_ssd.runtime.residency import apply_residency_policy, load_residency_profile
from rwkv_ssd.runtime.scheduler import LayerScheduler
from rwkv_ssd.runtime.staging import PingPongStaging
from rwkv_ssd.runtime.state_cache import PrefixStateCache
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_provider import (
    ManifestWeightProvider,
    pack_uses_quant_codec,
)
from rwkv_ssd.runtime.weight_store import WeightStore, open_weight_store

logger = logging.getLogger(__name__)

# Backwards-compatible alias
RuntimeConfig = EngineConfig


def _default_cpu_threads(
    *, n_embd: int | None = None, fused_lut: bool = False
) -> int:
    """Pick a sane Torch intra-op thread count for CPU ChatRWKV.

    Small models (0.1B, ``n_embd`` ≤ 1024) prefer 1 thread on Windows/XPU Torch
    because tiny BF16 GEMVs lose to OpenMP overhead. Larger models (2.9B+) need
    more threads — measured ~1.7 tok/s @1 vs ~3.0 tok/s @8 on 2.9B bf16.
    Quantized fused CPU decode is different: its native AVX2/OpenMP GEMVs
    already own the large matrix parallelism, while the remaining Torch work
    is mostly layer/group normalization and small BF16 projections.  On the
    2.9B grouped-U8 path, eight Torch threads beat the old four-thread default
    after the small TMix projections moved to native grouped kernels; keep the
    count bounded by the host CPU and let callers override it explicitly.
    """
    import os as _os

    cpu = _os.cpu_count() or 4
    if n_embd is not None and n_embd >= 2048:
        if fused_lut:
            return max(1, min(8, cpu))
        return max(1, min(8, cpu))
    if n_embd is not None and n_embd >= 1280:
        return max(1, min(4, cpu))
    return 1


def _apply_cpu_thread_defaults(
    device: torch.device, *, n_embd: int | None = None, fused_lut: bool = False
) -> None:
    """Tune Torch CPU threading for RWKV GEMVs.

    Respects explicit ``RWKV_CPU_THREADS`` / OMP / MKL / TORCH thread env vars.
    ``auto`` (default) scales with model width when ``n_embd`` is known.  The
    native grouped-U8 GEMV has its own OpenMP pool; when it is active, seed a
    conservative ``OMP_NUM_THREADS`` default unless the caller supplied one.
    """
    if device.type != "cpu":
        return
    cpu = os.cpu_count() or 4
    raw = os.environ.get("RWKV_CPU_THREADS")
    if raw is None and any(
        os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
    ):
        return
    if raw is None or raw.strip().lower() in ("", "auto"):
        threads = _default_cpu_threads(n_embd=n_embd, fused_lut=fused_lut)
    elif raw.strip().lower() in ("0", "false", "off", "no"):
        return
    else:
        try:
            threads = max(1, int(raw.strip()))
        except ValueError:
            logger.warning("Ignoring invalid RWKV_CPU_THREADS=%r", raw)
            return
    torch.set_num_threads(threads)
    interop_raw = os.environ.get("RWKV_CPU_INTEROP_THREADS")
    if interop_raw is not None:
        try:
            torch.set_num_interop_threads(max(1, int(interop_raw)))
        except (RuntimeError, ValueError):
            logger.warning("Ignoring invalid RWKV_CPU_INTEROP_THREADS=%r", interop_raw)
    elif "TORCH_NUM_INTEROP_THREADS" not in os.environ:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if fused_lut and not os.environ.get("OMP_NUM_THREADS"):
        native_raw = os.environ.get("RWKV_LUT2_OMP_THREADS", "auto").strip().lower()
        if native_raw in ("", "auto"):
            native_threads = max(1, min(8, cpu))
        elif native_raw in ("0", "false", "off", "no"):
            native_threads = 0
        else:
            try:
                native_threads = max(1, int(native_raw))
            except ValueError:
                logger.warning("Ignoring invalid RWKV_LUT2_OMP_THREADS=%r", native_raw)
                native_threads = max(1, min(8, cpu))
        if native_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(native_threads)


def _record_cache_write_stats(
    metrics: MetricsCollector,
    last_submits: int = 0,
    last_sync_ms: float = 0.0,
) -> tuple[int, float]:
    """Copy module-level disk-cache write counters into ``metrics`` (DNV-2 telemetry).

    ``cache_write_sync_ms`` is the sum of encode_ms (sync CPU work on
    the calling thread) and write_ms (actual disk write, async via
    thread pool). Near-zero on the write side means writes fully
    overlap with the next layer's compute.

    The module-level counters are global (accumulate across all
    ``generate()`` calls in the process). To get per-call metrics we
    track the last seen values and report the delta. Returns the
    updated last-seen tuple so the caller can carry it forward.
    """
    from rwkv_ssd.runtime.decode_disk_cache import cache_write_stats

    submits, sync_ms = cache_write_stats()
    delta_submits = int(submits) - last_submits
    delta_sync_ms = float(sync_ms) - last_sync_ms
    metrics.cache_write_submits = max(0, delta_submits)
    metrics.cache_write_sync_ms = max(0.0, delta_sync_ms)
    return int(submits), float(sync_ms)


def _resident_layer_ids(tensors) -> set[int]:
    """Block layer ids with at least one resident block tensor (excluding ln0-only)."""
    by_layer: dict[int, list[str]] = {}
    for entry in tensors:
        if entry.residency != "resident" or not entry.name.startswith("blocks."):
            continue
        lid = entry.layer_id
        if 0 <= lid < 9000:
            by_layer.setdefault(lid, []).append(entry.name)
    out: set[int] = set()
    for lid, names in by_layer.items():
        if any(".ln0." not in n for n in names):
            out.add(lid)
    return out


class InferenceEngine:
    """Orchestrates pack load, weight residency, and generation."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.metrics = MetricsCollector()
        self.backend = create_backend(config.backend)
        self.manifest: Manifest | None = None
        self.store: WeightStore | None = None
        self.scheduler: LayerScheduler | None = None
        self.staging: PingPongStaging | None = None
        self._device = resolve_device(config.device)
        # An Intel XPU can still accelerate packed LUT gathers when its
        # matrix backend is unavailable. Preserve that explicitly requested
        # placement after demoting model computation to CPU.
        if (
            config.device.lower().strip().startswith("xpu")
            and self._device.type == "cpu"
            and config.trinity_decode_device.strip().lower() == "auto"
        ):
            from rwkv_ssd.runtime.device import xpu_runtime_available

            if xpu_runtime_available():
                config.trinity_decode_device = "xpu"
                logger.info(
                    "XPU matrix compute is unavailable; using CPU model compute "
                    "with XPU Trinity LUT decode."
                )
        _apply_cpu_thread_defaults(self._device)
        self._prefix_cache = (
            PrefixStateCache(
                max_entries=config.prefix_cache_max_entries,
                disk_dir=config.pack_dir,
            )
            if config.state_cache
            else None
        )
        self._pack_provider: ManifestWeightProvider | None = None
        self._streaming_provider: ManifestWeightProvider | None = None
        # Exactly one full startup reuse sweep is selected for a streaming
        # engine.  Keeping this explicit prevents provider, native, and disk
        # warmers from independently traversing the same pack.
        self._startup_warm_plan = "none"
        # Last seen module-level disk-cache write counters (so per-call
        # metrics report the delta, not the process-wide total).
        self._last_cache_write_submits = 0
        self._last_cache_write_sync_ms = 0.0
        self._cache_format_was_explicit = config.cache_format != "auto"
        self._adaptive_residency = None
        self._adaptive_pending = None
        self._session_promotion_pending = None
        self._session_promotion_last_plan = None
        self._session_tokens_seen = 0
        self._session_request_index = 0
        self._session_observation_start = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def prefix_cache(self) -> PrefixStateCache | None:
        return self._prefix_cache

    @contextmanager
    def _sampling_scope(
        self,
        *,
        temperature: float | None = None,
        greedy: bool | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ):
        """Apply request sampling options without leaking them to the engine."""

        old_temperature = self.config.temperature
        old_greedy = self.config.greedy
        requested_temperature = (
            old_temperature if temperature is None else float(temperature)
        )
        requested_greedy = old_greedy if greedy is None else bool(greedy)
        if temperature is not None and greedy is None and requested_temperature > 0.0:
            requested_greedy = False
        self.config.temperature = requested_temperature
        self.config.greedy = requested_greedy
        requested_top_p = self.config.top_p if top_p is None else float(top_p)
        requested_seed = self.config.seed if seed is None else int(seed)
        try:
            with sampling_context(seed=requested_seed, top_p=requested_top_p):
                yield
        finally:
            self.config.temperature = old_temperature
            self.config.greedy = old_greedy

    def __enter__(self) -> InferenceEngine:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def save_snapshot(self, path: str | Path, *, prompt: str = "") -> Path:
        """Snapshot the engine's current recurrent state to a file.

        Distinct from :class:`PrefixStateCache`, which is keyed by prefix text
        for TTFT amortization. Snapshots are portable artifacts that include
        enough engine config to reconstruct an ``InferenceEngine`` and resume.
        """
        from rwkv_ssd.runtime.snapshot import (
            SnapshotMeta,
            pack_identity,
            save_snapshot as _save_snapshot,
        )

        state = self.backend.get_recurrent_state()
        if state is None:
            raise RuntimeError(
                "backend has no current recurrent state — call engine.generate() first"
            )
        sampling_meta: dict[str, object] = {
            "greedy": bool(self.config.greedy),
            "temperature": float(self.config.temperature),
        }
        # Keep old snapshot metadata byte-for-byte compatible for the default
        # sampler while recording the additive controls when they are active.
        if float(self.config.top_p) != 1.0:
            sampling_meta["top_p"] = float(self.config.top_p)
        if self.config.seed is not None:
            sampling_meta["seed"] = int(self.config.seed)
        model_family = (
            str(self.manifest.meta.get("model_family", "rwkv7"))
            if self.manifest is not None
            else str(getattr(self.backend, "model_family", "rwkv7"))
        )
        meta = SnapshotMeta(
            backend=self.config.backend,
            mode=self.config.mode,
            model_family=model_family,
            max_layers_in_z=int(self.config.max_layers_in_z),
            decouple_provider_cache=bool(self.config.decouple_provider_cache),
            max_provider_cache_layers=int(self.config.max_provider_cache_layers),
            prompt=prompt,
            last_token_id=int(state.last_token_id),
            extras={
                "pack_identity": pack_identity(self.config.pack_dir),
                "sampling": sampling_meta,
            },
        )
        return _save_snapshot(path, state, meta)

    def load_snapshot(self, path: str | Path) -> None:
        """Restore a previously saved snapshot into the current engine."""
        from rwkv_ssd.runtime.snapshot import (
            load_snapshot as _load_snapshot,
            verify_snapshot_compatibility,
        )

        state, meta = _load_snapshot(path)
        verify_snapshot_compatibility(
            meta, self.config.pack_dir, backend=self.config.backend
        )
        self.backend.set_recurrent_state(state)

    @classmethod
    def from_snapshot(
        cls,
        snapshot_path: str | Path,
        pack_dir: Path,
        *,
        max_tokens: int = 64,
        device: str = "cpu",
        strategy: str = "cpu bf16",
        checkpoint_path: str | None = None,
        system_prefix: str = "",
    ) -> InferenceEngine:
        """Open a new engine from a snapshot file and restore its state.

        Convenience: a fresh engine is configured from the snapshot's stored
        config, the snapshot's recurrent state is restored, and the engine is
        returned ready for ``engine.generate(snapshot_meta.prompt)``.
        """
        from rwkv_ssd.runtime.config import EngineConfig
        from rwkv_ssd.runtime.snapshot import (
            load_snapshot as _load_snapshot,
            verify_snapshot_compatibility,
        )

        state, meta = _load_snapshot(snapshot_path)
        verify_snapshot_compatibility(meta, pack_dir, backend=meta.backend)
        sampling = dict(meta.extras.get("sampling", {}) or {})
        cfg = EngineConfig(
            pack_dir=pack_dir,
            mode=meta.mode,
            backend=meta.backend,
            device=device,
            max_tokens=max_tokens,
            checkpoint_path=checkpoint_path,
            strategy=strategy,
            max_layers_in_z=meta.max_layers_in_z,
            decouple_provider_cache=meta.decouple_provider_cache,
            max_provider_cache_layers=meta.max_provider_cache_layers,
            system_prefix=system_prefix,
            greedy=bool(sampling.get("greedy", True)),
            temperature=float(sampling.get("temperature", 1.0)),
            top_p=float(sampling.get("top_p", 1.0)),
            seed=(int(sampling["seed"]) if sampling.get("seed") is not None else None),
        )
        eng = cls(cfg)
        eng.load()
        eng.backend.set_recurrent_state(state)
        return eng

    def load(self) -> None:
        from rwkv_ssd.runtime.pack_profiles import resolve_trinity_pack

        if self._device.type == "xpu" and isinstance(self.backend, RWKVCppBackend):
            raise RuntimeError(
                "rwkv.cpp does not currently expose the engine's Intel XPU path; "
                "use --backend chatrwkv with the RWKV-7 skeleton instead."
            )
        if self.config.session_promotion and self.config.adaptive_residency:
            raise ValueError(
                "session_promotion and adaptive_residency are alternative controllers; "
                "enable only one"
            )
        if self.config.session_promotion_policy not in {
            "benefit_per_byte",
            "highest_stall",
            "lru",
        }:
            raise ValueError(
                f"unsupported session_promotion_policy: "
                f"{self.config.session_promotion_policy!r}"
            )

        resolved = resolve_trinity_pack(self.config.pack_dir)
        if resolved != self.config.pack_dir:
            logger.info("Trinity pack profile: %s", resolved)
            self.config.pack_dir = resolved
        logger.info("verifying pack %s ...", self.config.pack_dir)
        self.manifest = load_manifest_checked(
            self.config.pack_dir, check_hash=self.config.verify_hash
        )
        self._merge_meta_json()
        if self._device.type == "xpu" and self.manifest.meta.get("deepembed"):
            from rwkv_ssd.runtime.deepembed import (
                DEEP_EMBED_QKV_DEA,
                DEEP_EMBED_RWKV7A_V1,
            )

            variant = str(self.manifest.meta.get("deepembed_variant", ""))
            if variant != DEEP_EMBED_RWKV7A_V1:
                raise RuntimeError(
                    "Intel XPU currently supports ordinary RWKV-7 and the "
                    "RWKV7a-v1 Torch path, but not qkv/DEA DeepEmbed. "
                    f"This pack is {variant or DEEP_EMBED_QKV_DEA!r}; use "
                    "--device cpu or the resident CPU reference path."
                )
        deepembed_sidecar_missing = bool(
            self.manifest.meta.get("deepembed_sidecar_required")
            and not (self.config.pack_dir / "DeepEmbed.bin").is_file()
        )
        if self.manifest.meta.get("deepembed") and isinstance(self.backend, RWKVCppBackend):
            raise CapabilityNotSupportedError(
                "deepembed",
                "rwkvcpp",
                "rwkv.cpp has no DeepEmbed tensor/sidecar execution ABI; "
                "use the ChatRWKV reference backend",
            )
        if (
            self.manifest.meta.get("deepembed")
            and self.config.mode != "resident"
            and (
                not self.manifest.meta.get("deepembed_streaming_supported", False)
                or deepembed_sidecar_missing
            )
        ):
            raise ValueError(
                "This DeepEmbed pack cannot use CPU streaming: qkv/DEA requires "
                "DeepEmbed.bin and a variant-aware adapter is required. Use "
                "--mode resident, or repack with its required sidecar."
            )
        # Re-apply after meta is known so large models get more CPU threads.
        n_embd_meta = self.manifest.meta.get("n_embd")
        try:
            n_embd_i = int(n_embd_meta) if n_embd_meta is not None else None
        except (TypeError, ValueError):
            n_embd_i = None
        lut_kernel = os.environ.get("RWKV_LUT_KERNEL", "auto").strip().lower()
        lut_fused_override = os.environ.get("RWKV_LUT_GEMM_FUSED", "auto").strip().lower()
        fused_lut_cpu_hint = (
            self._device.type == "cpu"
            and self.config.backend == "chatrwkv"
            and self.config.mode in ("streaming", "partial")
            and pack_uses_quant_codec(self.manifest.tensors)
            and lut_kernel not in ("numpy", "numba", "python", "np")
            and lut_fused_override not in ("0", "false", "off", "no")
        )
        _apply_cpu_thread_defaults(
            self._device,
            n_embd=n_embd_i,
            fused_lut=fused_lut_cpu_hint,
        )
        if isinstance(self.backend, RWKVCppBackend):
            # rwkv.cpp must receive its native ggml thread count at model
            # construction time.  The manifest is already validated here,
            # so provide its width before backend.load(); explicit
            # RWKV_CPU_THREADS=N still wins inside the backend.
            self.backend.set_model_width_hint(n_embd_i)
            native_u8_pack = any(
                (entry.dequant or "").strip().lower() == "scale_u8_grouped"
                and len(entry.shape) == 2
                for entry in self.manifest.tensors
            )
            matrix_entries = [
                entry for entry in self.manifest.tensors if len(entry.shape) == 2
            ]
            native_u8_pack_complete = bool(matrix_entries) and all(
                (entry.dequant or "").strip().lower() == "scale_u8_grouped"
                for entry in matrix_entries
            )
            self.backend.set_native_u8_hint(
                self.config.mode != "resident" and native_u8_pack
            )
            set_packed_only = getattr(
                self.backend, "set_native_u8_packed_only_hint", None
            )
            if callable(set_packed_only):
                set_packed_only(
                    self.config.mode != "resident" and native_u8_pack_complete
                )
            set_layer_streaming = getattr(
                self.backend, "set_layer_streaming_hint", None
            )
            if callable(set_layer_streaming):
                set_layer_streaming(self.config.mode != "resident")

        num_layers = int(self.manifest.meta.get("n_layer", 0))
        if isinstance(self.backend, PackBackend):
            self.backend.load_pack(self.manifest, str(self._device))
            num_layers = self.backend.num_layers
        else:
            num_layers = num_layers or 32

        from rwkv_ssd.runtime.throughput_defaults import (
            apply_bounded_fused_defaults,
            apply_low_ram_defaults,
            apply_partial_defaults,
            apply_partial_fused_defaults,
            apply_partial_hot4_ssd_defaults,
            apply_partial_ssd_tier_defaults,
            apply_ssd_health_defaults,
            apply_ssd_stream_defaults,
            apply_ssd_tier_fused_defaults,
            apply_cache_format_defaults,
            apply_streaming_defaults,
        )

        ssd_health = os.environ.get("RWKV_SSD_HEALTH", "").strip().lower()
        bounded_stream = os.environ.get("RWKV_BOUNDED_STREAM", "").strip().lower()
        partial_fused = os.environ.get("RWKV_PARTIAL_FUSED", "").strip().lower()
        partial_hot4 = os.environ.get("RWKV_PARTIAL_HOT4", "").strip().lower()
        partial_ssd = os.environ.get("RWKV_PARTIAL_SSD_TIER", "").strip().lower()
        ssd_tier = os.environ.get("RWKV_SSD_TIER", "auto").strip().lower()
        if bounded_stream in ("1", "true", "on", "yes"):
            apply_bounded_fused_defaults(self.config, self.manifest)
        elif partial_fused in ("1", "true", "on", "yes"):
            apply_partial_fused_defaults(self.config, self.manifest)
        elif partial_hot4 in ("1", "true", "on", "yes"):
            apply_partial_hot4_ssd_defaults(self.config, self.manifest)
        elif partial_ssd in ("1", "true", "on", "yes"):
            apply_partial_ssd_tier_defaults(self.config, self.manifest)
        elif ssd_tier in (
            "1",
            "true",
            "on",
            "fused",
        ):
            apply_ssd_tier_fused_defaults(self.config, self.manifest)
        elif ssd_tier in (
            "stream",
            "ssd_stream",
        ):
            apply_ssd_stream_defaults(self.config, self.manifest)
        elif self.config.ram_budget_gb and self.config.ram_budget_gb > 0:
            from rwkv_ssd.runtime.ram_budget import (
                apply_ram_budget_to_config,
                apply_ram_budget_tier,
                select_ram_budget_tier,
            )

            tier = select_ram_budget_tier(float(self.config.ram_budget_gb))
            apply_ram_budget_tier(self.config, self.manifest, tier=tier)
            plan = apply_ram_budget_to_config(
                self.config, self.manifest, n_layer=num_layers or None
            )
            per_layer_mb = plan.per_layer_bytes / 1e6
            resident_count = len(plan.resident_layer_ids)
            logger.info(
                "ram_budget=%.1f GB -> tier %s: pin %d layers (%.0f MB/layer), "
                "max_z=%d, provider=%.0f MB, est_peak=%.1f GB (globals=%.0f MB)",
                plan.budget_gb,
                tier,
                resident_count,
                per_layer_mb,
                plan.max_layers_in_z,
                plan.max_provider_cache_bytes / 1e6,
                plan.estimated_peak_gb,
                plan.global_bytes / 1e6,
            )
        elif self.config.low_ram:
            apply_low_ram_defaults(self.config, self.manifest)
        elif ssd_health in ("1", "true", "on", "yes"):
            apply_ssd_health_defaults(self.config, self.manifest)
        else:
            apply_streaming_defaults(self.config, self.manifest)
        if self.config.cache_budget_auto and self.config.cache_budget_gb is None:
            from rwkv_ssd.runtime.cache_budget import resolve_auto_cache_budget_gb

            self.config.cache_budget_gb = resolve_auto_cache_budget_gb(
                self.config, self.manifest
            )
            logger.info(
                "cache_budget_auto -> %.2f GB provider cache cap",
                float(self.config.cache_budget_gb),
            )
        if self.config.cache_budget_gb is not None:
            cache_bytes = int(max(0.0, float(self.config.cache_budget_gb)) * 1e9)
            self.config.decouple_provider_cache = True
            self.config.max_provider_cache_bytes = cache_bytes
            self.config.max_provider_cache_layers = 0
            logger.info(
                "cache_budget=%.2f GB -> provider cache cap %.0f MB",
                float(self.config.cache_budget_gb),
                cache_bytes / 1e6,
            )
        if ssd_health in ("1", "true", "on", "yes"):
            from rwkv_ssd.runtime.throughput_defaults import apply_ssd_health_overlay

            apply_ssd_health_overlay(self.config, self.manifest)
        apply_partial_defaults(self.config, self.manifest)
        from rwkv_ssd.runtime.throughput_defaults import apply_auto_residency_policy

        if self.config.adaptive_residency and not self._cache_format_was_explicit:
            self.config.residency_policy = "auto"
        selected_cache = apply_auto_residency_policy(self.config, self.manifest)
        if self.config.residency_policy == "auto":
            logger.info("auto residency policy selected cache_format=%s", selected_cache)
        apply_cache_format_defaults(self.config)
        # F5/promote is the native resident-graph fast path.  Keep F1-F4 on
        # the bounded one-block ABI, but do not instantiate a layer-local
        # model for a profile that explicitly requests all weights to remain
        # hot.  The config remains ``streaming`` so provider/cache metrics
        # and the existing HTTP/CLI mode contract stay compatible; the
        # backend capability is the source of truth for which execution plan
        # was actually selected.
        set_layer_streaming = getattr(
            self.backend, "set_layer_streaming_hint", None
        )
        if isinstance(self.backend, RWKVCppBackend) and callable(set_layer_streaming):
            promote_raw = os.environ.get("RWKV_PROMOTE_FULL_Z", "auto").strip().lower()
            promote_full_native = self.config.warm_z or self.config.cache_format == "dense" or promote_raw in {
                "1",
                "true",
                "on",
                "yes",
            }
            set_layer_streaming(
                self.config.mode != "resident" and not promote_full_native
            )
            # The native layer-local plan has one active GGML weight arena.
            # F2-F4 may additionally retain a bounded decoded-payload LRU;
            # F1 stays strict and F5 remains the full resident graph.  An
            # explicit RWKVCPP_LAYER_CACHE_BYTES value overrides this auto
            # profile through the backend resolver.
            set_layer_cache = getattr(
                self.backend, "set_layer_cache_bytes_hint", None
            )
            if callable(set_layer_cache):
                native_layer_cache_default = 0
                if (
                    self.config.mode != "resident"
                    and not promote_full_native
                    and (
                        self.config.mode == "partial"
                        or self.config.stream_layer_cache
                    )
                ):
                    # Keep the native cache materially smaller than the
                    # provider/z budgets of F2-F4.  This is a profile default,
                    # not an unbounded model-sized allocation.
                    native_layer_cache_default = 64 * 1024 * 1024
                elif (
                    self.config.mode == "streaming"
                    and not promote_full_native
                ):
                    # F1 is the strictest tier, but the measured native
                    # payload cache is still bounded well below its provider
                    # budget and is enough to eliminate repeated SG8 decode
                    # work on revisit.  Keep it smaller than F2-F4.
                    native_layer_cache_default = 16 * 1024 * 1024
                set_layer_cache(native_layer_cache_default)
        # Expose the load-time decision even before the first generation call.
        # HTTP/CLI consumers can inspect metrics immediately after ``load``;
        # the provider refreshes the byte counters after generation.
        self.metrics.cache_format = self.config.cache_format
        if self.config.adaptive_residency and not self._cache_format_was_explicit:
            from rwkv_ssd.runtime.adaptive_residency import AdaptiveResidencyController

            self._adaptive_residency = AdaptiveResidencyController(
                initial_format=self.config.cache_format,
                window=self.config.adaptive_residency_window,
                min_dwell_tokens=self.config.adaptive_residency_min_dwell_tokens,
                hysteresis=self.config.adaptive_residency_hysteresis,
                max_changes=self.config.adaptive_residency_max_changes,
            )

        profile = self.config.residency_profile_inline
        if profile is None and self.config.residency_profile:
            profile = load_residency_profile(self.config.residency_profile)

        self.manifest.tensors = apply_residency_policy(
            self.manifest.tensors,
            self.config.mode,
            num_layers,
            profile=profile,
        )

        if isinstance(self.backend, PackBackend):
            refresh = getattr(self.backend, "refresh_manifest", None)
            if callable(refresh):
                refresh(self.manifest)
        else:
            ckpt = self.config.checkpoint_path
            if ckpt is None:
                source = self.manifest.meta.get("source_checkpoint")
                if source:
                    candidate = Path(str(source))
                    if not candidate.is_absolute():
                        beside_pack = self.config.pack_dir / candidate
                        if beside_pack.exists():
                            candidate = beside_pack
                    ckpt = candidate
            if not ckpt:
                raise ValueError(
                    f"{self.config.backend} backend requires --checkpoint or "
                    "meta.json source_checkpoint"
                )
            if not Path(str(ckpt)).exists():
                raise ValueError(
                    f"{self.config.backend} backend checkpoint was not found at "
                    f"{ckpt}; provide --checkpoint explicitly (pack metadata "
                    "stores only portable provenance)"
                )
            resident_layers = _resident_layer_ids(self.manifest.tensors)
            use_skeleton = (
                isinstance(self.backend, ChatRWKVBackend)
                and self.config.skeleton_load
                and (
                    self.config.mode != "resident"
                    or self._device.type == "xpu"
                )
                and self.manifest.meta.get("rwkv_version") == 7
            )
            if use_skeleton:
                logger.info(
                    "loading RWKV-7 skeleton from pack (no full .pth in RAM) ..."
                )
            else:
                logger.info(
                    "loading checkpoint %s (first run can take 10-30s on CPU) ...", ckpt
                )
            if isinstance(self.backend, ChatRWKVBackend):
                self.backend.load(
                    str(ckpt),
                    self.config.strategy,
                    str(self._device),
                    pack_meta=self.manifest.meta,
                    pack_dir=self.config.pack_dir if use_skeleton else None,
                    skeleton_load=use_skeleton,
                    resident_layer_ids=resident_layers,
                )
            else:
                self.backend.load(str(ckpt), self.config.strategy, str(self._device))
            num_layers = self.backend.num_layers or num_layers
            if use_skeleton and isinstance(self.backend, ChatRWKVBackend):
                z_mb = estimate_z_bytes(self.backend._model.z) / 1e6
                logger.info(
                    "skeleton z size ~%.1f MB (block weights stream from pack)", z_mb
                )
            if (
                isinstance(self.backend, ChatRWKVBackend)
                and self.backend._model is not None
            ):
                z = getattr(self.backend._model, "z", None)
                if isinstance(z, dict) and "emb.weight" in z:
                    self._device = z["emb.weight"].device
                    logger.info("weight/compute device synced to %s", self._device)

        # Sharded pack (M-class): when the manifest records multiple
        # shard files, open them all and route reads by ``shard_file``.
        # On a single SSD this is identical to the legacy single-file
        # store; on multiple SSDs (one shard per mount point) the
        # thread pool in the sharded store gives Kx aggregate
        # bandwidth for K shards.
        if self.manifest.is_sharded():
            from rwkv_ssd.runtime.weight_store_sharded import (
                open_sharded_weight_store,
            )

            self.store = open_sharded_weight_store(
                self.manifest,
                parallel_workers=max(1, len(self.manifest.shard_files)),
            )
        else:
            self.store = open_weight_store(
                self.manifest.weights_path,
                backend=self.config.io_backend,
                mmap_sequential=self.config.mmap_sequential,
                hedged=self.config.io_hedged,
            )
        self.shadow_store = None
        shadow_paths = self.manifest.shadow_paths()
        if len(shadow_paths) > 1:
            from rwkv_ssd.runtime.weight_store_sharded import ShardedWeightStore

            self.shadow_store = ShardedWeightStore(
                self.manifest,
                parallel_workers=len(shadow_paths),
                files=shadow_paths,
            )
            logger.info(
                "bf16 decode shadow: %d striped files (%s)",
                len(shadow_paths),
                ", ".join(path.name for path in shadow_paths),
            )
        elif shadow_paths:
            self.shadow_store = open_weight_store(
                shadow_paths[0],
                backend=self.config.io_backend,
                mmap_sequential=self.config.mmap_sequential,
            )
            logger.info("bf16 decode shadow: %s", shadow_paths[0].name)
        self.scheduler = LayerScheduler(self.manifest.by_layer())
        self._init_staging_if_needed()
        self._startup_warm_plan = self._select_startup_warm_plan()
        self._warm_z_if_needed()
        if self._startup_warm_plan == "provider":
            self._warm_provider_cache_if_needed()
        elif self._startup_warm_plan == "disk":
            self._warm_decode_disk_cache_if_needed()
        if self._startup_warm_plan == "provider":
            self._warm_native_layer_cache_if_needed()

        logger.info(
            "loaded pack=%s mode=%s backend=%s device=%s layers=%d tensors=%d",
            self.config.pack_dir,
            self.config.mode,
            self.config.backend,
            self._device,
            num_layers,
            len(self.manifest.tensors),
        )

    def _merge_meta_json(self) -> None:
        assert self.manifest is not None
        meta_path = self.config.pack_dir / "meta.json"
        if not meta_path.is_file():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in meta.items():
            self.manifest.meta.setdefault(key, value)

    def _init_staging_if_needed(self) -> None:
        assert self.scheduler is not None
        if self.config.mode == "resident" or self._device.type != "cuda":
            return
        if not self.scheduler.layer_ids:
            return
        max_payload = max(
            self.scheduler.payload_bytes(lid) for lid in self.scheduler.layer_ids
        )
        if max_payload > 0:
            self.staging = PingPongStaging(
                max_payload, self._device, dtype=torch.float32
            )

    def _warm_z_if_needed(self) -> None:
        if self.config.mode == "resident":
            if self._device.type == "xpu":
                self._warm_xpu_resident_model()
            return
        if not self.config.warm_z:
            return
        if isinstance(self.backend, RWKVCppBackend):
            assert self.manifest and self.store
            provider = self._get_or_create_streaming_provider()
            layer_ids = manifest_block_layers(self.manifest)
            self.backend.prepare_streaming_layers(
                provider, self.manifest.by_layer(), layer_ids, self.metrics
            )
            logger.info("rwkv.cpp warm-z: %d block layers uploaded", len(layer_ids))
            return
        if not isinstance(self.backend, ChatRWKVBackend):
            return
        model = self.backend._model
        if model is None or not hasattr(model, "z"):
            return
        if getattr(self.backend, "_deepembed_streaming_reference", False):
            # The qkv/DEA reference intentionally loads a temporary layer
            # window inside forward_streaming(); promoting those tensors into
            # the global z would defeat its bounded-RAM contract.
            return
        assert self.manifest and self.store
        # Reuse the long-lived streaming provider so generate-time warm
        # sees already-materialized layers and skips a second full decode.
        provider = self._get_or_create_streaming_provider(model_z=model.z)
        from rwkv_ssd.runtime.rwkv7_weights import warm_stream_cache_layers_into_z

        layer_ids = manifest_block_layers(self.manifest)
        n = warm_stream_cache_layers_into_z(
            model.z,
            provider,
            self.manifest.by_layer(),
            layer_ids,
            metrics=self.metrics,
        )
        if n:
            logger.info("warm-z: %d block layers preloaded into z", n)
            # Weights now live in ``z``; drop provider duplicates so peak RAM
            # is not 2× (old path closed a throwaway provider for the same reason).
            provider.release_all_streamed_layers()

    def _warm_xpu_resident_model(self) -> None:
        """Materialize a skeleton into ``z`` for resident Intel-XPU inference.

        ChatRWKV's upstream strategy validator does not know ``xpu``.  The
        pack-only skeleton avoids that validator, but resident generation
        still expects every block projection in ``model.z``.  Decode the pack
        directly to XPU and use the same prepared layout as the native RWKV-7
        forward path.
        """
        if not isinstance(self.backend, ChatRWKVBackend):
            return
        model = self.backend._model
        if model is None or not isinstance(getattr(model, "z", None), dict):
            return
        if getattr(self.backend, "_deepembed_streaming_reference", False):
            return
        assert self.manifest and self.store
        from rwkv_ssd.runtime.rwkv7_weights import (
            layer_weights_in_z,
            warm_stream_cache_layers_into_z,
        )

        layer_ids = manifest_block_layers(self.manifest)
        if all(layer_weights_in_z(model.z, layer_id) for layer_id in layer_ids):
            return
        provider = self._get_or_create_streaming_provider(model_z=model.z)
        try:
            loaded = warm_stream_cache_layers_into_z(
                model.z,
                provider,
                self.manifest.by_layer(),
                layer_ids,
                metrics=self.metrics,
            )
            if loaded:
                logger.info(
                    "XPU resident warm: %d block layers materialized on %s",
                    loaded,
                    self._device,
                )
        finally:
            # Resident generation uses model.z directly.  Do not retain a
            # second full copy in the provider's decode cache.
            provider.close()
            if self._streaming_provider is provider:
                self._streaming_provider = None

    def _create_streaming_provider(
        self,
        model_z: dict | None = None,
    ) -> ManifestWeightProvider:
        assert self.manifest and self.store
        native_layer_streaming = bool(
            isinstance(self.backend, RWKVCppBackend)
            and self.backend._native_layer_streaming_active()
        )
        return create_weight_provider(
            self.config,
            self.store,
            self.manifest.tensors,
            self._device,
            self.metrics,
            self.staging,
            model_z=model_z,
            shadow_store=self.shadow_store,
            pack_dir=self.config.pack_dir,
            manifest_meta=self.manifest.meta,
            native_layer_streaming=native_layer_streaming,
        )

    def _get_or_create_streaming_provider(
        self,
        model_z: dict | None = None,
    ) -> ManifestWeightProvider:
        """Reuse decoded provider cache across ``generate()`` calls when streaming."""
        if self._streaming_provider is not None:
            self._streaming_provider.set_model_z(
                model_z if isinstance(model_z, dict) else None
            )
            return self._streaming_provider
        # ``_create_streaming_provider`` already passes ``model_z`` to
        # the provider constructor, so no separate ``set_model_z`` call
        # is needed on the fresh path.
        provider = self._create_streaming_provider(model_z=model_z)
        self._streaming_provider = provider
        return provider

    def _warm_provider_cache_if_needed(self) -> None:
        """Decode all streamed layers into provider RAM before first token (Trinity auto)."""
        if (
            not self.config.stream_layer_cache
            or self.config.warm_z
            or self.config.mode == "resident"
        ):
            return
        if not isinstance(self.backend, (ChatRWKVBackend, RWKVCppBackend)):
            return
        from rwkv_ssd.runtime.throughput_defaults import _warm_provider_cache_auto

        if not _warm_provider_cache_auto(self.manifest):
            return
        assert self.manifest and self.store
        model_z = None
        model = self.backend._model
        if model is not None and hasattr(model, "z"):
            model_z = model.z
        provider = self._get_or_create_streaming_provider(model_z=model_z)
        from rwkv_ssd.runtime.layer_keys import manifest_block_layers
        from rwkv_ssd.runtime.rwkv7_weights import all_block_layers_in_z

        layer_ids = manifest_block_layers(self.manifest)
        by_layer = self.manifest.by_layer()
        if model_z is not None and all_block_layers_in_z(model_z, layer_ids, by_layer):
            logger.info("warm-provider-cache: skipped (all block layers already in z)")
            return
        warmed = 0
        with torch.no_grad():
            for layer_id in layer_ids:
                entries = by_layer.get(layer_id, [])
                if not entries:
                    continue
                if provider._layer_resident_in_z(layer_id):
                    warmed += 1
                    continue
                if not provider.layer_has_streamed_tensors(entries):
                    continue
                if layer_id in provider._prepared_layers:
                    warmed += 1
                    continue
                # ``load_layer_tensors`` calls ``begin_layer`` which
                # creates a metrics row and records read_ms. The
                # subsequent ``prepare_layer_for_z`` records staging_ms
                # on that same row (the timing arg is the current
                # row, looked up via ``metrics.layers[-1]``).
                tensors = (
                    provider.load_layer_tensors_materialized(entries)
                    if isinstance(self.backend, RWKVCppBackend)
                    else provider.load_layer_tensors(entries)
                )
                if isinstance(self.backend, ChatRWKVBackend) and self.metrics.layers:
                    provider.prepare_layer_for_z(
                        layer_id, tensors, self.metrics.layers[-1]
                    )
                elif isinstance(self.backend, ChatRWKVBackend):
                    provider.prepare_layer_for_z(layer_id, tensors)
                warmed += 1
        if warmed:
            logger.info(
                "warm-provider-cache: %d block layers decoded into provider RAM",
                warmed,
            )

    def _select_startup_warm_plan(self) -> str:
        """Pick one full-pack warm strategy for the current residency tier."""
        if (
            self.config.mode == "resident"
            or self.config.warm_z
            or self.manifest is None
            or not isinstance(self.backend, (ChatRWKVBackend, RWKVCppBackend))
        ):
            return "none"
        from rwkv_ssd.runtime.throughput_defaults import (
            _warm_disk_cache_auto,
            _warm_provider_cache_auto,
        )

        import os

        def explicit(name: str) -> bool | None:
            raw = os.environ.get(name, "auto").strip().lower()
            if raw in {"1", "true", "on", "yes"}:
                return True
            if raw in {"0", "false", "off", "no"}:
                return False
            return None

        want_provider = explicit("RWKV_WARM_PROVIDER_CACHE")
        want_disk = explicit("RWKV_WARM_DISK_CACHE")
        if want_provider and want_disk:
            raise ValueError(
                "RWKV_WARM_PROVIDER_CACHE and RWKV_WARM_DISK_CACHE cannot both "
                "be enabled: choose one startup warm plan"
            )
        if want_provider:
            return "provider"
        if want_disk:
            return "disk"

        layer_count = int(self.manifest.meta.get("n_layer", 0))
        provider_holds_all = (
            self.config.stream_layer_cache
            and layer_count > 0
            and self.config.max_provider_cache_layers >= layer_count
        )
        if provider_holds_all and _warm_provider_cache_auto(self.manifest):
            return "provider"
        if _warm_disk_cache_auto(self.manifest):
            return "disk"
        return "none"

    def _warm_native_layer_cache_if_needed(self) -> None:
        """Populate bounded native layer plans before the first request.

        The native layer ABI can retain borrowed packed views only when their
        owner is stable for the engine lifetime (mmap) or when the configured
        provider cache already owns them.  Prewarming in those cases moves the
        one-time layer sweep into ``load()`` while preserving F1-F4's
        one-block native weight residency.  Pread/cold F1 stays lazy because
        retaining its temporary read buffers would change its RAM contract.
        """
        if (
            self.config.mode == "resident"
            or self.config.warm_z
            or not isinstance(self.backend, RWKVCppBackend)
            or self.manifest is None
            or self.store is None
            or not bool(getattr(self.backend, "_native_layer_streaming_active", lambda: False)())
        ):
            return
        raw = os.environ.get("RWKVCPP_PREWARM_LAYERS", "auto").strip().lower()
        if raw in {"0", "false", "off", "no"}:
            return
        if raw not in {"", "auto", "1", "true", "on", "yes"}:
            logger.warning("Ignoring invalid RWKVCPP_PREWARM_LAYERS=%r", raw)
            return
        stable_store = bool(getattr(self.store, "stable_memoryviews", False))
        bounded_provider = bool(self.config.stream_layer_cache)
        if raw in {"", "auto"} and not (stable_store or bounded_provider):
            return
        if not (stable_store or bounded_provider):
            logger.info(
                "rwkv.cpp native prewarm skipped: provider buffers are not stable"
            )
            return
        provider = self._get_or_create_streaming_provider()
        layer_ids = manifest_block_layers(self.manifest)
        by_layer = self.manifest.by_layer()
        if not layer_ids:
            return
        self.backend.prepare_streaming_layers(
            provider,
            by_layer,
            layer_ids,
            self.metrics,
            upload_layers=True,
        )
        logger.info(
            "rwkv.cpp native prewarm: %d bounded block layers uploaded",
            len(layer_ids),
        )

    def _warm_decode_disk_cache_if_needed(self) -> None:
        """Decode streamed layers once at load and persist to ``.decode_cache/``."""
        if self.config.mode == "resident" or self.config.warm_z:
            return
        if not isinstance(self.backend, (ChatRWKVBackend, RWKVCppBackend)):
            return
        from rwkv_ssd.runtime.throughput_defaults import _warm_disk_cache_auto

        if not _warm_disk_cache_auto(self.manifest):
            return
        assert self.manifest and self.store
        model_z = None
        model = self.backend._model
        if model is not None and hasattr(model, "z"):
            model_z = model.z
        provider = self._get_or_create_streaming_provider(model_z=model_z)
        if provider._disk_cache is None:
            return
        from rwkv_ssd.runtime.decode_disk_cache import warm_disk_cache_layers

        layer_ids = manifest_block_layers(self.manifest)
        by_layer = self.manifest.by_layer()
        warmed = warm_disk_cache_layers(
            provider._disk_cache,
            provider,
            by_layer,
            layer_ids,
        )
        if warmed:
            logger.info(
                "warm-disk-cache: %d streamed layers written to .decode_cache/",
                warmed,
            )
        provider.clear_prepared_pinned_layers()

    def close(self) -> None:
        # Native rwkv.cpp layer plans may borrow mmap-backed packed views.
        # Release the backend/model owners before closing the provider/store,
        # otherwise Windows correctly rejects closing an mmap with exported
        # ctypes/NumPy buffers still alive.
        if hasattr(self.backend, "close"):
            self.backend.close()
        if self._streaming_provider is not None:
            self._streaming_provider.close()
            self._streaming_provider = None
        if self._pack_provider is not None:
            self._pack_provider.close()
            self._pack_provider = None
        if self.store:
            self.store.close()
            self.store = None
        if getattr(self, "shadow_store", None):
            self.shadow_store.close()
            self.shadow_store = None
        if self.staging is not None:
            self.staging.close()
            self.staging = None

    def _get_or_create_pack_provider(self) -> ManifestWeightProvider:
        if self._pack_provider is not None:
            return self._pack_provider
        assert self.manifest and self.store
        self._pack_provider = create_weight_provider(
            self.config,
            self.store,
            self.manifest.tensors,
            self._device,
            self.metrics,
            self.staging,
            shadow_store=self.shadow_store,
            pack_dir=self.config.pack_dir,
            manifest_meta=self.manifest.meta,
        )
        prepare = getattr(self.backend, "prepare_provider", None)
        if callable(prepare):
            try:
                prepare(self._pack_provider, config=self.config)
            except Exception:
                self._pack_provider.close()
                self._pack_provider = None
                raise
        return self._pack_provider

    def _prompt_token_count(self, prompt: str) -> int:
        """Count the backend's actual prompt IDs for OpenAI usage metrics."""
        text = str(prompt)
        prefix = self.config.system_prefix
        if prefix and not text.startswith(prefix):
            text = prefix + text
        for owner, name in (
            (self.backend, "_encode"),
            (self.backend, "_encode_fn"),
        ):
            encoder = getattr(owner, name, None)
            if callable(encoder):
                try:
                    return len(encoder(text))
                except (TypeError, ValueError, RuntimeError):
                    pass
        pipeline = getattr(self.backend, "_pipeline", None)
        encoder = getattr(pipeline, "encode", None)
        if callable(encoder):
            try:
                return len(encoder(text))
            except (TypeError, ValueError, RuntimeError):
                pass
        tokenizer = getattr(self.backend, "_tokenizer", None)
        encoder = getattr(tokenizer, "encode", None)
        if callable(encoder):
            try:
                return len(encoder(text))
            except (TypeError, ValueError, RuntimeError):
                pass
        return len(text.encode("utf-8"))

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        """Generate text with optional request-scoped sampling controls."""

        with self._sampling_scope(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
        ):
            return self._generate_impl(
                prompt,
                cancel_event=cancel_event,
                deadline=deadline,
            )

    def _generate_impl(
        self,
        prompt: str,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        self._apply_adaptive_residency_boundary()
        t0 = time.perf_counter()
        self.metrics.reset_for_generate()
        self.metrics.prompt_tokens = self._prompt_token_count(prompt)
        self.metrics.power_percent = int(self.config.power_percent)
        self.metrics.observe_memory(
            budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
        )
        self._apply_session_promotion_boundary()
        if self.config.progress:
            logger.info(
                "generating up to %d tokens (mode=%s backend=%s)",
                self.config.max_tokens,
                self.config.mode,
                self.config.backend,
            )

        if isinstance(self.backend, PackBackend):
            text = self._generate_pack(
                prompt,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        elif (
            isinstance(self.backend, ChatRWKVBackend) and self.config.mode != "resident"
        ):
            text = self._generate_chatrwkv_streaming(
                prompt,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        elif isinstance(self.backend, RWKVCppBackend) and self.config.mode != "resident":
            text = self._generate_rwkvcpp_streaming(
                prompt,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        elif self.config.mode != "resident":
            from rwkv_ssd.backends.capabilities import supports_true_streaming

            if not supports_true_streaming(self.backend):
                raise CapabilityNotSupportedError(
                    "incremental_streaming",
                    self.config.backend,
                    f"Backend {self.config.backend!r} does not support true streaming yet. "
                    "Use --mode resident, or --backend synthetic or chatrwkv with RWKV-7."
                )
            raise CapabilityNotSupportedError(
                "incremental_streaming",
                self.config.backend,
                f"Backend {self.config.backend!r} does not support true streaming yet. "
                "Use --backend synthetic or chatrwkv with RWKV-7."
            )
        else:
            text = self._generate_resident(
                prompt,
                cancel_event=cancel_event,
                deadline=deadline,
            )

        self.metrics.total_wall_s = time.perf_counter() - t0
        self.metrics.observe_memory(
            budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
        )
        self._observe_adaptive_residency()
        self._observe_session_promotion()
        if self.config.trace_path:
            from rwkv_ssd.runtime.runtime_intelligence import append_decision_trace

            append_decision_trace(
                self.config.trace_path,
                self.metrics,
                request={"prompt_chars": len(prompt), "max_tokens": self.config.max_tokens},
            )
        if self.config.metrics_csv:
            self.metrics.write_csv(self.config.metrics_csv)
        return text

    def _apply_adaptive_residency_boundary(self) -> None:
        """Apply a pending retier only before a new request starts."""
        decision = self._adaptive_pending
        if decision is None:
            return
        if self._streaming_provider is not None:
            self._streaming_provider.release_all_streamed_layers()
            self._streaming_provider.close()
            self._streaming_provider = None
        if self._pack_provider is not None:
            self._pack_provider.release_all_streamed_layers()
            self._pack_provider.close()
            self._pack_provider = None
        self.config.cache_format = decision.new_format
        from rwkv_ssd.runtime.throughput_defaults import apply_cache_format_defaults

        apply_cache_format_defaults(self.config)
        self.metrics.cache_format = decision.new_format
        self._adaptive_pending = None
        logger.info(
            "adaptive residency retier %s -> %s at request boundary: %s",
            decision.old_format,
            decision.new_format,
            decision.reason,
        )

    def _observe_adaptive_residency(self) -> None:
        controller = self._adaptive_residency
        if controller is None or self._adaptive_pending is not None:
            return
        controller.observe(
            self.config.cache_format,
            read_ms=sum(layer.read_ms for layer in self.metrics.layers),
            staging_ms=sum(layer.staging_ms + layer.h2d_ms for layer in self.metrics.layers),
            compute_ms=sum(layer.compute_ms for layer in self.metrics.layers),
            tokens=max(1, int(self.metrics.tokens_generated)),
        )
        self._adaptive_pending = controller.maybe_retier(
            explicit_cache_format=(
                self.config.cache_format if self._cache_format_was_explicit else "auto"
            ),
            current_format=self.config.cache_format,
        )

    def _apply_session_promotion_boundary(self) -> None:
        """Materialize a profitable pending plan before the next request."""
        plan = self._session_promotion_pending
        self._session_promotion_pending = None
        self._session_observation_start = 0
        if plan is None or not plan.selected:
            return
        if not isinstance(self.backend, ChatRWKVBackend) or self.config.mode == "resident":
            return
        model = self.backend._model
        if model is None or not isinstance(getattr(model, "z", None), dict):
            return
        assert self.manifest is not None
        z = model.z
        provider = self._get_or_create_streaming_provider(model_z=z)
        from rwkv_ssd.runtime.rwkv7_skeleton import estimate_z_bytes
        from rwkv_ssd.runtime.rwkv7_weights import (
            layer_weights_in_z,
            warm_stream_cache_layers_into_z,
        )
        from rwkv_ssd.runtime.z_layer_retention import evict_block_layer_from_z

        selected = [item.layer_id for item in plan.selected]
        preexisting = {layer_id for layer_id in selected if layer_weights_in_z(z, layer_id)}
        before_total = estimate_z_bytes(z) + provider.cached_weight_bytes()
        before_z = estimate_z_bytes(z)
        row_start = len(self.metrics.layers)
        started = time.perf_counter()
        warm_stream_cache_layers_into_z(
            z,
            provider,
            self.manifest.by_layer(),
            selected,
            self.metrics,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        added_z = max(0, estimate_z_bytes(z) - before_z)
        cap = max(0, int(self.config.session_promotion_bytes))
        if added_z > cap:
            for layer_id in selected:
                if layer_id not in preexisting:
                    evict_block_layer_from_z(z, layer_id)
                provider.evict_streamed_layer(layer_id, force=True)
            logger.warning(
                "session promotion rejected: actual z delta %d exceeds cap %d",
                added_z,
                cap,
            )
            self._session_observation_start = len(self.metrics.layers)
            return

        provider._z_retention.pinned_layer_ids.update(selected)
        for layer_id in selected:
            provider.evict_streamed_layer(layer_id, force=True)
        after_total = estimate_z_bytes(z) + provider.cached_weight_bytes()
        actual_added = max(0, after_total - before_total)
        if actual_added > cap:
            provider._z_retention.pinned_layer_ids.difference_update(selected)
            for layer_id in selected:
                if layer_id not in preexisting:
                    evict_block_layer_from_z(z, layer_id)
            logger.warning(
                "session promotion rolled back: actual RAM delta %d exceeds cap %d",
                actual_added,
                cap,
            )
            self._session_observation_start = len(self.metrics.layers)
            return

        self.metrics.session_promoted_layers = selected
        self.metrics.session_promotion_bytes = actual_added
        self.metrics.session_promotion_ms = elapsed_ms
        self.metrics.session_promotion_estimated_net_ms = plan.net_benefit_ms
        self.metrics.session_expected_remaining_tokens = plan.expected_remaining_tokens
        self._session_observation_start = len(self.metrics.layers)
        logger.info(
            "session promotion pinned layers=%s added=%dB cost=%.1fms expected_net=%.1fms",
            selected,
            actual_added,
            elapsed_ms,
            plan.net_benefit_ms,
        )

    def _observe_session_promotion(self) -> None:
        """Build the next request-boundary plan from this request's timings."""
        if not self.config.session_promotion:
            return
        if not isinstance(self.backend, ChatRWKVBackend) or self.config.mode == "resident":
            return
        if self.manifest is None or self.config.session_promotion_bytes <= 0:
            return
        model = self.backend._model
        if model is None or not isinstance(getattr(model, "z", None), dict):
            return
        from collections import defaultdict

        from rwkv_ssd.runtime.promotion_planner import (
            PromotionCandidate,
            plan_promotions,
        )
        from rwkv_ssd.runtime.rwkv7_weights import layer_weights_in_z

        generated = max(0, int(self.metrics.tokens_generated))
        self._session_tokens_seen += generated
        self._session_request_index += 1
        if self.config.session_expected_tokens > 0:
            remaining = max(
                0, int(self.config.session_expected_tokens) - self._session_tokens_seen
            )
        else:
            remaining = max(0, int(self.config.max_tokens))
        if remaining <= 0:
            self._session_promotion_pending = None
            return

        rows: dict[int, list[float]] = defaultdict(list)
        for row in self.metrics.layers[self._session_observation_start :]:
            if row.layer_id < 0:
                continue
            rows[row.layer_id].append(
                max(0.0, row.read_ms + row.staging_ms + row.h2d_ms)
            )
        by_layer = self.manifest.by_layer()
        act_bytes = 2
        emb = model.z.get("emb.weight")
        if isinstance(emb, torch.Tensor):
            act_bytes = emb.element_size()
        candidates = []
        for layer_id, costs in rows.items():
            entries = by_layer.get(layer_id, [])
            if not entries or layer_weights_in_z(model.z, layer_id):
                continue
            if not any(entry.residency == "streamed" for entry in entries):
                continue
            total_cost = sum(costs)
            if total_cost <= 0:
                continue
            candidates.append(
                PromotionCandidate(
                    layer_id=layer_id,
                    resident_bytes=sum(entry.numel for entry in entries) * act_bytes,
                    promotion_ms=total_cost / max(1, len(costs)),
                    staging_ms_saved_per_token=total_cost / max(1, generated),
                    last_used_token=self._session_request_index,
                )
            )
        plan = plan_promotions(
            candidates,
            expected_remaining_tokens=remaining,
            ram_cap_bytes=int(self.config.session_promotion_bytes),
            policy=self.config.session_promotion_policy,
        )
        self._session_promotion_last_plan = plan
        self._session_promotion_pending = plan if plan.selected else None

    def generate_tokens(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        """Return generated token ids with optional sampling controls."""

        with self._sampling_scope(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
        ):
            return self._generate_tokens_with_scope(
                prompt,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )

    def _generate_tokens_with_scope(
        self,
        prompt: str,
        *,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        """Return generated token ids without decoding to text.

        Token generation is also the entry point used by the parity harness
        and the process-worker bridge.  Keep the same request boundary and
        metrics finalization as :meth:`generate`, including on cancellation
        or deadline failure, so those callers never receive a zero wall-time
        or stale RSS observation.
        """
        self._apply_adaptive_residency_boundary()
        started = time.perf_counter()
        try:
            self._apply_session_promotion_boundary()
            return self._generate_tokens_impl(
                prompt,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        finally:
            self.metrics.total_wall_s = time.perf_counter() - started
            self.metrics.observe_memory(
                budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
            )
            self._observe_adaptive_residency()
            self._observe_session_promotion()

    def _generate_tokens_impl(
        self,
        prompt: str,
        *,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        """Dispatch token generation after the request boundary is entered."""
        self.metrics.reset_for_generate()
        self.metrics.prompt_tokens = self._prompt_token_count(prompt)
        self.metrics.power_percent = int(self.config.power_percent)
        self.metrics.observe_memory(
            budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
        )
        if isinstance(self.backend, PackBackend):
            return self._generate_pack_tokens(
                prompt,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        if (
            isinstance(self.backend, ChatRWKVBackend)
            and self.config.mode != "resident"
        ):
            return self._generate_chatrwkv_streaming_tokens(
                prompt,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        if isinstance(self.backend, RWKVCppBackend) and self.config.mode != "resident":
            return self._generate_rwkvcpp_streaming_tokens(
                prompt,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        if self.config.mode != "resident":
            raise CapabilityNotSupportedError(
                "incremental_streaming",
                self.config.backend,
                f"Backend {self.config.backend!r} does not support true streaming yet."
            )
        if (
            isinstance(self.backend, ChatRWKVBackend)
            and self.backend._rwkv7
            and hasattr(self.backend, "generate_greedy_native")
        ):
            return self.backend.generate_greedy_native(
                prompt,
                self.config.max_tokens,
                metrics=self.metrics,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        if (
            isinstance(self.backend, RWKVCppBackend)
            and hasattr(self.backend, "generate_greedy_native")
        ):
            return self.backend.generate_greedy_native(
                prompt,
                self.config.max_tokens,
                metrics=self.metrics,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        raise RuntimeError("generate_tokens unsupported for this backend path")

    def probe_generation(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        thresholds: object | None = None,
        reference: object | None = None,
    ):
        """Run an opt-in diagnostic generation and return a parity trace.

        The public generation methods remain unchanged.  ``reference`` may be
        a ``BackendProbeResult``, a mapping with ``token_ids``/``logits``/
        ``states``, or a token-ID list.  Supplying it enables exact greedy and
        guardrail comparisons; without it the returned trace is still useful
        for timing, memory, tokenizer, and native-path diagnostics.
        """
        from rwkv_ssd.runtime.parity import (
            ParityStep,
            ParityThresholds,
            ParityTrace,
            _coerce_probe_result,
            kl_candidate_to_reference,
            relative_state_error,
            top_k_overlap,
        )

        count = int(self.config.max_tokens if max_tokens is None else max_tokens)
        configured = ParityThresholds(
            min_top10_overlap=float(
                getattr(self.config, "parity_min_top10_overlap", 0.80)
            ),
            max_kl=float(getattr(self.config, "parity_max_kl", 0.05)),
            max_state_relative_error=float(
                getattr(self.config, "parity_max_state_relative_error", 0.10)
            ),
        )
        limits = thresholds if isinstance(thresholds, ParityThresholds) else ParityThresholds.from_mapping(thresholds if isinstance(thresholds, dict) else None)
        if thresholds is None:
            limits = configured
        started = time.perf_counter()
        old_max_tokens = self.config.max_tokens
        self.config.max_tokens = count
        try:
            token_ids = self.generate_tokens(prompt)
        finally:
            self.config.max_tokens = old_max_tokens
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        state = self.backend.get_recurrent_state()
        logits = self.backend.probe_logits(state)
        prompt_ids: list[int] = []
        for encoder_name in ("_encode", "_encode_fn"):
            encoder = getattr(self.backend, encoder_name, None)
            if callable(encoder):
                try:
                    prompt_ids = [int(value) for value in encoder(prompt)]
                    break
                except (TypeError, ValueError, RuntimeError):
                    pass
        trace = ParityTrace(
            backend=type(self.backend).__name__,
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            generated_token_ids=[int(value) for value in token_ids],
            decoded_text=self.backend.decode_text([int(value) for value in token_ids]),
            thresholds=limits,
            prefill_ms=float(getattr(self.metrics, "prefill_wall_s", 0.0) * 1000.0),
            decode_ms=float(getattr(self.metrics, "decode_wall_s", 0.0) * 1000.0),
            rss_bytes=int(getattr(self.metrics, "process_rss_bytes", 0)),
            rss_peak_bytes=int(getattr(self.metrics, "process_rss_peak_bytes", 0)),
            streamed_bytes=int(getattr(self.metrics, "streamed_bytes", 0)),
            cache_hits=int(getattr(self.metrics, "cache_hits", 0)),
        )
        if trace.decode_ms <= 0.0:
            trace.decode_ms = elapsed_ms
        if reference is not None:
            ref = _coerce_probe_result(reference)
            for index in range(max(len(ref.token_ids), len(token_ids))):
                ref_id = ref.token_ids[index] if index < len(ref.token_ids) else None
                cand_id = token_ids[index] if index < len(token_ids) else None
                step = ParityStep(
                    token_index=index,
                    reference_token_id=ref_id,
                    candidate_token_id=(int(cand_id) if cand_id is not None else None),
                    greedy_match=ref_id == cand_id,
                )
                if index < len(ref.logits) and logits is not None:
                    candidate_logits = logits[index] if isinstance(logits, (list, tuple)) and index < len(logits) else logits
                    step.top10_overlap = top_k_overlap(ref.logits[index], candidate_logits)
                    step.kl = kl_candidate_to_reference(ref.logits[index], candidate_logits)
                if index < len(ref.states) and state is not None:
                    candidate_state = state[index] if isinstance(state, (list, tuple)) and index < len(state) else state
                    try:
                        step.state_relative_error = relative_state_error(ref.states[index], candidate_state)
                    except (TypeError, ValueError):
                        trace.notes.append("state representation was not directly comparable")
                trace.add_step(step)
        else:
            for index, token_id in enumerate(token_ids):
                trace.add_step(
                    ParityStep(token_index=index, candidate_token_id=int(token_id))
                )
        return trace

    # ``parity_trace`` is the concise name used by benchmark callers.
    parity_trace = probe_generation

    def capabilities(self) -> dict[str, object]:
        """Return the backend capability report without changing execution."""
        from rwkv_ssd.backends.capabilities import probe_backend_capabilities

        report = probe_backend_capabilities(self.backend)
        report["config"] = {
            "backend": self.config.backend,
            "mode": self.config.mode,
            "device": str(self._device),
        }
        return report

    def supports_capability(self, capability: str) -> bool:
        """Query whether the selected backend declares a capability."""
        from rwkv_ssd.backends.capabilities import capability_supported

        return capability_supported(self.backend, capability)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Generate several pack-backed sessions in one weight-stationary sweep."""
        if not isinstance(prompts, list) or any(not isinstance(item, str) for item in prompts):
            raise TypeError("prompts must be a list of strings")
        if not prompts:
            return []
        if not self.config.greedy:
            raise CapabilityNotSupportedError(
                "sampling",
                self.config.backend,
                "generate_batch currently supports greedy decoding only; use "
                "generate() per session for temperature sampling"
            )
        if not isinstance(self.backend, (PackBackend, ChatRWKVBackend)):
            raise CapabilityNotSupportedError(
                "batching",
                self.config.backend,
                f"backend {self.config.backend!r} does not support weight-stationary batching"
            )
        self._apply_adaptive_residency_boundary()
        self.metrics.reset_for_generate()
        self.metrics.prompt_tokens = sum(
            self._prompt_token_count(prompt) for prompt in prompts
        )
        self.metrics.power_percent = int(self.config.power_percent)
        provider = self._get_or_create_pack_provider() if isinstance(self.backend, PackBackend) else self._get_or_create_streaming_provider(model_z=self.backend._model.z)
        started = time.perf_counter()
        count = int(max_tokens if max_tokens is not None else self.config.max_tokens)
        if isinstance(self.backend, PackBackend):
            token_batches = self.backend.generate_greedy_batch(prompts, provider, count, self.metrics)
        else:
            assert self.manifest is not None
            token_batches = self.backend.generate_greedy_batch_streaming(prompts, count, provider, self.manifest.by_layer(), manifest_block_layers(self.manifest), self.metrics)
        self.metrics.total_wall_s = time.perf_counter() - started
        self.metrics.observe_memory(
            budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
        )
        self._observe_adaptive_residency()
        return [self.backend.decode_text(tokens) for tokens in token_batches]

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ):
        """Yield decoded chunks as each token is produced.

        Backends historically returned a completed list.  A worker bridge keeps
        that compatibility while exposing tokens immediately and gives the
        HTTP layer a cancellation point when a client disconnects.
        """
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        worker_cancel = cancel_event or threading.Event()
        def on_token(token_id: int) -> None:
            events.put(("token", int(token_id)))

        def run() -> None:
            try:
                self.generate_tokens(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    greedy=greedy,
                    token_callback=on_token,
                    cancel_event=worker_cancel,
                    deadline=deadline,
                )
                events.put(("done", None))
            except BaseException as exc:  # propagate backend errors to caller
                events.put(("error", exc))

        worker = threading.Thread(target=run, name="rwkv-generation", daemon=True)
        worker.start()
        try:
            while True:
                kind, value = events.get()
                if kind == "done":
                    return
                if kind == "error":
                    if isinstance(value, GenerationCancelled):
                        return
                    raise value  # type: ignore[misc]
                yield self.backend.decode_text([int(value)])
        finally:
            worker_cancel.set()
            # The engine owns mutable recurrent state and provider/cache
            # objects.  Do not let the caller release or reuse it while a
            # native layer call is still running after cancellation.
            worker.join()

    def continue_generate(
        self,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        """Decode from the backend's current recurrent state without re-prefill."""
        return self.decode_from_state(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_followup(
        self,
        suffix: str,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> str:
        """Prefill ``suffix`` onto the current state, then decode.

        Follow-up generation has several backend-specific early-return paths.
        Keep their implementation unchanged behind one metrics boundary so
        successful, cancelled, and failed requests all publish a consistent
        wall-time/RSS observation to HTTP and parity diagnostics.
        """
        with self._sampling_scope(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
        ):
            return self._generate_followup_with_scope(
                suffix,
                max_tokens=max_tokens,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )

    def _generate_followup_with_scope(
        self,
        suffix: str,
        max_tokens: int | None = None,
        *,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> str:
        started = time.perf_counter()
        try:
            return self._generate_followup_impl(
                suffix,
                max_tokens=max_tokens,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        finally:
            self.metrics.total_wall_s = time.perf_counter() - started
            self.metrics.observe_memory(
                budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
            )
            self._observe_adaptive_residency()
            self._observe_session_promotion()

    def _generate_followup_impl(
        self,
        suffix: str,
        max_tokens: int | None = None,
        *,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> str:
        """Prefill ``suffix`` onto the current state, then decode."""
        n = int(max_tokens if max_tokens is not None else self.config.max_tokens)
        self.metrics.reset_for_generate()
        self.metrics.prompt_tokens = self._prompt_token_count(suffix)
        self.metrics.power_percent = int(self.config.power_percent)
        self.metrics.observe_memory(
            budget_bytes=int(float(self.config.ram_budget_gb or 0) * 1e9)
        )
        state = self.backend.get_recurrent_state()
        if state is None:
            token_ids = self.generate_tokens(
                suffix,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            return self.backend.decode_text(token_ids)
        self.metrics.power_percent = int(self.config.power_percent)
        if isinstance(self.backend, PackBackend):
            provider = self._get_or_create_pack_provider()
            merged = self.backend.prefill_text(
                suffix,
                provider,
                self.metrics,
                initial_state=state,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            token_ids = self.backend.decode_greedy(
                merged,
                provider,
                n,
                self.metrics,
                temperature=self.config.temperature if not self.config.greedy else 0.0,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            return self.backend.decode_text(token_ids)
        if isinstance(self.backend, RWKVCppBackend) and state.external_state is not None:
            assert self.manifest and self.store
            provider = self._get_or_create_streaming_provider()
            layer_ids = manifest_block_layers(self.manifest)
            by_layer = self.manifest.by_layer()
            current = state
            suffix_ids = self.backend._encode_fn(suffix) if self.backend._encode_fn else []
            layers_already_synced = False
            if suffix_ids:
                self.backend._sync_provider_layers(provider, by_layer, layer_ids, self.metrics)
                layers_already_synced = True
                _, external = self.backend._model.eval_sequence_in_chunks(
                    suffix_ids,
                    np.asarray(state.external_state, dtype=np.float32),
                    None,
                    None,
                    use_numpy=True,
                )
                from rwkv_ssd.runtime.state_cache import RecurrentState

                current = RecurrentState(
                    last_token_id=int(suffix_ids[-1]),
                    external_state=external,
                )
            token_ids = self.backend.generate_greedy_from_state(
                current,
                n,
                provider,
                by_layer,
                layer_ids,
                self.metrics,
                layers_already_synced=layers_already_synced,
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            return self.backend.decode_text(token_ids)
        if isinstance(self.backend, ChatRWKVBackend) and state.rwkv7_state is not None:
            from rwkv_ssd.backends.rwkv7_forward import (
                decode_greedy_from_state,
                prefill_text_streaming,
            )
            from rwkv_ssd.runtime.state_cache import RecurrentState

            assert self.backend._model is not None
            assert self.backend._pipeline is not None
            streaming = (
                self.config.mode != "resident"
                and self.manifest is not None
                and self.store is not None
                and self.scheduler is not None
            )
            if streaming:
                model_z = None
                model = self.backend._model
                if model is not None and hasattr(model, "z"):
                    model_z = model.z
                provider = self._get_or_create_streaming_provider(model_z=model_z)
                layer_ids = manifest_block_layers(self.manifest)
                by_layer = self.manifest.by_layer()
                rwkv_state = [t.clone() for t in state.rwkv7_state]
                rwkv_state, last_id = prefill_text_streaming(
                    self.backend._model,
                    self.backend._pipeline,
                    suffix,
                    rwkv_state,
                    provider,
                    by_layer,
                    layer_ids,
                    self.metrics,
                )
                token_ids = decode_greedy_from_state(
                    self.backend._model,
                    rwkv_state,
                    last_id,
                    n,
                    metrics=self.metrics,
                    power_percent=int(self.config.power_percent),
                    provider=provider,
                    by_layer=by_layer,
                    layer_ids=layer_ids,
                    temperature=float(self.config.temperature),
                    greedy=bool(self.config.greedy),
                    token_callback=token_callback,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
            else:
                suffix_ids = self.backend._pipeline.encode(suffix)
                rwkv_state = [t.clone() for t in state.rwkv7_state]
                last_id = int(state.last_token_id)
                if suffix_ids:
                    with torch.no_grad():
                        for tid in suffix_ids:
                            _, rwkv_state = self.backend._model.forward(
                                [int(tid)], rwkv_state
                            )
                        last_id = int(suffix_ids[-1])
                token_ids = decode_greedy_from_state(
                    self.backend._model,
                    rwkv_state,
                    last_id,
                    n,
                    metrics=self.metrics,
                    power_percent=int(self.config.power_percent),
                    temperature=float(self.config.temperature),
                    greedy=bool(self.config.greedy),
                    token_callback=token_callback,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
            restored = self.backend.get_recurrent_state()
            if restored is not None:
                self.backend.set_recurrent_state(restored)
            else:
                self.backend.set_recurrent_state(
                    RecurrentState(
                        last_token_id=token_ids[-1] if token_ids else state.last_token_id,
                        rwkv7_state=state.rwkv7_state,
                    )
                )
            return self.backend.decode_text(token_ids)
        raise RuntimeError("generate_followup unsupported for this backend path")

    def decode_from_state(
        self,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool | None = None,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        """Decode from the backend's current recurrent state without re-prefill."""
        with self._sampling_scope(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
        ):
            return self._decode_from_state_with_scope(
                max_tokens=max_tokens,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )

    def _decode_from_state_with_scope(
        self,
        max_tokens: int | None = None,
        *,
        token_callback=None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        """Implementation for :meth:`decode_from_state` inside sampling scope."""
        n = int(max_tokens if max_tokens is not None else self.config.max_tokens)
        state = self.backend.get_recurrent_state()
        if state is None:
            raise RuntimeError(
                "no recurrent state to continue from — run generate() or load_snapshot() first"
            )
        self.metrics.power_percent = int(self.config.power_percent)
        if isinstance(self.backend, PackBackend):
            provider = self._get_or_create_pack_provider()
            token_ids = self.backend.decode_greedy(
                state,
                provider,
                n,
                self.metrics,
                temperature=self.config.temperature if not self.config.greedy else 0.0,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            return self.backend.decode_text(token_ids)
        if isinstance(self.backend, RWKVCppBackend) and state.external_state is not None:
            assert self.manifest and self.store
            provider = self._get_or_create_streaming_provider()
            layer_ids = manifest_block_layers(self.manifest)
            token_ids = self.backend.generate_greedy_from_state(
                state,
                n,
                provider,
                self.manifest.by_layer(),
                layer_ids,
                self.metrics,
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            return self.backend.decode_text(token_ids)
        if isinstance(self.backend, ChatRWKVBackend) and state.rwkv7_state is not None:
            from rwkv_ssd.backends.rwkv7_forward import decode_greedy_from_state
            from rwkv_ssd.runtime.state_cache import RecurrentState

            assert self.backend._model is not None
            streaming = (
                self.config.mode != "resident"
                and self.manifest is not None
                and self.store is not None
                and self.scheduler is not None
            )
            if streaming:
                model_z = None
                model = self.backend._model
                if model is not None and hasattr(model, "z"):
                    model_z = model.z
                provider = self._get_or_create_streaming_provider(model_z=model_z)
                layer_ids = manifest_block_layers(self.manifest)
                by_layer = self.manifest.by_layer()
                token_ids = decode_greedy_from_state(
                    self.backend._model,
                    state.rwkv7_state,
                    state.last_token_id,
                    n,
                    metrics=self.metrics,
                    power_percent=int(self.config.power_percent),
                    provider=provider,
                    by_layer=by_layer,
                    layer_ids=layer_ids,
                    temperature=float(self.config.temperature),
                    greedy=bool(self.config.greedy),
                    token_callback=token_callback,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
            else:
                token_ids = decode_greedy_from_state(
                    self.backend._model,
                    state.rwkv7_state,
                    state.last_token_id,
                    n,
                    metrics=self.metrics,
                    power_percent=int(self.config.power_percent),
                    temperature=float(self.config.temperature),
                    greedy=bool(self.config.greedy),
                    token_callback=token_callback,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
            restored = self.backend.get_recurrent_state()
            if restored is not None:
                self.backend.set_recurrent_state(restored)
            else:
                self.backend.set_recurrent_state(
                    RecurrentState(
                        last_token_id=token_ids[-1] if token_ids else state.last_token_id,
                        rwkv7_state=state.rwkv7_state,
                    )
                )
            return self.backend.decode_text(token_ids)
        raise RuntimeError("continue_generate unsupported for this backend path")

    def _generate_pack_tokens(
        self,
        prompt: str,
        *,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        assert self.manifest and self.store
        provider = self._get_or_create_pack_provider()
        from rwkv_ssd.runtime.pack_generation import generate_greedy_tokens

        return generate_greedy_tokens(
            self.backend,
            provider,
            prompt,
            self.config.max_tokens,
            self.metrics,
            config=self.config,
            prefix_cache=self._prefix_cache,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def _generate_chatrwkv_streaming_tokens(
        self,
        prompt: str,
        *,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        assert self.manifest and self.store and self.scheduler
        model_z = None
        model = self.backend._model
        if model is not None and hasattr(model, "z"):
            model_z = model.z
        provider = self._get_or_create_streaming_provider(model_z=model_z)
        layer_ids = manifest_block_layers(self.manifest)
        return self.backend.generate_greedy_pack_streaming(
            prompt,
            self.config.max_tokens,
            provider,
            self.scheduler.layers,
            layer_ids,
            self.metrics,
            system_prefix=self.config.system_prefix,
            prefix_cache=self._prefix_cache,
            prefix_cache_mode=self.config.prefix_cache_mode,
            power_percent=int(self.config.power_percent),
            temperature=float(self.config.temperature),
            greedy=bool(self.config.greedy),
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def _generate_rwkvcpp_streaming_tokens(
        self,
        prompt: str,
        *,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        assert self.manifest and self.store and self.scheduler
        provider = self._get_or_create_streaming_provider()
        layer_ids = manifest_block_layers(self.manifest)
        return self.backend.generate_greedy_pack_streaming(
            prompt,
            self.config.max_tokens,
            provider,
            self.manifest.by_layer(),
            layer_ids,
            self.metrics,
            system_prefix=self.config.system_prefix,
            prefix_cache=self._prefix_cache,
            prefix_cache_mode=self.config.prefix_cache_mode,
            power_percent=int(self.config.power_percent),
            temperature=float(self.config.temperature),
            greedy=bool(self.config.greedy),
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def _generate_pack(self, prompt: str, *, cancel_event=None, deadline=None) -> str:
        assert self.manifest and self.store
        provider = self._get_or_create_pack_provider()
        token_ids = generate_greedy_tokens(
            self.backend,
            provider,
            prompt,
            self.config.max_tokens,
            self.metrics,
            config=self.config,
            prefix_cache=self._prefix_cache,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        (
            self._last_cache_write_submits,
            self._last_cache_write_sync_ms,
        ) = _record_cache_write_stats(
            self.metrics,
            self._last_cache_write_submits,
            self._last_cache_write_sync_ms,
        )
        return self.backend.decode_text(token_ids)

    def _generate_chatrwkv_streaming(
        self, prompt: str, *, cancel_event=None, deadline=None
    ) -> str:
        assert self.manifest and self.store and self.scheduler
        model_z = None
        model = self.backend._model
        if model is not None and hasattr(model, "z"):
            model_z = model.z
        provider = self._get_or_create_streaming_provider(model_z=model_z)
        try:
            layer_ids = manifest_block_layers(self.manifest)
            token_ids = self.backend.generate_greedy_pack_streaming(
                prompt,
                self.config.max_tokens,
                provider,
                self.scheduler.layers,
                layer_ids,
                self.metrics,
                system_prefix=self.config.system_prefix,
                prefix_cache=self._prefix_cache,
                prefix_cache_mode=self.config.prefix_cache_mode,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                cancel_event=cancel_event,
                deadline=deadline,
            )
            self.metrics.tokens_generated = len(token_ids)
            model = self.backend._model
            if model is not None and hasattr(model, "z"):
                from rwkv_ssd.runtime.rwkv7_skeleton import estimate_z_bytes

                self.metrics.z_bytes = estimate_z_bytes(model.z)
                self.metrics.weight_cache_bytes = self.metrics.z_bytes
            else:
                self.metrics.z_bytes = 0
            stats = provider.cache_stats()
            self.metrics.cache_format = str(stats["cache_format"])
            self.metrics.provider_cache_bytes = int(stats["provider_cache_bytes"])
            self.metrics.provider_layer_cache_bytes = int(
                stats.get("provider_layer_cache_bytes", 0)
            )
            self.metrics.provider_resident_bytes = int(
                stats.get("provider_resident_bytes", 0)
            )
            self.metrics.provider_mmap_bytes = int(
                stats.get("provider_mmap_bytes", 0)
            )
            self.metrics.packed_cache_bytes = int(stats["packed_cache_bytes"])
            self.metrics.prepared_cache_bytes = int(stats["prepared_cache_bytes"])
            self.metrics.lut2_index_cache_bytes = int(stats["lut2_index_cache_bytes"])
            self.metrics.packed_cache_evictions = int(stats["packed_cache_evictions"])
            self.metrics.provider_cache_evictions = int(
                stats.get("provider_cache_evictions", 0)
            )
            total_cmix = int(stats.get("cmix_total_elements", 0))
            self.metrics.cmix_samples = int(stats.get("cmix_samples", 0))
            self.metrics.cmix_zero_fraction = (
                int(stats.get("cmix_zero_elements", 0)) / total_cmix
                if total_cmix
                else 0.0
            )
            self.metrics.cmix_active_fraction = (
                int(stats.get("cmix_active_elements", 0)) / total_cmix
                if total_cmix
                else 0.0
            )
            self.metrics.cmix_tile_size = int(stats.get("cmix_tile_size", 0))
            self.metrics.cmix_tile_samples = int(stats.get("cmix_tile_samples", 0))
            self.metrics.cmix_tile_active_fraction = float(
                stats.get("cmix_tile_active_fraction", 0.0)
            )
            self.metrics.cmix_tile_occupancy = dict(
                stats.get("cmix_tile_occupancy", {})
            )
            self.metrics.cmix_selective_tiles_read = int(
                stats.get("cmix_selective_tiles_read", 0)
            )
            self.metrics.cmix_selective_tiles_skipped = int(
                stats.get("cmix_selective_tiles_skipped", 0)
            )
            self.metrics.cmix_selective_bytes_read = int(
                stats.get("cmix_selective_bytes_read", 0)
            )
            if not self.metrics.weight_cache_bytes:
                self.metrics.weight_cache_bytes = self.metrics.provider_cache_bytes
        except Exception:
            if self._streaming_provider is provider:
                provider.close()
                self._streaming_provider = None
            raise
        (
            self._last_cache_write_submits,
            self._last_cache_write_sync_ms,
        ) = _record_cache_write_stats(
            self.metrics,
            self._last_cache_write_submits,
            self._last_cache_write_sync_ms,
        )
        return self.backend.decode_text(token_ids)

    def _generate_rwkvcpp_streaming(
        self, prompt: str, *, cancel_event=None, deadline=None
    ) -> str:
        assert self.manifest and self.store and self.scheduler
        provider = self._get_or_create_streaming_provider()
        try:
            layer_ids = manifest_block_layers(self.manifest)
            token_ids = self.backend.generate_greedy_pack_streaming(
                prompt,
                self.config.max_tokens,
                provider,
                self.manifest.by_layer(),
                layer_ids,
                self.metrics,
                system_prefix=self.config.system_prefix,
                prefix_cache=self._prefix_cache,
                prefix_cache_mode=self.config.prefix_cache_mode,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                cancel_event=cancel_event,
                deadline=deadline,
            )
            self.metrics.tokens_generated = len(token_ids)
            stats = provider.cache_stats()
            self.metrics.cache_format = str(stats["cache_format"])
            self.metrics.provider_cache_bytes = int(stats["provider_cache_bytes"])
            self.metrics.provider_layer_cache_bytes = int(
                stats.get("provider_layer_cache_bytes", 0)
            )
            self.metrics.provider_resident_bytes = int(
                stats.get("provider_resident_bytes", 0)
            )
            self.metrics.provider_mmap_bytes = int(
                stats.get("provider_mmap_bytes", 0)
            )
            self.metrics.packed_cache_bytes = int(stats["packed_cache_bytes"])
            self.metrics.prepared_cache_bytes = int(stats["prepared_cache_bytes"])
            self.metrics.lut2_index_cache_bytes = int(stats["lut2_index_cache_bytes"])
            self.metrics.packed_cache_evictions = int(stats["packed_cache_evictions"])
            self.metrics.provider_cache_evictions = int(
                stats.get("provider_cache_evictions", 0)
            )
            total_cmix = int(stats.get("cmix_total_elements", 0))
            self.metrics.cmix_samples = int(stats.get("cmix_samples", 0))
            self.metrics.cmix_zero_fraction = (
                int(stats.get("cmix_zero_elements", 0)) / total_cmix
                if total_cmix
                else 0.0
            )
            self.metrics.cmix_active_fraction = (
                int(stats.get("cmix_active_elements", 0)) / total_cmix
                if total_cmix
                else 0.0
            )
            self.metrics.cmix_tile_size = int(stats.get("cmix_tile_size", 0))
            self.metrics.cmix_tile_samples = int(stats.get("cmix_tile_samples", 0))
            self.metrics.cmix_tile_active_fraction = float(
                stats.get("cmix_tile_active_fraction", 0.0)
            )
            self.metrics.cmix_tile_occupancy = dict(
                stats.get("cmix_tile_occupancy", {})
            )
            self.metrics.cmix_selective_tiles_read = int(
                stats.get("cmix_selective_tiles_read", 0)
            )
            self.metrics.cmix_selective_tiles_skipped = int(
                stats.get("cmix_selective_tiles_skipped", 0)
            )
            self.metrics.cmix_selective_bytes_read = int(
                stats.get("cmix_selective_bytes_read", 0)
            )
            self.metrics.weight_cache_bytes = self.metrics.provider_cache_bytes
        except Exception:
            if self._streaming_provider is provider:
                provider.close()
                self._streaming_provider = None
            raise
        return self.backend.decode_text(token_ids)

    def _generate_resident(
        self, prompt: str, *, cancel_event=None, deadline=None
    ) -> str:
        from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend

        if (
            isinstance(self.backend, ChatRWKVBackend)
            and self.backend._rwkv7
            and hasattr(self.backend, "generate_greedy_native")
        ):
            token_ids = self.backend.generate_greedy_native(
                prompt,
                self.config.max_tokens,
                metrics=self.metrics,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                cancel_event=cancel_event,
                deadline=deadline,
            )
            self.metrics.tokens_generated = len(token_ids)
            model = self.backend._model
            if model is not None and hasattr(model, "z"):
                from rwkv_ssd.runtime.rwkv7_skeleton import estimate_z_bytes

                self.metrics.z_bytes = estimate_z_bytes(model.z)
                self.metrics.weight_cache_bytes = self.metrics.z_bytes
            return self.backend.decode_text(token_ids)

        if (
            isinstance(self.backend, RWKVCppBackend)
            and hasattr(self.backend, "generate_greedy_native")
        ):
            token_ids = self.backend.generate_greedy_native(
                prompt,
                self.config.max_tokens,
                metrics=self.metrics,
                power_percent=int(self.config.power_percent),
                temperature=float(self.config.temperature),
                greedy=bool(self.config.greedy),
                cancel_event=cancel_event,
                deadline=deadline,
            )
            self.metrics.tokens_generated = len(token_ids)
            return self.backend.decode_text(token_ids)

        if not hasattr(self.backend, "generate_simple"):
            raise RuntimeError(
                f"backend {self.config.backend!r} has no generate_simple"
            )
        text = self.backend.generate_simple(
            prompt,
            self.config.max_tokens,
            greedy=self.config.greedy,
            temperature=float(self.config.temperature),
        )
        self.metrics.tokens_generated = self.config.max_tokens
        return text


if __name__ == "__main__":
    from rwkv_ssd.runtime.__main__ import main

    main()
