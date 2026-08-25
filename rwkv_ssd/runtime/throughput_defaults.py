"""Apply thesis-aligned I/O defaults for streaming decode (P2.a / Ch.10)."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_provider import pack_uses_quant_codec

_REPO = Path(__file__).resolve().parents[2]
_PARTIAL_HOT3 = _REPO / "deploy" / "rwkv7_0.1b_partial_hot3.json"
_PARTIAL_HOT4 = _REPO / "deploy" / "rwkv7_0.1b_partial_hot4.json"
_PARTIAL_HOT7 = _REPO / "deploy" / "rwkv7_0.1b_partial_hot7.json"
_PARTIAL_2_9B_HOT3 = _REPO / "deploy" / "rwkv7_2.9b_partial_hot3.json"


def _partial_hot3_profile(manifest: Manifest | None) -> Path:
    """Pick a hot3 residency profile scaled to the pack's layer count.

    The 0.1B hot7/hot4 profiles hardcode layer 11 as near-last — wrong on
    32-layer 2.9B packs (layer 11 is mid-stack). Prefer the 2.9B profile
    when ``n_layer >= 24``.
    """
    n_layer = 0
    if manifest is not None:
        n_layer = int(manifest.meta.get("n_layer", 0))
    if n_layer >= 24 and _PARTIAL_2_9B_HOT3.is_file():
        return _PARTIAL_2_9B_HOT3
    return _PARTIAL_HOT3


def _pack_uses_trinity(manifest: Manifest | None) -> bool:
    return pack_uses_quant_codec(manifest.tensors) if manifest is not None else False


def capture_active_toggles() -> dict:
    """Snapshot the throughput-defining env toggles as a machine-readable dict.

    Includes a derived ``stacked_count`` and ``stacked`` flag — the
    machine-enforceable no-stacking signal (see IDEAS.md §P2 and thesis §2.5).
    """
    import os

    def _on(name: str, default: str = "auto") -> bool:
        raw = os.environ.get(name, default).strip().lower()
        if raw in ("0", "false", "off", "no"):
            return False
        if raw in ("1", "true", "on", "yes"):
            return True
        return default == "1"

    shadow_active = _on("RWKV_DECODE_SHADOW", "auto")
    disk_cache_active = _on("RWKV_WARM_DISK_CACHE", "auto")
    fused_active = _on("RWKV_LUT_GEMM_FUSED", "auto")
    stream_cache_active = _on("RWKV_STREAM_LAYER_CACHE", "auto")
    # Historical key name ``warm_z_active`` actually tracks provider-cache warm
    # (``RWKV_WARM_PROVIDER_CACHE``), not ``config.warm_z`` / ``RWKV_WARM_Z``.
    warm_provider_cache_active = _on("RWKV_WARM_PROVIDER_CACHE", "auto")

    stacked_count = sum(
        bool(x)
        for x in (
            shadow_active,
            disk_cache_active,
            fused_active,
            stream_cache_active,
            warm_provider_cache_active,
        )
    )
    return {
        "shadow_active": shadow_active,
        "decode_disk_cache_active": disk_cache_active,
        "fused_gemm_active": fused_active,
        "stream_layer_cache_active": stream_cache_active,
        "warm_provider_cache_active": warm_provider_cache_active,
        "warm_z_active": warm_provider_cache_active,  # alias (legacy benches)
        "stacked_count": stacked_count,
        "stacked": stacked_count >= 2,
    }


def _stream_layer_cache_auto(manifest: Manifest | None) -> bool:
    """Throughput default: RAM decode cache + bounded ``z`` (not strict per-token reload)."""
    import os

    raw = os.environ.get("RWKV_STREAM_LAYER_CACHE", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    if manifest is None:
        return False
    return _pack_uses_trinity(manifest) or pack_uses_quant_codec(manifest.tensors)


def _warm_provider_cache_auto(manifest: Manifest | None) -> bool:
    """Pre-decode streamed layers at load so first token skips SSD+LUT staging."""
    import os

    raw = os.environ.get("RWKV_WARM_PROVIDER_CACHE", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    if manifest is None:
        return False
    return _pack_uses_trinity(manifest)


def _warm_disk_cache_auto(manifest: Manifest | None) -> bool:
    """Pre-build ``.decode_cache/`` for streamed layers at load (strict / bounded paths)."""
    import os

    raw = os.environ.get("RWKV_WARM_DISK_CACHE", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    if manifest is None:
        return False
    if not _pack_uses_trinity(manifest):
        return False
    promote = os.environ.get("RWKV_PROMOTE_FULL_Z", "auto").strip().lower()
    if promote in ("1", "true", "on", "yes"):
        return False
    if promote in ("0", "false", "off", "no"):
        return True
    n_layer = int(manifest.meta.get("n_layer", 0))
    from rwkv_ssd.runtime.rwkv7_weights import promote_full_z_enabled

    return not promote_full_z_enabled(n_layer)


def warm_disk_cache_active(manifest: Manifest | None = None) -> bool:
    """True when warm ``.decode_cache/`` is enabled for this session."""
    import os

    raw = os.environ.get("RWKV_WARM_DISK_CACHE", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    return _warm_disk_cache_auto(manifest)


def apply_cache_format_defaults(config: EngineConfig) -> None:
    """Normalize the explicit packed/prepared/dense residency axis.

    Legacy presets remain unchanged when ``cache_format=auto``.  Explicit
    formats are applied after the normal throughput preset so a user can ask
    for a truthful disk-backed profile without an auto preset silently
    re-enabling a decoded provider cache.
    """
    fmt = (config.cache_format or "auto").strip().lower()
    if fmt not in {"auto", "none", "packed", "prepared", "dense"}:
        raise ValueError(f"unsupported cache_format: {config.cache_format!r}")
    if fmt == "none":
        config.stream_layer_cache = False
        config.warm_z = False
        config.max_layers_in_z = 0
        config.max_provider_cache_layers = 0
        config.max_provider_cache_bytes = 0
        config.prepared_cache_bytes = 0
    elif fmt == "packed":
        # Packed mode uses fused LUT blobs but does not retain decoded tensors
        # in the ordinary provider LRU.  A zero cap means unlimited legacy
        # packed retention; a positive cap is enforced by the provider.
        config.stream_layer_cache = False
        config.warm_z = False
        config.max_layers_in_z = 0
        config.max_provider_cache_layers = 0
        config.max_provider_cache_bytes = 0
        config.prepared_cache_bytes = 0
    elif fmt == "prepared":
        config.stream_layer_cache = True
        config.decouple_provider_cache = True
    elif fmt == "dense":
        config.warm_z = True
        config.stream_layer_cache = True


def apply_auto_residency_policy(
    config: EngineConfig, manifest: Manifest
) -> str:
    """Select a deterministic Pareto cache format from pack bytes and budget.

    This policy is intentionally static for one engine lifetime; changing
    residency during a token sweep would create cache thrash and unstable
    latency. A later engine reload can select a new point from new telemetry.
    Explicit ``cache_format`` always wins.
    """
    policy = (config.residency_policy or "static").strip().lower()
    if policy not in {"static", "auto"}:
        raise ValueError(f"unsupported residency_policy: {policy!r}")
    if policy == "static" or config.cache_format != "auto":
        return config.cache_format

    tier = str(getattr(config, "_ram_budget_tier_applied", ""))
    if tier:
        selected = {
            "F1": "packed",
            "F2": "prepared",
            "F3": "prepared",
            "F4": "prepared",
            "F5": "dense",
        }.get(tier, "packed")
        config.cache_format = selected
        return selected

    streamed = manifest.streamed_tensors() or manifest.tensors
    packed_total = sum(entry.length for entry in streamed)
    dense_total = sum(entry.numel * 2 for entry in streamed)
    by_layer: dict[int, tuple[int, int]] = {}
    for entry in streamed:
        packed, dense = by_layer.get(entry.layer_id, (0, 0))
        by_layer[entry.layer_id] = (
            packed + entry.length,
            dense + entry.numel * 2,
        )
    largest_packed_layer = max((pair[0] for pair in by_layer.values()), default=0)
    largest_dense_layer = max((pair[1] for pair in by_layer.values()), default=0)
    budget_gb = config.cache_budget_gb
    budget = int(float(budget_gb) * 1e9) if budget_gb is not None else 0
    uses_quant = pack_uses_quant_codec(streamed)
    if budget_gb is not None:
        if budget >= dense_total and dense_total > 0:
            selected = "dense"
        elif budget >= largest_dense_layer and largest_dense_layer > 0:
            selected = "prepared"
        elif uses_quant and budget >= largest_packed_layer and packed_total > 0:
            selected = "packed"
        else:
            selected = "none"
    else:
        selected = "packed" if uses_quant else "prepared"
    config.cache_format = selected
    return selected


def apply_streaming_defaults(
    config: EngineConfig, manifest: Manifest | None = None
) -> None:
    """
    Turn on practical latency hiders when the user did not set explicit I/O flags.

    - ``layer_size`` chunk policy: micro-pipeline within large layers (thesis §2.5 stage 5).
    - ``mmap_sequential``: sequential madvise on layer sweep (Ch.10).
    - Trinity streaming: persistent ``.decode_cache/`` on by default (SSD trade for tok/s).
    - Trinity/quant streaming: ``stream_layer_cache`` + optional provider warm (min SSD tax).
    """
    if config.mode not in ("streaming", "partial"):
        return
    if not config.warm_z and _stream_layer_cache_auto(manifest):
        config.stream_layer_cache = True
    if config.stream_layer_cache and not config.warm_z:
        config.decouple_provider_cache = True
        if manifest is not None and config.max_provider_cache_layers <= 0:
            n_layer = int(manifest.meta.get("n_layer", 0))
            if n_layer > 0:
                import os

                promote_raw = (
                    os.environ.get("RWKV_PROMOTE_FULL_Z", "auto").strip().lower()
                )
                promote_on = promote_raw in ("1", "true", "on", "yes") or (
                    promote_raw in ("", "auto") and n_layer < 8
                )
                if promote_on or n_layer < 8:
                    # Full-z promote (n<8 auto, or RWKV_PROMOTE_FULL_Z=1)
                    # keeps every block layer in the provider cache.
                    config.max_provider_cache_layers = n_layer
                elif _pack_uses_trinity(manifest):
                    # Keep the default honest: a streaming tier has a
                    # bounded decoded working set.  Full provider retention
                    # is a valid F5 throughput profile, but it must be
                    # explicitly requested through a cache budget or a layer
                    # cap rather than silently replacing the SSD contract.
                    config.max_provider_cache_layers = min(2, n_layer)
    policy = config.io_chunk_policy.strip().lower()
    if config.io_chunk_bytes <= 0 and policy in ("", "off", "none", "uniform"):
        config.io_chunk_policy = "layer_size"
    if config.io_backend == "mmap" and not config.mmap_sequential:
        config.mmap_sequential = True
    if config.decode_disk_cache is None and _pack_uses_trinity(manifest):
        config.decode_disk_cache = "auto"
    if config.prefetch_io_only is None and manifest is not None:
        from rwkv_ssd.runtime.device import resolve_device
        from rwkv_ssd.runtime.weight_provider import resolve_prefetch_io_only

        config.prefetch_io_only = resolve_prefetch_io_only(
            None,
            device=resolve_device(config.device),
            entries=manifest.tensors,
            mode=config.mode,
        )
    if manifest is not None and manifest.has_bf16_shadow():
        import logging
        import os

        logging.getLogger(__name__).info(
            "bf16 shadow present — decode uses shadow.bin (set RWKV_DECODE_SHADOW=0 to disable)"
        )
        if os.environ.get("RWKV_DECODE_SHADOW", "auto").strip().lower() in (
            "",
            "auto",
        ):
            os.environ.setdefault("RWKV_DECODE_SHADOW", "1")
    if _pack_uses_trinity(manifest):
        import os

        os.environ.setdefault("RWKV_LUT_BF16_NATIVE", "auto")


def apply_bounded_stream_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    Stream+cache with small RAM: 2-layer ``z`` + 2-layer provider LRU + SSD ``.decode_cache/``.

    Skips full-z promote (~380MB) so revisit layers load from disk cache instead of
    holding all blocks in ``z``. Second-token I/O cap approaches full stream+cache.
    """
    import os

    if config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = True
    config.decouple_provider_cache = True
    if config.max_layers_in_z <= 0 or config.max_layers_in_z > 2:
        config.max_layers_in_z = 2
    # ``max_provider_cache_layers`` is left at 0 here so the
    # ``resolve_max_provider_cache_layers`` resolver in
    # ``provider_factory`` picks the right value based on ``n_block``
    # (full cache for n>=8 decoupled, smaller for budget-limited).
    # The old hardcoded 2 was wrong for any model with more than 2
    # block layers: a 2-layer LRU on a 12-layer model has a 16.7%
    # hit rate and the management overhead exceeds the savings.
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    # Provider RAM already holds decoded layers; warm disk cache is redundant
    # I/O at load and contends with decode.
    os.environ.setdefault("RWKV_WARM_DISK_CACHE", "0")
    # Keep the 2-layer z window honest (no extra first/last accuracy pins).
    os.environ.setdefault("RWKV_PIN_ACCURACY_LAYERS", "0")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "1")
    apply_streaming_defaults(config, manifest)


def apply_partial_ssd_tier_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
    *,
    residency_profile: Path | None = None,
) -> None:
    """
    Partial hot3 + strict fused (~262 MB ``z`` skeleton on 0.1B).

    Pins blocks 0–2 (+ auto last) in skeleton; layers 3–10 mmap+fused each token
    with **no** stream cache or provider RAM. Fewer layers per token than full
    ``RWKV_SSD_TIER=1`` strict path — intended RAM/speed compromise.
    """
    import os

    if config.mode == "resident":
        config.mode = "partial"
    elif config.mode not in ("streaming", "partial"):
        return
    else:
        config.mode = "partial"
    profile = residency_profile or _partial_hot3_profile(manifest)
    if profile.is_file():
        config.residency_profile = profile
    config.stream_layer_cache = False
    config.decouple_provider_cache = True
    config.warm_z = False
    config.max_layers_in_z = 0
    config.max_provider_cache_layers = 0
    has_shadow = manifest is not None and manifest.has_bf16_shadow()
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "0" if has_shadow else "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_PREFER_FUSED_LUT", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    os.environ.setdefault("RWKV_WARM_PROVIDER_CACHE", "0")
    apply_streaming_defaults(config, manifest)
    config.stream_layer_cache = False
    config.max_provider_cache_layers = 0
    os.environ.setdefault("RWKV_STRICT_FUSED_RETAIN", "auto")
    os.environ.setdefault("RWKV_STRICT_FUSED_LEAN_Z", "auto")


def apply_partial_hot4_ssd_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """Partial hot4 + strict fused — pin 0,1,2,11; stream middle layers only."""
    apply_partial_ssd_tier_defaults(config, manifest, residency_profile=_PARTIAL_HOT4)


def apply_partial_fused_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    Partial hot3 + bounded stream cache + fused att/ffn/head (~263 MB ``z`` on 0.1B).

    **Deprecated for frontier benches** — usually slower than
    ``apply_partial_ssd_tier_defaults`` on fast NVMe. Kept for ``--legacy`` regression.
    """
    import os

    if config.mode == "resident":
        config.mode = "partial"
    elif config.mode not in ("streaming", "partial"):
        return
    else:
        config.mode = "partial"
    profile = _partial_hot3_profile(manifest)
    if profile.is_file():
        config.residency_profile = profile
    config.stream_layer_cache = True
    config.decouple_provider_cache = True
    config.warm_z = False
    config.max_layers_in_z = 2
    # Leave ``max_provider_cache_layers`` at 0 so the resolver picks.
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    os.environ.setdefault("RWKV_WARM_PROVIDER_CACHE", "0")
    apply_streaming_defaults(config, manifest)


def apply_bounded_fused_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    Low RAM + speed: 2-layer ``z`` window, fused att/ffn/head, SSD ``.decode_cache/``.

    ~203 MB ``z`` on 0.1B; no full promote. Steady tokens reload evicted layers from
    ``.decode_cache/`` or mmap+fused on revisit.

    Note: the *provider* cache (decoupled from ``z``) is sized by the
    ``resolve_max_provider_cache_layers`` resolver — full cache for
    n>=8 in unconstrained mode, capped for budget-limited. The old
    hardcoded 2-layer LRU was a net loss on any model with more than
    2 block layers (16.7% hit rate on 0.1B; management overhead
    exceeded the savings). The 2-layer cap here is only for ``z``;
    the provider cache is independent.
    """
    import os

    if config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = True
    config.decouple_provider_cache = True
    config.max_layers_in_z = 2
    # Leave ``max_provider_cache_layers`` at 0 so the resolver picks.
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "0"
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    # Full decoupled provider cache makes load-time warm disk cache redundant.
    os.environ.setdefault("RWKV_WARM_DISK_CACHE", "0")
    # Accuracy pins of layers 0 + n-1 stacked on the 2-layer LRU → 4 layers
    # in z (BASELINE_BUGS F2 z-cap). Disable for this preset.
    os.environ.setdefault("RWKV_PIN_ACCURACY_LAYERS", "0")
    apply_streaming_defaults(config, manifest)


def apply_promote_max_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    LUT Trinity stream+cache + full ``z`` promote (~382 MB on 0.1B).

    Frontier max tok/s on LUT packs without shadow sidecar. After warm, native
    ``forward`` with staging ≈ 0.
    """
    import os

    if config.mode == "resident":
        config.mode = "streaming"
    elif config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = True
    if config.warm_z:
        # warm_z bypasses the LRU cap, so we can request "all block layers" and
        # let ZLayerRetention retain them after warm. The previous default of 1
        # only worked because warm_z happened to override it; spelling it out
        # avoids surprising the resolver on small packs.
        if manifest is not None and config.max_layers_in_z <= 0:
            from rwkv_ssd.runtime.layer_keys import layer_ids

            n_block = len(layer_ids(manifest.tensors))
            if n_block > 0:
                config.max_layers_in_z = n_block
    elif config.max_layers_in_z <= 0:
        config.max_layers_in_z = 1
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "1"
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    apply_streaming_defaults(config, manifest)


def apply_stacked_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    Stack throughput optimizations (not the SSD-tier minimal-RAM path).

    selective shadow pack + stream layer cache + warm provider + full-z promote
    + ``.decode_cache/`` + shadow decode + native LUT. Steady state holds all
    block weights in ``z`` and runs native ``forward`` (~382 MB on 0.1B).
    """
    import os

    if config.mode == "resident":
        config.mode = "streaming"
    elif config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = True
    if config.max_layers_in_z <= 0:
        config.max_layers_in_z = 1
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ["RWKV_PROMOTE_FULL_Z"] = "1"
    os.environ.setdefault("RWKV_DECODE_SHADOW", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    apply_streaming_defaults(config, manifest)


def apply_stacked_strict_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """Stack strict-path wins: shadow pack + fused att/ffn/head + disk cache."""
    import os

    if config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = False
    config.max_layers_in_z = 0
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ.setdefault("RWKV_PROMOTE_FULL_Z", "0")
    os.environ.setdefault("RWKV_DECODE_SHADOW", "1")
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    os.environ.setdefault("RWKV_WARM_DISK_CACHE", "auto")
    apply_streaming_defaults(config, manifest)


def apply_ssd_tier_fused_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    Infinite-SSD / finite-RAM preset (~201 MB ``z`` skeleton).

    Fused LUT GEMV for att+FFN+head (no bf16 weight slabs in RAM). Pack bytes
    come from mmap; ``.decode_cache/`` is optional cold backup (uncompressed).
    """
    import os

    if config.mode == "resident":
        config.mode = "streaming"
    elif config.mode not in ("streaming", "partial"):
        return
    config.max_layers_in_z = 0
    has_shadow = manifest is not None and manifest.has_bf16_shadow()
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "0" if has_shadow else "auto"
    os.environ.setdefault("RWKV_PROMOTE_FULL_Z", "0")
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_PREFER_FUSED_LUT", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    os.environ.setdefault("RWKV_WARM_DISK_CACHE", "auto")
    apply_streaming_defaults(config, manifest)
    config.stream_layer_cache = False
    os.environ.setdefault("RWKV_STRICT_FUSED_RETAIN", "auto")
    os.environ.setdefault("RWKV_STRICT_FUSED_LEAN_Z", "auto")
    if manifest is not None and manifest.has_bf16_shadow():
        os.environ.setdefault("RWKV_DECODE_SHADOW", "1")
        os.environ.setdefault("RWKV_STRICT_FUSED_RETAIN_SHADOW", "1")


def apply_ssd_stream_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    SSD spill tier: small provider LRU + disk ``.decode_cache/``, no full-z promote.

    Uses fused matmul when weights are not resident; disk cache for shadow-heavy packs.
    """
    import os

    if config.mode == "resident":
        config.mode = "streaming"
    elif config.mode not in ("streaming", "partial"):
        return
    config.stream_layer_cache = True
    config.decouple_provider_cache = True
    config.max_layers_in_z = 0
    # Leave ``max_provider_cache_layers`` at 0 so the resolver picks.
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ.setdefault("RWKV_PROMOTE_FULL_Z", "0")
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
    apply_streaming_defaults(config, manifest)


def apply_ssd_health_overlay(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """Gentle SSD tweaks applied on top of any throughput tier."""
    import os

    config.mmap_dontneed = False
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_WRITERS", "1")
    _ = manifest  # reserved for future pack-size-aware caps


def apply_ssd_health_defaults(
    config: EngineConfig,
    manifest: Manifest | None = None,
) -> None:
    """
    SSD health / temperature preset: fewer bytes per token, warm disk cache,
    no page-cache eviction, gentler cache writes.

    Trades some tok/s vs F5 for lower sustained NVMe read duty. Best paired
    with offline ``build_decode_cache --compress``. See ``docs/SSD_HEALTH.md``.
    """
    import os

    if config.mode == "resident":
        return
    if config.mode not in ("streaming", "partial"):
        config.mode = "streaming"
    profile = _partial_hot3_profile(manifest)
    if profile.is_file():
        config.residency_profile = profile
    config.stream_layer_cache = True
    config.decouple_provider_cache = True
    config.warm_z = False
    config.max_layers_in_z = 2
    config.mmap_dontneed = False
    if config.decode_disk_cache is None:
        config.decode_disk_cache = "auto"
    os.environ.setdefault("RWKV_PROMOTE_FULL_Z", "0")
    os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "1")
    os.environ.setdefault("RWKV_DECODE_CACHE_WRITERS", "1")
    os.environ.setdefault("RWKV_WARM_DISK_CACHE", "1")
    os.environ.setdefault("RWKV_LUT_GEMM_FUSED", "1")
    apply_streaming_defaults(config, manifest)
    config.stream_layer_cache = True
    apply_ssd_health_overlay(config, manifest)


def apply_low_ram_defaults(
    config: EngineConfig, manifest: Manifest | None = None
) -> None:
    """
    Thesis low-RAM preset.

    Default: partial hot3 + strict fused (F3t / ``apply_partial_ssd_tier_defaults``).
    When ``ram_budget_gb`` is set, select the matching frontier tier first
    (``apply_ram_budget_tier``), then let the planner fill residual caps
    without clobbering tier fields — same order as ``InferenceEngine.load``.
    """
    import os

    if config.ram_budget_gb is None and manifest is not None:
        n_layer = int(manifest.meta.get("n_layer", 0))
        if n_layer >= 16:
            config.ram_budget_gb = 10.0
    if config.ram_budget_gb and config.ram_budget_gb > 0 and manifest is not None:
        from rwkv_ssd.runtime.ram_budget import (
            apply_ram_budget_to_config,
            apply_ram_budget_tier,
            select_ram_budget_tier,
        )

        n_layer = int(manifest.meta.get("n_layer", 0)) or None
        tier = select_ram_budget_tier(float(config.ram_budget_gb))
        apply_ram_budget_tier(config, manifest, tier=tier)
        apply_ram_budget_to_config(config, manifest, n_layer=n_layer)
        # Tier presets already set PROMOTE / LUT / compress; only fill gaps.
        os.environ.setdefault("RWKV_DECODE_CACHE_COMPRESS", "0")
        apply_streaming_defaults(config, manifest)
        return
    apply_partial_ssd_tier_defaults(config, manifest)


def apply_partial_defaults(
    config: EngineConfig, manifest: Manifest | None = None
) -> None:
    """Use bundled hot-layer profile for 0.1B-class packs when none was passed."""
    if config.mode != "partial":
        return
    if (
        config.residency_profile is not None
        or config.residency_profile_inline is not None
    ):
        return
    n_layer = 0
    if manifest is not None:
        n_layer = int(manifest.meta.get("n_layer", 0))
    if n_layer >= 24:
        # 2.9B-class: use early-hot3, not the 0.1B hot7 (which pins layer 11).
        config.residency_profile = _partial_hot3_profile(manifest)
    elif n_layer == 12:
        # 0.1B-class: hot7 pins 0–5 and last layer 11.
        if _PARTIAL_HOT7.is_file():
            config.residency_profile = _PARTIAL_HOT7
        elif _PARTIAL_HOT4.is_file():
            config.residency_profile = _PARTIAL_HOT4
    elif n_layer >= 8:
        # Mid-size packs: do not reuse 0.1B hot7 (layer 11 is mid-stack).
        config.residency_profile = _partial_hot3_profile(manifest)
