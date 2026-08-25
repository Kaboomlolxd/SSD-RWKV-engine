"""Runtime configuration from CLI, env, and optional YAML file."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None


# Native rwkv.cpp is the default for real CPU RWKV use. Synthetic packs and
# the PyTorch compatibility path remain available through an explicit backend.
DEFAULT_BACKEND = "rwkvcpp"


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse booleans from YAML/JSON values without treating ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _as_bool(value, default=False)


def _config_path(value: Any, config_path: Path | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if config_path is not None and not path.is_absolute():
        path = (config_path.resolve().parent / path).resolve()
    return path


@dataclass
class EngineConfig:
    pack_dir: Path
    mode: str = "resident"
    backend: str = DEFAULT_BACKEND
    device: str = "cpu"
    max_tokens: int = 64
    temperature: float = 1.0
    greedy: bool = True
    # ``top_p`` and ``seed`` are request/config defaults shared by Torch and
    # native NumPy backends.  The sampler validates top_p at request entry so
    # a malformed HTTP or direct-engine request cannot silently sample from
    # an empty distribution.
    top_p: float = 1.0
    seed: int | None = None
    checkpoint_path: str | None = None
    strategy: str = "cpu fp32"
    metrics_csv: Path | None = None
    config_path: Path | None = None
    io_backend: str = "mmap"
    io_chunk_bytes: int = 0
    prefetch_policy: str = "layer_aware"
    # CPU SSD streaming defaults to one synchronous read path. Async prefetch
    # remains opt-in for measured deployments.
    prefetch_enabled: bool = False
    log_level: str = "INFO"
    state_cache: bool = False
    prefix_cache_mode: str = "system"
    prefix_cache_max_entries: int = 16
    system_prefix: str | None = None
    progress: bool = False
    residency_profile: Path | None = None
    verify_hash: bool = False
    skeleton_load: bool = True
    stream_layer_cache: bool = False
    warm_z: bool = False
    max_layers_in_z: int = 1
    mmap_sequential: bool = False
    mmap_willneed: bool = True
    mmap_dontneed: bool = False
    io_chunk_policy: str = "uniform"
    io_hedged: bool = False
    ngram_weight_cache: bool = False
    mtp_speculative: bool = False
    mtp_draft_tokens: int = 4
    mtp_min_prefill_tokens: int = 32
    # Trinity LUT gather: auto | cpu | xpu (Intel iGPU via torch+xpu wheels)
    trinity_decode_device: str = "auto"
    decouple_provider_cache: bool = True
    max_provider_cache_layers: int = 0
    max_provider_cache_bytes: int = 0
    # Packed and prepared caches are separate residency axes.  ``auto``
    # preserves the historical behavior; ``none`` is the true disk-backed
    # strict profile, while ``packed`` retains only compressed LUT payloads.
    cache_format: str = "auto"
    packed_cache_bytes: int = 0
    prepared_cache_bytes: int = 0
    residency_policy: str = "static"
    adaptive_residency: bool = False
    adaptive_residency_window: int = 8
    adaptive_residency_min_dwell_tokens: int = 32
    adaptive_residency_hysteresis: float = 0.15
    adaptive_residency_max_changes: int = 2
    # Request-lifetime-aware dense layer promotion. This is separate from
    # adaptive cache-format retiering and is intentionally opt-in.
    session_promotion: bool = False
    session_expected_tokens: int = 0
    session_promotion_bytes: int = 0
    session_promotion_policy: str = "benefit_per_byte"
    decode_disk_cache: str | None = None
    low_ram: bool = False
    ram_budget_gb: float | None = None
    cache_budget_gb: float | None = None
    cache_budget_auto: bool = False
    power_percent: int = 100
    trace_path: Path | None = None
    residency_profile_inline: dict[str, Any] | None = None
    prefetch_io_only: bool | None = None
    codec_policy: str = "auto"
    # Opt-in parity diagnostics.  These are guardrails for backend
    # qualification, not requirements for identical internal state layouts.
    parity_min_top10_overlap: float = 0.80
    parity_max_kl: float = 0.05
    parity_max_state_relative_error: float = 0.10

    @classmethod
    def from_file(cls, path: Path) -> EngineConfig:
        if yaml is None:
            raise ImportError("PyYAML required for config files: pip install pyyaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a mapping/object")
        return cls.from_mapping(raw, config_path=path)

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any], config_path: Path | None = None
    ) -> EngineConfig:
        if not isinstance(raw, dict):
            raise ValueError("config must be a mapping/object")
        pack = raw.get("pack_dir") or raw.get("model")
        if not pack:
            raise ValueError("config must set pack_dir or model")
        metrics = raw.get("metrics_csv")
        profile = raw.get("residency_profile")
        config_path = Path(config_path) if config_path is not None else None
        return cls(
            pack_dir=_config_path(pack, config_path) or Path(pack),
            mode=str(raw.get("mode", "resident")),
            backend=str(raw.get("backend", DEFAULT_BACKEND)),
            device=str(raw.get("device", "cpu")),
            max_tokens=int(raw.get("max_tokens", 64)),
            temperature=float(raw.get("temperature", 1.0)),
            greedy=_as_bool(raw.get("greedy"), default=True),
            top_p=float(raw.get("top_p", 1.0)),
            seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
            checkpoint_path=str(
                _config_path(
                    raw.get("checkpoint_path") or raw.get("checkpoint"), config_path
                )
            )
            if (raw.get("checkpoint_path") or raw.get("checkpoint"))
            else None,
            strategy=str(raw.get("strategy", "cpu fp32")),
            metrics_csv=_config_path(metrics, config_path),
            config_path=config_path,
            io_backend=str(raw.get("io_backend", "mmap")),
            io_chunk_bytes=int(raw.get("io_chunk_bytes", 0)),
            prefetch_policy=str(raw.get("prefetch_policy", "layer_aware")),
            prefetch_enabled=_as_bool(raw.get("prefetch_enabled"), default=False),
            log_level=str(raw.get("log_level", "INFO")),
            state_cache=_as_bool(raw.get("state_cache"), default=False),
            prefix_cache_mode=str(raw.get("prefix_cache_mode", "system")),
            prefix_cache_max_entries=int(raw.get("prefix_cache_max_entries", 16)),
            system_prefix=raw.get("system_prefix"),
            progress=_as_bool(raw.get("progress"), default=False),
            residency_profile=_config_path(profile, config_path),
            verify_hash=_as_bool(raw.get("verify_hash"), default=False),
            skeleton_load=_as_bool(raw.get("skeleton_load"), default=True),
            stream_layer_cache=_as_bool(raw.get("stream_layer_cache"), default=False),
            warm_z=_as_bool(raw.get("warm_z"), default=False),
            max_layers_in_z=int(raw.get("max_layers_in_z", 1)),
            mmap_sequential=_as_bool(raw.get("mmap_sequential"), default=False),
            mmap_willneed=_as_bool(raw.get("mmap_willneed"), default=True),
            mmap_dontneed=_as_bool(raw.get("mmap_dontneed"), default=False),
            io_chunk_policy=str(raw.get("io_chunk_policy", "uniform")),
            io_hedged=_as_bool(raw.get("io_hedged"), default=False),
            ngram_weight_cache=_as_bool(raw.get("ngram_weight_cache"), default=False),
            mtp_speculative=_as_bool(raw.get("mtp_speculative"), default=False),
            mtp_draft_tokens=int(raw.get("mtp_draft_tokens", 4)),
            mtp_min_prefill_tokens=int(raw.get("mtp_min_prefill_tokens", 32)),
            trinity_decode_device=str(raw.get("trinity_decode_device", "auto")),
            decouple_provider_cache=_as_bool(
                raw.get("decouple_provider_cache"), default=True
            ),
            max_provider_cache_layers=int(raw.get("max_provider_cache_layers", 0)),
            max_provider_cache_bytes=int(raw.get("max_provider_cache_bytes", 0)),
            cache_format=str(raw.get("cache_format", "auto")).strip().lower(),
            packed_cache_bytes=int(raw.get("packed_cache_bytes", 0)),
            prepared_cache_bytes=int(raw.get("prepared_cache_bytes", 0)),
            residency_policy=str(raw.get("residency_policy", "static")).strip().lower(),
            adaptive_residency=_as_bool(raw.get("adaptive_residency"), default=False),
            adaptive_residency_window=int(raw.get("adaptive_residency_window", 8)),
            adaptive_residency_min_dwell_tokens=int(
                raw.get("adaptive_residency_min_dwell_tokens", 32)
            ),
            adaptive_residency_hysteresis=float(
                raw.get("adaptive_residency_hysteresis", 0.15)
            ),
            adaptive_residency_max_changes=int(
                raw.get("adaptive_residency_max_changes", 2)
            ),
            session_promotion=_as_bool(raw.get("session_promotion"), default=False),
            session_expected_tokens=max(0, int(raw.get("session_expected_tokens", 0))),
            session_promotion_bytes=max(0, int(raw.get("session_promotion_bytes", 0))),
            session_promotion_policy=str(
                raw.get("session_promotion_policy", "benefit_per_byte")
            ).strip().lower(),
            decode_disk_cache=raw.get("decode_disk_cache"),
            low_ram=_as_bool(raw.get("low_ram"), default=False),
            ram_budget_gb=(
                float(raw["ram_budget_gb"])
                if raw.get("ram_budget_gb") is not None
                else None
            ),
            cache_budget_gb=(
                float(raw["cache_budget_gb"])
                if raw.get("cache_budget_gb") is not None
                else None
            ),
            cache_budget_auto=_as_bool(raw.get("cache_budget_auto"), default=False),
            power_percent=int(raw.get("power_percent", 100)),
            trace_path=_config_path(raw.get("trace_path"), config_path),
            prefetch_io_only=_optional_bool(raw.get("prefetch_io_only")),
            codec_policy=str(raw.get("codec_policy", "auto")),
            parity_min_top10_overlap=float(raw.get("parity_min_top10_overlap", 0.80)),
            parity_max_kl=float(raw.get("parity_max_kl", 0.05)),
            parity_max_state_relative_error=float(
                raw.get("parity_max_state_relative_error", 0.10)
            ),
        )

    @classmethod
    def merge_env(cls, cfg: EngineConfig) -> EngineConfig:
        if os.environ.get("RWKV_SSD_PACK"):
            cfg.pack_dir = Path(os.environ["RWKV_SSD_PACK"])
        if os.environ.get("RWKV_SSD_MODE"):
            cfg.mode = os.environ["RWKV_SSD_MODE"]
        if os.environ.get("RWKV_SSD_BACKEND"):
            cfg.backend = os.environ["RWKV_SSD_BACKEND"]
        if os.environ.get("RWKV_SSD_DEVICE"):
            cfg.device = os.environ["RWKV_SSD_DEVICE"]
        if os.environ.get("RWKV_SSD_TOP_P"):
            cfg.top_p = float(os.environ["RWKV_SSD_TOP_P"])
        if os.environ.get("RWKV_SSD_SEED"):
            cfg.seed = int(os.environ["RWKV_SSD_SEED"])
        if os.environ.get("RWKV_TRINITY_DECODE_DEVICE"):
            cfg.trinity_decode_device = os.environ["RWKV_TRINITY_DECODE_DEVICE"]
        if os.environ.get("RWKV_SSD_SYSTEM_PREFIX"):
            cfg.system_prefix = os.environ["RWKV_SSD_SYSTEM_PREFIX"]
            cfg.state_cache = True
        if os.environ.get("RWKV_RAM_BUDGET_GB"):
            cfg.ram_budget_gb = float(os.environ["RWKV_RAM_BUDGET_GB"])
        if os.environ.get("RWKV_CACHE_BUDGET_GB"):
            cfg.cache_budget_gb = float(os.environ["RWKV_CACHE_BUDGET_GB"])
        if os.environ.get("RWKV_CACHE_BUDGET_AUTO", "").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        ):
            cfg.cache_budget_auto = True
        if os.environ.get("RWKV_CACHE_FORMAT"):
            cfg.cache_format = os.environ["RWKV_CACHE_FORMAT"].strip().lower()
        if os.environ.get("RWKV_PACKED_CACHE_BYTES"):
            cfg.packed_cache_bytes = int(os.environ["RWKV_PACKED_CACHE_BYTES"])
        if os.environ.get("RWKV_PREPARED_CACHE_BYTES"):
            cfg.prepared_cache_bytes = int(os.environ["RWKV_PREPARED_CACHE_BYTES"])
        if os.environ.get("RWKV_PARITY_MIN_TOP10_OVERLAP"):
            cfg.parity_min_top10_overlap = float(
                os.environ["RWKV_PARITY_MIN_TOP10_OVERLAP"]
            )
        if os.environ.get("RWKV_PARITY_MAX_KL"):
            cfg.parity_max_kl = float(os.environ["RWKV_PARITY_MAX_KL"])
        if os.environ.get("RWKV_PARITY_MAX_STATE_RELATIVE_ERROR"):
            cfg.parity_max_state_relative_error = float(
                os.environ["RWKV_PARITY_MAX_STATE_RELATIVE_ERROR"]
            )
        if os.environ.get("RWKV_RESIDENCY_POLICY"):
            cfg.residency_policy = os.environ["RWKV_RESIDENCY_POLICY"].strip().lower()
        if "RWKV_ADAPTIVE_RESIDENCY" in os.environ:
            cfg.adaptive_residency = _as_bool(
                os.environ["RWKV_ADAPTIVE_RESIDENCY"], default=False
            )
        if os.environ.get("RWKV_ADAPTIVE_RESIDENCY_WINDOW"):
            cfg.adaptive_residency_window = int(
                os.environ["RWKV_ADAPTIVE_RESIDENCY_WINDOW"]
            )
        if os.environ.get("RWKV_ADAPTIVE_RESIDENCY_MIN_DWELL_TOKENS"):
            cfg.adaptive_residency_min_dwell_tokens = int(
                os.environ["RWKV_ADAPTIVE_RESIDENCY_MIN_DWELL_TOKENS"]
            )
        if os.environ.get("RWKV_ADAPTIVE_RESIDENCY_HYSTERESIS"):
            cfg.adaptive_residency_hysteresis = float(
                os.environ["RWKV_ADAPTIVE_RESIDENCY_HYSTERESIS"]
            )
        if os.environ.get("RWKV_ADAPTIVE_RESIDENCY_MAX_CHANGES"):
            cfg.adaptive_residency_max_changes = int(
                os.environ["RWKV_ADAPTIVE_RESIDENCY_MAX_CHANGES"]
            )
        if os.environ.get("RWKV_SESSION_PROMOTION"):
            cfg.session_promotion = _as_bool(
                os.environ["RWKV_SESSION_PROMOTION"], default=False
            )
        if os.environ.get("RWKV_SESSION_EXPECTED_TOKENS"):
            cfg.session_expected_tokens = max(
                0, int(os.environ["RWKV_SESSION_EXPECTED_TOKENS"])
            )
        if os.environ.get("RWKV_SESSION_PROMOTION_BYTES"):
            cfg.session_promotion_bytes = max(
                0, int(os.environ["RWKV_SESSION_PROMOTION_BYTES"])
            )
        if os.environ.get("RWKV_SESSION_PROMOTION_POLICY"):
            cfg.session_promotion_policy = os.environ[
                "RWKV_SESSION_PROMOTION_POLICY"
            ].strip().lower()
        if os.environ.get("RWKV_PREFIX_CACHE_MODE"):
            cfg.prefix_cache_mode = os.environ["RWKV_PREFIX_CACHE_MODE"].strip().lower()
        if os.environ.get("RWKV_PREFIX_CACHE_MAX"):
            cfg.prefix_cache_max_entries = int(os.environ["RWKV_PREFIX_CACHE_MAX"])
        if os.environ.get("RWKV_POWER"):
            cfg.power_percent = int(os.environ["RWKV_POWER"])
        if os.environ.get("RWKV_PREFETCH_IO_ONLY") is not None:
            raw_io = os.environ["RWKV_PREFETCH_IO_ONLY"].strip().lower()
            if raw_io in ("0", "false", "off", "no"):
                cfg.prefetch_io_only = False
            elif raw_io in ("1", "true", "on", "yes"):
                cfg.prefetch_io_only = True
        if os.environ.get("RWKV_CODEC_POLICY"):
            cfg.codec_policy = os.environ["RWKV_CODEC_POLICY"].strip().lower()
        return cfg


def load_meta(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
