"""Shared CLI / serve argument wiring for EngineConfig."""

from __future__ import annotations

import argparse
from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig


def add_engine_args(p: argparse.ArgumentParser, *, for_serve: bool = False) -> None:
    p.set_defaults(_engine_args_for_serve=for_serve)
    p.add_argument("--model", required=False, help="Runtime pack directory")
    p.add_argument("--config", help="Optional YAML config file")
    p.add_argument("--checkpoint", help="Original RWKV .pth for ChatRWKV")
    p.add_argument(
        "--mode",
        choices=["resident", "partial", "streaming"],
        default=None,
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=[
            "synthetic",
            "chatrwkv",
            "rwkvcpp",
            "albatross",
        ],
    )
    p.add_argument(
        "--device",
        default=None,
        help="cpu (default), cuda, or xpu (Intel Arc/iGPU via ChatRWKV)",
    )
    p.add_argument(
        "--strategy",
        default=None,
        help="ChatRWKV strategy; e.g. cpu bf16, cuda fp16, xpu bf16",
    )
    p.add_argument(
        "--trinity-decode-device",
        choices=["auto", "cpu", "cuda", "xpu"],
        default=None,
        help="device for Trinity LUT decode; auto follows the compute accelerator",
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--io-backend", choices=["mmap", "pread", "threaded"], default=None)
    p.add_argument("--io-chunk-bytes", type=int, default=None)
    p.add_argument(
        "--prefetch-policy",
        choices=["layer", "gate", "layer_aware"],
        default=None,
    )
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--stream-layer-cache", action="store_true")
    p.add_argument("--max-layers-in-z", type=int, default=None)
    p.add_argument("--warm-z", action="store_true")
    p.add_argument("--low-ram", action="store_true")
    p.add_argument("--ram-budget-gb", type=float, default=None)
    p.add_argument("--cache-budget-gb", type=float, default=None)
    p.add_argument(
        "--cache-budget-auto",
        action="store_true",
        help="auto-select provider cache budget from RAM tier or pack size",
    )
    p.add_argument("--power", type=int, default=None)
    p.add_argument("--decode-disk-cache", choices=["auto", "0", "1"], default=None)
    p.add_argument("--no-decouple-provider-cache", action="store_true")
    p.add_argument("--max-provider-cache-layers", type=int, default=None)
    p.add_argument(
        "--cache-format",
        choices=["auto", "none", "packed", "prepared", "dense"],
        default=None,
        help="packed/prepared residency axis; auto preserves legacy behavior",
    )
    p.add_argument("--packed-cache-bytes", type=int, default=None)
    p.add_argument("--prepared-cache-bytes", type=int, default=None)
    p.add_argument(
        "--residency-policy",
        choices=["static", "auto"],
        default=None,
        help="static profiles or measured cost-per-byte residency planning",
    )
    p.add_argument(
        "--adaptive-residency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="retier cache format between requests using measured costs",
    )
    p.add_argument("--adaptive-residency-window", type=int, default=None)
    p.add_argument("--adaptive-residency-min-dwell-tokens", type=int, default=None)
    p.add_argument("--adaptive-residency-hysteresis", type=float, default=None)
    p.add_argument("--adaptive-residency-max-changes", type=int, default=None)
    p.add_argument(
        "--session-promotion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="promote profitable dense layers between requests using expected session length",
    )
    p.add_argument("--session-expected-tokens", type=int, default=None)
    p.add_argument("--session-promotion-bytes", type=int, default=None)
    p.add_argument(
        "--session-promotion-policy",
        choices=["benefit_per_byte", "highest_stall", "lru"],
        default=None,
    )
    p.add_argument("--residency-profile", help="JSON partial residency profile")
    p.add_argument("--mmap-sequential", action="store_true")
    p.add_argument("--no-mmap-willneed", action="store_true")
    p.add_argument("--mmap-dontneed", action="store_true")
    p.add_argument("--system-prefix", help="Fixed system prefix for state cache")
    p.add_argument("--state-cache", action="store_true")
    p.add_argument(
        "--stateless-prefix-cache",
        action="store_true",
        help="enable disk-backed prefix reuse for stateless HTTP clients",
    )
    p.add_argument(
        "--prefix-cache-mode",
        choices=["system", "transcript"],
        default=None,
        help="system=cache system_prefix only; transcript=content-addressed chat history",
    )
    p.add_argument("--prefix-cache-max-entries", type=int, default=None)
    if not for_serve:
        p.add_argument("--metrics-csv", help="Write per-layer timings CSV")
        p.add_argument("--progress", action="store_true")
        p.add_argument("--verify-hash", action="store_true")
        p.add_argument("--no-skeleton-load", action="store_true")


def build_engine_config(args: argparse.Namespace) -> EngineConfig:
    if getattr(args, "config", None):
        cfg = EngineConfig.from_file(Path(args.config))
    else:
        if not args.model:
            raise SystemExit("--model or --config is required")
        cfg = EngineConfig(
            pack_dir=Path(args.model),
            mode=args.mode
            or ("streaming" if getattr(args, "_engine_args_for_serve", False) else "resident"),
            backend=args.backend or "rwkvcpp",
            device=args.device or "cpu",
            max_tokens=args.max_tokens if args.max_tokens is not None else 64,
            checkpoint_path=args.checkpoint,
            strategy=args.strategy or "cpu fp32",
            metrics_csv=Path(args.metrics_csv) if getattr(args, "metrics_csv", None) else None,
            greedy=True if args.greedy is None else bool(args.greedy),
            top_p=(
                float(getattr(args, "top_p"))
                if getattr(args, "top_p", None) is not None
                else 1.0
            ),
            seed=(
                int(getattr(args, "seed"))
                if getattr(args, "seed", None) is not None
                else None
            ),
        )
    cfg = EngineConfig.merge_env(cfg)
    if args.mode:
        cfg.mode = args.mode
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.pack_dir = Path(args.model)
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.device:
        cfg.device = args.device
    if args.strategy:
        cfg.strategy = args.strategy
    if getattr(args, "trinity_decode_device", None):
        cfg.trinity_decode_device = args.trinity_decode_device
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if getattr(args, "metrics_csv", None):
        cfg.metrics_csv = Path(args.metrics_csv)
    if args.temperature is not None:
        cfg.temperature = float(args.temperature)
        if args.temperature > 0 and args.greedy is None:
            cfg.greedy = False
    if getattr(args, "top_p", None) is not None:
        cfg.top_p = float(args.top_p)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if args.greedy is not None:
        cfg.greedy = bool(args.greedy)
    if args.system_prefix:
        cfg.system_prefix = args.system_prefix
        cfg.state_cache = True
    if getattr(args, "state_cache", False) or getattr(args, "stateless_prefix_cache", False):
        cfg.state_cache = True
    if getattr(args, "prefix_cache_mode", None):
        cfg.prefix_cache_mode = args.prefix_cache_mode
    elif getattr(args, "stateless_prefix_cache", False):
        cfg.prefix_cache_mode = "transcript"
    if getattr(args, "prefix_cache_max_entries", None) is not None:
        cfg.prefix_cache_max_entries = int(args.prefix_cache_max_entries)
    if getattr(args, "progress", False):
        cfg.progress = True
    if getattr(args, "verify_hash", False):
        cfg.verify_hash = True
    if getattr(args, "no_skeleton_load", False):
        cfg.skeleton_load = False
    if args.io_backend:
        cfg.io_backend = args.io_backend
    if args.io_chunk_bytes is not None:
        cfg.io_chunk_bytes = args.io_chunk_bytes
    if args.prefetch_policy:
        cfg.prefetch_policy = args.prefetch_policy
    if getattr(args, "no_prefetch", False):
        cfg.prefetch_enabled = False
    if getattr(args, "stream_layer_cache", False):
        cfg.stream_layer_cache = True
    if getattr(args, "warm_z", False):
        cfg.warm_z = True
    if args.max_layers_in_z is not None:
        cfg.max_layers_in_z = int(args.max_layers_in_z)
    if getattr(args, "low_ram", False):
        cfg.low_ram = True
    if args.ram_budget_gb is not None:
        cfg.ram_budget_gb = float(args.ram_budget_gb)
    if getattr(args, "cache_budget_auto", False):
        cfg.cache_budget_auto = True
    if args.cache_budget_gb is not None:
        cfg.cache_budget_gb = float(args.cache_budget_gb)
    if args.power is not None:
        cfg.power_percent = int(args.power)
    if getattr(args, "no_decouple_provider_cache", False):
        cfg.decouple_provider_cache = False
    if args.max_provider_cache_layers is not None:
        cfg.max_provider_cache_layers = int(args.max_provider_cache_layers)
    if getattr(args, "cache_format", None):
        cfg.cache_format = args.cache_format
    if getattr(args, "packed_cache_bytes", None) is not None:
        cfg.packed_cache_bytes = max(0, int(args.packed_cache_bytes))
    if getattr(args, "prepared_cache_bytes", None) is not None:
        cfg.prepared_cache_bytes = max(0, int(args.prepared_cache_bytes))
    if getattr(args, "residency_policy", None):
        cfg.residency_policy = args.residency_policy
    if getattr(args, "adaptive_residency", None) is not None:
        cfg.adaptive_residency = bool(args.adaptive_residency)
    if getattr(args, "adaptive_residency_window", None) is not None:
        cfg.adaptive_residency_window = int(args.adaptive_residency_window)
    if getattr(args, "adaptive_residency_min_dwell_tokens", None) is not None:
        cfg.adaptive_residency_min_dwell_tokens = int(
            args.adaptive_residency_min_dwell_tokens
        )
    if getattr(args, "adaptive_residency_hysteresis", None) is not None:
        cfg.adaptive_residency_hysteresis = float(args.adaptive_residency_hysteresis)
    if getattr(args, "adaptive_residency_max_changes", None) is not None:
        cfg.adaptive_residency_max_changes = int(args.adaptive_residency_max_changes)
    if getattr(args, "session_promotion", None) is not None:
        cfg.session_promotion = bool(args.session_promotion)
    if getattr(args, "session_expected_tokens", None) is not None:
        cfg.session_expected_tokens = max(0, int(args.session_expected_tokens))
    if getattr(args, "session_promotion_bytes", None) is not None:
        cfg.session_promotion_bytes = max(0, int(args.session_promotion_bytes))
    if getattr(args, "session_promotion_policy", None):
        cfg.session_promotion_policy = str(args.session_promotion_policy)
    if getattr(args, "decode_disk_cache", None) is not None:
        cfg.decode_disk_cache = args.decode_disk_cache
    if getattr(args, "mmap_sequential", False):
        cfg.mmap_sequential = True
    if getattr(args, "no_mmap_willneed", False):
        cfg.mmap_willneed = False
    if getattr(args, "mmap_dontneed", False):
        cfg.mmap_dontneed = True
    if getattr(args, "residency_profile", None):
        cfg.residency_profile = Path(args.residency_profile)
    return cfg
