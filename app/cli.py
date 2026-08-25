#!/usr/bin/env python3
"""CLI for RWKV SSD inference engine."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _find_chatrwkv_root() -> Path | None:
    env = os.environ.get("CHATRWKV_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[1]
    if os.environ.get("RWKV_SSD_SKIP_BUNDLED_CHATRWKV") != "1":
        candidates.append(here / "test_model" / "ChatRWKV")
    candidates.extend(
        [
            here.parent / "ChatRWKV",
            here / "backends" / "chatrwkv_ref",
        ]
    )
    for candidate in candidates:
        pip = candidate / "rwkv_pip_package" / "src" / "rwkv" / "model.py"
        legacy = candidate / "src" / "model_run.py"
        if pip.is_file() or legacy.is_file():
            return candidate
    return None


logger = logging.getLogger(__name__)

MODE_HELP = """
Runtime modes:
  resident   - all weights cached in RAM; fastest, highest memory (reference path)
  partial    - embed/head + first/last layer resident; middle layers streamed
  streaming  - per-token layer reads from weights.bin (synthetic or chatrwkv RWKV-7)

Streaming options:
  --stream-layer-cache   retain a small hot set of block layers in model.z (LRU cap)
  --max-layers-in-z N    max streamed block layers in z (default 1; skeleton layers pinned)
  --warm-z               opt-in: preload all block layers into z (throughput experiments)
  --low-ram              bounded z + decoupled provider cache + Trinity disk decode cache
  --ram-budget-gb N      partial + pinned early layers; cap RAM near N GB (e.g. 10 for 200B)
"""


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def build_config(args: argparse.Namespace) -> EngineConfig:
    from rwkv_ssd.runtime.config import EngineConfig

    if args.config:
        cfg = EngineConfig.from_file(Path(args.config))
    else:
        if not args.model:
            raise SystemExit("--model or --config is required")
        cfg = EngineConfig(
            pack_dir=Path(args.model),
            mode=args.mode or "resident",
            backend=args.backend or "rwkvcpp",
            device=args.device or "cpu",
            max_tokens=args.max_tokens if args.max_tokens is not None else 64,
            checkpoint_path=args.checkpoint,
            strategy=args.strategy or "cpu fp32",
            metrics_csv=Path(args.metrics_csv) if args.metrics_csv else None,
            greedy=True if args.greedy is None else args.greedy,
            temperature=(float(args.temperature) if args.temperature is not None else 1.0),
            top_p=(float(args.top_p) if args.top_p is not None else 1.0),
            seed=(int(args.seed) if args.seed is not None else None),
        )

    cfg = EngineConfig.merge_env(cfg)

    # Apply CLI overrides on top of env-merged config.
    # Note: tier env vars (RWKV_SSD_TIER, RWKV_PARTIAL_FUSED, RWKV_PARTIAL_SSD_TIER)
    # are applied later by engine.load() - do not re-parse them here.
    if args.mode:
        cfg.mode = args.mode
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.pack_dir = Path(args.model)
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.temperature is not None:
        cfg.temperature = float(args.temperature)
        if args.greedy is None and args.temperature > 0:
            cfg.greedy = False
    if args.top_p is not None:
        cfg.top_p = float(args.top_p)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.device:
        cfg.device = args.device
    if args.strategy:
        cfg.strategy = args.strategy
    if getattr(args, "trinity_decode_device", None):
        cfg.trinity_decode_device = args.trinity_decode_device
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.greedy is not None:
        cfg.greedy = bool(args.greedy)
    if args.metrics_csv:
        cfg.metrics_csv = Path(args.metrics_csv)
    if args.system_prefix:
        cfg.system_prefix = args.system_prefix
        cfg.state_cache = True
    if args.state_cache is not None:
        cfg.state_cache = args.state_cache
    if args.progress:
        cfg.progress = True
    if args.verify_hash:
        cfg.verify_hash = True
    if getattr(args, "no_skeleton_load", False):
        cfg.skeleton_load = False
    if getattr(args, "io_backend", None):
        cfg.io_backend = args.io_backend
    if getattr(args, "io_chunk_bytes", None):
        cfg.io_chunk_bytes = args.io_chunk_bytes
    if getattr(args, "prefetch_policy", None):
        cfg.prefetch_policy = args.prefetch_policy
    if getattr(args, "no_prefetch", False):
        cfg.prefetch_enabled = False
    if getattr(args, "stream_layer_cache", False):
        cfg.stream_layer_cache = True
    if getattr(args, "warm_z", False):
        cfg.warm_z = True
    if getattr(args, "max_layers_in_z", None) is not None:
        cfg.max_layers_in_z = int(args.max_layers_in_z)
    if getattr(args, "low_ram", False):
        cfg.low_ram = True
    if getattr(args, "ram_budget_gb", None) is not None:
        cfg.ram_budget_gb = float(args.ram_budget_gb)
    if getattr(args, "cache_budget_gb", None) is not None:
        cfg.cache_budget_gb = float(args.cache_budget_gb)
    if getattr(args, "cache_budget_auto", False):
        cfg.cache_budget_auto = True
    if getattr(args, "prefix_cache_mode", None):
        cfg.prefix_cache_mode = args.prefix_cache_mode
    if getattr(args, "prefix_cache_max_entries", None) is not None:
        cfg.prefix_cache_max_entries = int(args.prefix_cache_max_entries)
    if getattr(args, "power", None) is not None:
        cfg.power_percent = int(args.power)
    if getattr(args, "no_decouple_provider_cache", False):
        cfg.decouple_provider_cache = False
    if getattr(args, "max_provider_cache_layers", None) is not None:
        cfg.max_provider_cache_layers = int(args.max_provider_cache_layers)
    if getattr(args, "decode_disk_cache", None) is not None:
        cfg.decode_disk_cache = args.decode_disk_cache
    if getattr(args, "mmap_sequential", False):
        cfg.mmap_sequential = True
    if getattr(args, "no_mmap_willneed", False):
        cfg.mmap_willneed = False
    if getattr(args, "mmap_dontneed", False):
        cfg.mmap_dontneed = True
    if getattr(args, "io_hedged", False):
        cfg.io_hedged = True
    if getattr(args, "io_chunk_policy", None):
        cfg.io_chunk_policy = args.io_chunk_policy
    if getattr(args, "ngram_weight_cache", False):
        cfg.ngram_weight_cache = True
    if getattr(args, "mtp_speculative", False):
        cfg.mtp_speculative = True
    if args.residency_profile:
        cfg.residency_profile = Path(args.residency_profile)
    return cfg


def _run_interactive(engine: "InferenceEngine", prompt: str = "") -> None:
    """Small snapshot-oriented REPL for local daily-driver smoke use."""
    transcript = prompt.strip()
    if transcript:
        print(engine.generate(transcript), flush=True)
    print("Commands: /save PATH, /load PATH, /switch PATH, /strip PATH, /list [DIR], /quit")
    has_state = False
    while True:
        try:
            line = input("rwkv-ssd> ")
        except EOFError:
            print()
            return
        text = line.strip()
        if not text:
            continue
        cmd, _, rest = text.partition(" ")
        if cmd in {"/quit", "/exit"}:
            return
        if cmd == "/save":
            if not rest:
                print("usage: /save PATH")
                continue
            engine.save_snapshot(Path(rest), prompt=transcript)
            print(f"saved snapshot to {rest}")
            continue
        if cmd in {"/load", "/switch"}:
            if not rest:
                print(f"usage: {cmd} PATH")
                continue
            path = Path(rest)
            if path.suffix == ".stripped.json":
                import json as _json

                data = _json.loads(path.read_text(encoding="utf-8"))
                transcript = str(data.get("prompt", ""))
                print(f"loaded stripped transcript from {path}")
            else:
                engine.load_snapshot(path)
                has_state = True
                print(f"loaded snapshot from {path}")
            continue
        if cmd == "/strip":
            if not rest:
                print("usage: /strip PATH")
                continue
            import json as _json

            path = Path(rest)
            if path.suffix != ".stripped.json":
                path = path.with_suffix(path.suffix + ".stripped.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                _json.dumps({"prompt": transcript}, indent=2), encoding="utf-8"
            )
            print(f"wrote stripped transcript to {path}")
            continue
        if cmd == "/list":
            root = Path(rest) if rest else Path.cwd()
            rows = sorted(
                list(root.glob("*.rws")) + list(root.glob("*.stripped.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for row in rows[:50]:
                print(row)
            continue

        if has_state and engine.backend.get_recurrent_state() is not None:
            out = engine.generate_followup(f"\nUser: {text}\nAssistant:")
        else:
            transcript = f"{transcript}\nUser: {text}\nAssistant:" if transcript else text
            out = engine.generate(transcript)
            has_state = True
        print(out, flush=True)
        transcript = (
            f"{transcript}\nUser: {text}\nAssistant:{out}"
            if transcript
            else f"User: {text}\nAssistant:{out}"
        )
        has_state = True


def validate_config(cfg: EngineConfig) -> None:
    from rwkv_ssd.backends.factory import ensure_v0_backend, supports_streaming_mode

    ensure_v0_backend(cfg.backend)
    if cfg.mode != "resident" and not supports_streaming_mode(cfg.backend):
        raise SystemExit(
            "ERROR: --mode partial|streaming requires a backend with true streaming "
            "support (rwkvcpp, synthetic, chatrwkv, or albatross)."
        )
    if cfg.backend == "chatrwkv" and _find_chatrwkv_root() is None:
        raise SystemExit(
            "ERROR: ChatRWKV not found. Clone https://github.com/BlinkDL/ChatRWKV\n"
            "       into test_model/ChatRWKV, set CHATRWKV_ROOT, "
            "or use --backend synthetic."
        )


def main() -> None:
    from rwkv_ssd.runtime.errors import BackendNotAvailableError

    _configure_stdout()

    p = argparse.ArgumentParser(
        description="RWKV SSD inference engine (V0)",
        epilog=MODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", help="Optional YAML config file")
    p.add_argument("--model", help="Runtime pack directory (manifest + weights.bin)")
    p.add_argument("--checkpoint", help="Original RWKV .pth for ChatRWKV")
    p.add_argument("--prompt", default="Hello", help="Input prompt")
    p.add_argument(
        "--mode",
        choices=["resident", "partial", "streaming"],
        default=None,
        help="weight residency (see epilog)",
    )
    p.add_argument(
        "--backend",
        choices=[
            "synthetic",
            "chatrwkv",
            "rwkvcpp",
            "albatross",
        ],
        default=None,
        help="rwkvcpp (default CPU RWKV backend); use chatrwkv for compatibility/reference",
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device",
        default=None,
        help="cpu (default), cuda, or xpu (Intel Arc/iGPU via the ChatRWKV path)",
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
    p.add_argument("--metrics-csv", help="Write per-layer timings CSV")
    p.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--system-prefix", help="System prompt prefix for state cache")
    p.add_argument(
        "--state-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--progress", action="store_true", help="Log load/generate progress")
    p.add_argument(
        "--verify-hash",
        action="store_true",
        help="SHA-256 verify full weights.bin on load (slow for large packs)",
    )
    p.add_argument(
        "--no-skeleton-load",
        action="store_true",
        help="Load full .pth into RAM (disable pack-only skeleton for streaming)",
    )
    p.add_argument(
        "--io-backend",
        choices=["mmap", "pread", "threaded"],
        default=None,
        help="weights.bin read path (mmap default; pread for profiling)",
    )
    p.add_argument(
        "--io-chunk-policy",
        choices=["uniform", "layer_size"],
        default=None,
        help="heterogeneous chunk schedule (P2.c); use with --io-chunk-bytes",
    )
    p.add_argument(
        "--io-hedged",
        action="store_true",
        help="hedged dual-read race for tail latency (P2.d)",
    )
    p.add_argument(
        "--ngram-weight-cache",
        action="store_true",
        help="reuse identical tensor blobs across prefill (P2.e)",
    )
    p.add_argument(
        "--mtp-speculative",
        action="store_true",
        help="evaluate MTP workload gate (speculative decode not wired; logs gate)",
    )
    p.add_argument(
        "--prefetch-policy",
        choices=["layer", "gate", "layer_aware"],
        default=None,
        help="prefetch planner: layer (N+1), gate (adaptive N+2), layer_aware (N+1..3)",
    )
    p.add_argument(
        "--no-prefetch",
        action="store_true",
        help="disable layer prefetch",
    )
    p.add_argument(
        "--stream-layer-cache",
        action="store_true",
        help="LRU-retain streamed block layers in model.z (see --max-layers-in-z)",
    )
    p.add_argument(
        "--max-layers-in-z",
        type=int,
        default=None,
        help="cap streamed block layers in z when using --stream-layer-cache (default 1)",
    )
    p.add_argument(
        "--warm-z",
        action="store_true",
        help="preload all block layers into z at load (opt-in; not low-RAM streaming)",
    )
    p.add_argument(
        "--low-ram",
        action="store_true",
        help="low-RAM preset; large packs default to --ram-budget-gb 10",
    )
    p.add_argument(
        "--ram-budget-gb",
        type=float,
        default=None,
        metavar="GB",
        help="partial mode: pin early layers, stream rest; cap decoded RAM near N GB",
    )
    p.add_argument(
        "--cache-budget-gb",
        type=float,
        default=None,
        metavar="GB",
        help="cap decoded provider cache near N GB (env: RWKV_CACHE_BUDGET_GB)",
    )
    p.add_argument(
        "--cache-budget-auto",
        action="store_true",
        help="auto-select provider cache budget from RAM tier or pack size",
    )
    p.add_argument(
        "--prefix-cache-mode",
        choices=["system", "transcript"],
        default=None,
        help="system=cache system_prefix only; transcript=hash chat history prefixes",
    )
    p.add_argument("--prefix-cache-max-entries", type=int, default=None)
    p.add_argument(
        "--power",
        type=int,
        default=None,
        metavar="PCT",
        help="cooperative power target 1..100; lower inserts sleeps between work units",
    )
    p.add_argument(
        "--no-decouple-provider-cache",
        action="store_true",
        help="evict decoded tensors when z evicts (legacy coupled cache)",
    )
    p.add_argument(
        "--max-provider-cache-layers",
        type=int,
        default=None,
        help="provider LRU cap (0=auto; decoupled default keeps all block layers)",
    )
    p.add_argument(
        "--decode-disk-cache",
        choices=["auto", "0", "1"],
        default=None,
        help="persistent .decode_cache/ for Trinity (auto on streaming Trinity packs)",
    )
    p.add_argument(
        "--mmap-sequential",
        action="store_true",
        help="Linux: madvise(MADV_SEQUENTIAL) on mmap (no-op on Windows)",
    )
    p.add_argument(
        "--no-mmap-willneed",
        action="store_true",
        help="disable madvise(MADV_WILLNEED) on upcoming layer ranges (Linux)",
    )
    p.add_argument(
        "--mmap-dontneed",
        action="store_true",
        help="Linux: madvise(MADV_DONTNEED) after consumed layers (strict streaming RAM)",
    )
    p.add_argument(
        "--residency-profile",
        help="JSON partial residency profile (see deploy/partial_profile.example.json)",
    )
    p.add_argument(
        "--json-metrics",
        action="store_true",
        help="Emit structured metrics JSON (MetricsCollector.to_dict()) instead of the summary line",
    )
    p.add_argument(
        "--save-snapshot",
        type=Path,
        help="Save engine recurrent state to this file after generation (P1 #13)",
    )
    p.add_argument(
        "--load-snapshot",
        type=Path,
        help="Restore engine recurrent state from this file before generation (P1 #13)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Start a small REPL with /save, /load, /switch, /strip, and /list",
    )
    args = p.parse_args()

    if args.backend == "chatrwkv" and _find_chatrwkv_root() is None:
        print(
            "ERROR: ChatRWKV not found. Clone https://github.com/BlinkDL/ChatRWKV\n"
            "       into test_model/ChatRWKV, set CHATRWKV_ROOT, "
            "or use --backend synthetic.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    _setup_logging(args.log_level)

    print("rwkv-ssd: starting...", file=sys.stderr, flush=True)

    try:
        cfg = build_config(args)
        validate_config(cfg)
    except BackendNotAvailableError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    from rwkv_ssd.runtime.engine import InferenceEngine

    with InferenceEngine(cfg) as engine:
        if getattr(args, "load_snapshot", None):
            engine.load_snapshot(args.load_snapshot)
            print(
                f"restored snapshot from {args.load_snapshot}",
                file=sys.stderr,
                flush=True,
            )
        if getattr(args, "interactive", False):
            _run_interactive(engine, args.prompt)
            out = ""
        else:
            print("Generating...", file=sys.stderr, flush=True)
            out = engine.generate(args.prompt)
            print(out, flush=True)
        if getattr(args, "json_metrics", False) and engine.metrics.layers:
            import json as _json

            print(_json.dumps(engine.metrics.to_dict(), indent=2))
        elif engine.metrics.layers:
            print("\n--- metrics ---")
            print(engine.metrics.summary())
        if getattr(args, "save_snapshot", None):
            engine.save_snapshot(args.save_snapshot, prompt=args.prompt)
            print(
                f"saved snapshot to {args.save_snapshot}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    from rwkv_ssd.runtime.engine import InferenceEngine
    from rwkv_ssd.runtime.errors import EngineError

    try:
        main()
    except EngineError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
