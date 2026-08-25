#!/usr/bin/env python3
"""Measure decode behavior at increasing prompt/context frontiers."""

from __future__ import annotations

from collections.abc import Callable
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack


def _frontiers(start: int, max_ctx: int, incr: int) -> list[int]:
    rows: list[int] = []
    cur = start
    while cur <= max_ctx:
        rows.append(cur)
        cur += incr
    return rows


def _prompt_for_tokens(frontier: int) -> str:
    return "x" * max(1, frontier)


def _count_prompt_tokens(engine: InferenceEngine, prompt: str) -> int:
    backend = engine.backend
    if hasattr(backend, "_encode"):
        return len(backend._encode(prompt))  # type: ignore[attr-defined]
    pipeline = getattr(backend, "_pipeline", None)
    if pipeline is not None and hasattr(pipeline, "encode"):
        return len(pipeline.encode(prompt))
    return len(prompt.encode("utf-8"))


def _prompt_reaching_tokens(
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> tuple[str, int]:
    """Grow a repeating unit until ``count_tokens(prompt) >= target_tokens``."""
    unit = "The quick brown fox jumps over the lazy dog. "
    parts: list[str] = []
    tokens = 0
    while tokens < max(1, target_tokens):
        parts.append(unit)
        prompt = "".join(parts)
        tokens = count_tokens(prompt)
    return prompt, tokens


def run_frontiers(args: argparse.Namespace) -> dict:
    pack = Path(args.model) if args.model else Path("_bench_pack") / "context_frontier"
    if not args.model:
        create_synthetic_pack(pack, n_layer=4, n_embd=32, quiet=True)
    cfg = EngineConfig(
        pack_dir=pack,
        backend=args.backend,
        mode=args.mode,
        device=args.device,
        max_tokens=args.gen_tokens,
        checkpoint_path=args.checkpoint,
        strategy=args.strategy,
        cache_budget_gb=args.cache_budget_gb,
        power_percent=args.power,
    )
    rows: list[dict] = []
    for frontier in _frontiers(args.ctx_start, args.ctx_max, args.step_incr):
        with InferenceEngine(cfg) as engine:
            engine.metrics = MetricsCollector(power_percent=args.power)
            if getattr(args, "token_frontier", False):
                prompt, prompt_tokens = _prompt_reaching_tokens(
                    frontier,
                    lambda text: _count_prompt_tokens(engine, text),
                )
            else:
                prompt = _prompt_for_tokens(frontier)
                prompt_tokens = _count_prompt_tokens(engine, prompt)
            t0 = time.perf_counter()
            engine.generate(prompt)
            wall_s = time.perf_counter() - t0
            m = engine.metrics
        read_ms = sum(x.read_ms for x in m.layers)
        staging_ms = sum(x.staging_ms for x in m.layers)
        compute_ms = sum(x.compute_ms for x in m.layers)
        rows.append(
            {
                "frontier_tokens": frontier,
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "gen_tokens": args.gen_tokens,
                "wall_s": round(wall_s, 4),
                "prefill_wall_s": round(m.prefill_wall_s, 4),
                "generation_tok_s": round(
                    args.gen_tokens / wall_s if wall_s > 0 else 0.0, 3
                ),
                "read_ms": round(read_ms, 3),
                "staging_ms": round(staging_ms, 3),
                "compute_ms": round(compute_ms, 3),
                "provider_cache_mb": round(m.provider_cache_bytes / 1e6, 3),
                "z_mb": round(m.z_bytes / 1e6, 3),
                "state_cache_hit": m.state_cache_hit,
            }
        )
    return {
        "pack": str(pack),
        "backend": args.backend,
        "mode": args.mode,
        "ctx_start": args.ctx_start,
        "ctx_max": args.ctx_max,
        "step_incr": args.step_incr,
        "token_frontier": bool(getattr(args, "token_frontier", False)),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Context-frontier benchmark")
    p.add_argument("--model", help="Runtime pack directory")
    p.add_argument("--checkpoint", help="RWKV .pth for chatrwkv")
    p.add_argument("--backend", default="synthetic", choices=["synthetic", "chatrwkv"])
    p.add_argument(
        "--mode", default="streaming", choices=["resident", "partial", "streaming"]
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--strategy", default="cpu fp32")
    p.add_argument("--ctx-start", type=int, default=512)
    p.add_argument("--ctx-max", type=int, default=4096)
    p.add_argument("--step-incr", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument(
        "--token-frontier",
        action="store_true",
        help="grow prompts until tokenizer reaches each frontier token count",
    )
    p.add_argument("--cache-budget-gb", type=float, default=None)
    p.add_argument("--power", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    out = run_frontiers(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(
        f"context frontier  pack={out['pack']} backend={args.backend} mode={args.mode}"
    )
    print(f"{'ctx':>8} {'ptok':>8} {'tok/s':>8} {'wall':>8} {'read_ms':>10} {'compute_ms':>10}")
    for row in out["rows"]:
        ptok = row.get("prompt_tokens", row["frontier_tokens"])
        print(
            f"{row['frontier_tokens']:>8} {ptok:>8} {row['generation_tok_s']:>8.2f} "
            f"{row['wall_s']:>8.3f} {row['read_ms']:>10.1f} {row['compute_ms']:>10.1f}"
        )


if __name__ == "__main__":
    main()
