#!/usr/bin/env python3
"""Quick tok/s sweep across the F1-F5 frontier for any available pack.

Usage:
  python scripts/bench_tok_s.py --pack test_model/trinity_eval/trinity_grouped_0.1b \\
      --checkpoint test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth

Skips packs that don't have the .pth for chatrwkv mode. Reports median of N
runs per scenario, plus a vs_resident column.

Frontier tiers (per docs/THROUGHPUT_PLAN.md):
  F1 min RAM          : ram_budget_gb=0.15  (RWKV_SSD_TIER=1)
  F2 bounded cache    : ram_budget_gb=0.21  (RWKV_BOUNDED_STREAM=1)
  F3 partial SSD tier : ram_budget_gb=0.5   (RWKV_PARTIAL_SSD_TIER=1)
  F4 mid              : stream_layer_cache + max_layers_in_z=2
  F5 warm-z promote   : stream_layer_cache + warm_z (RWKV_PROMOTE_FULL_Z=1)
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def run_one(
    pack: Path, ckpt: Path | None, mode: str, max_tokens: int, **kw
) -> float | None:
    import os

    if kw.get("warm_z"):
        os.environ["RWKV_PROMOTE_FULL_Z"] = "1"
    cfg_kwargs: dict = dict(
        pack_dir=pack,
        mode=mode,
        backend="chatrwkv" if ckpt else "synthetic",
        strategy="cpu bf16",
        device="cpu",
        max_tokens=max_tokens,
        greedy=True,
        skeleton_load=True,
    )
    if ckpt:
        cfg_kwargs["checkpoint_path"] = str(ckpt)
    cfg_kwargs.update(kw)
    cfg = EngineConfig(**cfg_kwargs)
    try:
        eng = InferenceEngine(cfg)
        eng.load()
    except Exception:
        return None
    try:
        # Warmup: cold first-touch is dominated by lazy decode + provider cache
        # build; without this the bench reports ~25-40% slower tok/s than steady
        # state. Discarded from timing.
        eng.generate("tok/s bench warmup")
        t0 = time.perf_counter()
        eng.generate("tok/s bench prompt")
        wall = time.perf_counter() - t0
        return max_tokens / wall
    finally:
        eng.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument("--samples", type=int, default=2)
    args = p.parse_args()

    if not args.pack.is_dir():
        raise SystemExit(f"pack not found: {args.pack}")
    if not (args.checkpoint and args.checkpoint.is_file()):
        print("# no checkpoint, running synthetic-only path")
        args.checkpoint = None

    scenarios: list[tuple[str, str, dict]] = [
        ("resident (FP16 in z, native path)", "resident", {}),
        (
            "streaming (raw, no cache)",
            "streaming",
            dict(stream_layer_cache=False),
        ),
        (
            "streaming (F4: max_z=2)",
            "streaming",
            dict(stream_layer_cache=True, max_layers_in_z=2),
        ),
        (
            "streaming (F5: warm-z promote)",
            "streaming",
            dict(stream_layer_cache=True, warm_z=True),
        ),
    ]
    note = (
        "\n  Note: F1/F2/F3 (ram_budget_gb tiers) are intentionally skipped on this\n"
        "        bench. They are calibrated for 7B+ models where the resident set is\n"
        "        a fraction of the full pack. On a 0.1B model, every layer is already\n"
        "        a large fraction of the RAM budget and the planner throttles streaming\n"
        "        by design, producing numbers that do NOT correspond to the F1-F3\n"
        "        frontier in docs/THROUGHPUT_PLAN.md. Use F4/F5 as the resident vs\n"
        "        streaming comparison on small models.\n"
        "  Note: For packs with shadow.bin (e.g. trinity_lut2_shadow_*), the\n"
        "        engine routes sensitive tensors to the FP16 shadow at runtime via\n"
        "        RWKV_CODEC_POLICY=auto|accuracy|hybrid. Default `auto` gives the\n"
        "        best tok/s on 0.1B. See docs/PRESETS.md for the routing table."
    )

    rows: list[dict] = []
    for label, mode, kw in scenarios:
        samples: list[float] = []
        for _ in range(args.samples):
            r = run_one(args.pack, args.checkpoint, mode, args.max_tokens, **kw)
            if r is None:
                break
            samples.append(r)
        if not samples:
            continue
        rows.append(
            {
                "label": label,
                "tok_s_median": round(statistics.median(samples), 2),
                "tok_s_min": round(min(samples), 2),
                "tok_s_max": round(max(samples), 2),
            }
        )

    if not rows:
        raise SystemExit("no scenarios ran successfully")
    resident = next(
        (r["tok_s_median"] for r in rows if r["label"].startswith("resident")), 0.0
    )
    print(
        f"\n{args.pack.name}  (median of {args.samples}, max_tokens={args.max_tokens})"
    )
    print(note)
    print(f"  {'mode':32}  {'tok/s':>8}  {'min':>8}  {'max':>8}  {'vs_resident':>12}")
    for r in rows:
        ratio = r["tok_s_median"] / resident if resident > 0 else 0
        ratio_str = f"{ratio:>11.0%}" if resident > 0 else "    n/a"
        print(
            f"  {r['label']:32}  {r['tok_s_median']:>8.2f}  {r['tok_s_min']:>8.2f}  {r['tok_s_max']:>8.2f}  {ratio_str}"
        )


if __name__ == "__main__":
    main()
