#!/usr/bin/env python3
"""Tiny local capability regression harness.

This is intentionally not a benchmark leaderboard. It runs a small prompt suite
through the same local engine path users run, then applies simple deterministic
graders so engine changes can be compared over time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

DEFAULT_SUITE_PATH = ROOT / "eval" / "default_suite.json"


def _load_suite(path: Path | None) -> list[dict[str, Any]]:
    suite_path = path or DEFAULT_SUITE_PATH
    if not suite_path.is_file():
        return [
            {
                "id": "smoke",
                "prompt": "x",
                "regex": ".*",
            }
        ]
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("suite JSON must be a list of cases")
    return [dict(item) for item in raw]


def _grade(case: dict[str, Any], text: str) -> tuple[bool, str]:
    if "exact" in case:
        expected = str(case["exact"])
        return text.strip() == expected, f"exact={expected!r}"
    if "contains" in case:
        needle = str(case["contains"])
        return needle in text, f"contains={needle!r}"
    if "regex" in case:
        pattern = str(case["regex"])
        return re.search(pattern, text) is not None, f"regex={pattern!r}"
    return True, "ungraded"


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    pack = Path(args.model) if args.model else Path("_bench_pack") / "capability_eval"
    if not args.model:
        create_synthetic_pack(pack, n_layer=4, n_embd=32, quiet=True)
    suite = _load_suite(args.suite)
    cfg = EngineConfig(
        pack_dir=pack,
        backend=args.backend,
        mode=args.mode,
        device=args.device,
        max_tokens=args.max_tokens,
        checkpoint_path=args.checkpoint,
        strategy=args.strategy,
        cache_budget_gb=args.cache_budget_gb,
        power_percent=args.power,
    )
    rows: list[dict[str, Any]] = []
    for case in suite:
        with InferenceEngine(cfg) as engine:
            engine.metrics = MetricsCollector(power_percent=args.power)
            text = engine.generate(str(case.get("prompt", "")))
            passed, rule = _grade(case, text)
            metrics = engine.metrics.to_dict()
        rows.append(
            {
                "id": str(case.get("id", len(rows) + 1)),
                "passed": passed,
                "rule": rule,
                "output": text,
                "tok_s": metrics.get("tok_s", 0.0),
            }
        )
    passed = sum(1 for row in rows if row["passed"])
    return {
        "pack": str(pack),
        "backend": args.backend,
        "mode": args.mode,
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Small local capability regression suite")
    p.add_argument("--model", help="Runtime pack directory")
    p.add_argument("--checkpoint", help="RWKV .pth for chatrwkv")
    p.add_argument("--backend", default="synthetic", choices=["synthetic", "chatrwkv"])
    p.add_argument(
        "--mode", default="streaming", choices=["resident", "partial", "streaming"]
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--strategy", default="cpu fp32")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--suite", type=Path, help="JSON list of {id,prompt,contains|exact|regex}")
    p.add_argument("--cache-budget-gb", type=float, default=None)
    p.add_argument("--power", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--trace-out", type=Path, help="Write JSON trace for regrade")
    args = p.parse_args()

    out = run_suite(args)
    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(
        f"capability eval  pack={out['pack']} backend={args.backend} "
        f"passed={out['passed']}/{out['total']}"
    )
    for row in out["rows"]:
        mark = "PASS" if row["passed"] else "FAIL"
        print(f"{mark} {row['id']} ({row['rule']})")


if __name__ == "__main__":
    main()
