#!/usr/bin/env python3
"""Certify ChatRWKV/rwkv.cpp resident and provider paths on one prompt corpus.

The ChatRWKV resident engine is the reference.  The other three variants are
run against that same prompt/token corpus and are accepted only when greedy
token IDs agree exactly and the optional logit/state guardrails pass.  This is
an evidence-producing harness; it never silently substitutes a backend when
one of the requested variants cannot load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.config import EngineConfig  # noqa: E402
from rwkv_ssd.runtime.engine import InferenceEngine  # noqa: E402
from rwkv_ssd.runtime.parity import (  # noqa: E402
    BackendProbeResult,
    ParityThresholds,
    probe_loaded_engine,
    run_backend_conformance,
)


def _prompts(args: argparse.Namespace) -> list[str]:
    values = [str(value) for value in args.prompt]
    if args.prompts_file:
        values.extend(
            line.rstrip("\r\n")
            for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        raise SystemExit("provide at least one --prompt or --prompts-file line")
    return values


def _config(
    pack: Path,
    *,
    backend: str,
    mode: str,
    checkpoint: Path,
    ggml: Path,
    max_tokens: int,
) -> EngineConfig:
    return EngineConfig(
        pack_dir=pack,
        backend=backend,
        mode=mode,
        checkpoint_path=str(checkpoint if backend == "chatrwkv" else ggml),
        device="cpu",
        strategy="cpu bf16",
        max_tokens=max_tokens,
        greedy=True,
        temperature=0.0,
        skeleton_load=mode != "resident" if backend == "chatrwkv" else False,
    )


def _load_variant(
    label: str,
    config: EngineConfig,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, BackendProbeResult]:
    engine = InferenceEngine(config)
    try:
        engine.load()
        return {
            prompt: probe_loaded_engine(engine, prompt, max_tokens=max_tokens)
            for prompt in prompts
        }
    except Exception as exc:
        raise RuntimeError(f"{label} failed to load or probe: {exc}") from exc
    finally:
        engine.close()


def _trace_payload(traces: list[Any]) -> list[dict[str, Any]]:
    return [trace.to_dict() for trace in traces]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="runtime pack directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="ChatRWKV .pth checkpoint")
    parser.add_argument("--ggml", type=Path, required=True, help="rwkv.cpp .bin checkpoint")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--min-top10-overlap", type=float, default=0.80)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--max-state-relative-error", type=float, default=0.10)
    parser.add_argument(
        "--quality-certificate",
        type=Path,
        help="read parity thresholds from a quality_certificate.json gate set",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    for path, label in (
        (args.pack, "pack"),
        (args.checkpoint, "checkpoint"),
        (args.ggml, "ggml"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    prompts = _prompts(args)
    thresholds = ParityThresholds(
        min_top10_overlap=float(args.min_top10_overlap),
        max_kl=float(args.max_kl),
        max_state_relative_error=float(args.max_state_relative_error),
    )
    if args.quality_certificate:
        certificate = json.loads(args.quality_certificate.read_text(encoding="utf-8"))
        thresholds = ParityThresholds.from_quality_certificate(certificate)

    variants = {
        "chatrwkv_resident": _config(
            args.pack,
            backend="chatrwkv",
            mode="resident",
            checkpoint=args.checkpoint,
            ggml=args.ggml,
            max_tokens=args.max_tokens,
        ),
        "chatrwkv_pack_streaming": _config(
            args.pack,
            backend="chatrwkv",
            mode="streaming",
            checkpoint=args.checkpoint,
            ggml=args.ggml,
            max_tokens=args.max_tokens,
        ),
        "rwkvcpp_resident": _config(
            args.pack,
            backend="rwkvcpp",
            mode="resident",
            checkpoint=args.checkpoint,
            ggml=args.ggml,
            max_tokens=args.max_tokens,
        ),
        "rwkvcpp_provider_streaming": _config(
            args.pack,
            backend="rwkvcpp",
            mode="streaming",
            checkpoint=args.checkpoint,
            ggml=args.ggml,
            max_tokens=args.max_tokens,
        ),
    }
    results: dict[str, dict[str, BackendProbeResult]] = {}
    for label, config in variants.items():
        print(f"probing {label} ...", file=sys.stderr)
        results[label] = _load_variant(label, config, prompts, args.max_tokens)

    reference = results["chatrwkv_resident"]
    trace_sets: dict[str, list[Any]] = {}
    for label, candidates in results.items():
        if label == "chatrwkv_resident":
            continue
        trace_sets[label] = run_backend_conformance(
            lambda prompt, _count, values=reference: values[prompt],
            lambda prompt, _count, values=candidates: values[prompt],
            prompts,
            max_tokens=args.max_tokens,
            reference_backend="chatrwkv_resident",
            candidate_backend=label,
            thresholds=thresholds,
            # The certification CLI is the hard qualification gate.  A
            # token-only smoke runner is still supported by the library API,
            # but a real backend certificate must prove that both sides
            # publish per-token logits and recurrent-state observations.
            require_guardrails=True,
        )

    payload = {
        "pack": str(args.pack),
        "checkpoint": str(args.checkpoint),
        "ggml": str(args.ggml),
        "prompts": prompts,
        "max_tokens": args.max_tokens,
        "thresholds": thresholds.__dict__,
        "reference": "chatrwkv_resident",
        "traces": {label: _trace_payload(traces) for label, traces in trace_sets.items()},
        "passed": all(trace.passed for traces in trace_sets.values() for trace in traces),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
