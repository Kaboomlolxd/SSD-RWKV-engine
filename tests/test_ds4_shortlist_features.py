"""Smoke tests for the ds4-inspired product slices."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from bench.bench_context_frontier import run_frontiers
from bench.capability_eval import run_suite


def test_context_frontier_bench_smoke(synthetic_pack: Path) -> None:
    out = run_frontiers(
        Namespace(
            model=str(synthetic_pack),
            checkpoint=None,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            strategy="cpu fp32",
            ctx_start=8,
            ctx_max=16,
            step_incr=8,
            gen_tokens=2,
            cache_budget_gb=None,
            power=100,
        )
    )
    assert [row["frontier_tokens"] for row in out["rows"]] == [8, 16]
    assert all(row["gen_tokens"] == 2 for row in out["rows"])


def test_token_frontier_bench_smoke(synthetic_pack: Path) -> None:
    out = run_frontiers(
        Namespace(
            model=str(synthetic_pack),
            checkpoint=None,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            strategy="cpu fp32",
            ctx_start=8,
            ctx_max=16,
            step_incr=8,
            gen_tokens=2,
            cache_budget_gb=None,
            power=100,
            token_frontier=True,
        )
    )
    assert out["token_frontier"] is True
    assert all(row["prompt_tokens"] >= row["frontier_tokens"] for row in out["rows"])


def test_capability_eval_harness_smoke(synthetic_pack: Path, tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        '[{"id":"smoke","prompt":"x","regex":".*"}]',
        encoding="utf-8",
    )
    out = run_suite(
        Namespace(
            model=str(synthetic_pack),
            checkpoint=None,
            backend="synthetic",
            mode="streaming",
            device="cpu",
            strategy="cpu fp32",
            max_tokens=2,
            suite=suite,
            cache_budget_gb=None,
            power=100,
        )
    )
    assert out["total"] == 1
    assert out["passed"] == 1
