"""Deterministic A/B probes for hardware-independent opportunity planners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.promotion_planner import PromotionCandidate, plan_promotions
from rwkv_ssd.runtime.storage_placement import (
    DriveTier,
    LayerPayload,
    place_layers,
    total_expected_stall_ms,
)


def run_planner_ab() -> dict:
    candidates = [
        PromotionCandidate(0, 80, 12.0, 1.8, 8),
        PromotionCandidate(1, 20, 8.0, 1.8, 2),
        PromotionCandidate(2, 60, 18.0, 2.2, 5),
        PromotionCandidate(3, 10, 4.0, 0.9, 1),
    ]
    workloads = {}
    for label, remaining in (("short", 4), ("medium", 24), ("long", 128)):
        policies = {
            policy: plan_promotions(
                candidates,
                expected_remaining_tokens=remaining,
                ram_cap_bytes=70,
                policy=policy,
            )
            for policy in ("lru", "highest_stall", "benefit_per_byte")
        }
        workloads[label] = {
            name: {
                "layers": [item.layer_id for item in plan.selected],
                "resident_bytes": plan.resident_bytes,
                "net_benefit_ms": plan.net_benefit_ms,
                "baseline_staging_ms": sum(
                    item.future_savings_ms(remaining) for item in candidates
                ),
                "projected_staging_plus_promotion_ms": (
                    sum(item.future_savings_ms(remaining) for item in candidates)
                    - plan.future_savings_ms
                    + plan.promotion_cost_ms
                ),
            }
            for name, plan in policies.items()
        }

    layers = [
        LayerPayload(0, 100_000_000, 1),
        LayerPayload(1, 800_000_000, 4),
        LayerPayload(2, 200_000_000, 1),
        LayerPayload(3, 700_000_000, 2),
    ]
    drives = [
        DriveTier("fast", 6000, 0.08, 1_000_000_000),
        DriveTier("slow", 700, 0.25, 1_000_000_000),
    ]
    rr = place_layers(layers, drives, policy="round_robin")
    optimized = place_layers(layers, drives, policy="optimized")
    rr_ms = total_expected_stall_ms(rr)
    optimized_ms = total_expected_stall_ms(optimized)
    return {
        "schema_version": 1,
        "promotion": workloads,
        "placement": {
            "round_robin_stall_ms": rr_ms,
            "optimized_stall_ms": optimized_ms,
            "modeled_speedup": rr_ms / optimized_ms,
            "physical_hardware_validated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = run_planner_ab()
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
