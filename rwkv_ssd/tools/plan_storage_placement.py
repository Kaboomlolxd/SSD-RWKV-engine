"""Plan runtime-pack layer placement for measured heterogeneous drive tiers.

The drive JSON is a list of objects with ``name``, ``bandwidth_mbps``, optional
``latency_ms``, and optional ``capacity_bytes``. The output is a plan only; it
does not move user files. Feed the mapping into a later repack operation after
reviewing the A/B cost estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.storage_placement import (
    DriveTier,
    LayerPayload,
    place_layers,
    total_expected_stall_ms,
)


def resolve_drive_specs(drive_specs: list[dict], *, base_dir: Path | None = None) -> list[dict]:
    """Resolve explicit tiers or storage-diagnostic JSON references."""
    resolved = []
    for item in drive_specs:
        row = dict(item)
        diagnostic_ref = row.pop("diagnostic_json", None)
        if diagnostic_ref is not None:
            path = Path(diagnostic_ref)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            diagnostic = json.loads(path.read_text(encoding="utf-8"))
            sequential = diagnostic.get("sequential_tensor_read", {})
            row.setdefault("bandwidth_mbps", float(sequential.get("median_mbps", 0)))
            layer_ms = [
                float(value.get("read_ms", 0))
                for value in diagnostic.get("per_layer", [])
                if float(value.get("read_ms", 0)) > 0
            ]
            row.setdefault("latency_ms", min(layer_ms) if layer_ms else 0.0)
            row["diagnostic_source"] = str(path)
        resolved.append(row)
    return resolved


def plan_pack_placement(pack: Path, drive_specs: list[dict]) -> dict:
    manifest = Manifest.load(pack)
    drives = [
        DriveTier(
            name=str(item["name"]),
            bandwidth_mbps=float(item["bandwidth_mbps"]),
            latency_ms=float(item.get("latency_ms", 0.0)),
            capacity_bytes=int(item.get("capacity_bytes", 0)),
        )
        for item in drive_specs
    ]
    layers = [
        LayerPayload(
            layer_id=layer_id,
            byte_count=sum(entry.length for entry in entries),
        )
        for layer_id, entries in sorted(manifest.by_layer().items())
    ]
    optimized = place_layers(layers, drives, policy="optimized")
    baseline = place_layers(layers, drives, policy="round_robin")
    optimized_ms = total_expected_stall_ms(optimized)
    baseline_ms = total_expected_stall_ms(baseline)
    return {
        "schema_version": 1,
        "pack": str(pack),
        "drive_count": len(drives),
        "baseline_policy": "round_robin",
        "optimized_policy": "measured_stall_cost",
        "round_robin_stall_ms": baseline_ms,
        "optimized_stall_ms": optimized_ms,
        "modeled_speedup": baseline_ms / optimized_ms if optimized_ms > 0 else 0.0,
        "physical_hardware_validated": False,
        "placements": [item.__dict__ for item in optimized],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--drives-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.drives_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("drive JSON must be a list")
    result = plan_pack_placement(
        args.pack, resolve_drive_specs(raw, base_dir=args.drives_json.parent)
    )
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
