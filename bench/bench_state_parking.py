"""Measure exact recurrent-state RAM/SSD parking and restore latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rwkv_ssd.runtime.state_cache import RecurrentState
from rwkv_ssd.runtime.state_parking import HierarchicalStateStore, state_nbytes


def run_state_parking_bench(root: Path, *, states: int = 8, elements: int = 65536) -> dict:
    state_list = [
        RecurrentState(
            last_token_id=index,
            h=torch.full((elements,), float(index), dtype=torch.float32),
        )
        for index in range(states)
    ]
    one_bytes = state_nbytes(state_list[0])
    store = HierarchicalStateStore(root, max_ram_bytes=one_bytes * 2)
    write_ms = []
    for index, state in enumerate(state_list):
        started = time.perf_counter()
        store.put(f"session-{index}", state, model_family="synthetic")
        write_ms.append((time.perf_counter() - started) * 1000.0)
    restore_ms = []
    for index, expected in enumerate(state_list):
        started = time.perf_counter()
        restored = store.get(f"session-{index}")
        restore_ms.append((time.perf_counter() - started) * 1000.0)
        assert restored is not None and restored.h is not None
        assert torch.equal(restored.h, expected.h)
    return {
        "schema_version": 1,
        "states": states,
        "state_payload_bytes": one_bytes,
        "ram_cap_bytes": store.max_ram_bytes,
        "peak_observed_ram_bytes": store.ram_bytes,
        "disk_bytes": store.disk_bytes(),
        "median_write_ms": statistics.median(write_ms),
        "median_restore_ms": statistics.median(restore_ms),
        "ram_evictions": store.stats.ram_evictions,
        "disk_hits": store.stats.disk_hits,
        "exact_parity": True,
        "scope": "synthetic CPU filesystem",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = run_state_parking_bench(args.root, states=args.states, elements=args.elements)
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
