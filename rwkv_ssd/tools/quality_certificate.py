"""Attach a load-enforced quality certificate to a runtime pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.quality_certificate import issue_quality_certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    path = issue_quality_certificate(
        args.pack, metrics, gates, evidence_scope=args.scope
    )
    print(json.dumps({"certificate": str(path), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
