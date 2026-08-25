#!/usr/bin/env python3
"""Verify manifest offsets and tensor byte lengths against weights.bin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rwkv_ssd.runtime.pack_verify import verify_pack


def verify(pack_dir: Path, check_hash: bool = True, quiet: bool = False) -> bool:
    ok, messages = verify_pack(pack_dir, check_hash=check_hash)
    if not quiet:
        for line in messages:
            if line.startswith("FAIL") or line.startswith("OK"):
                print(line)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pack_dir")
    p.add_argument("--no-hash", action="store_true")
    args = p.parse_args()
    sys.exit(0 if verify(Path(args.pack_dir), check_hash=not args.no_hash) else 1)


if __name__ == "__main__":
    main()
