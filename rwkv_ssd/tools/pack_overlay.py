"""Create or materialize content-addressed pack overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.pack_overlay import PackChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--profile", required=True)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--profile", required=True)
    materialize.add_argument("--output", type=Path, required=True)
    sub.add_parser("stats")
    remove = sub.add_parser("remove")
    remove.add_argument("--profile", required=True)
    sub.add_parser("gc")
    args = parser.parse_args()
    store = PackChunkStore(args.store, chunk_bytes=args.chunk_bytes)
    if args.command == "ingest":
        print(json.dumps(store.ingest_directory(args.source, args.profile), indent=2))
    elif args.command == "materialize":
        output = store.materialize(args.profile, args.output)
        print(json.dumps({"profile": args.profile, "output": str(output)}, indent=2))
    elif args.command == "stats":
        print(json.dumps(store.store_stats(), indent=2))
    elif args.command == "remove":
        print(json.dumps({"profile": args.profile, "removed": store.remove_profile(args.profile)}, indent=2))
    else:
        print(json.dumps(store.garbage_collect(), indent=2))


if __name__ == "__main__":
    main()
