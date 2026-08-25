#!/usr/bin/env python3

"""Run the three active throughput benches (synthetic, frontier, real)."""



from __future__ import annotations



import argparse

import subprocess

import sys

from pathlib import Path





def _run(cmd: list[str], cwd: Path) -> int:

    print("\n$", " ".join(cmd), flush=True)

    return subprocess.call(cmd, cwd=cwd)





def main() -> int:

    p = argparse.ArgumentParser(description="Active throughput bench suite")

    p.add_argument("--heavy", action="store_true", help="0.1B real + Trinity frontier (quick profile)")

    p.add_argument(

        "--full",

        action="store_true",

        help="Full depth: all F scenarios, 48 tokens, warm disk cache at load",

    )

    p.add_argument("--skip-real", action="store_true", help="Skip ChatRWKV bench")

    p.add_argument("--skip-synthetic", action="store_true")

    p.add_argument("--skip-frontier", action="store_true")

    args = p.parse_args()



    root = Path(__file__).resolve().parents[2]

    code = 0

    eval_dir = root / "test_model/trinity_eval"

    eval_dir.mkdir(parents=True, exist_ok=True)



    if not args.skip_synthetic:

        synth_cmd = [

            sys.executable,

            "bench/bench_generate.py",

            "--json-out",

            str(eval_dir / "synthetic_bench.json"),

        ]

        if args.full:

            synth_cmd.append("--full")

        code |= _run(synth_cmd, root)



    if not args.skip_frontier:

        frontier_cmd = [

            sys.executable,

            "bench/bench_io_ceiling.py",

            "--json-out",

            str(eval_dir / "ram_frontier.json"),

        ]

        if args.heavy:

            frontier_cmd.append("--heavy")

        if args.full:

            frontier_cmd.append("--full")

        code |= _run(frontier_cmd, root)



    ckpt = root / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"

    pack = (

        root / "test_model/runtime_pack"

        if args.heavy

        else root / "test_model/runtime_pack_0.01b"

    )

    if not args.skip_real and ckpt.is_file() and pack.is_dir():

        real_cmd = [

            sys.executable,

            "bench/bench_throughput.py",

            "--backend",

            "chatrwkv",

            "--model",

            str(pack),

            "--checkpoint",

            str(ckpt),

            "--strategy",

            "cpu bf16",

            "--json-out",

            str(eval_dir / "throughput_real.json"),

        ]

        if args.heavy:

            real_cmd.append("--heavy")

        if args.full:

            real_cmd.append("--full")

        code |= _run(real_cmd, root)

    else:

        print("skip real bench (no checkpoint/pack or --skip-real)")



    return code





if __name__ == "__main__":

    raise SystemExit(main())

