#!/usr/bin/env python3

"""End-to-end ChatRWKV tok/s across packs (streaming modes + promote + prefix cache)."""



from __future__ import annotations



import argparse

import json

import statistics

import time

from pathlib import Path



from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root

from rwkv_ssd.runtime.config import EngineConfig

from rwkv_ssd.runtime.engine import InferenceEngine

from rwkv_ssd.runtime.metrics import MetricsCollector

from rwkv_ssd.runtime.z_layer_retention import block_layer_ids_in_z



ROOT = Path(__file__).resolve().parents[1]



CKPT_0_01B = ROOT / "test_model/rwkv7-g1d-0.01b-bench.pth"

CKPT_0_1B = ROOT / "test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth"

SYSTEM_PREFIX = "You are a helpful assistant. "





def _row_from_run(

    eng: InferenceEngine,

    wall: float,

    max_tokens: int,

) -> dict:

    model = getattr(eng.backend, "_model", None)

    layers_in_z = (

        len(block_layer_ids_in_z(model.z))

        if model is not None and hasattr(model, "z")

        else 0

    )

    m = eng.metrics

    read_ms = sum(L.read_ms for L in m.layers)

    staging_ms = sum(L.staging_ms for L in m.layers)

    shadow_hits = sum(getattr(L, "shadow_hits", 0) for L in m.layers)

    disk_hits = sum(getattr(L, "disk_cache_hits", 0) for L in m.layers)

    return {

        "tok_s": round(max_tokens / wall, 2) if wall > 0 else 0.0,

        "wall_s": round(wall, 3),

        "layers_in_z": layers_in_z,

        "read_ms": round(read_ms, 1),

        "staging_ms": round(staging_ms, 1),

        "shadow_hits": shadow_hits,

        "disk_cache_hits": disk_hits,

        "state_cache_hit": bool(getattr(m, "state_cache_hit", False)),

        "prefill_wall_s": round(getattr(m, "prefill_wall_s", 0.0), 4),

    }





def _median_tok_s(

    pack: Path,

    ckpt: Path,

    *,

    mode: str,

    stream_layer_cache: bool,

    max_layers_in_z: int,

    max_tokens: int,

    samples: int,

    warmup: int,

    state_cache: bool = False,

    system_prefix: str | None = None,

    user_prompt: str = "Throughput bench prompt",

) -> dict:

    cfg = EngineConfig(

        pack_dir=pack,

        checkpoint_path=str(ckpt),

        backend="chatrwkv",

        mode=mode,

        strategy="cpu bf16",

        device="cpu",

        max_tokens=max_tokens,

        stream_layer_cache=stream_layer_cache,

        max_layers_in_z=max_layers_in_z,

        greedy=True,

        skeleton_load=True,

        state_cache=state_cache,

        system_prefix=system_prefix,

    )

    eng = InferenceEngine(cfg)

    eng.load()

    rates: list[float] = []

    last: dict = {}

    try:

        for attempt in range(warmup + samples):

            eng.metrics = MetricsCollector()

            t0 = time.perf_counter()

            eng.generate(user_prompt)

            wall = time.perf_counter() - t0

            last = _row_from_run(eng, wall, max_tokens)

            if attempt >= warmup:

                rates.append(last["tok_s"])

    finally:

        eng.close()

    last["tok_s"] = round(statistics.median(rates), 2) if rates else 0.0

    last["samples"] = len(rates)

    return last





def _median_prefix_warm(

    pack: Path,

    ckpt: Path,

    *,

    max_tokens: int,

    samples: int,

    warmup: int,

) -> dict:

    """Second+ request with same system prefix (prefix cache hit)."""

    cfg = EngineConfig(

        pack_dir=pack,

        checkpoint_path=str(ckpt),

        backend="chatrwkv",

        mode="streaming",

        strategy="cpu bf16",

        device="cpu",

        max_tokens=max_tokens,

        stream_layer_cache=True,

        max_layers_in_z=2,

        greedy=True,

        skeleton_load=True,

        state_cache=True,

        system_prefix=SYSTEM_PREFIX,

    )

    eng = InferenceEngine(cfg)

    eng.load()

    rates: list[float] = []

    last: dict = {}

    try:

        eng.generate("Cold prefix fill request.")

        for attempt in range(warmup + samples):

            eng.metrics = MetricsCollector()

            t0 = time.perf_counter()

            eng.generate("Warm user follow-up question.")

            wall = time.perf_counter() - t0

            last = _row_from_run(eng, wall, max_tokens)

            if attempt >= warmup:

                rates.append(last["tok_s"])

    finally:

        eng.close()

    last["tok_s"] = round(statistics.median(rates), 2) if rates else 0.0

    last["samples"] = len(rates)

    return last





def main() -> None:

    if find_chatrwkv_root() is None:

        raise SystemExit("ChatRWKV not found under test_model/ChatRWKV")



    p = argparse.ArgumentParser()

    p.add_argument("--max-tokens", type=int, default=32)

    p.add_argument("--samples", type=int, default=3)

    p.add_argument("--warmup", type=int, default=1)

    p.add_argument(

        "--skip-prefix",

        action="store_true",

        help="Skip prefix-cache warm scenario (faster)",

    )

    args = p.parse_args()



    suites = [

        (

            "0.01B",

            CKPT_0_01B,

            [

                ("FP16", ROOT / "test_model/trinity_eval/fp16_grouped"),

                ("FP16_alt", ROOT / "test_model/runtime_pack_0.01b"),

                ("trinity_lut2", ROOT / "test_model/trinity_eval/trinity_lut2"),

                (

                    "trinity_lut2+shadow",

                    ROOT / "test_model/trinity_eval/trinity_lut2_shadow",

                ),

            ],

        ),

        (

            "0.1B",

            CKPT_0_1B,

            [

                ("FP16", ROOT / "test_model/trinity_eval/fp16_grouped_0.1b"),
                ("FP16_default", ROOT / "test_model/runtime_pack"),

                ("trinity_lut2", ROOT / "test_model/trinity_eval/trinity_lut2_0.1b"),

                (

                    "trinity_lut2+shadow",

                    ROOT / "test_model/trinity_eval/trinity_lut2_shadow_0.1b",

                ),

            ],

        ),

    ]



    scenarios = [

        ("streaming", "streaming", False, 1),

        ("streaming+cache", "streaming", True, 2),

    ]



    results: list[dict] = []

    meta = {

        "max_tokens": args.max_tokens,

        "samples": args.samples,

        "warmup": args.warmup,

        "strategy": "cpu bf16",

        "notes": "v0.6.12+ promote; native/numba bf16 gather; selective shadow; async decode cache",

    }

    print(

        f"ChatRWKV tok/s  max_tokens={args.max_tokens}  "

        f"samples={args.samples} warmup={args.warmup}\n"

    )



    for size, ckpt, packs in suites:

        if not ckpt.is_file():

            print(f"SKIP {size}: checkpoint missing {ckpt}")

            continue

        print(f"=== {size} ===")

        for pack_label, pack_path in packs:

            if not pack_path.is_dir():

                print(f"  SKIP {pack_label}: {pack_path}")

                continue

            for scen_label, mode, cache, max_z in scenarios:

                r = _median_tok_s(

                    pack_path,

                    ckpt,

                    mode=mode,

                    stream_layer_cache=cache,

                    max_layers_in_z=max_z,

                    max_tokens=args.max_tokens,

                    samples=args.samples,

                    warmup=args.warmup,

                )

                row = {

                    "size": size,

                    "pack": pack_label,

                    "scenario": scen_label,

                    **r,

                }

                results.append(row)

                sh = (

                    f" shadow_hits={r['shadow_hits']}"

                    if r.get("shadow_hits")

                    else ""

                )

                print(

                    f"  {pack_label:22} {scen_label:18} "

                    f"{r['tok_s']:7.2f} tok/s  z={r['layers_in_z']}  "

                    f"read={r['read_ms']:.0f}ms staging={r['staging_ms']:.0f}ms{sh}"

                )

            if not args.skip_prefix and pack_label.startswith("FP16"):

                r = _median_prefix_warm(

                    pack_path,

                    ckpt,

                    max_tokens=args.max_tokens,

                    samples=args.samples,

                    warmup=args.warmup,

                )

                row = {

                    "size": size,

                    "pack": pack_label,

                    "scenario": "streaming+cache+prefix(warm)",

                    **r,

                }

                results.append(row)

                print(

                    f"  {pack_label:22} {'prefix(warm)':18} "

                    f"{r['tok_s']:7.2f} tok/s  z={r['layers_in_z']}  "

                    f"prefill={r['prefill_wall_s']:.3f}s hit={r['state_cache_hit']}"

                )

        print()



    out = ROOT / "test_model/trinity_eval/tok_s_compare.json"

    payload = {"meta": meta, "results": results}

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out}")





if __name__ == "__main__":

    main()

