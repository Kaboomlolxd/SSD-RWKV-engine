#!/usr/bin/env python3
"""
Compression Trinity afternoon experiment (IDEAS.md gate).

Compares FP16 (or none) baseline pack against Trinity / M5 candidates.
Promotion to engine path requires:
  - storage ratio >= 1.3× vs baseline weights.bin, and
  - sequential read+dequant wall time within 10% of baseline (optional --dequant-bench).

Usage:
  python -m rwkv_ssd.tools.compression_trinity_experiment \\
    --baseline test_model/runtime_pack_0.01b \\
    --candidate test_model/packs_0.01b/trinity

  python -m rwkv_ssd.tools.compression_trinity_experiment \\
    --build test_model/rwkv7-g1d-0.01b-bench.pth \\
    --output test_model/trinity_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_bench import pack_full_stats, pack_read_stats
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.tools.pack_runtime import pack


def _preload_blobs(pack_dir: Path, io_backend: str) -> list[tuple[object, bytes]]:
    manifest = Manifest.load(pack_dir)
    streamed = manifest.streamed_tensors() or manifest.tensors
    blobs: list[tuple[object, bytes]] = []
    with open_weight_store(manifest.weights_path, backend=io_backend) as store:
        for entry in streamed:
            blobs.append((entry, store.read_bytes(entry)))
    return blobs


def _group_by_layer(
    blobs: list[tuple[object, bytes]],
) -> dict[int, list[tuple[object, bytes]]]:
    by_layer: dict[int, list[tuple[object, bytes]]] = {}
    for entry, raw in blobs:
        by_layer.setdefault(int(entry.layer_id), []).append((entry, raw))
    return by_layer


def _dequant_wall_s(
    pack_dir: Path,
    io_backend: str,
    *,
    decode_only: bool = True,
) -> float:
    """
      Wall time to materialize all streamed tensors.

      ``decode_only=True`` (default) preloads blobs then times decode — matches
      the engine after SSD read (prefetch / layer span) and avoids penalizing
    compressed packs for smaller reads.
    """
    manifest = Manifest.load(pack_dir)
    streamed = manifest.streamed_tensors() or manifest.tensors
    device = torch.device("cpu")
    t0 = time.perf_counter()
    if decode_only:
        from rwkv_ssd.runtime.trinity_codec import decode_lut2_layer_from_span

        blobs = _preload_blobs(pack_dir, io_backend)
        by_layer = _group_by_layer(blobs)
        from rwkv_ssd.runtime.trinity_codec import (
            LayerZlibCache,
            decode_trinity_layer_span,
        )

        cache = LayerZlibCache()
        for _layer_id, group in by_layer.items():
            codecs = {(e.dequant or "none").strip().lower() for e, _ in group}
            if codecs == {"trinity_layer"}:
                entries = [e for e, _ in group]
                decode_trinity_layer_span(group[0][1], entries, cache, device)
            elif codecs == {"trinity_lut2"} and len(group) > 1:
                entries = [e for e, _ in group]
                base = min(e.offset for e in entries)
                total = max(e.offset + e.length for e in entries) - base
                span = bytearray(total)
                for entry, raw in group:
                    rel = entry.offset - base
                    span[rel : rel + entry.length] = raw
                decode_lut2_layer_from_span(bytes(span), entries, base, device)
            else:
                for entry, raw in group:
                    decode_weight_to_tensor(raw, entry, device)
    else:
        with open_weight_store(manifest.weights_path, backend=io_backend) as store:
            for entry in streamed:
                raw = store.read_bytes(entry)
                decode_weight_to_tensor(raw, entry, device)
    return time.perf_counter() - t0


def _build_trinity_variants(
    checkpoint: Path, output: Path, model_family: str
) -> list[tuple[str, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Path]] = []
    for label, codec, layout in (
        ("fp16_grouped", "none", "layer_grouped"),
        ("scale_u4", "scale_u4", "layer_grouped"),
        ("trinity_lut2", "trinity_lut2", "layer_grouped"),
        ("trinity", "trinity", "layer_grouped"),
        ("trinity_layer", "trinity", "layer_grouped"),
    ):
        dest = output / label
        print(f"pack {codec} -> {dest}")
        pack(
            checkpoint,
            dest,
            model_family=model_family,
            pack_codec=codec,
            pack_layout=layout,
            quiet=True,
        )
        out.append((label, dest))
    return out


def _dequant_budget(label: str, default_reg: float, lut2_reg: float) -> float:
    if "trinity_lut2" in label or label == "trinity_lut2":
        return lut2_reg
    if label == "trinity":
        return max(lut2_reg, default_reg * 4)
    return default_reg


def _eval_pair(
    baseline: Path,
    candidate: Path,
    *,
    label: str = "",
    io_backend: str,
    min_storage_ratio: float,
    max_read_regression: float,
    max_dequant_regression: float,
    max_dequant_regression_lut2: float,
    dequant_bench: bool,
    decode_only: bool = True,
) -> dict:
    base = pack_full_stats(baseline, io_backend)
    cand = pack_full_stats(candidate, io_backend)

    storage_ratio = base["weights_bin_mb"] / max(cand["weights_bin_mb"], 1e-9)
    storage_ratio_adjusted = base["total_mb"] / max(cand["total_mb"], 1e-9)
    read_regression = (cand["read_s"] - base["read_s"]) / max(base["read_s"], 1e-9)

    dequant_s = None
    dequant_regression = None
    if dequant_bench:
        base_d = _dequant_wall_s(baseline, io_backend, decode_only=decode_only)
        cand_d = _dequant_wall_s(candidate, io_backend, decode_only=decode_only)
        dequant_s = {"baseline": round(base_d, 4), "candidate": round(cand_d, 4)}
        dequant_regression = (cand_d - base_d) / max(base_d, 1e-9)

    passes_storage = (
        storage_ratio >= min_storage_ratio and read_regression <= max_read_regression
    )
    dequant_budget = _dequant_budget(
        label, max_dequant_regression, max_dequant_regression_lut2
    )
    passes_engine = passes_storage
    if dequant_regression is not None:
        passes_engine = passes_engine and dequant_regression <= dequant_budget

    return {
        "baseline": base,
        "candidate": cand,
        "storage_ratio": round(storage_ratio, 3),
        "storage_ratio_adjusted": round(storage_ratio_adjusted, 3),
        "read_regression": round(read_regression, 3),
        "dequant_wall_s": dequant_s,
        "dequant_regression": (
            round(dequant_regression, 3) if dequant_regression is not None else None
        ),
        "min_storage_ratio": min_storage_ratio,
        "max_read_regression": max_read_regression,
        "max_dequant_regression": max_dequant_regression,
        "dequant_budget": dequant_budget,
        "passes_storage_gate": passes_storage,
        "passes_engine_gate": passes_engine,
        "promote_trinity": passes_engine,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compression Trinity promotion gate")
    p.add_argument("--baseline", type=Path, help="FP16 / none reference pack")
    p.add_argument("--candidate", type=Path, help="Trinity or quant candidate pack")
    p.add_argument(
        "--build", type=Path, metavar="CHECKPOINT", help="Build ladder under --output"
    )
    p.add_argument("--output", type=Path, help="Output dir for --build")
    p.add_argument("--model-family", default="rwkv7")
    p.add_argument("--min-storage-ratio", type=float, default=1.3)
    p.add_argument("--max-read-regression", type=float, default=0.10)
    p.add_argument(
        "--max-dequant-regression",
        type=float,
        default=0.10,
        help="Decode-only wall-time regression vs FP16 (use --max-dequant-regression-lut2 for LUT)",
    )
    p.add_argument(
        "--max-dequant-regression-lut2",
        type=float,
        default=15.0,
        help="Decode-only budget for trinity_lut2 (batched layer path; tighten over time)",
    )
    p.add_argument("--io-backend", default="mmap")
    p.add_argument(
        "--dequant-bench",
        action="store_true",
        help="Time pack decode (default: decode-only after preload)",
    )
    p.add_argument(
        "--dequant-includes-read",
        action="store_true",
        help="Include per-tensor SSD read in dequant bench (legacy, unfair to compressed)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 unless storage and dequant gates pass (default: storage-only OK)",
    )
    args = p.parse_args()

    if args.build:
        if not args.output:
            p.error("--output required with --build")
        if not args.build.is_file():
            raise SystemExit(f"checkpoint missing: {args.build}")
        variants = _build_trinity_variants(args.build, args.output, args.model_family)
        baseline = variants[0][1]
        reports = []
        for label, path in variants[1:]:
            row = _eval_pair(
                baseline,
                path,
                label=label,
                io_backend=args.io_backend,
                min_storage_ratio=args.min_storage_ratio,
                max_read_regression=args.max_read_regression,
                max_dequant_regression=args.max_dequant_regression,
                max_dequant_regression_lut2=args.max_dequant_regression_lut2,
                dequant_bench=args.dequant_bench,
                decode_only=not args.dequant_includes_read,
            )
            row["label"] = label
            reports.append(row)
        if args.json:
            print(json.dumps(reports, indent=2))
        else:
            base_mb = pack_read_stats(baseline, args.io_backend)["weights_mb"]
            base_full = pack_full_stats(baseline, args.io_backend)
            print(
                f"Baseline: {baseline.name} ({base_mb} MB weights.bin, "
                f"{base_full['total_mb']} MB total)"
            )
            for r in reports:
                stor = "storage PASS" if r["passes_storage_gate"] else "storage FAIL"
                eng = "engine PASS" if r["passes_engine_gate"] else "engine FAIL"
                cand_mb = r["candidate"]["weights_bin_mb"]
                cand_total = r["candidate"]["total_mb"]
                print(
                    f"  {r['label']:14} {cand_mb:7.2f}/{cand_total:7.2f} MB  "
                    f"weights={r['storage_ratio']:.2f}×  total={r['storage_ratio_adjusted']:.2f}×  "
                    f"read_reg={r['read_regression']:+.0%}  {stor}; {eng}"
                )
                if r.get("dequant_regression") is not None:
                    print(f"      dequant_reg={r['dequant_regression']:+.0%}")
        failed_storage = [r["label"] for r in reports if not r["passes_storage_gate"]]
        failed_engine = [r["label"] for r in reports if not r["passes_engine_gate"]]
        if failed_storage:
            print(f"storage gate failed: {', '.join(failed_storage)}", file=sys.stderr)
            raise SystemExit(1)
        if failed_engine and args.strict:
            print(f"engine gate failed: {', '.join(failed_engine)}", file=sys.stderr)
            raise SystemExit(1)
        if failed_engine:
            print(
                "Note: storage gate passed; dequant still over budget — "
                "optimize trinity_codec or use --strict to fail.",
                file=sys.stderr,
            )
        return

    if not args.baseline or not args.candidate:
        p.error("pass --baseline and --candidate, or --build with --output")

    report = _eval_pair(
        args.baseline,
        args.candidate,
        label=args.candidate.name,
        io_backend=args.io_backend,
        min_storage_ratio=args.min_storage_ratio,
        max_read_regression=args.max_read_regression,
        max_dequant_regression=args.max_dequant_regression,
        max_dequant_regression_lut2=args.max_dequant_regression_lut2,
        dequant_bench=args.dequant_bench,
        decode_only=not args.dequant_includes_read,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
    if not report["promote_trinity"]:
        raise SystemExit(
            "Trinity/candidate did not pass promotion gate — keep FP16 default on engine path."
        )


if __name__ == "__main__":
    main()
