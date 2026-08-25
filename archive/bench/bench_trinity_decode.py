#!/usr/bin/env python3
"""
Compare Trinity / FP16 pack decode cost: SSD read, dequant CPU, and effective throughput.

Reports decode-only (blobs preloaded) and read+decode (streaming-like), per-tensor
vs layer-batched LUT2. Optional Intel XPU probe (needs Intel Extension for PyTorch).

Usage:
  python bench/bench_trinity_decode.py --pack test_model/trinity_eval/fp16_grouped \\
      --pack test_model/trinity_eval/trinity_lut2 --pack test_model/trinity_eval/trinity
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_bench import pack_read_stats
from rwkv_ssd.runtime.trinity_codec import decode_lut2_layer_from_span
from rwkv_ssd.runtime.weight_store import open_weight_store


def _fp16_equiv_mb(entries: list) -> float:
    return sum(e.numel * 2 for e in entries) / 1e6


def _group_by_layer(entries, blobs: list[tuple[object, bytes]]):
    by: dict[int, list[tuple[object, bytes]]] = defaultdict(list)
    for entry, raw in blobs:
        by[int(entry.layer_id)].append((entry, raw))
    return by


def _read_all(pack: Path, io_backend: str) -> tuple[list, list[tuple[object, bytes]], float]:
    manifest = Manifest.load(pack)
    entries = manifest.streamed_tensors() or manifest.tensors
    t0 = time.perf_counter()
    with open_weight_store(manifest.weights_path, backend=io_backend) as store:
        blobs = [(e, store.read_bytes(e)) for e in entries]
    read_s = time.perf_counter() - t0
    return entries, blobs, read_s


def _decode_per_tensor(
    blobs: list[tuple[object, bytes]],
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> float:
    from rwkv_ssd.runtime.trinity_accel import resolve_trinity_decode_device

    dec = decode_device or resolve_trinity_decode_device(
        "xpu" if device.type == "xpu" else "cpu", fallback=device
    )
    out_dev = torch.device("cpu") if dec.type == "xpu" else device
    t0 = time.perf_counter()
    for entry, raw in blobs:
        decode_weight_to_tensor(
            raw, entry, out_dev, decode_device=dec
        )
    return time.perf_counter() - t0


def _decode_trinity_layer_batched(
    blobs: list[tuple[object, bytes]],
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> float:
    from rwkv_ssd.runtime.trinity_accel import resolve_trinity_decode_device
    from rwkv_ssd.runtime.trinity_codec import LayerZlibCache, decode_trinity_layer_span

    dec = decode_device or resolve_trinity_decode_device(
        "xpu" if device.type == "xpu" else "cpu", fallback=device
    )
    out_dev = torch.device("cpu") if dec.type == "xpu" else device
    t0 = time.perf_counter()
    by_layer = _group_by_layer(None, blobs)
    cache = LayerZlibCache()
    for group in by_layer.values():
        codecs = {(e.dequant or "none").strip().lower() for e, _ in group}
        if codecs == {"trinity_layer"}:
            entries = [e for e, _ in group]
            packed = group[0][1]
            decode_trinity_layer_span(
                packed, entries, cache, out_dev, decode_device=dec
            )
        else:
            for entry, raw in group:
                decode_weight_to_tensor(
                    raw, entry, out_dev, decode_device=dec
                )
    return time.perf_counter() - t0


def _decode_lut2_layer_batched(
    blobs: list[tuple[object, bytes]],
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> float:
    from rwkv_ssd.runtime.trinity_accel import resolve_trinity_decode_device

    dec = decode_device or resolve_trinity_decode_device(
        "xpu" if device.type == "xpu" else "cpu", fallback=device
    )
    out_dev = torch.device("cpu") if dec.type == "xpu" else device
    t0 = time.perf_counter()
    by_layer = _group_by_layer(None, blobs)
    for group in by_layer.values():
        codecs = {(e.dequant or "none").strip().lower() for e, _ in group}
        if codecs == {"trinity_lut2"} and len(group) > 0:
            entries = [e for e, _ in group]
            base = min(e.offset for e in entries)
            end = max(e.offset + e.length for e in entries)
            span = bytearray(end - base)
            for entry, raw in group:
                rel = entry.offset - base
                span[rel : rel + entry.length] = raw
            decode_lut2_layer_from_span(
                bytes(span), entries, base, out_dev, decode_device=dec
            )
        else:
            for entry, raw in group:
                decode_weight_to_tensor(
                    raw, entry, out_dev, decode_device=dec
                )
    return time.perf_counter() - t0


def _probe_intel_xpu() -> dict:
    from rwkv_ssd.runtime.trinity_accel import probe_intel_xpu

    return probe_intel_xpu()


def bench_pack(
    pack: Path,
    *,
    label: str,
    io_backend: str,
    device: torch.device,
) -> dict:
    stats = pack_read_stats(pack, io_backend)
    entries, blobs, read_s = _read_all(pack, io_backend)
    fp16_mb = _fp16_equiv_mb(entries)

    decode_per_tensor_s = _decode_per_tensor(blobs, device)
    codecs = {(e.dequant or "none").strip().lower() for e in entries}
    if codecs == {"trinity_layer"}:
        decode_layer_s = _decode_trinity_layer_batched(blobs, device)
    elif codecs <= {"trinity_lut2"}:
        decode_layer_s = _decode_lut2_layer_batched(blobs, device)
    else:
        decode_layer_s = decode_per_tensor_s

    read_decode_s = read_s + decode_per_tensor_s
    eff_decode_gbs = (fp16_mb / 1024.0) / decode_per_tensor_s if decode_per_tensor_s > 0 else 0.0
    eff_e2e_gbs = (fp16_mb / 1024.0) / read_decode_s if read_decode_s > 0 else 0.0

    baseline_note = ""
    return {
        "label": label,
        "pack": str(pack),
        "weights_mb": stats["weights_mb"],
        "fp16_equiv_mb": round(fp16_mb, 2),
        "tensors": len(entries),
        "codecs": stats["dequant_codecs"],
        "read_s": round(read_s, 4),
        "read_gbs": stats["read_gbs"],
        "decode_per_tensor_s": round(decode_per_tensor_s, 4),
        "decode_layer_batch_s": round(decode_layer_s, 4),
        "read_plus_decode_s": round(read_decode_s, 4),
        "effective_decode_gbs": round(eff_decode_gbs, 2),
        "effective_e2e_gbs": round(eff_e2e_gbs, 2),
        "layer_batch_speedup": round(
            decode_per_tensor_s / max(decode_layer_s, 1e-9), 3
        ),
        "baseline_note": baseline_note,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Trinity decode vs FP16 comparison")
    p.add_argument("--pack", action="append", required=True, type=Path)
    p.add_argument("--label", action="append")
    p.add_argument("--io-backend", default="mmap")
    p.add_argument("--device", default="cpu", help="cpu or xpu (if IPEX available)")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--probe-xpu", action="store_true")
    p.add_argument(
        "--compare-cpu-xpu",
        action="store_true",
        help="If XPU is available, print CPU vs XPU decode times for first pack",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    if args.device.startswith("xpu") and not torch.xpu.is_available():
        raise SystemExit(
            "torch.xpu not available. Install Intel Extension for PyTorch "
            "(intel-extension-for-pytorch) and Intel GPU drivers."
        )

    if args.probe_xpu:
        print("Intel GPU probe:", json.dumps(_probe_intel_xpu(), indent=2))

    if args.compare_cpu_xpu and torch.xpu.is_available() and args.pack:
        entries, blobs, _ = _read_all(args.pack[0], args.io_backend)
        codecs = {(e.dequant or "none").strip().lower() for e in entries}

        def _layer_batch_decode(decode_device: torch.device) -> None:
            if codecs == {"trinity_layer"}:
                _decode_trinity_layer_batched(
                    blobs, torch.device("cpu"), decode_device=decode_device
                )
            else:
                _decode_lut2_layer_batched(
                    blobs, torch.device("cpu"), decode_device=decode_device
                )

        for _ in range(2):
            _layer_batch_decode(torch.device("cpu"))
        t0 = time.perf_counter()
        for _ in range(3):
            _layer_batch_decode(torch.device("cpu"))
        cpu_t = (time.perf_counter() - t0) / 3
        for _ in range(2):
            _layer_batch_decode(torch.device("xpu"))
        t0 = time.perf_counter()
        for _ in range(3):
            _layer_batch_decode(torch.device("xpu"))
        xpu_t = (time.perf_counter() - t0) / 3
        winner = "CPU" if cpu_t <= xpu_t else "XPU"
        pack_gb = sum(e.numel for e in entries) / 1e9
        print(
            f"\nCPU vs XPU layer-batch decode ({args.pack[0].name}, engine path): "
            f"cpu={cpu_t:.4f}s xpu={xpu_t:.4f}s ratio={cpu_t / max(xpu_t, 1e-9):.2f}x "
            f"({winner} faster)\n"
            f"Logical payload: {pack_gb:.3f}B values. "
            "Set RWKV_TRINITY_XPU_MIN_NUMEL to tune the accelerator threshold.\n"
        )

    rows = []
    for i, pack in enumerate(args.pack):
        label = args.label[i] if args.label and i < len(args.label) else pack.name
        rows.append(
            bench_pack(pack, label=label, io_backend=args.io_backend, device=device)
        )

    baseline = rows[0]
    print(
        f"baseline={baseline['label']}  weights={baseline['weights_mb']} MB  "
        f"read={baseline['read_s']}s  decode={baseline['decode_per_tensor_s']}s"
    )
    print(
        f"{'label':16} {'diskMB':>8} {'read_s':>7} {'decode_s':>8} {'layer_s':>8} "
        f"{'batch×':>6} {'e2eGB/s':>8} {'vs_base':>8}"
    )
    for r in rows:
        vs = r["decode_per_tensor_s"] / max(baseline["decode_per_tensor_s"], 1e-9)
        print(
            f"{r['label']:16} {r['weights_mb']:8.2f} {r['read_s']:7.4f} "
            f"{r['decode_per_tensor_s']:8.4f} {r['decode_layer_batch_s']:8.4f} "
            f"{r['layer_batch_speedup']:6.2f} {r['effective_e2e_gbs']:8.2f} "
            f"{vs:7.2f}x"
        )

    if len(rows) >= 2:
        # Per-token sweep cost: one layer's tensors per step (streaming mental model)
        from rwkv_ssd.runtime.layer_keys import manifest_block_layers

        manifest0 = Manifest.load(Path(rows[0]["pack"]))
        entries0 = manifest0.streamed_tensors() or manifest0.tensors
        block_layers = manifest_block_layers(manifest0)
        n_layers = len(block_layers)
        tensors_per_layer = defaultdict(int)
        for e in entries0:
            if e.layer_id in block_layers:
                tensors_per_layer[int(e.layer_id)] += 1
        avg_tensors = (
            sum(tensors_per_layer.values()) / max(n_layers, 1) if n_layers else 0
        )
        print()
        print(
            f"Streaming hint (0.01B pack): ~{n_layers} block layers, "
            f"~{avg_tensors:.0f} tensors/layer — per token you decode ~1 layer, "
            "not the full pack."
        )
        lut = next((r for r in rows if "lut2" in r["label"]), None)
        if lut:
            per_layer_decode = lut["decode_layer_batch_s"] / max(n_layers, 1)
            per_layer_fp16 = baseline["decode_per_tensor_s"] / max(n_layers, 1)
            print(
                f"  Amortized decode per token (1 layer, engine path): "
                f"~{per_layer_decode*1000:.1f} ms vs FP16 ~{per_layer_fp16*1000:.1f} ms "
                f"({per_layer_decode / max(per_layer_fp16, 1e-9):.1f}x)"
            )
        tl = next((r for r in rows if "trinity_layer" in r["label"]), None)
        if tl and lut:
            print(
                f"  trinity_layer vs trinity (per-tensor zlib): disk "
                f"{tl['weights_mb']:.2f} vs {next((r for r in rows if 'trinity_old' in r['label']), lut)['weights_mb']:.2f} MB, "
                f"layer decode {tl['decode_layer_batch_s']:.3f}s vs "
                f"{lut.get('decode_layer_batch_s', lut['decode_per_tensor_s']):.3f}s"
            )
        if lut:
            storage_ratio = baseline["weights_mb"] / max(lut["weights_mb"], 1e-9)
            decode_ratio = lut["decode_per_tensor_s"] / max(
                baseline["decode_per_tensor_s"], 1e-9
            )
            print()
            print(
                f"SSD: {storage_ratio:.2f}x smaller weights.bin vs FP16 "
                f"({lut['weights_mb']:.1f} vs {baseline['weights_mb']:.1f} MB)"
            )
            print(
                f"CPU decode-only: {decode_ratio:.2f}x slower than FP16 memcpy "
                f"({lut['decode_per_tensor_s']:.3f}s vs {baseline['decode_per_tensor_s']:.3f}s)"
            )
            print(
                f"End-to-end read+decode: {lut['read_plus_decode_s']:.3f}s vs "
                f"{baseline['read_plus_decode_s']:.3f}s "
                f"({baseline['read_plus_decode_s'] / max(lut['read_plus_decode_s'], 1e-9):.2f}x "
                "wall-clock — smaller disk often wins despite decode tax)"
            )
            print(
                f"RAM during decode: materializes ~{lut['fp16_equiv_mb']:.1f} MB "
                "FP16-equivalent weights in tensors (same logical model as FP16)."
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
