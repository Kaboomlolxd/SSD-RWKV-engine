#!/usr/bin/env python3
"""
LUT2 / LUT3 / LUT4 research bench — pack size, decode time, MSE vs bf16.

Usage:
  python bench/bench_lut_bitwidth_research.py
  python bench/bench_lut_bitwidth_research.py --checkpoint test_model/...pth --layers 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_ssd.runtime.lut_bitwidth_research import (
    blob_byte_size,
    decode_lut_tensor,
    encode_lut_tensor,
    mse_vs_bf16,
    SPECS,
)
from rwkv_ssd.runtime.pack_codec import encode_scale_u4, encode_scale_u8


def _bench_decode(blob: bytes, shape: tuple[int, ...], variant: str, trials: int = 8) -> float:
    for _ in range(3):
        decode_lut_tensor(blob, shape, variant)
    t0 = time.perf_counter()
    for _ in range(trials):
        decode_lut_tensor(blob, shape, variant)
    return (time.perf_counter() - t0) / trials * 1000


def _bench_scale(blob: bytes, shape: tuple[int, ...], bits: int, trials: int = 8) -> float:
    from rwkv_ssd.runtime.manifest import TensorEntry
    from rwkv_ssd.runtime.pack_codec import decode_scale_to_tensor

    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=list(shape),
        offset=0,
        length=len(blob),
        alignment=4096,
        residency="streamed",
        dequant=f"scale_u{bits}",
    )
    for _ in range(3):
        decode_scale_to_tensor(blob, entry, torch.device("cpu"), bits=bits)
    t0 = time.perf_counter()
    for _ in range(trials):
        decode_scale_to_tensor(blob, entry, torch.device("cpu"), bits=bits)
    return (time.perf_counter() - t0) / trials * 1000


def bench_tensor(name: str, tensor: torch.Tensor) -> list[dict]:
    shape = tuple(tensor.shape)
    numel = tensor.numel()
    bf16_bytes = numel * 2
    rows: list[dict] = []

    for variant in ("lut2", "lut3", "lut4"):
        blob = encode_lut_tensor(tensor, variant)
        decoded = decode_lut_tensor(blob, shape, variant)
        rows.append(
            {
                "tensor": name,
                "codec": variant,
                "shape": list(shape),
                "bf16_bytes": bf16_bytes,
                "packed_bytes": len(blob),
                "ratio_vs_bf16": round(bf16_bytes / len(blob), 2),
                "mse": round(mse_vs_bf16(tensor, decoded), 8),
                "decode_ms": round(_bench_decode(blob, shape, variant), 3),
            }
        )

    u8 = encode_scale_u8(tensor)
    from rwkv_ssd.runtime.manifest import TensorEntry
    from rwkv_ssd.runtime.pack_codec import decode_scale_to_tensor

    ent = TensorEntry(
        name="w", layer_id=0, dtype="bfloat16", shape=list(shape),
        offset=0, length=len(u8), alignment=4096, residency="streamed", dequant="scale_u8",
    )
    d8 = decode_scale_to_tensor(u8, ent, torch.device("cpu"), bits=8)
    rows.append(
        {
            "tensor": name,
            "codec": "scale_u8",
            "shape": list(shape),
            "bf16_bytes": bf16_bytes,
            "packed_bytes": len(u8),
            "ratio_vs_bf16": round(bf16_bytes / len(u8), 2),
            "mse": round(mse_vs_bf16(tensor, d8), 8),
            "decode_ms": round(_bench_scale(u8, shape, 8), 3),
        }
    )

    u4 = encode_scale_u4(tensor)
    ent4 = TensorEntry(
        name="w", layer_id=0, dtype="bfloat16", shape=list(shape),
        offset=0, length=len(u4), alignment=4096, residency="streamed", dequant="scale_u4",
    )
    d4 = decode_scale_to_tensor(u4, ent4, torch.device("cpu"), bits=4)
    rows.append(
        {
            "tensor": name,
            "codec": "scale_u4",
            "shape": list(shape),
            "bf16_bytes": bf16_bytes,
            "packed_bytes": len(u4),
            "ratio_vs_bf16": round(bf16_bytes / len(u4), 2),
            "mse": round(mse_vs_bf16(tensor, d4), 8),
            "decode_ms": round(_bench_scale(u4, shape, 4), 3),
        }
    )
    return rows


def synthetic_shapes() -> list[tuple[str, tuple[int, int]]]:
    """RWKV-0.1B-like mats; head omitted here (run with --checkpoint for full table)."""
    return [
        ("att_receptance", (768, 768)),
        ("att_key", (768, 768)),
        ("ffn_key", (768, 3072)),
        ("ffn_value", (3072, 768)),
    ]


def load_checkpoint_tensors(ckpt: Path, n_layers: int) -> list[tuple[str, torch.Tensor]]:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    out: list[tuple[str, torch.Tensor]] = []
    for key, t in state.items():
        if not key.endswith(".weight") or not isinstance(t, torch.Tensor):
            continue
        if key.startswith("blocks."):
            lid = int(key.split(".")[1])
            if lid >= n_layers:
                continue
        out.append((key, t))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="LUT bitwidth research bench")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--layers", type=int, default=2, help="max block layers from ckpt")
    p.add_argument("--json-out", type=Path, default=ROOT / "test_model/trinity_eval/lut_bitwidth_research.json")
    args = p.parse_args()

    tensors: list[tuple[str, torch.Tensor]] = []
    if args.checkpoint and args.checkpoint.is_file():
        tensors = load_checkpoint_tensors(args.checkpoint, args.layers)
        print(f"Loaded {len(tensors)} weight tensors from {args.checkpoint}")
    else:
        for name, shape in synthetic_shapes():
            tensors.append((name, torch.randn(*shape, dtype=torch.bfloat16)))
        print("Using synthetic bf16 tensors (no checkpoint)")

    all_rows: list[dict] = []
    for name, t in tensors:
        all_rows.extend(bench_tensor(name, t))

    # Summary by codec
    print("\n=== Mean by codec ===")
    codecs = sorted({r["codec"] for r in all_rows})
    for codec in codecs:
        sub = [r for r in all_rows if r["codec"] == codec]
        print(
            f"{codec:10} ratio={statistics.mean(r['ratio_vs_bf16'] for r in sub):.2f}x "
            f"mse={statistics.mean(r['mse'] for r in sub):.6f} "
            f"decode_ms={statistics.mean(r['decode_ms'] for r in sub):.2f}"
        )

    out = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "specs": {k: {"entries": v.entries, "bits": v.bits} for k, v in SPECS.items()},
        "rows": all_rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
