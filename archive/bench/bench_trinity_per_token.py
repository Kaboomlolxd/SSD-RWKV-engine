#!/usr/bin/env python3
"""Simulate per-token Trinity cost: read + decode ONE block layer (streaming hot path)."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from rwkv_ssd.runtime.layer_io import entries_contiguous_span
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.trinity_codec import LayerZlibCache, decode_trinity_layer_span
from rwkv_ssd.runtime.weight_store import open_weight_store

ROOT = Path(__file__).resolve().parents[1]


def _one_layer_ms(pack: Path, layer_id: int, *, iters: int, use_shadow: bool) -> dict:
    manifest = Manifest.load(pack)
    block_layers = manifest_block_layers(manifest)
    if layer_id not in block_layers:
        layer_id = block_layers[0]
    entries = [e for e in manifest.tensors if e.layer_id == layer_id]
    if not entries:
        raise SystemExit(f"no tensors for layer {layer_id}")
    codecs = {(e.dequant or "none").strip().lower() for e in entries}
    trinity_layer = codecs == {"trinity_layer"}
    use_shadow = use_shadow and manifest.has_bf16_shadow()
    span = entries_contiguous_span(entries)
    if use_shadow:
        from rwkv_ssd.runtime.decode_shadow import entries_shadow_contiguous_span

        shadow_span = entries_shadow_contiguous_span(entries)
        if shadow_span is None:
            raise SystemExit("shadow offsets not contiguous for layer")
        base, span_len = shadow_span
    elif trinity_layer:
        base = entries[0].offset
        span_len = entries[0].length
    elif span is not None:
        base, span_len = span
    else:
        base = 0
        span_len = 0

    def _read_layer(store, shadow_store=None) -> bytes | None:
        read_span = getattr(store, "read_bytes_span", None)
        if use_shadow and shadow_store is not None:
            read_span = getattr(shadow_store, "read_bytes_span", None)
            st = shadow_store
        else:
            st = store
        if span_len > 0 and read_span is not None:
            return read_span(base, span_len)
        return None

    def _decode_layer(raw: bytes | None, store) -> None:
        if use_shadow and raw is not None:
            from rwkv_ssd.runtime.decode_shadow import decode_shadow_layer_from_span

            decode_shadow_layer_from_span(raw, entries, base, torch.device("cpu"))
            return
        if trinity_layer and raw is not None:
            cache = LayerZlibCache()
            decode_trinity_layer_span(raw, entries, cache, torch.device("cpu"))
            return
        if raw is not None and span is not None and codecs <= {"trinity_lut2"}:
            from rwkv_ssd.runtime.trinity_codec import decode_lut2_layer_from_span

            decode_lut2_layer_from_span(raw, entries, base, torch.device("cpu"))
            return
        if raw is not None and span is not None:
            for entry in entries:
                rel = entry.offset - base
                decode_weight_to_tensor(
                    raw[rel : rel + entry.length], entry, torch.device("cpu")
                )
            return
        for entry in entries:
            decode_weight_to_tensor(
                store.read_bytes(entry), entry, torch.device("cpu")
            )

    shadow_path = manifest.shadow_path()
    with open_weight_store(manifest.weights_path, backend="mmap") as store:
        shadow_cm = (
            open_weight_store(shadow_path, backend="mmap")
            if use_shadow and shadow_path
            else nullcontext(None)
        )
        with shadow_cm as shadow_store:
            for _ in range(2):
                _decode_layer(_read_layer(store, shadow_store), store)

            t_read = t_dec = 0.0
            for _ in range(iters):
                t0 = time.perf_counter()
                raw = _read_layer(store, shadow_store)
                if raw is None:
                    for entry in entries:
                        store.read_bytes(entry)
                t_read += time.perf_counter() - t0

                t0 = time.perf_counter()
                _decode_layer(raw, store)
                t_dec += time.perf_counter() - t0

    read_ms = (t_read / iters) * 1000
    dec_ms = (t_dec / iters) * 1000
    return {
        "layer_id": layer_id,
        "span_kb": round(span_len / 1024, 1),
        "read_ms": round(read_ms, 3),
        "decode_ms": round(dec_ms, 3),
        "total_ms": round(read_ms + dec_ms, 3),
        "tok_s_ceiling": round(1000.0 / max(read_ms + dec_ms, 0.001), 1),
        "shadow": use_shadow,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--layer", type=int, default=-1, help="block layer id (default: first)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--shadow", action="store_true", help="use shadow.bin if present")
    args = p.parse_args()
    r = _one_layer_ms(args.pack, args.layer, iters=args.iters, use_shadow=args.shadow)
    tag = " shadow" if r.get("shadow") else ""
    print(f"pack={args.pack.name} layer={r['layer_id']} span={r['span_kb']} KB{tag}")
    print(
        f"  read={r['read_ms']:.2f} ms  decode={r['decode_ms']:.2f} ms  "
        f"total={r['total_ms']:.2f} ms  (~{r['tok_s_ceiling']:.0f} tok/s ceiling from I/O+decode)"
    )


if __name__ == "__main__":
    main()
