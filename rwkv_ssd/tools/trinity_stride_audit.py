#!/usr/bin/env python3
"""Compare Trinity LUT2 decoded tensors vs FP16 pack after ``prepare_rwkv7_tensor_for_z``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.layer_keys import entries_for_layer
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.rwkv7_weights import prepare_rwkv7_tensor_for_z
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes
from rwkv_ssd.runtime.weight_store import open_weight_store


@dataclass
class TensorAudit:
    name: str
    shape_ok: bool
    stride_ok: bool
    contiguous: bool
    max_abs_diff: float
    fp16_shape: list[int]
    trinity_shape: list[int]


@dataclass
class LayerAudit:
    layer_id: int
    tensors: list[TensorAudit]
    ok: bool


def audit_layer(
    fp16_manifest: Manifest,
    trinity_manifest: Manifest,
    fp16_store,
    trinity_store,
    layer_id: int,
    *,
    weight_atol: float = 0.35,
    vector_atol: float = 10.0,
) -> LayerAudit:
    fp_n = int(fp16_manifest.meta.get("n_embd", 0))
    tr_n = int(trinity_manifest.meta.get("n_embd", 0))
    if fp_n and tr_n and fp_n != tr_n:
        raise ValueError(
            f"pack size mismatch: fp16 n_embd={fp_n} vs trinity n_embd={tr_n}"
        )
    fp_entries = entries_for_layer(fp16_manifest.tensors, layer_id)
    tr_entries = entries_for_layer(trinity_manifest.tensors, layer_id)
    tr_by_name = {e.name: e for e in tr_entries}
    rows: list[TensorAudit] = []
    for entry in fp_entries:
        if entry.name not in tr_by_name:
            continue
        tr_entry = tr_by_name[entry.name]
        fp_raw = fp16_store.read_bytes(entry)
        tr_raw = trinity_store.read_bytes(tr_entry)
        fp_t = prepare_rwkv7_tensor_for_z(
            entry.name,
            tensor_from_bytes(fp_raw, entry, torch.device("cpu")),
        )
        tr_t = prepare_rwkv7_tensor_for_z(
            tr_entry.name,
            decode_weight_to_tensor(tr_raw, tr_entry, torch.device("cpu")),
        )
        shape_ok = list(fp_t.shape) == list(tr_t.shape)
        stride_ok = fp_t.stride() == tr_t.stride()
        contiguous = tr_t.is_contiguous()
        diff = (fp_t.float() - tr_t.float()).abs().max().item() if shape_ok else float("inf")
        is_weight = any(
            suffix in entry.name
            for suffix in (
                "key.weight",
                "value.weight",
                "receptance.weight",
                "output.weight",
                "head.weight",
            )
        )
        atol = weight_atol if is_weight else vector_atol
        close = shape_ok and diff <= atol
        rows.append(
            TensorAudit(
                name=entry.name,
                shape_ok=shape_ok,
                stride_ok=stride_ok,
                contiguous=contiguous,
                max_abs_diff=round(diff, 6),
                fp16_shape=list(fp_t.shape),
                trinity_shape=list(tr_t.shape),
            )
        )
        if not close:
            rows[-1].max_abs_diff = diff
    ok = all(r.shape_ok and r.stride_ok for r in rows) and len(rows) > 0

    def _is_matmul_weight(name: str) -> bool:
        return any(
            suffix in name
            for suffix in (
                "key.weight",
                "value.weight",
                "receptance.weight",
                "output.weight",
                "head.weight",
            )
        )

    quant_ok = all(
        (
            r.max_abs_diff <= weight_atol
            if _is_matmul_weight(r.name)
            else r.max_abs_diff <= vector_atol
        )
        for r in rows
    )
    return LayerAudit(layer_id=layer_id, tensors=rows, ok=ok and quant_ok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp16", type=Path, required=True, help="FP16 grouped pack dir")
    p.add_argument("--trinity", type=Path, required=True, help="Trinity LUT2 pack dir")
    p.add_argument("--layers", type=str, default="0,3,6", help="Comma-separated layer ids")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    fp16_m = Manifest.load(args.fp16)
    tr_m = Manifest.load(args.trinity)
    fp16_store = open_weight_store(fp16_m.weights_path)
    tr_store = open_weight_store(tr_m.weights_path)
    audits: list[LayerAudit] = []
    try:
        for lid in layer_ids:
            audit = audit_layer(fp16_m, tr_m, fp16_store, tr_store, lid)
            audits.append(audit)
            status = "OK" if audit.ok else "FAIL"
            bad_stride = sum(1 for t in audit.tensors if not t.stride_ok)
            print(
                f"layer {lid}: {status}  tensors={len(audit.tensors)}  "
                f"stride_mismatch={bad_stride}"
            )
            for t in audit.tensors:
                if not t.stride_ok or t.max_abs_diff > 0.05:
                    print(
                        f"  {t.name}: diff={t.max_abs_diff:.4f} "
                        f"stride_ok={t.stride_ok} shape={t.trinity_shape}"
                    )
    finally:
        fp16_store.close()
        tr_store.close()

    if args.json_out:
        payload = [asdict(a) for a in audits]
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0 if all(a.ok for a in audits) else 1


if __name__ == "__main__":
    sys.exit(main())
