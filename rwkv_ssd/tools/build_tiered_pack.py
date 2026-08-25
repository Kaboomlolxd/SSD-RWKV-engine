"""
Build a tiered Trinity pack: hot block layers as FP16, cold layers as trinity_lut2.

Hot layers decode at skeleton load (native matmul). Cold layers stream with fused LUT
or ``.decode_cache/`` on revisit.

Example (0.1B hot3):

  python -m rwkv_ssd.tools.build_tiered_pack \\
    --input test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth \\
    --output test_model/trinity_eval/trinity_tiered_hot3_0.1b \\
    --resident-layers 0,1,2 \\
    --pack-layout layer_grouped
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rwkv_ssd.tools.pack_runtime import (
    ALIGNMENT,
    _layer_id_from_name,
    _load_state_dict,
    _tensor_to_bytes,
    _write_manifest,
    align_offset,
    pad_between_layers,
    sort_tensor_names,
)
from rwkv_ssd.runtime.trinity_codec import encode_trinity_lut2


def _parse_layer_ids(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if part:
            out.add(int(part))
    return out


def pack_tiered(
    input_path: Path,
    output_dir: Path,
    *,
    resident_layer_ids: set[int],
    model_family: str = "rwkv7",
    pack_layout: str = "layer_grouped",
    sector_bytes: int = 262144,
    hash_weights: bool = True,
    hf_repo: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state_dict(input_path)
    layout = pack_layout.strip().lower()
    names = sort_tensor_names(list(state.keys()), layout, _layer_id_from_name)

    tensors_meta: list[dict] = []
    offset = 0
    weights_path = output_dir / "weights.bin"
    prev_layer_id: int | None = None

    with weights_path.open("wb") as out:
        for name in names:
            layer_id = _layer_id_from_name(name)
            if prev_layer_id is not None and layer_id != prev_layer_id:
                offset = pad_between_layers(
                    offset, prev_layer_id, layer_id, sector_bytes
                )
                if offset > out.tell():
                    out.write(b"\x00" * (offset - out.tell()))
            prev_layer_id = layer_id

            t = state[name].contiguous()
            hot = layer_id in resident_layer_ids
            if hot:
                raw = _tensor_to_bytes(t)
                dequant = "none"
            else:
                raw = encode_trinity_lut2(t)
                dequant = "trinity_lut2"

            offset = align_offset(offset, ALIGNMENT)
            if offset > out.tell():
                out.write(b"\x00" * (offset - out.tell()))
            start = out.tell()
            out.write(raw)
            length = len(raw)
            tensors_meta.append(
                {
                    "name": name,
                    "layer_id": layer_id,
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "offset": start,
                    "length": length,
                    "alignment": ALIGNMENT,
                    "residency": "resident" if hot else "streamed",
                    "dequant": dequant,
                }
            )
            offset = start + length

    _write_manifest(
        output_dir,
        weights_path,
        tensors_meta,
        model_family,
        "tiered_lut2",
        layout,
        sector_bytes,
        state,
        hf_repo,
        hash_weights,
        input_path,
        meta_extra={"tiered_resident_layer_ids": sorted(resident_layer_ids)},
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Tiered FP16 hot + LUT2 cold pack")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--resident-layers",
        required=True,
        help="Comma-separated block layer ids for FP16 resident (e.g. 0,1,2)",
    )
    p.add_argument("--pack-layout", default="layer_grouped")
    p.add_argument("--sector-bytes", type=int, default=262144)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--no-hash", action="store_true")
    args = p.parse_args()
    pack_tiered(
        args.input,
        args.output,
        resident_layer_ids=_parse_layer_ids(args.resident_layers),
        pack_layout=args.pack_layout,
        sector_bytes=args.sector_bytes,
        hash_weights=not args.no_hash,
        hf_repo=args.hf_repo,
    )
    print(f"tiered pack written to {args.output}")


if __name__ == "__main__":
    main()
