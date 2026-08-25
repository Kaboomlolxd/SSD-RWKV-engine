#!/usr/bin/env python3
"""Add shadow.bin + fast_offset fields to an existing Trinity pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_ssd.runtime.bf16_shadow import SHADOW_FILENAME, write_bf16_shadow
from rwkv_ssd.runtime.checkpoint_meta import load_checkpoint_tensors
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_layout import sort_tensor_names
from rwkv_ssd.tools.pack_runtime import _layer_id_from_name


def add_shadow(
    pack_dir: Path,
    checkpoint: Path | None,
    *,
    shadow_min_numel: int = 0,
    quiet: bool = False,
) -> None:
    manifest = Manifest.load(pack_dir)
    meta_path = pack_dir / "manifest.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors_meta: list[dict] = raw["tensors"]

    ckpt = checkpoint
    if ckpt is None:
        src = manifest.meta.get("source_checkpoint")
        if not src:
            raise SystemExit("provide --checkpoint or pack meta.source_checkpoint")
        candidate = Path(str(src))
        # New packs store only a portable basename.  Resolve it beside the
        # pack when a caller deliberately stages the source there, while
        # retaining support for legacy packs that recorded an absolute path.
        if not candidate.is_absolute():
            beside_pack = pack_dir / candidate
            if beside_pack.exists():
                candidate = beside_pack
        if not candidate.exists():
            raise SystemExit(
                "pack metadata contains only portable checkpoint provenance; "
                "provide --checkpoint with the source model"
            )
        ckpt = candidate
    state = load_checkpoint_tensors(ckpt)
    layout = manifest.meta.get("pack_layout", "layer_grouped")
    names = sort_tensor_names(list(state.keys()), layout, _layer_id_from_name)
    sector = int(manifest.meta.get("sector_bytes", 0))
    write_bf16_shadow(
        state,
        names,
        pack_dir,
        tensors_meta,
        layer_id_fn=_layer_id_from_name,
        sector_bytes=sector,
        shadow_min_numel=shadow_min_numel,
        quiet=quiet,
    )
    raw["tensors"] = tensors_meta
    raw["meta"]["shadow_file"] = SHADOW_FILENAME
    meta_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    if not quiet:
        print(f"Updated {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Add bf16 shadow.bin to a Trinity pack")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--shadow-min-numel",
        type=int,
        default=0,
        help="only shadow tensors with numel >= N (0=all; try 4096 for weight mats only)",
    )
    args = p.parse_args()
    add_shadow(args.pack, args.checkpoint, shadow_min_numel=args.shadow_min_numel)


if __name__ == "__main__":
    main()
