"""Safely apply a reviewed heterogeneous-drive layer placement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from rwkv_ssd.runtime.manifest import Manifest


def _safe_drive_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return safe.strip("_") or "drive"


def repack_storage_placement(src: Path, dst: Path, plan: dict) -> dict:
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"output directory is not empty: {dst}")
    manifest = Manifest.load(src)
    if manifest.is_sharded():
        raise ValueError("placement repack currently requires a single-file source pack")
    mapping = {
        int(item["layer_id"]): str(item["drive"])
        for item in plan.get("placements", [])
    }
    missing = sorted(set(manifest.by_layer()) - set(mapping))
    if missing:
        raise ValueError(f"placement plan is missing layers: {missing}")
    drive_names = list(dict.fromkeys(mapping[layer] for layer in sorted(mapping)))
    dst.mkdir(parents=True, exist_ok=True)
    shard_names = {
        drive: f"weights.{index}.{_safe_drive_name(drive)}.bin"
        for index, drive in enumerate(drive_names)
    }
    tmp_paths = {drive: dst / (name + ".tmp") for drive, name in shard_names.items()}
    handles = {drive: path.open("wb") for drive, path in tmp_paths.items()}
    offsets = {drive: 0 for drive in drive_names}
    updated = []
    try:
        with manifest.weights_path.open("rb") as source:
            for entry in manifest.tensors:
                drive = mapping[entry.layer_id]
                alignment = max(1, int(entry.alignment))
                offset = (offsets[drive] + alignment - 1) // alignment * alignment
                if offset > offsets[drive]:
                    handles[drive].write(b"\x00" * (offset - offsets[drive]))
                source.seek(entry.offset)
                payload = source.read(entry.length)
                if len(payload) != entry.length:
                    raise OSError(f"short source read for {entry.name}")
                handles[drive].write(payload)
                offsets[drive] = offset + len(payload)
                row = {
                    "name": entry.name,
                    "layer_id": entry.layer_id,
                    "dtype": entry.dtype,
                    "shape": list(entry.shape),
                    "offset": offset,
                    "length": entry.length,
                    "alignment": entry.alignment,
                    "residency": entry.residency,
                    "dequant": entry.dequant,
                    "inner_offset": entry.inner_offset,
                    "inner_length": entry.inner_length,
                    "fast_offset": entry.fast_offset,
                    "fast_length": entry.fast_length,
                    "fast_shard_file": entry.fast_shard_file,
                    "fast_stripes": list(entry.fast_stripes),
                    "shard_file": shard_names[drive],
                    "stripes": [],
                }
                updated.append(row)
    finally:
        for handle in handles.values():
            handle.close()
    for drive, tmp in tmp_paths.items():
        os.replace(tmp, dst / shard_names[drive])

    raw = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    raw["weights_file"] = shard_names[drive_names[0]]
    raw["weights_files"] = [shard_names[drive] for drive in drive_names]
    raw["tensors"] = updated
    raw.setdefault("meta", {})
    raw["meta"].pop("weights_sha256", None)
    raw["meta"]["shard_strategy"] = "heterogeneous_cost_optimized"
    raw["meta"]["placement_drives"] = drive_names
    raw["meta"]["weights_sha256_by_file"] = {
        shard_names[drive]: hashlib.sha256(
            (dst / shard_names[drive]).read_bytes()
        ).hexdigest()
        for drive in drive_names
    }
    manifest_tmp = dst / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, dst / "manifest.json")
    for name in ("meta.json",):
        source = src / name
        if source.is_file():
            shutil.copy2(source, dst / name)
    for shadow in manifest.shadow_paths():
        target = dst / shadow.name
        if shadow != target:
            shutil.copy2(shadow, target)

    verified = Manifest.load(dst)
    for original, placed in zip(manifest.tensors, verified.tensors):
        with manifest.weights_path.open("rb") as source:
            source.seek(original.offset)
            expected = source.read(original.length)
        shard = dst / placed.shard_file
        with shard.open("rb") as handle:
            handle.seek(placed.offset)
            actual = handle.read(placed.length)
        if actual != expected:
            raise ValueError(f"repacked tensor verification failed: {original.name}")
    return {
        "output": str(dst),
        "drives": drive_names,
        "shard_files": [shard_names[drive] for drive in drive_names],
        "shard_bytes": {drive: offsets[drive] for drive in drive_names},
        "tensors_verified": len(verified.tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = repack_storage_placement(
        args.input, args.output, json.loads(args.plan.read_text(encoding="utf-8"))
    )
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
