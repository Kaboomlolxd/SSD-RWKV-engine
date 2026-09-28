#!/usr/bin/env python3
"""
Shard an existing pack across multiple SSD-targeted files (M-class).

Use case: split a single ``weights.bin`` into K files so each can be
placed on a separate physical SSD. The sharded weight store
(``rwkv_ssd.runtime.weight_store_sharded``) reads them in parallel
through a thread pool, multiplying aggregate SSD bandwidth by up to K.

For 0.1B-class packs (49 MB) the entire pack fits in RAM and SSD
parallelism doesn't help. For 7B+ packs (~14 GB) where the SSD is the
bottleneck, sharding with K=2-4 SSDs gives 2-4x aggregate read
bandwidth — close to halving the per-token I/O cost on the F2 / F5 /
F6 paths.

Usage:
  python -m rwkv_ssd.tools.shard_pack --input test_model/trinity_eval/trinity_grouped_0.1b --shards 4
  python -m rwkv_ssd.tools.shard_pack --input /path/to/pack --shards 2 --output /path/to/sharded

The tool reads the existing pack, splits the tensors round-robin
across the requested number of shards, and writes a new pack
directory with the same manifest but ``weights_files`` listing the
shards and a ``shard_file`` per tensor.

The original pack is left untouched; pass ``--in-place`` to
overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rwkv_ssd.runtime.manifest import Manifest


def shard_pack(
    src: Path,
    dst: Path,
    n_shards: int,
    *,
    in_place: bool = False,
    strategy: str = "layer",
    stripe_bytes: int = 64 * 1024 * 1024,
) -> dict:
    """Split a pack into ``n_shards`` files; return a stats dict.

    Each tensor is assigned to shard ``layer_id % n_shards`` (so all
    tensors for the same layer end up in the same shard — important
    for layer-span decode which reads the whole layer from one file
    with a single pread). Global tensors (no ``layer_id``) are distributed
    round-robin so a large embedding/head pair does not make shard 0 the
    bottleneck.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    strategy = strategy.strip().lower()
    if strategy not in {"layer", "stripe"}:
        raise ValueError("strategy must be 'layer' or 'stripe'")
    if stripe_bytes <= 0:
        raise ValueError("stripe_bytes must be positive")
    manifest = Manifest.load(src)
    if manifest.is_sharded():
        raise ValueError(
            f"pack is already sharded ({len(manifest.shard_files)} shards)"
        )
    if str(manifest.meta.get("weights_compression", "")).strip().lower() == "zstd":
        raise ValueError(
            "sharding compressed zstd packs is not supported: tensor offsets "
            "refer to the decompressed image; decompress/repack first"
        )
    if strategy == "stripe" and any(entry.fast_stripes for entry in manifest.tensors):
        raise ValueError("pack already contains striped BF16 shadow extents")

    # Resolve output directory.
    if in_place:
        out_dir = src
    else:
        out_dir = dst
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocate shard files.
    shard_names: list[str] = []
    shard_paths: list[Path] = []
    for i in range(n_shards):
        name = f"weights.shard.{i}.bin"
        shard_names.append(name)
        shard_paths.append(out_dir / name)
    # Open the original weights for reading (mmap via simple open).
    shadow_source = manifest.shadow_path()
    shadow_paths_out: list[Path] = []
    with open(manifest.weights_path, "rb") as src_f:
        shadow_src_f = open(shadow_source, "rb") if shadow_source is not None else None
        # Open each shard for writing.
        shard_fhs = [open(p, "wb") for p in shard_paths]
        shadow_fhs = []
        shadow_names: list[str] = []
        shadow_offsets: list[int] = [0] * n_shards
        if strategy == "stripe" and manifest.has_bf16_shadow():
            shadow_names = [f"shadow.shard.{i}.bin" for i in range(n_shards)]
            shadow_paths_out = [out_dir / name for name in shadow_names]
            shadow_fhs = [open(path, "wb") for path in shadow_paths_out]
        try:
            # Walk tensors in manifest order; assign by layer_id.
            # Globals (layer_id < 0) are round-robin across shards so
            # shard 0 is not 2x the size of the others (1 GB of
            # emb.weight + head.weight on a 7B model would otherwise
            # make shard 0 the bottleneck).
            updated_tensors: list[dict] = []
            shard_offsets: list[int] = [0] * n_shards
            shard_bytes: list[int] = [0] * n_shards
            global_rr = 0

            for entry in manifest.tensors:
                if entry.layer_id < 0:
                    shard_id = global_rr % n_shards
                    global_rr += 1
                else:
                    shard_id = entry.layer_id % n_shards
                # Read the tensor bytes from the source.
                src_f.seek(entry.offset)
                payload = src_f.read(entry.length)
                if len(payload) != entry.length:
                    raise IOError(
                        f"short read: wanted {entry.length} bytes for "
                        f"{entry.name!r}, got {len(payload)}"
                    )
                # Preserve each tensor's alignment in the new shard.  Apart
                # from making verification quiet, this keeps layer-span reads
                # aligned for direct I/O capable filesystems.
                alignment = max(1, int(entry.alignment))
                stripe_extents: list[dict[str, int | str]] = []
                if strategy == "stripe" and len(payload) > stripe_bytes:
                    # Round-robin physical extents.  Each extent is aligned
                    # independently; logical offsets reconstruct the original
                    # tensor without requiring a contiguous layer span.
                    logical = 0
                    part = 0
                    while logical < len(payload):
                        n = min(stripe_bytes, len(payload) - logical)
                        sid = (entry.layer_id if entry.layer_id >= 0 else global_rr) + part
                        sid %= n_shards
                        current = shard_offsets[sid]
                        physical = (current + alignment - 1) // alignment * alignment
                        if physical > current:
                            shard_fhs[sid].write(b"\x00" * (physical - current))
                        shard_fhs[sid].write(payload[logical : logical + n])
                        physical_length = (
                            (n + alignment - 1) // alignment * alignment
                        )
                        if physical_length > n:
                            shard_fhs[sid].write(b"\x00" * (physical_length - n))
                        shard_offsets[sid] = physical + physical_length
                        shard_bytes[sid] = shard_offsets[sid]
                        stripe_extents.append(
                            {
                                "shard_file": shard_names[sid],
                                "offset": physical,
                                "length": n,
                                "logical_offset": logical,
                                "physical_length": physical_length,
                            }
                        )
                        logical += n
                        part += 1
                    offset_in_shard = 0
                    shard_id = 0
                else:
                    target_fh = shard_fhs[shard_id]
                    current = shard_offsets[shard_id]
                    offset_in_shard = (current + alignment - 1) // alignment * alignment
                    if offset_in_shard > current:
                        target_fh.write(b"\x00" * (offset_in_shard - current))
                    target_fh.write(payload)
                    shard_offsets[shard_id] = offset_in_shard + len(payload)
                    shard_bytes[shard_id] = shard_offsets[shard_id]
                fast_stripe_extents: list[dict[str, int | str]] = []
                fast_shard_file = entry.fast_shard_file
                if entry.fast_stripes:
                    # Re-sharding an already striped shadow is intentionally
                    # rejected below; retaining this field keeps the layer
                    # strategy lossless for future manifests.
                    fast_stripe_extents = [dict(item) for item in entry.fast_stripes]
                elif strategy == "stripe" and entry.fast_offset >= 0 and entry.fast_length > 0:
                    if shadow_src_f is None or not shadow_fhs:
                        raise RuntimeError("BF16 shadow metadata has no readable source file")
                    shadow_src_f.seek(entry.fast_offset)
                    shadow_payload = shadow_src_f.read(entry.fast_length)
                    if len(shadow_payload) != entry.fast_length:
                        raise IOError(
                            f"short shadow read: wanted {entry.fast_length} bytes for "
                            f"{entry.name!r}, got {len(shadow_payload)}"
                        )
                    logical = 0
                    part = 0
                    alignment = max(1, int(entry.alignment))
                    while logical < len(shadow_payload):
                        n = min(stripe_bytes, len(shadow_payload) - logical)
                        sid = (entry.layer_id if entry.layer_id >= 0 else global_rr) + part
                        sid %= n_shards
                        current = shadow_offsets[sid]
                        physical = (current + alignment - 1) // alignment * alignment
                        if physical > current:
                            shadow_fhs[sid].write(b"\x00" * (physical - current))
                        shadow_fhs[sid].write(shadow_payload[logical : logical + n])
                        physical_length = (n + alignment - 1) // alignment * alignment
                        if physical_length > n:
                            shadow_fhs[sid].write(b"\x00" * (physical_length - n))
                        shadow_offsets[sid] = physical + physical_length
                        fast_stripe_extents.append(
                            {
                                "shard_file": shadow_names[sid],
                                "offset": physical,
                                "length": n,
                                "logical_offset": logical,
                                "physical_length": physical_length,
                            }
                        )
                        logical += n
                        part += 1
                # Build the manifest record (keep all existing fields).
                updated_tensors.append(
                    {
                        "name": entry.name,
                        "layer_id": entry.layer_id,
                        "dtype": entry.dtype,
                        "shape": list(entry.shape),
                        "offset": offset_in_shard,
                        "length": entry.length,
                        "alignment": entry.alignment,
                        "residency": entry.residency,
                        "dequant": entry.dequant,
                        "inner_offset": entry.inner_offset,
                        "inner_length": entry.inner_length,
                        "fast_offset": entry.fast_offset,
                        "fast_length": entry.fast_length,
                        "fast_shard_file": (
                            "" if fast_stripe_extents else fast_shard_file
                        ),
                        "fast_stripes": fast_stripe_extents,
                        "shard_file": "" if stripe_extents else shard_names[shard_id],
                        "stripes": stripe_extents,
                    }
                )
        finally:
            for fh in shard_fhs:
                fh.close()
            for fh in shadow_fhs:
                fh.close()
            if shadow_src_f is not None:
                shadow_src_f.close()

    # Build the updated manifest.
    manifest_path = out_dir / "manifest.json"
    source_manifest_path = src / "manifest.json"
    raw = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    raw["version"] = 2 if strategy == "stripe" else int(raw.get("version", 1))
    raw["model_family"] = raw.get("model_family", "rwkv")
    raw["weights_file"] = shard_names[0]
    raw["weights_files"] = shard_names
    raw["tensors"] = updated_tensors
    raw.setdefault("meta", {})
    raw["meta"]["n_shards"] = n_shards
    raw["meta"]["shard_strategy"] = (
        "striped_round_robin" if strategy == "stripe" else "layer_id % n_shards"
    )
    if strategy == "stripe":
        raw["meta"]["stripe_bytes"] = stripe_bytes
        raw["meta"]["shadow_strategy"] = (
            "striped_round_robin" if shadow_names else "absent"
        )
        if shadow_names:
            raw["meta"]["shadow_files"] = shadow_names
            raw["meta"]["shadow_file"] = shadow_names[0]
    old_hash = raw["meta"].pop("weights_sha256", None)
    if old_hash:
        raw["meta"]["source_weights_sha256"] = old_hash
    raw["meta"]["weights_sha256_by_file"] = {
        name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        for name in shard_names
    }
    # Per-shard sizes for the bench.
    raw["meta"]["shard_bytes"] = shard_bytes
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # Copy the shadow sidecar(s) verbatim for layer-affinity packs.  Stripe
    # packs already wrote the sidecar extents above.
    if strategy != "stripe":
        for shadow_path in manifest.shadow_paths():
            try:
                shadow_name = shadow_path.relative_to(src).as_posix()
            except ValueError:
                shadow_name = shadow_path.name
            target = out_dir / shadow_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if shadow_path != target:
                shutil.copy2(shadow_path, target)

    # Copy meta.json if present.
    src_meta = src / "meta.json"
    if src_meta.is_file() and src_meta != out_dir / "meta.json":
        shutil.copy2(src_meta, out_dir / "meta.json")

    return {
        "n_shards": n_shards,
        "shard_bytes": shard_bytes,
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source pack directory (containing manifest.json + weights.bin)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pack directory (defaults to <input>_sharded_N)",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=2,
        help="Number of shard files to create (default 2)",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input pack (creates the shard files next to the original weights.bin)",
    )
    ap.add_argument(
        "--strategy",
        choices=("layer", "stripe"),
        default="layer",
        help="layer affinity (legacy) or striped tensor extents (manifest-v2)",
    )
    ap.add_argument(
        "--stripe-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="logical bytes per stripe when --strategy stripe (default 64 MiB)",
    )
    args = ap.parse_args()

    src = args.input.resolve()
    if not (src / "manifest.json").is_file():
        raise SystemExit(f"manifest.json not found in {src}")
    if args.in_place:
        out = src
    elif args.output is not None:
        out = args.output.resolve()
    else:
        out = src.parent / f"{src.name}_sharded_{args.shards}"
    if (out / "weights.bin").is_file() and not args.in_place:
        raise SystemExit(
            f"{out / 'weights.bin'} exists — pass --in-place to overwrite "
            f"or pick a different --output"
        )

    stats = shard_pack(
        src,
        out,
        args.shards,
        in_place=args.in_place,
        strategy=args.strategy,
        stripe_bytes=args.stripe_bytes,
    )
    print(f"sharded {stats['n_shards']} files into {stats['out_dir']}")
    for i, n in enumerate(stats["shard_bytes"]):
        print(f"  weights.shard.{i}.bin: {n} bytes ({n / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
