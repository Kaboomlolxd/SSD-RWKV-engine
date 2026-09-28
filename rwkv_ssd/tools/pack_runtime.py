#!/usr/bin/env python3
"""
Pack checkpoint tensors into weights.bin + manifest.json for streaming runtime.

Supports:
  - PyTorch .pth / .pt (state_dict or raw dict)
  - safetensors single file
  - Hugging Face repo ID (--hf-repo) downloads a .pth if huggingface_hub is installed

Usage:
  python -m rwkv_ssd.tools.pack_runtime --input model.pth --output ./runtime_pack
  python -m rwkv_ssd.tools.pack_runtime --hf-repo BlinkDL/rwkv-0.4 --output ./runtime_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import ALIGNMENT
from rwkv_ssd.runtime.pack_codec import (
    encode_scale_u4,
    encode_scale_u8,
    encode_scale_u8_grouped,
)
from rwkv_ssd.runtime.pack_bench import pack_full_stats
from rwkv_ssd.runtime.safetensors_loader import (
    download_hf_repo,
    is_safetensors_dir,
    is_safetensors_file,
    load_safetensors_dir,
    load_safetensors_file,
    looks_like_hf_dir,
)
from rwkv_ssd.runtime.trinity_codec import (
    encode_trinity,
    encode_trinity_layer_bundle,
    encode_trinity_lut2,
)
from rwkv_ssd.runtime.pack_layout import (
    DEFAULT_SECTOR_BYTES,
    align_offset,
    pad_between_layers,
    sort_tensor_names,
)


def _compute_pack_composition(pack_dir: Path) -> dict:
    """Honest byte composition of a pack on disk (B3).

    Mirrors :func:`pack_full_stats` but only the on-disk byte counts (no I/O
    timing). Written into ``meta.json`` at build time so a future
    "why is my pack 600 MB" question has a one-line answer.
    """
    return pack_full_stats(pack_dir)


def _require_zstd():
    """Import the optional zstandard dependency with an actionable error."""
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "zstandard compression requires the optional dependency; install "
            "with `pip install 'rwkv-ssd[zstd]'`"
        ) from exc
    return zstd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_source(path: Path) -> str:
    """Hash a checkpoint file or a deterministic HF/sharded input directory."""
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint directory is empty: {path}")
    digest = hashlib.sha256()
    root = path.resolve()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as source:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _finalize_weight_compression(
    output_dir: Path, compression: str, *, quiet: bool = False
) -> None:
    """Replace the logical pack with an optional cold-storage zstd frame.

    Tensor offsets stay relative to the decompressed image.  The manifest
    records both sizes, while ``weights_sha256`` continues to protect the
    physical file that is actually distributed.
    """
    codec = compression.strip().lower()
    if codec in ("", "none", "off"):
        return
    if codec != "zstd":
        raise ValueError("compression must be 'none' or 'zstd'")

    zstd = _require_zstd()
    raw_path = output_dir / "weights.bin"
    if not raw_path.is_file():
        raise FileNotFoundError(f"cannot compress missing pack: {raw_path}")
    logical_size = raw_path.stat().st_size
    compressed_path = output_dir / "weights.bin.zst"
    temp_path = compressed_path.with_suffix(compressed_path.suffix + ".tmp")
    compressor = zstd.ZstdCompressor(level=3).compressobj(logical_size)
    # Stream through a size-aware frame so the runtime can preallocate one
    # logical buffer. This keeps pack creation bounded by the checkpoint reader
    # instead of loading a second full uncompressed image into Python memory.
    with raw_path.open("rb") as source, temp_path.open("wb") as target:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            encoded = compressor.compress(block)
            if encoded:
                target.write(encoded)
        tail = compressor.flush()
        if tail:
            target.write(tail)
    os.replace(temp_path, compressed_path)
    compressed_size = compressed_path.stat().st_size
    raw_path.unlink()

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    meta = manifest.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("manifest.json 'meta' must be an object")
    manifest["weights_file"] = "weights.bin.zst"
    meta["weights_compression"] = "zstd"
    meta["weights_uncompressed_bytes"] = logical_size
    meta["weights_compressed_bytes"] = compressed_size
    # ``total_bytes`` historically meant the backing file size.  Keep that
    # meaning and expose the logical size explicitly for offset validation.
    meta["total_bytes"] = compressed_size
    if meta.get("weights_sha256"):
        meta["weights_sha256"] = _sha256_file(compressed_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Keep the optional sidecar honest when it exists.  Its manifest metadata
    # remains authoritative, but operators commonly inspect meta.json alone.
    sidecar_path = output_dir / "meta.json"
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if isinstance(sidecar, dict):
            sidecar["weights_compression"] = "zstd"
            sidecar["weights_uncompressed_bytes"] = logical_size
            sidecar["weights_compressed_bytes"] = compressed_size
            composition = sidecar.get("pack_composition")
            if isinstance(composition, dict):
                delta_mb = (compressed_size - logical_size) / (1024 * 1024)
                composition["weights_bin_mb"] = round(
                    compressed_size / (1024 * 1024), 2
                )
                composition["weights_mb"] = round(
                    compressed_size / (1024 * 1024), 2
                )
                if isinstance(composition.get("total_mb"), (int, float)):
                    composition["total_mb"] = round(
                        float(composition["total_mb"]) + delta_mb, 2
                    )
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    if not quiet:
        ratio = logical_size / compressed_size if compressed_size else 0.0
        print(
            f"Compressed weights.bin -> weights.bin.zst: "
            f"{logical_size / (1024**2):.2f} -> "
            f"{compressed_size / (1024**2):.2f} MiB ({ratio:.2f}x)"
        )


from rwkv_ssd.runtime.checkpoint_meta import (
    infer_checkpoint_meta,
    load_checkpoint_tensors,
    load_hf_config,
)
from rwkv_ssd.runtime.deepembed import (
    DEEP_EMBED_QKV_DEA,
    detect_deepembed_variant,
    infer_deepembed_meta,
    write_deepembed_sidecar,
)


def _source_checkpoint_label(input_path: Path) -> str:
    """Return portable checkpoint provenance for pack metadata.

    Pack metadata is copied between machines and may be committed alongside
    release manifests.  An absolute build-host path is neither portable nor
    useful to a consumer who must provide the checkpoint explicitly.  Keep
    the final file/directory name as a human-readable lineage hint instead;
    legacy packs with absolute paths remain readable by the runtime.
    """
    label = input_path.name
    return label or input_path.as_posix()


def _align(offset: int, alignment: int = ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _resolve_input(
    input_path: Path | None,
    hf_repo: str | None,
    *,
    hf_prefer: str = "safetensors",
) -> tuple[Path, str | None]:
    """Resolve ``--input`` / ``--hf-repo`` to a local path.

    Returns ``(path, hf_repo_or_none)``. The returned ``path`` can be
    a single file OR a directory; ``_load_state_dict`` handles both.
    """
    if hf_repo:
        path = download_hf_repo(hf_repo, prefer=hf_prefer)
        return path, hf_repo
    if input_path is None:
        raise ValueError("provide --input or --hf-repo")
    return input_path, None


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a state_dict from a file, a HF model directory, or a sharded dir.

    Accepts:
      * ``*.pth`` / ``*.pt`` (RWKV legacy) — single file
      * ``*.safetensors`` — single file
      * a directory containing ``model.safetensors`` /
        ``model.safetensors.index.json`` (Hugging Face layout)
      * a directory containing any ``*.safetensors`` files (single or
        multi-shard, but multi-shard requires a HF index file)
    """
    if path.is_dir():
        if is_safetensors_dir(path) or looks_like_hf_dir(path):
            return load_safetensors_dir(path)
        # Last-ditch: try a .pth/.pt inside (some HF repos ship that).
        for ext in (".pth", ".pt", ".bin"):
            cand = path / f"pytorch_model{ext}"
            if cand.is_file():
                return load_checkpoint_tensors(cand)
        raise FileNotFoundError(
            f"directory {path} contains no safetensors or pytorch_model.*"
        )
    if is_safetensors_file(path):
        return load_safetensors_file(path)
    return load_checkpoint_tensors(path)


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _quality_first_codec(name: str, tensor: torch.Tensor, requested: str) -> str:
    """Conservative mixed precision for Trinity quality-sensitive tensors.

    Two-bit coding is most useful on large projection matrices.  Applying it
    to normalization vectors, time constants, embeddings, and the output head
    saves comparatively little or causes disproportionate recurrent drift.
    """
    if requested not in ("trinity_lut2", "trinity"):
        return requested
    if tensor.ndim != 2 or tensor.numel() < 4096:
        return "none"
    if name in ("emb.weight", "head.weight") or name.endswith(".head.weight"):
        return "scale_u8"
    return requested


def _native_safe_codec(
    name: str,
    requested: str,
    model_family: str = "rwkv7",
    *,
    tensor: torch.Tensor | None = None,
) -> str:
    """Choose the compact codec while protecting recurrent control tensors.

    RWKV-7's non-matrix norms/biases and control tensors are small enough that
    dense BF16 costs little in the layer-aligned pack, but grouped affine U8
    error in those tensors compounds through the recurrent state. Keep them
    lossless and reserve grouped U8 for matrices. This also protects the large
    embedding and output-head matrices from the much higher recurrent drift of
    the legacy LUT2 codebook path.
    """
    if model_family.strip().lower() == "rwkv7":
        safe_requests = ("trinity_lut2", "trinity", "scale_u8_grouped")
        if requested in safe_requests:
            if tensor is not None and tensor.ndim != 2:
                return "none"
            if requested in ("trinity_lut2", "trinity"):
                # A legacy LUT2 request is still allowed to produce the
                # native grouped-U8 matrix path, including emb/head.
                return "scale_u8_grouped"
            if tensor is not None and tensor.ndim == 2:
                return "scale_u8_grouped"
    return requested


def _pack_trinity_layer_grouped(
    state: dict[str, torch.Tensor],
    names: list[str],
    output_dir: Path,
    model_family: str,
    hf_repo: str | None,
    hash_weights: bool,
    sector_bytes: int,
    quiet: bool,
    input_path: Path,
    *,
    layer_compress: bool = True,
    bf16_shadow: bool = False,
    shadow_min_numel: int = 0,
    hf_meta_info: dict | None = None,
    compression: str = "none",
    checkpoint_sha256: str | None = None,
) -> None:
    """``trinity`` + layer_grouped: one layer blob per block (zlib or raw TCL)."""
    weights_path = output_dir / "weights.bin"
    tensors_meta: list[dict] = []
    offset = 0
    prev_layer_id: int | None = None
    layer_buf: list[tuple[str, torch.Tensor]] = []

    with weights_path.open("wb") as out:

        def flush_layer(layer_id: int) -> None:
            nonlocal offset
            if not layer_buf:
                return
            lut2_blobs = [encode_trinity_lut2(t) for _, t in layer_buf]
            disk_blob, spans = encode_trinity_layer_bundle(
                lut2_blobs, compress=layer_compress
            )
            offset = align_offset(offset, ALIGNMENT)
            if offset > out.tell():
                out.write(b"\x00" * (offset - out.tell()))
            start = out.tell()
            out.write(disk_blob)
            disk_len = len(disk_blob)
            for (name, t), (inner_off, inner_len) in zip(layer_buf, spans, strict=True):
                tensors_meta.append(
                    {
                        "name": name,
                        "layer_id": layer_id,
                        "dtype": str(t.dtype).replace("torch.", ""),
                        "shape": list(t.shape),
                        "offset": start,
                        "length": disk_len,
                        "inner_offset": inner_off,
                        "inner_length": inner_len,
                        "alignment": ALIGNMENT,
                        "residency": "streamed",
                        "dequant": "trinity_layer",
                    }
                )
            offset = start + disk_len
            layer_buf.clear()

        for name in names:
            layer_id = _layer_id_from_name(name)
            if prev_layer_id is not None and layer_id != prev_layer_id:
                offset = pad_between_layers(
                    offset, prev_layer_id, layer_id, sector_bytes
                )
                flush_layer(prev_layer_id)
            layer_buf.append((name, state[name].contiguous()))
            prev_layer_id = layer_id
        if layer_buf and prev_layer_id is not None:
            flush_layer(prev_layer_id)

    meta_extra = {"trinity_layer_compress": "zlib" if layer_compress else "raw"}
    if hf_meta_info:
        meta_extra["hf_metadata"] = hf_meta_info
    if bf16_shadow:
        from rwkv_ssd.runtime.bf16_shadow import SHADOW_FILENAME, write_bf16_shadow

        write_bf16_shadow(
            state,
            names,
            output_dir,
            tensors_meta,
            layer_id_fn=_layer_id_from_name,
            sector_bytes=sector_bytes,
            shadow_min_numel=shadow_min_numel,
            quiet=quiet,
        )
        meta_extra["shadow_file"] = SHADOW_FILENAME
    _write_manifest(
        output_dir,
        weights_path,
        tensors_meta,
        model_family,
        "trinity_layer",
        "layer_grouped",
        sector_bytes,
        state,
        hf_repo,
        hash_weights,
        input_path=input_path,
        meta_extra=meta_extra,
        checkpoint_sha256=checkpoint_sha256,
    )
    _finalize_weight_compression(output_dir, compression, quiet=quiet)
    if not quiet:
        final_path = output_dir / ("weights.bin.zst" if compression == "zstd" else "weights.bin")
        print(f"Packed {len(tensors_meta)} tensors (trinity_layer) -> {final_path}")
        print(f"Total size: {final_path.stat().st_size / (1024**2):.2f} MiB")


def _write_manifest(
    output_dir: Path,
    weights_path: Path,
    tensors_meta: list[dict],
    model_family: str,
    pack_codec: str,
    pack_layout: str,
    sector_bytes: int,
    state: dict[str, torch.Tensor],
    hf_repo: str | None,
    hash_weights: bool,
    input_path: Path | None = None,
    meta_extra: dict | None = None,
    checkpoint_sha256: str | None = None,
) -> None:
    meta_codec = {
        "pack_codec": pack_codec,
        "pack_layout": pack_layout,
        "sector_bytes": sector_bytes,
        "trinity_layer_bundled": pack_codec == "trinity_layer",
    }
    if meta_extra:
        meta_codec.update(meta_extra)
    weights_sha256 = (
        hashlib.sha256(weights_path.read_bytes()).hexdigest() if hash_weights else None
    )
    manifest = {
        "version": 1,
        "model_family": model_family,
        "weights_file": "weights.bin",
        "tensors": tensors_meta,
        "meta": {
            "tensor_count": len(tensors_meta),
            "total_bytes": weights_path.stat().st_size,
            **meta_codec,
        },
    }
    if weights_sha256:
        manifest["meta"]["weights_sha256"] = weights_sha256
    if hf_repo:
        manifest["meta"]["hf_repo_id"] = hf_repo
    hf_config = load_hf_config(input_path) if input_path is not None else None
    checkpoint_meta = infer_checkpoint_meta(state, config=hf_config)
    manifest["meta"].update(checkpoint_meta)
    if hf_config:
        manifest["meta"]["hf_model_type"] = str(hf_config.get("model_type", ""))
        architectures = hf_config.get("architectures")
        if architectures:
            manifest["meta"]["hf_architectures"] = architectures
    if input_path is not None:
        manifest["meta"]["source_checkpoint"] = _source_checkpoint_label(input_path)
        if checkpoint_sha256:
            manifest["meta"]["checkpoint_sha256"] = checkpoint_sha256
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    meta = {"model_family": model_family, **checkpoint_meta}
    if hf_config:
        meta["hf_model_type"] = str(hf_config.get("model_type", ""))
        architectures = hf_config.get("architectures")
        if architectures:
            meta["hf_architectures"] = architectures
    if input_path is not None:
        meta["source_checkpoint"] = _source_checkpoint_label(input_path)
        if checkpoint_sha256:
            meta["checkpoint_sha256"] = checkpoint_sha256
    if hf_repo:
        meta["hf_repo_id"] = hf_repo
    meta["pack_composition"] = _compute_pack_composition(output_dir)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _layer_id_from_name(name: str) -> int:
    """Group a tensor name into a layer id for streaming-order layout.

    Recognized families:

    * RWKV v5/v6/v7:
        ``blocks.{N}.*`` -> N
        ``emb.*`` / ``ln0.*`` / ``ln_out.*`` -> 0
        ``head.*`` -> 9999

    Returned ids are stable, contiguous-ish, and the engine already
    treats -1 (misc) as a single shared group, 0 as "first / always
    resident" and 9999 as "always resident" — same convention used by
    the existing RWKV pipeline.
    """
    if "blocks." in name:
        try:
            part = name.split("blocks.")[1].split(".")[0]
            return int(part)
        except (IndexError, ValueError):
            pass
    # RWKV globals.
    lower = name.lower()
    if name in {"emb.weight", "embed.weight"} or name.startswith(("emb.", "embed.")):
        return 0
    if name.startswith(("ln_out.", "ln0.")):
        return 0
    if lower.endswith("head.weight") or name.startswith("head."):
        return 9999
    return -1


def detect_model_family_from_state(
    state: dict[str, torch.Tensor],
) -> str:
    """Best-effort guess of the model family from tensor names.

    Used to set a sensible default for ``manifest.model_family`` when
    the user did not pass ``--model-family``. Falls back to ``"unknown"``
    so unsupported architectures are never presented as runnable packs.
    """
    if not state:
        return "unknown"
    keys = list(state.keys())
    if detect_deepembed_variant(state) is not None:
        return "rwkv7_deepembed"
    has_rwkv_blocks = any(k.startswith("blocks.") for k in keys)
    if has_rwkv_blocks:
        if any("att.x_r" in k or "att.x_w" in k for k in keys):
            return "rwkv7"
        if any("att.time_maa_x" in k for k in keys):
            return "rwkv6"
        return "rwkv5"
    if any("token_embd" in k or k.startswith("blk.") for k in keys):
        return "ggml"  # llama.cpp / GGML-style names
    return "unknown"


def _resolve_codec_for_tensor(
    name: str,
    codec_map: dict[str, str],
    default_codec: str,
    known_codecs: frozenset[str],
) -> str:
    exact = codec_map.get(name)
    if exact is not None and exact in known_codecs:
        return exact
    prefix = codec_map.get(f"{name.split('.weight')[0] if '.weight' in name else name}")
    if prefix is not None and prefix in known_codecs:
        return prefix
    for pattern, mapped in codec_map.items():
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            if mapped in known_codecs:
                return mapped
    return default_codec


def pack(
    input_path: Path,
    output_dir: Path,
    model_family: str = "rwkv7",
    hf_repo: str | None = None,
    hash_weights: bool = True,
    pack_codec: str = "none",
    pack_layout: str = "default",
    sector_bytes: int = 0,
    quiet: bool = False,
    *,
    trinity_layer_compress: bool = True,
    bf16_shadow: bool = False,
    shadow_min_numel: int = 0,
    codec_map: dict[str, str] | None = None,
    trinity_codebook: str = "groupwise_kmeans_residual",
    trinity_group_size: int = 128,
    trinity_quality_preset: str = "native_safe",
    activation_stats_path: Path | None = None,
    scale_group_size: int | None = 64,
    copy_hf_metadata: bool = False,
    hf_metadata_overwrite: bool = False,
    write_deepembed: bool = True,
    compression: str = "none",
) -> None:
    if codec_map is None:
        codec_map = {}
    codec = pack_codec.strip().lower()
    compression = compression.strip().lower()
    if compression not in ("none", "zstd"):
        raise ValueError("compression must be 'none' or 'zstd'")
    if compression == "zstd":
        _require_zstd()
    quality_preset = trinity_quality_preset.strip().lower()
    if quality_preset not in ("native_safe", "legacy", "balanced"):
        raise ValueError(
            "trinity_quality_preset must be native_safe, legacy, or balanced"
        )
    if trinity_group_size <= 0:
        raise ValueError("trinity_group_size must be positive")
    resolved_scale_group_size = int(scale_group_size or trinity_group_size)
    if resolved_scale_group_size <= 0:
        raise ValueError("scale_group_size must be positive")
    if codec not in ("none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity"):
        raise ValueError(
            f"unsupported pack_codec {pack_codec!r} "
            "(none, scale_u8, scale_u8_grouped, scale_u4, trinity_lut2, trinity)"
        )
    KNOWN_CODECS = frozenset(
        {"none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity"}
    )
    layout = pack_layout.strip().lower()
    if sector_bytes < 0:
        raise ValueError("sector_bytes must be >= 0")

    # Compute provenance before writing anything. This matters when an
    # operator places the output pack under an HF input directory: generated
    # pack artifacts must not become part of the source checkpoint identity.
    checkpoint_sha256 = _sha256_source(input_path) if hash_weights else None
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state_dict(input_path)
    deepembed_info: dict[str, object] = {}
    if detect_deepembed_variant(state) == DEEP_EMBED_QKV_DEA:
        if write_deepembed:
            deepembed_info = write_deepembed_sidecar(
                state,
                output_dir / "DeepEmbed.bin",
            )
        else:
            deepembed_info = {
                "path": "DeepEmbed.bin",
                "format": "rwkv_deepembed_mmap_v1",
                "sidecar_missing": True,
            }
    activation_rms: dict[str, torch.Tensor] = {}
    if activation_stats_path is not None:
        from rwkv_ssd.runtime.activation_calibration import load_activation_rms

        activation_rms = load_activation_rms(activation_stats_path)

    # Auto-detect model family if the CLI left the family at ``auto``. Keep
    # the historical ``rwkv7`` function default backwards-compatible: older
    # callers relied on it promoting DeepEmbed and non-RWKV tensor layouts,
    # while ordinary RWKV block layouts remain explicitly rwkv7 unless the
    # public CLI asks for ``auto``.
    requested_family = str(model_family).strip().lower()
    if requested_family in {"", "auto", "rwkv7"}:
        detected = detect_model_family_from_state(state)
        should_promote = requested_family in {"", "auto"} or (
            not any(k.startswith("blocks.") for k in state)
            or detected == "rwkv7_deepembed"
        )
        if detected != "unknown" and should_promote:
            model_family = detected
            if not quiet:
                print(f"Auto-detected model_family: {model_family}")

    # Optionally copy the HF metadata bundle (config.json, tokenizer
    # files, generation_config.json) into the pack dir so downstream
    # code can instantiate the model + tokenizer from the pack alone.
    # Skipped for plain .pth inputs (they have no sibling metadata).
    hf_meta_info: dict[str, dict[str, object]] = {}
    if copy_hf_metadata and input_path.is_dir():
        from rwkv_ssd.runtime.safetensors_loader import copy_hf_metadata as _copy_meta
        from rwkv_ssd.runtime.safetensors_loader import find_hf_metadata_files as _find_meta

        hf_meta_info = _copy_meta(
            input_path,
            output_dir,
            overwrite=hf_metadata_overwrite,
        )
        if not quiet:
            for name, info in hf_meta_info.items():
                tag = "copied" if info["copied"] else "kept"
                print(
                    f"  HF metadata [{info['kind']}]: {name} "
                    f"({int(info['size']):,} B, {tag})"
                )
    elif copy_hf_metadata and not input_path.is_dir():
        if not quiet:
            print(
                "  (--copy-hf-metadata: ignored; --input is a file, "
                "not a HF directory)"
            )

    names = sort_tensor_names(list(state.keys()), layout, _layer_id_from_name)
    if codec == "trinity" and layout == "layer_grouped":
        _pack_trinity_layer_grouped(
            state,
            names,
            output_dir,
            model_family,
            hf_repo,
            hash_weights,
            sector_bytes,
            quiet,
            input_path,
            layer_compress=trinity_layer_compress,
            bf16_shadow=bf16_shadow,
            shadow_min_numel=shadow_min_numel,
            hf_meta_info=hf_meta_info or None,
            compression=compression,
            checkpoint_sha256=checkpoint_sha256,
        )
        return

    tensors_meta = []
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
            tensor_codec = (
                _resolve_codec_for_tensor(name, codec_map, codec, KNOWN_CODECS)
                if codec_map
                else codec
            )
            if tensor_codec == codec:
                if quality_preset == "native_safe":
                    tensor_codec = _native_safe_codec(
                        name,
                        tensor_codec,
                        model_family=model_family,
                        tensor=t,
                    )
                elif quality_preset == "balanced":
                    tensor_codec = _quality_first_codec(name, t, tensor_codec)
            if tensor_codec == "scale_u8":
                raw = encode_scale_u8(t)
                dequant = "scale_u8"
            elif tensor_codec == "scale_u8_grouped":
                raw = encode_scale_u8_grouped(
                    t, group_size=resolved_scale_group_size
                )
                dequant = "scale_u8_grouped"
            elif tensor_codec == "scale_u4":
                raw = encode_scale_u4(t)
                dequant = "scale_u4"
            elif tensor_codec == "trinity_lut2":
                raw = encode_trinity_lut2(
                    t,
                    codebook=trinity_codebook,
                    group_size=trinity_group_size,
                    importance=activation_rms.get(name),
                )
                dequant = "trinity_lut2"
            elif tensor_codec == "trinity":
                raw = encode_trinity(t)
                dequant = "trinity"
            else:
                raw = _tensor_to_bytes(t)
                dequant = "none"
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
                    "residency": "streamed",
                    "dequant": dequant,
                }
            )
            offset = start + length

    meta_codec = {
        "pack_codec": codec,
        "pack_layout": layout,
        "sector_bytes": sector_bytes,
        "trinity_codebook": trinity_codebook,
        "trinity_group_size": (
            trinity_group_size
            if trinity_codebook.startswith("groupwise_")
            else 0
        ),
        "trinity_quality_preset": quality_preset,
        "activation_calibrated": bool(activation_rms),
        "scale_group_size": resolved_scale_group_size,
    }
    if deepembed_info:
        meta_codec["deepembed_sidecar"] = deepembed_info
    if codec_map:
        meta_codec["codec_map"] = codec_map
    if bf16_shadow and codec in ("trinity_lut2", "trinity"):
        from rwkv_ssd.runtime.bf16_shadow import SHADOW_FILENAME, write_bf16_shadow

        write_bf16_shadow(
            state,
            names,
            output_dir,
            tensors_meta,
            layer_id_fn=_layer_id_from_name,
            sector_bytes=sector_bytes,
            shadow_min_numel=shadow_min_numel,
            quiet=quiet,
        )
        meta_codec["shadow_file"] = SHADOW_FILENAME

    weights_sha256 = (
        hashlib.sha256(weights_path.read_bytes()).hexdigest() if hash_weights else None
    )

    manifest = {
        "version": 1,
        "model_family": model_family,
        "weights_file": "weights.bin",
        "tensors": tensors_meta,
        "meta": {
            "source_checkpoint": _source_checkpoint_label(input_path),
            "tensor_count": len(tensors_meta),
            "total_bytes": weights_path.stat().st_size,
            **meta_codec,
        },
    }
    if hf_repo:
        manifest["meta"]["hf_repo_id"] = hf_repo
    if weights_sha256:
        manifest["meta"]["weights_sha256"] = weights_sha256
    if checkpoint_sha256:
        manifest["meta"]["checkpoint_sha256"] = checkpoint_sha256
    if hf_meta_info:
        manifest["meta"]["hf_metadata"] = hf_meta_info

    ckpt_meta = infer_checkpoint_meta(
        state,
        config=load_hf_config(input_path) if input_path.is_dir() else None,
    )
    meta = {
        "source_checkpoint": _source_checkpoint_label(input_path),
        "model_family": model_family,
        **ckpt_meta,
    }
    if checkpoint_sha256:
        meta["checkpoint_sha256"] = checkpoint_sha256
    manifest["meta"].update(ckpt_meta)
    if hf_repo:
        meta["hf_repo_id"] = hf_repo
    if hf_meta_info:
        meta["hf_metadata"] = hf_meta_info
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _finalize_weight_compression(output_dir, compression, quiet=quiet)
    if not quiet:
        final_path = output_dir / ("weights.bin.zst" if compression == "zstd" else "weights.bin")
        print(f"Packed {len(tensors_meta)} tensors -> {final_path}")
        print(f"Total size: {final_path.stat().st_size / (1024**2):.2f} MiB")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pack RWKV weights for SSD streaming runtime"
    )
    p.add_argument("--input", help="Path to .pth / .pt / .safetensors")
    p.add_argument("--hf-repo", help="Hugging Face repo ID to download weights from")
    p.add_argument("--output", required=True, help="Output directory for runtime pack")
    p.add_argument(
        "--model-family",
        default="auto",
        help="RWKV architecture family (default: auto-detect; rwkv5/rwkv6/rwkv7).",
    )
    p.add_argument(
        "--pack-codec",
        default="none",
        choices=["none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity"],
        help="P2.b codec (one per deploy; trinity = LUT2 + zlib ANS stand-in)",
    )
    p.add_argument(
        "--trinity-codebook",
        default="groupwise_kmeans_residual",
        choices=[
            "linspace",
            "kmeans",
            "groupwise_kmeans",
            "groupwise_kmeans_fp16",
            "groupwise_kmeans_residual",
            "groupwise_kmeans_salient",
            "groupwise_symmetric_fp16",
            "groupwise_input_fp16",
            "groupwise_input_residual",
            "groupwise_kmeans_activation",
            "groupwise_residual_activation",
            "hadamard_kmeans",
        ],
        help="Codebook algo for trinity_lut2/trinity. kmeans is +20dB SNR vs "
        "linspace on real RWKV weights; linspace is the legacy back-compat "
        "default for old packs (override with RWKV_TRINITY_CODEBOOK env).",
    )
    p.add_argument(
        "--trinity-group-size",
        type=int,
        default=128,
        help="Weights per learned LUT2 codebook group (default 128).",
    )
    p.add_argument(
        "--scale-group-size",
        type=int,
        default=64,
        help="Weights per grouped-U8 scale/min-max group (default 64).",
    )
    p.add_argument(
        "--trinity-quality-preset",
        choices=["native_safe", "legacy", "balanced"],
        default="native_safe",
        help="native_safe (default) uses quality-gated grouped U8 g64 for all "
        "RWKV matrices, keeps non-matrix controls dense, and avoids legacy "
        "LUT2 for large embedding/head matrices; balanced keeps "
        "small controls dense and uses LUT2 for large matrices; legacy applies "
        "the requested codec to every tensor.",
    )
    p.add_argument(
        "--activation-stats",
        type=Path,
        help="Calibration artifact from rwkv_ssd.tools.calibrate_activations.",
    )
    p.add_argument(
        "--pack-layout",
        default="default",
        choices=["default", "layer_grouped"],
        help="tensor order in weights.bin (layer_grouped = P2.b channel-friendly)",
    )
    p.add_argument(
        "--compress",
        choices=["none", "zstd"],
        default="none",
        help="optional cold-storage compression; zstd decompresses once into RAM at load",
    )
    p.add_argument(
        "--sector-bytes",
        type=int,
        default=0,
        help=f"pad between layers (0=off; try {DEFAULT_SECTOR_BYTES} for layer_grouped)",
    )
    p.add_argument("--no-hash", action="store_true")
    p.add_argument(
        "--no-layer-zlib",
        action="store_true",
        help="trinity+layer_grouped: TCL\\x02 raw layer blobs (faster decode, larger on disk)",
    )
    p.add_argument(
        "--bf16-shadow",
        action="store_true",
        help="also write shadow.bin (raw bf16 per tensor) for fast decode",
    )
    p.add_argument(
        "--shadow-min-numel",
        type=int,
        default=0,
        help="with --bf16-shadow: only shadow tensors with numel >= N (0=all)",
    )
    p.add_argument(
        "--codec-map",
        type=str,
        default=None,
        help="Per-tensor codec overrides as COMMA=separated NAME=codec pairs. "
        "Use 'default=trinity_lut2,head.weight=scale_u8'. "
        "Glob suffix: 'blocks.0.*=trinity_lut2'. "
        "Known codecs: none, scale_u8, scale_u8_grouped, scale_u4, "
        "trinity_lut2, trinity.",
    )
    p.add_argument(
        "--codec-map-json",
        type=Path,
        help="JSON object or planner report containing a codec_map object.",
    )
    p.add_argument(
        "--hf-prefer",
        choices=["safetensors", "pth"],
        default="safetensors",
        help="with --hf-repo: which weight format to download "
        "(default safetensors; legacy 'pth' for the old behaviour).",
    )
    p.add_argument(
        "--copy-hf-metadata",
        action="store_true",
        help="When --input is a HF model directory, also copy config.json, "
        "tokenizer files, and generation_config.json into the pack dir. "
        "The pack then becomes self-contained — downstream code can "
        "instantiate the model + tokenizer from the pack alone.",
    )
    p.add_argument(
        "--hf-metadata-overwrite",
        action="store_true",
        help="with --copy-hf-metadata: overwrite existing files in pack dir.",
    )
    p.add_argument(
        "--no-deepembed-sidecar",
        action="store_true",
        help="do not materialize DeepEmbed.bin when a DeepEmbed checkpoint is detected",
    )
    args = p.parse_args()
    inp, hf = _resolve_input(
        Path(args.input) if args.input else None,
        args.hf_repo,
        hf_prefer=args.hf_prefer,
    )

    codec_map_raw = args.codec_map
    parsed_codec_map: dict[str, str] = {}
    if codec_map_raw:
        for pair in codec_map_raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, val = pair.split("=", 1)
            key, val = key.strip(), val.strip().lower()
            if val not in ("none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity"):
                print(f"Warning: unknown codec {val!r} for {key!r}, skipping")
                continue
            parsed_codec_map[key] = val
    if args.codec_map_json:
        loaded_map = json.loads(args.codec_map_json.read_text(encoding="utf-8"))
        if isinstance(loaded_map, dict) and isinstance(loaded_map.get("codec_map"), dict):
            loaded_map = loaded_map["codec_map"]
        if not isinstance(loaded_map, dict):
            raise ValueError("--codec-map-json must contain a JSON object")
        for key, val in loaded_map.items():
            value = str(val).strip().lower()
            if value not in ("none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity"):
                raise ValueError(f"unknown codec {value!r} for {key!r}")
            parsed_codec_map[str(key)] = value

    pack(
        inp,
        Path(args.output),
        args.model_family,
        hf_repo=hf,
        hash_weights=not args.no_hash,
        pack_codec=args.pack_codec,
        pack_layout=args.pack_layout,
        sector_bytes=args.sector_bytes,
        trinity_layer_compress=not args.no_layer_zlib,
        bf16_shadow=args.bf16_shadow,
        shadow_min_numel=args.shadow_min_numel,
        codec_map=parsed_codec_map or None,
        trinity_codebook=args.trinity_codebook,
        trinity_group_size=args.trinity_group_size,
        trinity_quality_preset=args.trinity_quality_preset,
        activation_stats_path=args.activation_stats,
        scale_group_size=args.scale_group_size,
        copy_hf_metadata=args.copy_hf_metadata,
        hf_metadata_overwrite=args.hf_metadata_overwrite,
        write_deepembed=not args.no_deepembed_sidecar,
        compression=args.compress,
    )


if __name__ == "__main__":
    main()
