"""
Compression Trinity pack codec (thesis Ch.6 — engine path, CPU v1).

Stages on disk (mutually exclusive with M5 scale_u8/u4 per deployment):

1. **trinity_lut2** — 4-entry per-tensor LUT, 2 bits/weight (~8× vs dense FP16).
   Default codebook: K-means (Lloyd's algorithm) on the flat weight tensor.
   Use ``trinity_lut2_linspace`` for the legacy equally-spaced codebook (back-compat
   for existing packs). Use ``trinity_lut2_hadamard_kmeans`` for the
   QuIP#-style Hadamard-rotated K-means (best 2-bit quality on LLM weights).
2. **trinity** — LUT payload + zlib entropy (ANS stand-in until nvCOMP is wired).

2:4 structured sparsity is **not** in the blob yet; thesis treats it as compute-side
(Tensor Core), not extra storage at 2-bit granularity.

Measured SNR on rwkv7-g1d-0.1b ``blocks.0.att.receptance.weight`` (768×768 bf16):

  +------------------------------+--------+
  | codebook                     | SNR    |
  +------------------------------+--------+
  | linspace (legacy)            | -13.8dB|
  | kmeans (default, this build) |  +7.8dB|
  | hadamard_kmeans (best)       |  +9.0dB|
  +------------------------------+--------+
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Literal

import numpy as np
import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.tensor_loader import dtype_from_entry
from rwkv_ssd.runtime.trinity_codebook import (
    codebook_linspace as _codebook_linspace,
    codebook_kmeans as _codebook_kmeans,
    codebooks_groupwise_kmeans as _codebooks_groupwise_kmeans,
    codebooks_groupwise_symmetric as _codebooks_groupwise_symmetric,
    random_hadamard_matrix as _random_hadamard_matrix,
    apply_rotation as _apply_rotation,
    invert_rotation as _invert_rotation,
)

_LUT2_MAGIC = b"TR2\x01"
_LUT2_MAGIC_FLUTE = b"TR2\x02"
_LUT2_MAGIC_GROUPED = b"TR2\x03"
_LUT2_MAGIC_GROUPED_FP16 = b"TR2\x04"
_LUT2_MAGIC_GROUPED_RESIDUAL = b"TR2\x05"
_LUT2_MAGIC_INPUT_FP16 = b"TR2\x06"
_LUT2_MAGIC_INPUT_RESIDUAL = b"TR2\x07"
LUT2_MAGIC = _LUT2_MAGIC  # public alias
ANS_MAGIC = b"TRA\x01"
TCL_MAGIC = b"TCL\x01"  # one zlib blob per layer (concat LUT2 payloads)
TCL_RAW_MAGIC = b"TCL\x02"  # concat LUT2 payloads, no zlib (throughput path)
CODEBOOK_FLOATS = 4
CODEBOOK_BYTES = CODEBOOK_FLOATS * 4
LUT2_HEADER_BYTES = len(LUT2_MAGIC) + CODEBOOK_BYTES


CodebookAlgo = Literal[
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
]


def is_grouped_lut2_blob(data: bytes | memoryview) -> bool:
    return len(data) >= 4 and bytes(data[:4]) in {
        _LUT2_MAGIC_GROUPED,
        _LUT2_MAGIC_GROUPED_FP16,
        _LUT2_MAGIC_GROUPED_RESIDUAL,
        _LUT2_MAGIC_INPUT_FP16,
        _LUT2_MAGIC_INPUT_RESIDUAL,
    }


def _indices_groupwise(
    flat: np.ndarray, codebooks: np.ndarray, group_size: int
) -> np.ndarray:
    """Assign local codebook indices in bounded vectorized batches."""
    indices = np.empty(flat.size, dtype=np.uint8)
    groups = codebooks.shape[0]
    batch_groups = max(1, 1_000_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, flat.size - start)
        values = flat[start : start + available]
        group_ids = np.arange(available, dtype=np.intp) // group_size
        local = codebooks[first : first + count]
        indices[start : start + available] = np.abs(
            values[:, None] - local[group_ids]
        ).argmin(axis=1).astype(np.uint8)
    return indices


def _refine_weighted_codebooks(
    flat: np.ndarray,
    codebooks: np.ndarray,
    weights: np.ndarray,
    group_size: int,
    *,
    iters: int = 8,
) -> np.ndarray:
    """Lloyd refinement minimizing activation-weighted squared error."""
    if weights.shape != flat.shape:
        raise ValueError("activation importance shape does not match weights")
    out = codebooks.astype(np.float32, copy=True)
    groups = out.shape[0]
    batch_groups = max(1, 750_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, flat.size - start)
        values = np.zeros((count, group_size), dtype=np.float32)
        importance = np.zeros((count, group_size), dtype=np.float32)
        values.reshape(-1)[:available] = flat[start : start + available]
        importance.reshape(-1)[:available] = weights[start : start + available]
        centers = out[first : first + count]
        for _ in range(max(1, int(iters))):
            labels = np.abs(values[:, :, None] - centers[:, None, :]).argmin(axis=2)
            updated = centers.copy()
            for index in range(CODEBOOK_FLOATS):
                selected = labels == index
                weighted = np.where(selected, importance, 0.0)
                denominator = weighted.sum(axis=1, dtype=np.float64)
                numerator = (weighted * values).sum(axis=1, dtype=np.float64)
                present = denominator > 1e-12
                updated[present, index] = (
                    numerator[present] / denominator[present]
                ).astype(np.float32)
            updated.sort(axis=1)
            if np.max(np.abs(updated - centers)) < 1e-7:
                centers = updated
                break
            centers = updated
        out[first : first + count] = centers
    return out


def _residual_metadata(
    flat: np.ndarray,
    codebooks_fp16: np.ndarray,
    indices: np.ndarray,
    group_size: int,
    *,
    salient: bool = False,
    saliency_weights: np.ndarray | None = None,
) -> bytes:
    """Two exact-ish sparse residuals per group in a fixed 16-byte record."""
    groups = codebooks_fp16.shape[0]
    records = np.zeros(
        groups,
        dtype=np.dtype(
            [
                ("codebook", "<f2", (4,)),
                ("positions", "u1", (2,)),
                ("deltas", "<f2", (2,)),
                ("reserved", "u1", (2,)),
            ]
        ),
    )
    records["codebook"] = codebooks_fp16
    codebooks = codebooks_fp16.astype(np.float32)
    batch_groups = max(1, 1_000_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, flat.size - start)
        values = flat[start : start + available]
        group_ids = np.arange(available, dtype=np.intp) // group_size
        reconstructed = codebooks[first : first + count][group_ids, indices[start : start + available]]
        residual = values - reconstructed
        padded = np.zeros((count, group_size), dtype=np.float32)
        padded.reshape(-1)[:available] = residual
        score = np.abs(padded)
        if saliency_weights is not None:
            weighted_score = np.zeros((count, group_size), dtype=np.float32)
            weighted_score.reshape(-1)[:available] = saliency_weights[
                start : start + available
            ]
            score *= weighted_score
        if salient:
            weights = np.zeros((count, group_size), dtype=np.float32)
            weights.reshape(-1)[:available] = np.abs(values)
            score *= weights
        if group_size == 1:
            positions = np.zeros((count, 2), dtype=np.uint8)
        else:
            positions = np.argpartition(score, -2, axis=1)[:, -2:].astype(
                np.uint8
            )
        selected = np.take_along_axis(padded, positions.astype(np.intp), axis=1)
        records["positions"][first : first + count] = positions
        records["deltas"][first : first + count] = selected.astype(np.float16)
    return records.tobytes()


def _build_codebook(
    flat: np.ndarray,
    *,
    algo: str,
    hadamard_seed: int = 42,
) -> np.ndarray:
    if algo == "linspace":
        return _codebook_linspace(flat)
    if algo == "kmeans":
        return _codebook_kmeans(flat)
    raise ValueError(f"unsupported codebook algo: {algo!r}")


def _indices_lut2(flat: torch.Tensor, codebook: np.ndarray) -> np.ndarray:
    cb = torch.from_numpy(codebook)
    diff = (flat.unsqueeze(-1) - cb).abs()
    return diff.argmin(dim=-1).to(torch.uint8).numpy()


def _pack_2bit(indices: np.ndarray) -> bytes:
    idx = indices.astype(np.uint8).ravel()
    pad = (4 - idx.size % 4) % 4
    if pad:
        idx = np.append(idx, np.zeros(pad, dtype=np.uint8))
    out = np.empty(idx.size // 4, dtype=np.uint8)
    out[:] = (
        (idx[0::4] & 3)
        | ((idx[1::4] & 3) << 2)
        | ((idx[2::4] & 3) << 4)
        | ((idx[3::4] & 3) << 6)
    )
    return out.tobytes()


def _unpack_2bit_flute(
    data: bytes | memoryview,
    numel: int,
    shape: tuple[int, ...],
    row_stride: int,
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("flute_row layout requires 2-D tensor")
    out_f, in_f = int(shape[0]), int(shape[1])
    packed = np.frombuffer(data, dtype=np.uint8)
    indices = np.empty(numel, dtype=np.uint8)
    pos = 0
    for row in range(out_f):
        base = row * row_stride
        for col in range(in_f):
            byte_off = base + col // 4
            shift = (col % 4) * 2
            indices[pos] = (packed[byte_off] >> shift) & 3
            pos += 1
    return indices


def _unpack_2bit(data: bytes | memoryview, numel: int) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    need = (numel + 3) // 4
    if packed.size < need:
        raise ValueError(f"trinity_lut2 indices short: {packed.size} vs {need}")
    packed = packed[:need]
    pos = np.arange(numel, dtype=np.intp)
    return (packed[pos // 4] >> ((pos % 4) * 2)) & 3


def _lut2_inner_view(
    data: bytes | memoryview, entry: TensorEntry
) -> tuple[np.ndarray, memoryview, int | None]:
    """Return codebook, packed index view, optional row_stride (FLUTE layout)."""
    if len(data) < LUT2_HEADER_BYTES:
        raise ValueError(f"trinity_lut2 blob too short for {entry.name}")
    magic = bytes(data[:4])
    if magic not in (_LUT2_MAGIC, _LUT2_MAGIC_FLUTE):
        raise ValueError(f"bad trinity_lut2 magic for {entry.name}")
    codebook = np.frombuffer(
        data, dtype=np.float32, offset=4, count=CODEBOOK_FLOATS
    ).copy()
    idx_off = LUT2_HEADER_BYTES
    idx_view = memoryview(data)[idx_off:] if isinstance(data, bytes) else data[idx_off:]
    row_stride: int | None = None
    if magic == _LUT2_MAGIC_FLUTE and len(entry.shape) == 2:
        out_f, in_f = int(entry.shape[0]), int(entry.shape[1])
        packed_cols = (in_f + 3) // 4
        row_stride = ((packed_cols + 15) // 16) * 16
    return codebook, idx_view, row_stride


def _gather_lut2(
    codebook: np.ndarray,
    packed: np.ndarray,
    entry: TensorEntry,
) -> torch.Tensor:
    from rwkv_ssd.runtime.lut_gather_kernel import gather_lut2_packed

    n = entry.numel
    flat = np.empty(n, dtype=np.float32)
    gather_lut2_packed(flat, 0, codebook, packed, n)
    dt = dtype_from_entry(entry)
    return torch.from_numpy(flat).reshape(entry.shape).to(dtype=dt)


def _default_codebook_algo() -> str:
    """Read codebook algo from env, default ``kmeans`` (new since v0.6.16)."""
    return os.environ.get("RWKV_TRINITY_CODEBOOK", "kmeans").strip().lower()


def encode_trinity_lut2(
    tensor: torch.Tensor,
    *,
    layout: str = "default",
    codebook: str | None = None,
    group_size: int = 256,
    importance: torch.Tensor | np.ndarray | None = None,
) -> bytes:
    flat = tensor.detach().float().cpu().flatten().numpy()
    algo = (codebook or _default_codebook_algo()).strip().lower()
    if algo in {
        "groupwise_kmeans",
        "groupwise_kmeans_fp16",
        "groupwise_kmeans_residual",
        "groupwise_kmeans_salient",
        "groupwise_symmetric_fp16",
        "groupwise_input_fp16",
        "groupwise_input_residual",
        "groupwise_kmeans_activation",
        "groupwise_residual_activation",
    }:
        if layout.strip().lower() not in ("default", ""):
            raise ValueError("groupwise_kmeans currently requires the default layout")
        if algo in {
            "groupwise_kmeans_residual",
            "groupwise_kmeans_salient",
            "groupwise_input_residual",
            "groupwise_kmeans_activation",
            "groupwise_residual_activation",
        } and group_size > 256:
            raise ValueError("groupwise residual positions require group_size <= 256")
        feature_importance = (
            np.asarray(
                torch.as_tensor(importance).detach().float().cpu().numpy(),
                dtype=np.float32,
            ).reshape(-1)
            if algo in {
                "groupwise_kmeans_activation",
                "groupwise_residual_activation",
            }
            and importance is not None
            else None
        )
        activation_transpose = bool(
            feature_importance is not None
            and tensor.ndim == 2
            and feature_importance.size == tensor.shape[0]
            and feature_importance.size != tensor.shape[1]
        )
        encoded_flat = (
            tensor.detach().float().cpu().transpose(0, 1).contiguous().flatten().numpy()
            if algo in {"groupwise_input_fp16", "groupwise_input_residual"}
            and tensor.ndim == 2
            or activation_transpose
            else flat
        )
        codebooks = (
            _codebooks_groupwise_symmetric(encoded_flat, group_size=group_size)
            if algo == "groupwise_symmetric_fp16"
            else _codebooks_groupwise_kmeans(encoded_flat, group_size=group_size)
        )
        saliency_weights: np.ndarray | None = None
        if feature_importance is not None:
            if activation_transpose:
                saliency_weights = np.tile(feature_importance, int(tensor.shape[1]))
            elif tensor.ndim == 2 and feature_importance.size == tensor.shape[1]:
                saliency_weights = np.tile(feature_importance, int(tensor.shape[0]))
            elif feature_importance.size == encoded_flat.size:
                saliency_weights = feature_importance
            else:
                raise ValueError(
                    f"activation importance width mismatch: {feature_importance.size} "
                    f"for tensor shape {tuple(tensor.shape)}"
                )
            positive = saliency_weights[saliency_weights > 0]
            floor = float(np.median(positive)) * 1e-4 if positive.size else 1.0
            saliency_weights = np.maximum(saliency_weights, floor)
            if algo == "groupwise_kmeans_activation":
                codebooks = _refine_weighted_codebooks(
                    encoded_flat,
                    codebooks,
                    np.square(saliency_weights),
                    group_size,
                )
        if algo != "groupwise_kmeans":
            codebooks = codebooks.astype(np.float16).astype(np.float32)
        indices = _indices_groupwise(encoded_flat, codebooks, group_size)
        if algo in {
            "groupwise_kmeans_residual",
            "groupwise_kmeans_salient",
            "groupwise_input_residual",
            "groupwise_kmeans_activation",
            "groupwise_residual_activation",
        }:
            metadata = _residual_metadata(
                encoded_flat,
                codebooks.astype(np.float16),
                indices,
                group_size,
                salient=algo == "groupwise_kmeans_salient",
                saliency_weights=saliency_weights,
            )
            return (
                (
                    _LUT2_MAGIC_INPUT_RESIDUAL
                    if (
                        algo == "groupwise_input_residual" and tensor.ndim == 2
                    ) or activation_transpose
                    else _LUT2_MAGIC_GROUPED_RESIDUAL
                )
                + struct.pack("<I", int(group_size))
                + metadata
                + _pack_2bit(indices)
            )
        if algo in {
            "groupwise_kmeans_fp16",
            "groupwise_symmetric_fp16",
            "groupwise_input_fp16",
        }:
            return (
                (
                    _LUT2_MAGIC_INPUT_FP16
                    if algo == "groupwise_input_fp16" and tensor.ndim == 2
                    else _LUT2_MAGIC_GROUPED_FP16
                )
                + struct.pack("<I", int(group_size))
                + codebooks.astype(np.float16).tobytes()
                + _pack_2bit(indices)
            )
        return (
            _LUT2_MAGIC_GROUPED
            + struct.pack("<I", int(group_size))
            + codebooks.astype(np.float32, copy=False).tobytes()
            + _pack_2bit(indices)
        )
    cb = _build_codebook(flat, algo=algo)
    indices = _indices_lut2(torch.from_numpy(flat), cb)
    packed = _pack_2bit(indices)
    if layout.strip().lower() in ("flute", "flute_row"):
        if len(tensor.shape) != 2:
            raise ValueError("flute_row layout requires 2-D weight matrix")
        out_f, in_f = int(tensor.shape[0]), int(tensor.shape[1])
        packed_cols = (in_f + 3) // 4
        row_stride = ((packed_cols + 15) // 16) * 16
        rows: list[bytes] = []
        pos = 0
        for _row in range(out_f):
            chunk = packed[pos : pos + packed_cols]
            if len(chunk) < row_stride:
                chunk = chunk + b"\x00" * (row_stride - len(chunk))
            rows.append(chunk)
            pos += packed_cols
        return _LUT2_MAGIC_FLUTE + cb.tobytes() + b"".join(rows)
    return _LUT2_MAGIC + cb.tobytes() + packed


def encode_trinity(tensor: torch.Tensor, *, level: int = zlib.Z_BEST_SPEED) -> bytes:
    """Per-tensor zlib (legacy); prefer ``encode_trinity_layer_bundle`` when layer_grouped."""
    inner = encode_trinity_lut2(tensor)
    compressed = zlib.compress(inner, level=level)
    return ANS_MAGIC + struct.pack("<I", len(inner)) + compressed


def encode_trinity_layer_bundle(
    lut2_blobs: list[bytes],
    *,
    level: int = zlib.Z_BEST_SPEED,
    compress: bool = True,
) -> tuple[bytes, list[tuple[int, int]]]:
    """
    Concatenate LUT2 payloads for one layer.

    ``compress=True`` (default): zlib once (TCL\\x01) — smallest disk.
    ``compress=False``: raw concat (TCL\\x02) — fastest decode on fast SSD.
    """
    parts: list[bytes] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for blob in lut2_blobs:
        spans.append((pos, len(blob)))
        parts.append(blob)
        pos += len(blob)
    uncompressed = b"".join(parts)
    if compress:
        compressed = zlib.compress(uncompressed, level=level)
        disk = TCL_MAGIC + struct.pack("<I", len(uncompressed)) + compressed
    else:
        disk = TCL_RAW_MAGIC + struct.pack("<I", len(uncompressed)) + uncompressed
    return disk, spans


def decompress_trinity_layer_blob(data: bytes | memoryview) -> bytes:
    if len(data) < 8:
        raise ValueError("trinity_layer blob too short")
    magic = bytes(data[:4])
    (uncompressed_len,) = struct.unpack("<I", data[4:8])
    if magic == TCL_RAW_MAGIC:
        uncompressed = bytes(data[8 : 8 + uncompressed_len])
        if len(uncompressed) != uncompressed_len:
            raise ValueError(
                f"trinity_layer raw length mismatch: {len(uncompressed)} vs {uncompressed_len}"
            )
        return uncompressed
    if magic == TCL_MAGIC:
        uncompressed = zlib.decompress(data[8:])
        if len(uncompressed) != uncompressed_len:
            raise ValueError(
                f"trinity_layer length mismatch: {len(uncompressed)} vs {uncompressed_len}"
            )
        return uncompressed
    raise ValueError("not a trinity_layer blob (expected TCL\\x01 or TCL\\x02)")


class LayerZlibCache:
    """One zlib decompress per compressed layer block (offset, length)."""

    def __init__(self) -> None:
        self._uncompressed: dict[tuple[int, int], bytes] = {}

    def get_uncompressed(
        self, entry: TensorEntry, packed_layer: bytes | memoryview
    ) -> bytes:
        key = (entry.offset, entry.length)
        if key not in self._uncompressed:
            self._uncompressed[key] = decompress_trinity_layer_blob(packed_layer)
        return self._uncompressed[key]

    def evict(self, offset: int, length: int) -> None:
        self._uncompressed.pop((offset, length), None)


def entries_as_inner_lut2_slices(entries: list[TensorEntry]) -> list[TensorEntry]:
    """Map trinity_layer manifest offsets to inner uncompressed byte ranges."""
    from dataclasses import replace

    out: list[TensorEntry] = []
    for e in entries:
        if (e.dequant or "").strip().lower() == "trinity_layer" and e.inner_length > 0:
            out.append(replace(e, offset=e.inner_offset, length=e.inner_length))
        else:
            out.append(e)
    return out


def decode_trinity_layer_to_tensor(
    packed_layer: bytes | memoryview,
    entry: TensorEntry,
    cache: LayerZlibCache,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> torch.Tensor:
    layer = cache.get_uncompressed(entry, packed_layer)
    io = entry.inner_offset
    ilen = entry.inner_length or entry.length
    blob = layer[io : io + ilen]
    return decode_trinity_lut2_to_tensor(
        blob, entry, device, decode_device=decode_device
    )


def decode_trinity_layer_span(
    packed_layer: bytes | memoryview,
    entries: list[TensorEntry],
    cache: LayerZlibCache,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
    timing: object | None = None,
) -> dict[str, torch.Tensor]:
    """One zlib decompress + batched LUT2 for all tensors in a layer."""
    if not entries:
        return {}
    layer = cache.get_uncompressed(entries[0], packed_layer)
    return decode_lut2_layer_from_span(
        layer,
        entries_as_inner_lut2_slices(entries),
        0,
        device,
        decode_device=decode_device,
        timing=timing,
    )


def decode_trinity_lut2_to_tensor(
    data: bytes | memoryview,
    entry: TensorEntry,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> torch.Tensor:
    from rwkv_ssd.runtime.device import is_accelerator_device
    from rwkv_ssd.runtime.trinity_accel import (
        gather_grouped_lut2_packed_on_device,
        gather_lut2_packed_on_device,
        gather_lut2_on_device,
    )

    dec = decode_device or device
    if is_grouped_lut2_blob(data):
        if len(data) < 8:
            raise ValueError(f"grouped trinity_lut2 blob too short for {entry.name}")
        (group_size,) = struct.unpack("<I", data[4:8])
        if group_size <= 0:
            raise ValueError(f"invalid grouped trinity_lut2 group size for {entry.name}")
        groups = (entry.numel + group_size - 1) // group_size
        magic = bytes(data[:4])
        residual_magic = magic in {
            _LUT2_MAGIC_GROUPED_RESIDUAL,
            _LUT2_MAGIC_INPUT_RESIDUAL,
        }
        fp16_magic = magic in {
            _LUT2_MAGIC_GROUPED_FP16,
            _LUT2_MAGIC_INPUT_FP16,
        }
        record_bytes = 16 if residual_magic else (
            8 if fp16_magic else CODEBOOK_BYTES
        )
        codebook_end = 8 + groups * record_bytes
        if len(data) < codebook_end + (entry.numel + 3) // 4:
            raise ValueError(f"grouped trinity_lut2 payload short for {entry.name}")
        if magic == _LUT2_MAGIC_GROUPED:
            codebooks = np.frombuffer(
                data, dtype=np.float32, offset=8, count=groups * CODEBOOK_FLOATS
            ).reshape(groups, CODEBOOK_FLOATS)
        else:
            codebooks = np.ndarray(
                shape=(groups, CODEBOOK_FLOATS),
                dtype=np.float16,
                buffer=data,
                offset=8,
                strides=(record_bytes, 2),
            ).astype(np.float32)
        packed = memoryview(data)[codebook_end:]
        residual_positions = None
        residual_deltas = None
        if residual_magic:
            residual_positions = np.ndarray(
                shape=(groups, 2),
                dtype=np.uint8,
                buffer=data,
                offset=16,
                strides=(record_bytes, 1),
            )
            residual_deltas = np.ndarray(
                shape=(groups, 2),
                dtype=np.float16,
                buffer=data,
                offset=18,
                strides=(record_bytes, 2),
            ).astype(np.float32)
        if is_accelerator_device(dec):
            t = gather_grouped_lut2_packed_on_device(
                codebooks,
                packed,
                entry,
                dec,
                group_size=group_size,
                residual_positions=residual_positions,
                residual_deltas=residual_deltas,
                input_layout=magic
                in {_LUT2_MAGIC_INPUT_FP16, _LUT2_MAGIC_INPUT_RESIDUAL},
            )
            return t if t.device == device else t.to(
                device=device, non_blocking=device.type == "cpu"
            )
        flat = np.empty(entry.numel, dtype=np.float32)
        indices = _unpack_2bit(packed, entry.numel)
        for group_id, cb in enumerate(codebooks):
            start = group_id * group_size
            end = min(start + group_size, entry.numel)
            flat[start:end] = cb[indices[start:end]]
        if residual_magic:
            for group_id in range(groups):
                base = group_id * group_size
                valid = min(group_size, entry.numel - base)
                assert residual_positions is not None
                assert residual_deltas is not None
                for position, delta in zip(
                    residual_positions[group_id], residual_deltas[group_id], strict=True
                ):
                    if int(position) < valid:
                        flat[base + int(position)] += float(delta)
        if magic in {_LUT2_MAGIC_INPUT_FP16, _LUT2_MAGIC_INPUT_RESIDUAL} and len(entry.shape) == 2:
            shaped = torch.from_numpy(flat).reshape(entry.shape[1], entry.shape[0]).transpose(0, 1)
        else:
            shaped = torch.from_numpy(flat).reshape(entry.shape)
        t = shaped.to(dtype=dtype_from_entry(entry))
        return t if device.type == "cpu" else t.to(device=device)
    codebook, idx_view, row_stride = _lut2_inner_view(data, entry)
    packed = np.frombuffer(idx_view, dtype=np.uint8)
    if is_accelerator_device(dec):
        if row_stride is not None:
            indices = _unpack_2bit_flute(
                idx_view, entry.numel, tuple(entry.shape), row_stride
            )
            if indices.max(initial=0) >= CODEBOOK_FLOATS:
                raise ValueError(f"trinity_lut2 index out of range for {entry.name}")
            t = gather_lut2_on_device(codebook, indices, entry, dec)
        else:
            need = (entry.numel + 3) // 4
            if packed.size < need:
                raise ValueError(f"trinity_lut2 indices short for {entry.name}")
            t = gather_lut2_packed_on_device(codebook, packed[:need], entry, dec)
        return (
            t
            if dec == device
            else t.to(device=device, non_blocking=device.type == "cpu")
        )
    if row_stride is not None:
        indices = _unpack_2bit_flute(
            idx_view, entry.numel, tuple(entry.shape), row_stride
        )
        flat = np.empty(entry.numel, dtype=np.float32)
        flat[:] = codebook[indices]
        dt = dtype_from_entry(entry)
        t = torch.from_numpy(flat).reshape(entry.shape).to(dtype=dt)
        return t if device.type == "cpu" else t.to(device=device)
    need = (entry.numel + 3) // 4
    if packed.size < need:
        raise ValueError(f"trinity_lut2 indices short for {entry.name}")
    t = _gather_lut2(codebook, packed[:need], entry)
    return t if device.type == "cpu" else t.to(device=device)


def decode_trinity_lut2_to_bytes(data: bytes, entry: TensorEntry) -> bytes:
    t = decode_trinity_lut2_to_tensor(data, entry, torch.device("cpu"))
    dt = dtype_from_entry(entry)
    if dt == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def decode_trinity_to_tensor(
    data: bytes | memoryview,
    entry: TensorEntry,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
) -> torch.Tensor:
    if len(data) < len(ANS_MAGIC) + 4:
        raise ValueError(f"trinity blob too short for {entry.name}")
    if bytes(data[:4]) != ANS_MAGIC:
        raise ValueError(f"bad trinity magic for {entry.name}")
    (inner_len,) = struct.unpack("<I", data[4:8])
    inner = zlib.decompress(data[8:])
    if len(inner) != inner_len:
        raise ValueError(
            f"trinity inner length mismatch for {entry.name}: {len(inner)} vs {inner_len}"
        )
    return decode_trinity_lut2_to_tensor(
        inner, entry, device, decode_device=decode_device
    )


def decode_trinity_to_bytes(data: bytes, entry: TensorEntry) -> bytes:
    t = decode_trinity_to_tensor(data, entry, torch.device("cpu"))
    dt = dtype_from_entry(entry)
    if dt == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def packed_length_trinity_lut2(
    numel: int, *, codebook: str = "kmeans", group_size: int = 256
) -> int:
    algo = codebook.strip().lower()
    if algo in {
        "groupwise_kmeans",
        "groupwise_kmeans_fp16",
        "groupwise_kmeans_residual",
        "groupwise_kmeans_salient",
        "groupwise_symmetric_fp16",
        "groupwise_input_fp16",
        "groupwise_input_residual",
        "groupwise_kmeans_activation",
        "groupwise_residual_activation",
    }:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        groups = (numel + group_size - 1) // group_size
        record_bytes = 8 if algo in {
            "groupwise_kmeans_fp16",
            "groupwise_symmetric_fp16",
            "groupwise_input_fp16",
        } else CODEBOOK_BYTES
        return 8 + groups * record_bytes + (numel + 3) // 4
    return LUT2_HEADER_BYTES + (numel + 3) // 4


def packed_length_trinity(numel: int) -> int:
    return len(ANS_MAGIC) + 4 + packed_length_trinity_lut2(numel)


def storage_ratio_estimate(numel: int, *, fp16: bool = True) -> float:
    dense = numel * (2 if fp16 else 4)
    return dense / max(packed_length_trinity_lut2(numel), 1)


def decode_lut2_layer_from_span(
    raw: bytes | memoryview,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
    timing: object | None = None,
) -> dict[str, torch.Tensor]:
    """One ``.to(bfloat16)`` per layer; preallocated gather buffer (no concat churn)."""
    from rwkv_ssd.runtime.device import is_accelerator_device
    from rwkv_ssd.runtime.trinity_accel import decode_lut2_layer_on_accel

    if not entries:
        return {}
    dec = decode_device or device
    if is_accelerator_device(dec):
        blobs: list[tuple[bytes | memoryview, TensorEntry]] = []
        for entry in entries:
            rel = entry.offset - base
            blobs.append((raw[rel : rel + entry.length], entry))
        return decode_lut2_layer_on_accel(
            blobs=blobs,
            decode_device=dec,
            output_device=device,
            timing=timing,
        )
    from rwkv_ssd.runtime.trinity_decode_fast import decode_lut2_layer_cpu_fast

    return decode_lut2_layer_cpu_fast(raw, entries, base, device)


def decode_entries_from_span(
    raw: bytes | memoryview,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
    *,
    decode_device: torch.device | None = None,
    timing: object | None = None,
    max_workers: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Decode multiple packed tensors from one contiguous ``read_bytes_span``.

    Uses a thread pool for ``trinity`` (zlib releases the GIL) so layer loads
    amortize decompression better than per-tensor serial calls.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not entries:
        return {}

    codecs = {(e.dequant or "none").strip().lower() for e in entries}
    if codecs == {"trinity_lut2"}:
        return decode_lut2_layer_from_span(
            raw,
            entries,
            base,
            device,
            decode_device=decode_device,
            timing=timing,
        )
    if codecs == {"trinity_layer"}:
        cache = LayerZlibCache()
        rel = entries[0].offset - base
        packed = raw[rel : rel + entries[0].length]
        return decode_trinity_layer_span(
            packed,
            entries,
            cache,
            device,
            decode_device=decode_device,
            timing=timing,
        )

    def _one(entry: TensorEntry) -> tuple[str, torch.Tensor]:
        rel = entry.offset - base
        blob = raw[rel : rel + entry.length]
        codec = (entry.dequant or "none").strip().lower()
        if codec == "trinity_lut2":
            t = decode_trinity_lut2_to_tensor(
                blob, entry, device, decode_device=decode_device
            )
        elif codec == "trinity":
            t = decode_trinity_to_tensor(
                blob, entry, device, decode_device=decode_device
            )
        else:
            raise ValueError(f"decode_entries_from_span: unexpected codec {codec!r}")
        return entry.name, t

    if codecs <= {"trinity"} and len(entries) > 1 and max_workers > 1:
        out: dict[str, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for name, tensor in pool.map(_one, entries):
                out[name] = tensor
        return out

    return dict(_one(e) for e in entries)


TrinityCodec = Literal["trinity_lut2", "trinity", "trinity_layer"]
