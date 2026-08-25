"""M5 pack codecs — mutually exclusive quantization paths."""

from __future__ import annotations

import struct

import numpy as np
import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.tensor_loader import dtype_from_entry

SCALE_HEADER_BYTES = 8  # float32 min + float32 max
SCALE_U8_GROUPED_MAGIC = b"SG8\x01"


def _quantize_minmax(flat: torch.Tensor, levels: int) -> tuple[float, float, torch.Tensor]:
    mn = float(flat.min().item())
    mx = float(flat.max().item())
    if mx - mn < 1e-12:
        mx = mn + 1.0
    q = ((flat - mn) / (mx - mn) * float(levels)).round().clamp(0, levels).to(torch.int64)
    return mn, mx, q


def encode_scale_u8(tensor: torch.Tensor) -> bytes:
    """Per-tensor min/max UINT8 quant (M5 ladder entry)."""
    flat = tensor.detach().float().cpu().flatten()
    mn, mx, q = _quantize_minmax(flat, 255)
    return struct.pack("<ff", mn, mx) + q.to(torch.uint8).numpy().tobytes()


def encode_scale_u8_grouped(tensor: torch.Tensor, *, group_size: int = 128) -> bytes:
    """Outlier-resistant affine UINT8 with one min/max pair per group."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat = tensor.detach().float().cpu().flatten().numpy()
    groups = (flat.size + group_size - 1) // group_size
    mins = np.empty(groups, dtype=np.float32)
    maxs = np.empty(groups, dtype=np.float32)
    quant = np.empty(flat.size, dtype=np.uint8)
    batch_groups = max(1, 1_000_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, flat.size - start)
        chunk = np.empty((count, group_size), dtype=np.float32)
        chunk.reshape(-1)[:available] = flat[start : start + available]
        if available < count * group_size:
            chunk.reshape(-1)[available:] = chunk.reshape(-1)[available - 1]
        batch_mins = chunk.min(axis=1)
        batch_maxs = chunk.max(axis=1)
        safe_maxs = np.where(batch_maxs - batch_mins < 1e-12, batch_mins + 1.0, batch_maxs)
        mins[first : first + count] = batch_mins
        maxs[first : first + count] = safe_maxs
        q = np.rint(
            (chunk - batch_mins[:, None])
            / (safe_maxs - batch_mins)[:, None]
            * 255.0
        ).clip(0, 255).astype(np.uint8)
        quant[start : start + available] = q.reshape(-1)[:available]
    scales = np.stack((mins, maxs), axis=1)
    return (
        SCALE_U8_GROUPED_MAGIC
        + struct.pack("<I", int(group_size))
        + scales.tobytes()
        + quant.tobytes()
    )


def encode_scale_u4(tensor: torch.Tensor) -> bytes:
    """Per-tensor min/max 4-bit quant (M5 ladder — ~2× smaller than scale_u8)."""
    flat = tensor.detach().float().cpu().flatten()
    mn, mx, q = _quantize_minmax(flat, 15)
    nibbles = q.to(torch.uint8).numpy()
    if nibbles.size % 2 == 1:
        nibbles = np.append(nibbles, 0)
    packed = (nibbles[0::2] << 4) | (nibbles[1::2] & 0x0F)
    return struct.pack("<ff", mn, mx) + packed.tobytes()


def _unpack_scale_u4(packed: np.ndarray, numel: int) -> np.ndarray:
    need = (numel + 1) // 2
    if packed.size < need:
        raise ValueError(f"scale_u4 packed short: {packed.size} vs {need}")
    packed = packed[:need]
    pos = np.arange(numel, dtype=np.intp)
    # encode: high nibble first weight, low nibble second (per byte)
    shift = np.where(pos % 2 == 0, 4, 0).astype(np.intp)
    return (packed[pos // 2] >> shift) & 0x0F


def decode_scale_to_tensor(
    data: bytes | memoryview, entry: TensorEntry, device: torch.device, *, bits: int
) -> torch.Tensor:
    if len(data) < SCALE_HEADER_BYTES:
        raise ValueError(f"scale_u{bits} blob too short for {entry.name}")
    mn, mx = struct.unpack("<ff", data[:SCALE_HEADER_BYTES])
    payload = memoryview(data)[SCALE_HEADER_BYTES:]
    numel = entry.numel
    scale = mx - mn
    if bits == 8:
        q = np.frombuffer(payload, dtype=np.uint8)
        if q.size != numel:
            raise ValueError(
                f"scale_u8 length mismatch for {entry.name}: {q.size} vs {numel}"
            )
        values = mn + scale * (q.astype(np.float32) * (1.0 / 255.0))
    else:
        packed = np.frombuffer(payload, dtype=np.uint8)
        q = _unpack_scale_u4(packed, numel)
        values = mn + scale * (q.astype(np.float32) * (1.0 / 15.0))
    dt = dtype_from_entry(entry)
    t = torch.from_numpy(values).reshape(entry.shape).to(dtype=dt)
    return t if device.type == "cpu" else t.to(device=device)


def decode_scale_u8_grouped_to_tensor(
    data: bytes | memoryview, entry: TensorEntry, device: torch.device
) -> torch.Tensor:
    if len(data) < 8 or bytes(data[:4]) != SCALE_U8_GROUPED_MAGIC:
        raise ValueError(f"bad scale_u8_grouped blob for {entry.name}")
    (group_size,) = struct.unpack("<I", data[4:8])
    if group_size <= 0:
        raise ValueError(f"invalid scale_u8_grouped size for {entry.name}")
    groups = (entry.numel + group_size - 1) // group_size
    scales_end = 8 + groups * 8
    if len(data) != scales_end + entry.numel:
        raise ValueError(f"scale_u8_grouped length mismatch for {entry.name}")
    scales = np.frombuffer(data, dtype=np.float32, offset=8, count=groups * 2).reshape(
        groups, 2
    )
    quant = np.frombuffer(data, dtype=np.uint8, offset=scales_end, count=entry.numel)
    values = np.empty(entry.numel, dtype=np.float32)
    batch_groups = max(1, 1_000_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, entry.numel - start)
        q = quant[start : start + available].astype(np.float32)
        group_ids = np.arange(available, dtype=np.intp) // group_size
        batch = scales[first : first + count]
        values[start : start + available] = (
            batch[group_ids, 0]
            + (batch[group_ids, 1] - batch[group_ids, 0]) * (q / 255.0)
        )
    t = torch.from_numpy(values).reshape(entry.shape).to(dtype=dtype_from_entry(entry))
    return t if device.type == "cpu" else t.to(device=device)


def decode_scale_u8_grouped_to_bytes(data: bytes, entry: TensorEntry) -> bytes:
    t = decode_scale_u8_grouped_to_tensor(data, entry, torch.device("cpu"))
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def _decode_scale_quant(
    data: bytes, entry: TensorEntry, *, bits: int
) -> bytes:
    t = decode_scale_to_tensor(data, entry, torch.device("cpu"), bits=bits)
    dt = dtype_from_entry(entry)
    if dt == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def decode_scale_u8_to_bytes(data: bytes, entry: TensorEntry) -> bytes:
    return _decode_scale_quant(data, entry, bits=8)


def decode_scale_u4_to_bytes(data: bytes, entry: TensorEntry) -> bytes:
    return _decode_scale_quant(data, entry, bits=4)


def packed_length_scale_u8(numel: int) -> int:
    return SCALE_HEADER_BYTES + numel


def packed_length_scale_u8_grouped(numel: int, *, group_size: int = 128) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return 8 + ((numel + group_size - 1) // group_size) * 8 + numel


def decode_scale_u8_grouped_layer_from_span(
    raw: bytes | memoryview,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        entry.name: decode_scale_u8_grouped_to_tensor(
            raw[entry.offset - base : entry.offset - base + entry.length], entry, device
        )
        for entry in entries
    }


def packed_length_scale_u4(numel: int) -> int:
    return SCALE_HEADER_BYTES + (numel + 1) // 2


def decode_scale_layer_from_span(
    raw: bytes | memoryview,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
    *,
    bits: int,
) -> dict[str, torch.Tensor]:
    """
    Layer-batched scale_u8 / scale_u4 decode — one float32 slab, one dtype cast.

    Matches the Trinity LUT2 fast path in ``trinity_decode_fast`` for tok/s.
    """
    if not entries:
        return {}
    ordered = sorted(entries, key=lambda e: e.offset)
    total = sum(e.numel for e in ordered)
    flat = np.empty(total, dtype=np.float32)
    meta: list[tuple[str, tuple[int, ...], int, int]] = []
    off = 0
    inv = 1.0 / 255.0 if bits == 8 else 1.0 / 15.0
    for entry in ordered:
        rel = entry.offset - base
        blob = raw[rel : rel + entry.length]
        if len(blob) < SCALE_HEADER_BYTES:
            raise ValueError(f"scale_u{bits} blob too short for {entry.name}")
        mn, mx = struct.unpack("<ff", blob[:SCALE_HEADER_BYTES])
        payload = memoryview(blob)[SCALE_HEADER_BYTES:]
        numel = entry.numel
        scale = mx - mn
        if bits == 8:
            q = np.frombuffer(payload, dtype=np.uint8, count=numel)
            if q.size != numel:
                raise ValueError(
                    f"scale_u8 length mismatch for {entry.name}: {q.size} vs {numel}"
                )
            flat[off : off + numel] = mn + scale * (q.astype(np.float32) * inv)
        else:
            packed = np.frombuffer(payload, dtype=np.uint8)
            q = _unpack_scale_u4(packed, numel)
            flat[off : off + numel] = mn + scale * (q.astype(np.float32) * inv)
        meta.append((entry.name, tuple(entry.shape), off, numel))
        off += numel

    dt = dtype_from_entry(ordered[0])
    t_flat = torch.from_numpy(flat)
    if dt != torch.float32:
        t_flat = t_flat.to(dtype=dt)
    if device.type != "cpu":
        t_flat = t_flat.to(device=device)
    out: dict[str, torch.Tensor] = {}
    for name, shape, start, numel in meta:
        out[name] = t_flat[start : start + numel].reshape(shape)
    return out
