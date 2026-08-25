"""
Research-only per-tensor LUT encode/decode for bitwidth ladder (LUT2/3/4).

Not wired into pack_runtime yet — use for benches and M5 codec decisions.
See docs/LUT_BITWIDTH_RESEARCH.md.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

LUTVariant = Literal["lut2", "lut3", "lut4"]


@dataclass(frozen=True)
class LutSpec:
    name: LutVariant
    entries: int
    bits: int
    magic: bytes


SPECS: dict[LutVariant, LutSpec] = {
    "lut2": LutSpec("lut2", 4, 2, b"TR2\x91"),
    "lut3": LutSpec("lut3", 8, 3, b"TR3\x91"),
    "lut4": LutSpec("lut4", 16, 4, b"TR4\x91"),
}


def _codebook_linspace(flat: torch.Tensor, entries: int) -> np.ndarray:
    mn = float(flat.min().item())
    mx = float(flat.max().item())
    if mx - mn < 1e-12:
        mx = mn + 1.0
    return np.linspace(mn, mx, entries, dtype=np.float32)


def _indices(flat: torch.Tensor, codebook: np.ndarray) -> np.ndarray:
    cb = torch.from_numpy(codebook)
    return (flat.unsqueeze(-1) - cb).abs().argmin(dim=-1).to(torch.uint8).numpy().ravel()


def _pack_indices(indices: np.ndarray, bits: int) -> bytes:
    idx = indices.astype(np.uint8).ravel()
    if bits == 2:
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
    if bits == 3:
        # 8 indices per 3 bytes (24 bits); pad indices to multiple of 8
        pad = (8 - idx.size % 8) % 8
        if pad:
            idx = np.append(idx, np.zeros(pad, dtype=np.uint8))
        n_groups = idx.size // 8
        out = bytearray(n_groups * 3)
        for g in range(n_groups):
            base = g * 8
            chunk = idx[base : base + 8].astype(np.uint32)
            val = 0
            for i in range(8):
                val |= (int(chunk[i]) & 7) << (3 * i)
            out[g * 3] = val & 0xff
            out[g * 3 + 1] = (val >> 8) & 0xff
            out[g * 3 + 2] = (val >> 16) & 0xff
        return bytes(out)
    if bits == 4:
        pad = (2 - idx.size % 2) % 2
        if pad:
            idx = np.append(idx, np.zeros(pad, dtype=np.uint8))
        out = np.empty(idx.size // 2, dtype=np.uint8)
        out[:] = (idx[0::2] & 15) | ((idx[1::2] & 15) << 4)
        return out.tobytes()
    raise ValueError(f"unsupported bits={bits}")


def _unpack_indices(packed: bytes, numel: int, bits: int) -> np.ndarray:
    if bits == 2:
        raw = np.frombuffer(packed, dtype=np.uint8)
        need = (numel + 3) // 4
        pos = np.arange(numel, dtype=np.intp)
        return (raw[:need][pos // 4] >> ((pos % 4) * 2)) & 3
    if bits == 3:
        raw = np.frombuffer(packed, dtype=np.uint8)
        n_groups = (numel + 7) // 8
        idx = np.zeros(n_groups * 8, dtype=np.uint8)
        for g in range(n_groups):
            b0 = int(raw[g * 3]) if g * 3 < len(raw) else 0
            b1 = int(raw[g * 3 + 1]) if g * 3 + 1 < len(raw) else 0
            b2 = int(raw[g * 3 + 2]) if g * 3 + 2 < len(raw) else 0
            val = b0 | (b1 << 8) | (b2 << 16)
            for i in range(8):
                idx[g * 8 + i] = (val >> (3 * i)) & 7
        return idx[:numel]
    if bits == 4:
        raw = np.frombuffer(packed, dtype=np.uint8)
        need = (numel + 1) // 2
        out = np.empty(numel, dtype=np.uint8)
        for i in range(numel):
            byte = raw[i // 2]
            out[i] = (byte >> (4 * (i % 2))) & 15
        return out
    raise ValueError(f"unsupported bits={bits}")


def encode_lut_tensor(tensor: torch.Tensor, variant: LutVariant) -> bytes:
    spec = SPECS[variant]
    flat = tensor.detach().float().cpu().flatten()
    cb = _codebook_linspace(flat, spec.entries)
    indices = _indices(flat, cb)
    packed = _pack_indices(indices, spec.bits)
    return spec.magic + struct.pack("<I", spec.entries) + cb.tobytes() + packed


def decode_lut_tensor(blob: bytes, shape: tuple[int, ...], variant: LutVariant) -> torch.Tensor:
    spec = SPECS[variant]
    if blob[:4] != spec.magic:
        raise ValueError(f"bad magic for {variant}")
    entries = struct.unpack_from("<I", blob, 4)[0]
    cb = np.frombuffer(blob, dtype=np.float32, offset=8, count=entries)
    packed_off = 8 + entries * 4
    packed = blob[packed_off:]
    numel = int(np.prod(shape))
    indices = _unpack_indices(packed, numel, spec.bits)
    flat = cb[indices]
    return torch.from_numpy(flat.astype(np.float32)).reshape(shape)


def blob_byte_size(numel: int, variant: LutVariant) -> int:
    spec = SPECS[variant]
    header = 4 + 4 + spec.entries * 4
    if spec.bits == 2:
        payload = (numel + 3) // 4
    elif spec.bits == 3:
        payload = ((numel + 7) // 8) * 3
    else:
        payload = (numel + 1) // 2
    return header + payload


def mse_vs_bf16(original: torch.Tensor, decoded: torch.Tensor) -> float:
    o = original.float().cpu()
    d = decoded.float().cpu()
    return float(((o - d) ** 2).mean().item())
