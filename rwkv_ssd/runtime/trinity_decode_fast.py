"""Fast CPU Trinity LUT2 layer decode (tok/s path — minimize gather + cast cost)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from rwkv_ssd.runtime.lut_gather_kernel import (
    _numba_available,
    gather_lut2_layer,
    resolve_lut_kernel,
)
from rwkv_ssd.runtime.tensor_loader import dtype_from_entry
from rwkv_ssd.runtime.trinity_codec import (
    _lut2_inner_view,
    decode_trinity_lut2_to_tensor,
    is_grouped_lut2_blob,
)

if TYPE_CHECKING:
    from rwkv_ssd.runtime.manifest import TensorEntry


def _use_bf16_direct_gather(dt: torch.dtype) -> bool:
    if dt != torch.bfloat16:
        return False
    if os.environ.get("RWKV_LUT_BF16_NATIVE", "auto").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return False
    kernel = resolve_lut_kernel()
    # ``resolve_lut_kernel`` resolves ``auto`` to one of the concrete
    # backends, so the check is just ``native`` (preferred) or ``numba``
    # (when available). The native path writes bf16 bits directly; the
    # numba path uses the ``gather_packed_bf16`` kernel.
    if kernel == "native":
        return True
    return kernel == "numba" and _numba_available()


def decode_lut2_layer_cpu_fast(
    raw: bytes | memoryview,
    entries: list[TensorEntry],
    base: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Layer-batched LUT2 → bf16 on CPU.

    One fused gather (native bf16 or numba when available), optional dtype conversion,
    then views into the slab (no per-tensor clone — ``prepare_layer_for_z``
    owns layout copies).
    """
    if not entries:
        return {}
    ordered = sorted(entries, key=lambda e: e.offset)
    blobs = [raw[e.offset - base : e.offset - base + e.length] for e in ordered]
    if any(is_grouped_lut2_blob(blob) for blob in blobs):
        return {
            entry.name: decode_trinity_lut2_to_tensor(blob, entry, device)
            for entry, blob in zip(ordered, blobs, strict=True)
        }
    total = sum(e.numel for e in ordered)
    dt = dtype_from_entry(ordered[0])
    native_bf16 = _use_bf16_direct_gather(dt)
    meta: list[tuple[str, tuple[int, ...], int, int]] = []
    slices: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    off = 0
    for entry in ordered:
        rel = entry.offset - base
        blob = raw[rel : rel + entry.length]
        codebook, idx_view, _row_stride = _lut2_inner_view(blob, entry)
        packed = np.frombuffer(idx_view, dtype=np.uint8)
        slices.append((off, codebook, packed, entry.numel))
        meta.append((entry.name, tuple(entry.shape), off, entry.numel))
        off += entry.numel

    if native_bf16:
        flat_u16 = np.empty(total, dtype=np.uint16)
        gather_lut2_layer(flat_u16, slices, bf16_out=True)
        t_flat = torch.frombuffer(flat_u16, dtype=torch.bfloat16)
    else:
        flat = np.empty(total, dtype=np.float32)
        gather_lut2_layer(flat, slices)
        t_flat = torch.from_numpy(flat)
        if dt != torch.float32:
            t_flat = t_flat.to(dtype=dt)
    if device.type != "cpu":
        t_flat = t_flat.to(device=device)
    out: dict[str, torch.Tensor] = {}
    for name, shape, start, numel in meta:
        out[name] = t_flat[start : start + numel].reshape(shape)
    return out
