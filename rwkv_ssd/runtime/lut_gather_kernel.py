"""LUT2 gather kernels (Numba / NumPy / Torch) for Trinity decode tok/s."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

_KERNEL = os.environ.get("RWKV_LUT_KERNEL", "auto").strip().lower()
_NUMBA_OK: bool | None = None


def _lut_kernel_env() -> str:
    return os.environ.get("RWKV_LUT_KERNEL", "auto").strip().lower()


def _numba_available() -> bool:
    global _NUMBA_OK
    if _NUMBA_OK is not None:
        return _NUMBA_OK
    try:
        import numba  # noqa: F401

        _NUMBA_OK = True
    except ImportError:
        _NUMBA_OK = False
    return _NUMBA_OK


def _native_available() -> bool:
    try:
        from rwkv_ssd.native.lut2_gather_loader import lib

        return lib() is not None
    except Exception:
        return False


def resolve_lut_kernel() -> str:
    kernel = _lut_kernel_env()
    if kernel in ("numpy", "np"):
        return "numpy"
    if kernel in ("numba", "jit"):
        return "numba"
    if kernel in ("torch", "pt"):
        return "torch"
    if kernel in ("native", "c", "omp"):
        return "native"
    if kernel == "auto":
        if _native_available():
            return "native"
        return "numba" if _numba_available() else "numpy"
    return "numpy"


def unpack_2bit_into(out: np.ndarray, packed: np.ndarray) -> None:
    """Unpack 2-bit indices into ``out`` (length = out.size)."""
    n = out.size
    kernel = resolve_lut_kernel()
    if kernel == "native":
        # The native backend fuses unpack+gather, so it has no standalone
        # unpack entry point.  Accelerator decode still needs host indices
        # for small tensors and legacy callers, however.  Falling back here
        # is safe; the native path remains fused wherever gather_lut2_packed
        # or gather_lut2_layer is used.
        if _numba_available():
            _unpack_2bit_numba(out, packed)
        else:
            need = (n + 3) // 4
            pos = np.arange(n, dtype=np.intp)
            out[:] = (packed[:need][pos // 4] >> ((pos % 4) * 2)) & 3
        return
    if kernel == "numba":
        _unpack_2bit_numba(out, packed)
    else:
        need = (n + 3) // 4
        pos = np.arange(n, dtype=np.intp)
        out[:] = (packed[:need][pos // 4] >> ((pos % 4) * 2)) & 3


def _writable_u8(packed: np.ndarray) -> np.ndarray:
    """Native ctypes needs a writable buffer (``frombuffer`` on mmap slices fails)."""
    if packed.flags.writeable and packed.flags.c_contiguous:
        return packed
    return np.array(packed, dtype=np.uint8, copy=True)


def gather_lut2_packed(
    flat: np.ndarray,
    off: int,
    codebook: np.ndarray,
    packed: np.ndarray,
    n: int,
) -> None:
    """Fused 2-bit unpack + LUT gather (preferred for native)."""
    if n <= 0:
        return
    out = flat[off : off + n]
    kernel = resolve_lut_kernel()
    if kernel == "native":
        from rwkv_ssd.native import lut2_gather_loader

        if lut2_gather_loader.lib() is not None:
            need = (n + 3) // 4
            pk = _writable_u8(packed[:need])
            lut2_gather_loader.gather_packed(
                out, codebook, pk, n
            )
            return
    if kernel in ("numba", "native") and _numba_available():
        _ensure_numba()
        assert _gather_packed_jit is not None
        _gather_packed_jit(out, codebook, packed, n)
        return
    indices = np.empty(n, dtype=np.uint8)
    unpack_2bit_into(indices, packed)
    gather_lut2_into(flat, off, codebook, indices)


def gather_lut2_into(
    flat: np.ndarray, off: int, codebook: np.ndarray, indices: np.ndarray
) -> None:
    """``flat[off:off+n] = codebook[indices]`` with the fastest available kernel."""
    n = indices.size
    if n == 0:
        return
    out = flat[off : off + n]
    kernel = resolve_lut_kernel()
    if kernel == "numba":
        _gather_numba(out, codebook, indices)
    elif kernel == "torch":
        _gather_torch(out, codebook, indices)
    else:
        np.take(codebook, indices, out=out)


def _native_layer_gather(
    flat: np.ndarray,
    slices: list[tuple[int, np.ndarray, np.ndarray, int]],
    *,
    bf16: bool = False,
) -> bool:
    """Call native layer gather; return False on failure (caller falls back)."""
    import ctypes

    from rwkv_ssd.native import lut2_gather_loader

    lib = lut2_gather_loader.lib()
    if lib is None or not slices:
        return False
    try:
        n_tensors = len(slices)
        codebooks = np.stack([cb for _, cb, _, _ in slices], axis=0).astype(
            np.float32, copy=False
        )
        offsets = np.array([off for off, _, _, _ in slices], dtype=np.uint64)
        numels = np.array([n for _, _, _, n in slices], dtype=np.uint64)
        ptrs = (ctypes.POINTER(ctypes.c_uint8) * n_tensors)()
        pk_arrays: list[np.ndarray] = []
        for i, (_, _, pk, _n) in enumerate(slices):
            pk_arr = _writable_u8(np.frombuffer(pk, dtype=np.uint8))
            pk_arrays.append(pk_arr)
            ptrs[i] = pk_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        off_a = offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        nel_a = numels.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        cb_f = codebooks.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if bf16:
            out_u16 = flat.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16)
            )
            lib.lut2_layer_gather_bf16_export(
                out_u16, cb_f, ptrs, off_a, nel_a, n_tensors
            )
        else:
            out_f = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.lut2_layer_gather_export(
                out_f, cb_f, ptrs, off_a, nel_a, n_tensors
            )
        del pk_arrays, ptrs
        return True
    except (OSError, ValueError):
        return False


def gather_lut2_layer(
    flat: np.ndarray,
    slices: list[tuple[int, np.ndarray, np.ndarray, int]],
    *,
    bf16_out: bool = False,
) -> None:
    """
    Fill ``flat`` from multiple LUT2 blobs.

    Each slice is ``(offset, codebook float32[4], packed uint8, numel)``.
    When ``bf16_out=True``, ``flat`` must be uint16/bf16 bit storage (length=total numel).
    """
    if not slices:
        return
    kernel = resolve_lut_kernel()
    if kernel == "native" and slices:
        if _native_layer_gather(flat, slices, bf16=bf16_out):
            return
    if bf16_out:
        if _numba_available():
            _ensure_numba()
            assert _gather_packed_bf16_jit is not None
            for off, cb, pk, n in slices:
                out_u16 = flat[off : off + n].view(np.uint16)
                need = (n + 3) // 4
                _gather_packed_bf16_jit(out_u16, cb, pk[:need], n)
            return
        tmp = np.empty(flat.size, dtype=np.float32)
        for off, cb, pk, n in slices:
            gather_lut2_packed(tmp, off, cb, np.frombuffer(pk, dtype=np.uint8), n)
        flat[:] = torch.from_numpy(tmp).to(torch.bfloat16).view(torch.uint16).numpy()
        return
    for off, cb, pk, n in slices:
        gather_lut2_packed(flat, off, cb, np.frombuffer(pk, dtype=np.uint8), n)


def _gather_torch(
    out: np.ndarray, codebook: np.ndarray, indices: np.ndarray
) -> None:
    import torch

    cb = torch.from_numpy(codebook)
    idx = torch.from_numpy(indices.astype(np.int64, copy=False))
    out[:] = cb[idx].numpy()


# --- Numba (compiled once, cached on disk) ---

_gather_jit = None
_unpack_2bit_jit = None
_gather_packed_jit = None
_gather_packed_bf16_jit = None
_float_to_bf16_bits_jit = None


def _ensure_numba() -> None:
    global _gather_jit, _unpack_2bit_jit, _gather_packed_jit
    global _gather_packed_bf16_jit, _float_to_bf16_bits_jit
    if _gather_packed_jit is not None:
        return
    import numba

    @numba.njit(cache=True, fastmath=False)
    def float_to_bf16_bits(val):
        arr = np.empty(1, dtype=np.float32)
        arr[0] = val
        u = arr.view(np.uint32)[0]
        return np.uint16(
            (u + np.uint32(0x7FFF) + ((u >> np.uint16(16)) & np.uint32(1)))
            >> np.uint16(16)
        )

    @numba.njit(cache=True, fastmath=False)
    def gather(out, codebook, indices):
        n = indices.shape[0]
        for i in range(n):
            out[i] = codebook[indices[i]]

    @numba.njit(cache=True, fastmath=False)
    def unpack_2bit(out, packed):
        n = out.shape[0]
        for i in range(n):
            out[i] = (packed[i // 4] >> ((i % 4) * 2)) & 3

    @numba.njit(cache=True, fastmath=False)
    def gather_packed_fused(out, codebook, packed, n):
        i = 0
        p = 0
        n4 = n - (n % 4)
        while i < n4:
            b = packed[p]
            p += 1
            out[i] = codebook[b & 3]
            out[i + 1] = codebook[(b >> 2) & 3]
            out[i + 2] = codebook[(b >> 4) & 3]
            out[i + 3] = codebook[(b >> 6) & 3]
            i += 4
        if i < n:
            b = packed[p]
            shift = 0
            while i < n:
                out[i] = codebook[(b >> shift) & 3]
                i += 1
                shift += 2

    @numba.njit(cache=True, fastmath=False)
    def gather_packed_bf16(out_u16, codebook, packed, n):
        cb0 = float_to_bf16_bits(codebook[0])
        cb1 = float_to_bf16_bits(codebook[1])
        cb2 = float_to_bf16_bits(codebook[2])
        cb3 = float_to_bf16_bits(codebook[3])
        i = 0
        p = 0
        n4 = n - (n % 4)
        while i < n4:
            b = packed[p]
            p += 1
            out_u16[i] = cb0 if (b & 3) == 0 else cb1 if (b & 3) == 1 else cb2 if (b & 3) == 2 else cb3
            out_u16[i + 1] = (
                cb0 if ((b >> 2) & 3) == 0 else cb1 if ((b >> 2) & 3) == 1 else cb2 if ((b >> 2) & 3) == 2 else cb3
            )
            out_u16[i + 2] = (
                cb0 if ((b >> 4) & 3) == 0 else cb1 if ((b >> 4) & 3) == 1 else cb2 if ((b >> 4) & 3) == 2 else cb3
            )
            out_u16[i + 3] = (
                cb0 if ((b >> 6) & 3) == 0 else cb1 if ((b >> 6) & 3) == 1 else cb2 if ((b >> 6) & 3) == 2 else cb3
            )
            i += 4
        if i < n:
            b = packed[p]
            shift = 0
            while i < n:
                idx = (b >> shift) & 3
                out_u16[i] = cb0 if idx == 0 else cb1 if idx == 1 else cb2 if idx == 2 else cb3
                i += 1
                shift += 2

    _gather_jit = gather
    _unpack_2bit_jit = unpack_2bit
    _gather_packed_jit = gather_packed_fused
    _gather_packed_bf16_jit = gather_packed_bf16
    _float_to_bf16_bits_jit = float_to_bf16_bits


def _gather_numba(
    out: np.ndarray, codebook: np.ndarray, indices: np.ndarray
) -> None:
    _ensure_numba()
    assert _gather_jit is not None
    _gather_jit(out, codebook, indices)


def _unpack_2bit_numba(out: np.ndarray, packed: np.ndarray) -> None:
    _ensure_numba()
    assert _unpack_2bit_jit is not None
    _unpack_2bit_jit(out, packed)
