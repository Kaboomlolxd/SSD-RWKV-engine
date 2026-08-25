"""
Track D — fused LUT2 decode + GEMV (AQLM-style).

Avoids materializing full bf16 weight matrices for vector-matrix products in
strict streaming (FFN CMix + head). Enable: ``RWKV_LUT_GEMM_FUSED=1`` or ``auto``
(strict streaming without stream_layer_cache).

Native OpenMP GEMV when ``lut2_gather`` is built; Numba JIT fallback otherwise.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import ctypes
import numpy as np
import torch

from rwkv_ssd.runtime.trinity_codec import _lut2_inner_view, is_grouped_lut2_blob

_SCALE_U8_GROUPED_MAGIC = b"SG8\x01"
_GROUPED_LUT2_MAGIC_BYTES = {
    b"TR2\x03",
    b"TR2\x04",
    b"TR2\x05",
    b"TR2\x06",
    b"TR2\x07",
}

_NUMBA_GEMV = None
_NUMBA_GROUPED_LUT2_GEMV = None
_NUMBA_GROUPED_U8_GEMV = None
_LUT2_PACKED_CACHE: OrderedDict[
    tuple[int, int, int], tuple[np.ndarray, np.ndarray, bytes | memoryview]
] = OrderedDict()
_LUT2_PACKED_CACHE_MAX = 512
_LUT2_NATIVE_PK_CACHE: dict[tuple[int, int, int], ctypes.Array] = {}
_LUT2_PACKED_CACHE_LIMIT_BYTES = 0
_LUT2_PACKED_CACHE_USED_BYTES = 0


def _lut_kernel_pref() -> str:
    raw = os.environ.get("RWKV_LUT_KERNEL", "auto").strip().lower()
    aliases = {
        "native": "c",
        "dll": "c",
        "python": "numpy",
        "np": "numpy",
    }
    return aliases.get(raw, raw)


def activation_int8_enabled(*, head: bool = False) -> bool:
    """Return whether experimental per-group INT8 activation GEMV is enabled.

    The packed SG8 weights are unchanged, but quantizing each activation
    group changes reduction values slightly.  Keep the optimization opt-in
    until a model-specific quality certificate accepts it.  The vocabulary
    head has a separate switch because a small final-logit perturbation can
    change greedy token selection even when intermediate activations remain
    close.
    """
    name = "RWKV_LUT_ACTIVATION_INT8_HEAD" if head else "RWKV_LUT_ACTIVATION_INT8"
    raw = os.environ.get(name, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def activation_fp32_enabled() -> bool:
    """Return whether fused CPU decode should keep activations in FP32.

    Native packed GEMV already consumes FP32 input arrays.  Keeping the
    surrounding RWKV-7 activation path in FP32 avoids a BF16 -> FP32 -> BF16
    round-trip around every packed projection on CPUs without native BF16
    arithmetic.  ``auto`` enables it only when Torch reports that the host
    lacks AVX512-BF16; explicit ``0``/``1`` remain available for benchmarking
    and troubleshooting.
    """
    raw = os.environ.get("RWKV_LUT_ACTIVATION_FP32", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    if raw not in ("", "auto"):
        return False
    try:
        probe = getattr(getattr(torch, "cpu", None), "_is_avx512_bf16_supported", None)
        if callable(probe):
            return not bool(probe())
    except Exception:
        pass
    # Unknown CPU feature reporting should retain the conservative behavior.
    return False


def grouped_u8_cmix_fused_enabled() -> bool:
    """Return whether the native grouped-U8 CMix fusion is enabled.

    The fused key/ReLU^2/value kernel is algebraically equivalent to the
    existing two-GEMV path, but an explicit off switch is useful when
    diagnosing a native ABI or reduction-order issue.  ``auto`` is the
    production default; the Python fallback remains available when an older
    DLL does not export the optional symbol.
    """
    raw = os.environ.get("RWKV_LUT_CMIX_FUSED", "auto").strip().lower()
    return raw not in ("0", "false", "no", "off")


def clear_lut2_packed_cache() -> None:
    global _LUT2_PACKED_CACHE_USED_BYTES
    _LUT2_PACKED_CACHE.clear()
    _LUT2_NATIVE_PK_CACHE.clear()
    _LUT2_PACKED_CACHE_USED_BYTES = 0


def set_lut2_packed_cache_limit(limit_bytes: int) -> None:
    """Set a process-wide byte cap for unpacked LUT index arrays.

    ``0`` means unlimited (apart from the historical entry-count safety cap).
    The cache is process-wide because native GEMV keeps stable pointers, so a
    provider can configure it once when it is constructed.
    """
    global _LUT2_PACKED_CACHE_LIMIT_BYTES
    _LUT2_PACKED_CACHE_LIMIT_BYTES = max(0, int(limit_bytes))
    _trim_lut2_packed_cache()


def lut2_packed_cache_stats() -> dict[str, int]:
    """Return allocator-visible packed-cache counters for telemetry/tests."""
    return {
        "bytes": int(_LUT2_PACKED_CACHE_USED_BYTES),
        "entries": len(_LUT2_PACKED_CACHE),
        "native_entries": len(_LUT2_NATIVE_PK_CACHE),
    }


def _cache_entry_bytes(entry: tuple[np.ndarray, np.ndarray, object]) -> int:
    return int(entry[0].nbytes + entry[1].nbytes)


def _trim_lut2_packed_cache() -> None:
    global _LUT2_PACKED_CACHE_USED_BYTES
    while _LUT2_PACKED_CACHE and (
        len(_LUT2_PACKED_CACHE) > _LUT2_PACKED_CACHE_MAX
        or (
            _LUT2_PACKED_CACHE_LIMIT_BYTES > 0
            and _LUT2_PACKED_CACHE_USED_BYTES > _LUT2_PACKED_CACHE_LIMIT_BYTES
        )
    ):
        key, entry = _LUT2_PACKED_CACHE.popitem(last=False)
        _LUT2_PACKED_CACHE_USED_BYTES -= _cache_entry_bytes(entry)
        _LUT2_NATIVE_PK_CACHE.pop(key, None)


def _blob_cache_key(
    weight_blob: bytes | memoryview, out_features: int, in_features: int
) -> tuple[int, int, int]:
    # Use the view identity, not ``id(view.obj)``.  Multiple tensors can be
    # slices of one mmap, and using the exporter identity aliases unrelated
    # layers with the same shape.
    blob_id = id(weight_blob)
    return (blob_id, out_features, in_features)


def _lut2_entry_stub(out_features: int, in_features: int) -> object:
    return type(
        "E",
        (),
        {
            "name": "w",
            "numel": out_features * in_features,
            "shape": [out_features, in_features],
        },
    )()


def _lut2_arrays(
    weight_blob: bytes | memoryview,
    *,
    out_features: int,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (codebook f32, packed uint8) with a bounded cache.

    The packed indices are a read-only NumPy view when the source exposes a
    stable buffer (mmap/bytes).  The native C path may create its own writable
    pointer fallback, but the Python/Numba paths no longer duplicate every
    layer unconditionally.
    """
    global _LUT2_PACKED_CACHE_USED_BYTES
    key = _blob_cache_key(weight_blob, out_features, in_features)
    hit = _LUT2_PACKED_CACHE.get(key)
    if hit is not None:
        _LUT2_PACKED_CACHE.move_to_end(key)
        return hit[0], hit[1]
    entry_stub = _lut2_entry_stub(out_features, in_features)
    codebook, idx_view, _row_stride = _lut2_inner_view(weight_blob, entry_stub)  # type: ignore[arg-type]
    packed = np.frombuffer(idx_view, dtype=np.uint8)
    cb = np.asarray(codebook, dtype=np.float32)
    entry = (cb, packed, weight_blob)
    _LUT2_PACKED_CACHE[key] = entry
    _LUT2_PACKED_CACHE_USED_BYTES += _cache_entry_bytes(entry)
    _trim_lut2_packed_cache()
    return cb, packed


def _lut2_native_packed_view(
    weight_blob: bytes | memoryview,
    packed: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> ctypes.Array:
    """Stable ctypes view for native GEMV (one per blob identity)."""
    key = _blob_cache_key(weight_blob, out_features, in_features)
    hit = _LUT2_NATIVE_PK_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        pk = (ctypes.c_uint8 * len(packed)).from_buffer(memoryview(packed))
    except (TypeError, BufferError):
        # mmap/bytes views are read-only.  Keep the zero-copy NumPy view for
        # Python/Numba and make one explicit native pointer copy only when the
        # C ABI requires writable storage.
        pk = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    _LUT2_NATIVE_PK_CACHE[key] = pk
    _trim_lut2_packed_cache()
    return pk


def _native_blob_view(
    blob: bytes | memoryview | bytearray,
) -> tuple[np.ndarray, ctypes.POINTER(ctypes.c_uint8)]:
    """Expose a packed blob to the native ABI without expanding its weights."""
    view = np.frombuffer(blob, dtype=np.uint8)
    ptr = view.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    return view, ptr


def _tensor_to_f32_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach()
    if x.device.type != "cpu":
        x = x.float().cpu()
    elif x.dtype != torch.float32:
        x = x.float()
    return x.numpy().ravel()


def fused_lut2_enabled(
    mode: str | None = None,
    stream_layer_cache: bool = False,
    pack_uses_quant: bool | None = None,
) -> bool:
    raw = os.environ.get("RWKV_LUT_GEMM_FUSED", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if pack_uses_quant is False:
        return False
    if raw in ("1", "true", "yes", "on"):
        if mode is not None and mode not in ("streaming", "partial"):
            return False
        return True
    if raw in ("", "auto"):
        if mode not in ("streaming", "partial"):
            return False
        if not stream_layer_cache:
            return True
        from rwkv_ssd.runtime.rwkv7_weights import promote_full_z_enabled

        return not promote_full_z_enabled()
    return False


_ATT_FUSED_SUFFIXES = (
    "receptance.weight",
    "key.weight",
    "value.weight",
    "output.weight",
)

_ATT_FUSED_ORDER = _ATT_FUSED_SUFFIXES

# These are the compact RWKV-7 TMix projections evaluated as ``x @ W``.
# They remain dense in ``z`` for compatibility, but grouped-U8 packs can also
# retain their raw blobs and use the transposed native GEMV without decoding a
# BF16 matrix on every layer/token.
_ATT_TRANSPOSED_FUSED_SUFFIXES = (
    "w1",
    "w2",
    "a1",
    "a2",
    "v1",
    "v2",
    "g1",
    "g2",
)


def att_fused_suffixes() -> tuple[str, ...]:
    return _ATT_FUSED_ORDER


def att_prefix_from_key(key: str) -> str | None:
    if ".att." not in key:
        return None
    for suffix in _ATT_FUSED_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def is_fused_lut_transposed_tensor_name(name: str) -> bool:
    """Return whether ``name`` is a compact TMix matrix for ``x @ W``."""
    if ".att." not in name:
        return False
    return any(name.endswith(f".att.{suffix}") for suffix in _ATT_TRANSPOSED_FUSED_SUFFIXES)


def small_transposed_lut_enabled() -> bool:
    """Whether the grouped-U8 small-TMix transpose path is enabled.

    ``auto`` is the production default.  The path is a win for the real
    grouped-U8 CPU engine because it avoids hundreds of tiny Torch ``mm``
    launches per token, but its NumPy fallback is not a safe default on a
    machine that has not built the native AVX2 kernel.  Keep explicit ``1``
    available for experiments and explicit ``0`` for dense-BF16 A/B tests.
    """
    raw = os.environ.get("RWKV_LUT_SMALL_TRANSPOSED", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    if raw not in ("", "auto"):
        return False
    kernel = _lut_kernel_pref()
    if kernel in ("numpy", "numba", "python", "np"):
        return False
    try:
        from rwkv_ssd.native.lut2_gather_loader import lib

        native = lib()
        return bool(
            native is not None
            and hasattr(native, "scale_u8_grouped_transposed_tmix_gemv_f32_export")
        )
    except (ImportError, OSError):
        return False


def prefer_fused_lut_over_disk_cache() -> bool:
    """Skip ``.decode_cache/`` hits when fused LUT blobs avoid bf16 materialization."""
    import os

    raw = os.environ.get("RWKV_PREFER_FUSED_LUT", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    from rwkv_ssd.runtime.rwkv7_weights import promote_full_z_enabled

    return not promote_full_z_enabled()


def filter_tensors_for_fused_inject(
    provider: object | None,
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop fused weight slabs when LUT blobs are registered — keeps ``z`` lean for packed forward."""
    if not tensors or provider is None:
        return tensors
    use_fused = getattr(provider, "_use_fused_lut_matmul", None)
    if not callable(use_fused) or not use_fused():
        return tensors
    get_blob = getattr(provider, "get_fused_lut_blob", None)
    get_tmix = getattr(provider, "get_fused_tmix_blobs", None)
    if get_blob is None:
        return tensors
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not is_fused_lut_tensor_name(name):
            # The compact grouped-U8 TMix adapters use the transposed
            # ``x @ W`` kernel.  They are not part of the four large
            # attention/FFN matrices above, but once their raw SG8 blob is
            # registered the dense BF16 decode is equally redundant.
            if (
                is_fused_lut_transposed_tensor_name(name)
                and get_blob(name)
            ):
                continue
            out[name] = tensor
            continue
        if get_blob(name):
            continue
        prefix = att_prefix_from_key(name)
        if prefix and get_tmix and get_tmix(prefix):
            continue
        out[name] = tensor
    return out


def is_fused_lut_tensor_name(name: str) -> bool:
    if name == "head.weight":
        return True
    if ".ffn." in name and name.endswith(("key.weight", "value.weight")):
        return True
    if ".att." in name:
        for suffix in _ATT_FUSED_SUFFIXES:
            if name.endswith(suffix):
                return True
    return False


def _lut2_gemv_numpy(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    cb, packed = _lut2_arrays(
        weight_blob, out_features=out_features, in_features=in_features
    )
    y = np.empty(out_features, dtype=np.float32)
    cols = in_features
    for row in range(out_features):
        acc = 0.0
        row_base = row * cols
        for c in range(cols):
            i = row_base + c
            b = packed[i // 4]
            shift = (i % 4) * 2
            acc += cb[(b >> shift) & 3] * x_np[c]
        y[row] = acc
    return y


def _lut2_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    import ctypes

    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    if native is None:
        raise RuntimeError("native lut2_gather not loaded")
    cb, packed = _lut2_arrays(
        weight_blob, out_features=out_features, in_features=in_features
    )
    y = np.empty(out_features, dtype=np.float32)
    pk = _lut2_native_packed_view(
        weight_blob, packed, out_features=out_features, in_features=in_features
    )
    cb_f = (ctypes.c_float * 4).from_buffer(memoryview(cb))
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(
        memoryview(x_np.astype(np.float32, copy=False))
    )
    native.lut2_gemv_f32_export(y_f, cb_f, pk, x_f, int(out_features), int(in_features))
    return y


def _grouped_lut2_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "trinity_grouped_lut2_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped LUT2 native GEMV not loaded")
    y = np.empty(out_features, dtype=np.float32)
    x_f32 = x_np.astype(np.float32, copy=False)
    blob_view, blob_ptr = _native_blob_view(weight_blob)
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(memoryview(x_f32))
    fn(y_f, blob_ptr, x_f, int(out_features), int(in_features))
    # Keep the NumPy exporter alive until ctypes returns from the call.
    del blob_view
    return y


def _scale_u8_grouped_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 native GEMV not loaded")
    y = np.empty(out_features, dtype=np.float32)
    x_f32 = x_np.astype(np.float32, copy=False)
    blob_view, blob_ptr = _native_blob_view(weight_blob)
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(memoryview(x_f32))
    fn(y_f, blob_ptr, x_f, int(out_features), int(in_features))
    del blob_view
    return y


def _scale_u8_grouped_i8_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Native SG8 GEMV with per-input-group INT8 activation quantization."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_i8_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 activation-INT8 GEMV not loaded")
    y = np.empty(out_features, dtype=np.float32)
    x_f32 = x_np.astype(np.float32, copy=False)
    blob_view, blob_ptr = _native_blob_view(weight_blob)
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(memoryview(x_f32))
    fn(y_f, blob_ptr, x_f, int(out_features), int(in_features))
    del blob_view
    return y


def _scale_u8_grouped_transposed_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Native SG8 GEMV for a row-major W used as ``x @ W``."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_transposed_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 transposed native GEMV not loaded")
    y = np.empty(out_features, dtype=np.float32)
    x_f32 = x_np.astype(np.float32, copy=False)
    blob_view, blob_ptr = _native_blob_view(weight_blob)
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(memoryview(x_f32))
    fn(y_f, blob_ptr, x_f, int(out_features), int(in_features))
    del blob_view
    return y


def _scale_u8_grouped_transposed_tmix_gemv_native(
    blobs: tuple[bytes | memoryview, ...],
    xs: tuple[torch.Tensor, ...],
    shapes: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, ...]:
    """Batch independent SG8 ``x @ W`` TMix adapter projections."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_transposed_tmix_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 transposed TMix native GEMV not loaded")
    n_mats = len(blobs)
    if not (0 < n_mats <= 4) or len(xs) != n_mats or len(shapes) != n_mats:
        raise ValueError("transposed TMix batch must contain one to four maps")
    outputs: list[np.ndarray] = []
    y_ptrs = (ctypes.POINTER(ctypes.c_float) * n_mats)()
    blob_ptrs = (ctypes.POINTER(ctypes.c_uint8) * n_mats)()
    x_ptrs = (ctypes.POINTER(ctypes.c_float) * n_mats)()
    out_dims = (ctypes.c_int * n_mats)()
    in_dims = (ctypes.c_int * n_mats)()
    keep_blobs: list[np.ndarray] = []
    keep_x: list[np.ndarray] = []
    keep_y: list[np.ndarray] = []
    for index, (blob, tensor, shape) in enumerate(zip(blobs, xs, shapes)):
        rows, cols = int(shape[0]), int(shape[1])
        x_np = _tensor_to_f32_np(tensor)
        if x_np.size != rows:
            raise ValueError(f"x size {x_np.size} != in_features {rows}")
        y_np = np.empty(cols, dtype=np.float32)
        blob_view, blob_ptr = _native_blob_view(blob)
        y_f = (ctypes.c_float * cols).from_buffer(y_np)
        x_f = (ctypes.c_float * rows).from_buffer(memoryview(x_np))
        y_ptrs[index] = ctypes.cast(y_f, ctypes.POINTER(ctypes.c_float))
        blob_ptrs[index] = blob_ptr
        x_ptrs[index] = ctypes.cast(x_f, ctypes.POINTER(ctypes.c_float))
        out_dims[index] = cols
        in_dims[index] = rows
        keep_blobs.append(blob_view)
        keep_x.append(x_np)
        keep_y.append(y_np)
    fn(y_ptrs, blob_ptrs, x_ptrs, out_dims, in_dims, int(n_mats))
    return tuple(torch.from_numpy(y) for y in keep_y)


def _scale_u8_grouped_transposed_tmix_fused_native(
    blobs: tuple[bytes | memoryview, ...],
    xs: tuple[torch.Tensor, ...],
    shapes: tuple[tuple[int, int], ...],
    a0: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    """Run the complete compact RWKV-7 TMix adapter pipeline in C.

    ``blobs`` is ordered as w1,w2,a1,a2,g1,g2 and optionally v1,v2.  The
    native function keeps the rank-sized intermediates and tanh/sigmoid
    operations out of Python/Torch.  Returning ``None`` means that an older
    native DLL does not have the optional symbol; callers retain the existing
    two-batch fallback for that case.
    """
    if len(blobs) not in (6, 8) or len(xs) != (4 if len(blobs) == 8 else 3):
        raise ValueError("fused TMix adapter batch has an invalid width")
    if len(shapes) != len(blobs):
        raise ValueError("fused TMix adapter shapes do not match blobs")
    if any(bytes(blob[:4]) != _SCALE_U8_GROUPED_MAGIC for blob in blobs):
        raise ValueError("fused TMix adapters require grouped-U8 blobs")
    if xs[0].device.type != "cpu":
        return None

    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_transposed_tmix_fused_f32_export", None)
    if fn is None:
        return None

    n_embd = int(shapes[1][1])
    ranks = (int(shapes[0][1]), int(shapes[2][1]), int(shapes[4][1]), 0)
    if len(blobs) == 8:
        ranks = (ranks[0], ranks[1], ranks[2], int(shapes[6][1]))
    if n_embd <= 0 or any(rank <= 0 for rank in ranks[:3]):
        raise ValueError("fused TMix adapter shapes must be positive")
    if a0.numel() != n_embd:
        raise ValueError(f"TMix a0 size {a0.numel()} != model width {n_embd}")

    blob_ptrs = (ctypes.POINTER(ctypes.c_uint8) * 8)()
    x_ptrs = (ctypes.POINTER(ctypes.c_float) * 4)()
    rank_dims = (ctypes.c_int * 4)(*ranks)
    keep_blobs: list[np.ndarray] = []
    keep_x: list[np.ndarray] = []
    for index, blob in enumerate(blobs):
        blob_view, blob_ptr = _native_blob_view(blob)
        blob_ptrs[index] = blob_ptr
        keep_blobs.append(blob_view)
    for index, tensor in enumerate(xs):
        x_np = _tensor_to_f32_np(tensor)
        expected = int(shapes[index * 2][0])
        if x_np.size != expected:
            raise ValueError(
                f"TMix adapter input {index} size {x_np.size} != {expected}"
            )
        x_f = (ctypes.c_float * expected).from_buffer(memoryview(x_np))
        x_ptrs[index] = ctypes.cast(x_f, ctypes.POINTER(ctypes.c_float))
        keep_x.append(x_np)

    a0_np = _tensor_to_f32_np(a0)
    a0_f = (ctypes.c_float * n_embd).from_buffer(memoryview(a0_np))
    w_np = np.empty(n_embd, dtype=np.float32)
    a_np = np.empty(n_embd, dtype=np.float32)
    g_np = np.empty(n_embd, dtype=np.float32)
    v_np = np.empty(n_embd, dtype=np.float32) if len(blobs) == 8 else None
    w_f = (ctypes.c_float * n_embd).from_buffer(w_np)
    a_f = (ctypes.c_float * n_embd).from_buffer(a_np)
    g_f = (ctypes.c_float * n_embd).from_buffer(g_np)
    v_f = (
        (ctypes.c_float * n_embd).from_buffer(v_np)
        if v_np is not None
        else ctypes.POINTER(ctypes.c_float)()
    )
    fn(
        w_f,
        a_f,
        g_f,
        v_f,
        blob_ptrs,
        x_ptrs,
        a0_f,
        n_embd,
        rank_dims,
        int(v_np is not None),
    )
    # Keep all exporters and ctypes views alive through the native call.
    del keep_blobs, keep_x, a0_f, w_f, a_f, g_f, v_f
    result = [
        _lut2_torch_from_f32(w_np, xs[0]),
        _lut2_torch_from_f32(a_np, xs[0]),
        _lut2_torch_from_f32(g_np, xs[0]),
    ]
    if v_np is not None:
        result.append(_lut2_torch_from_f32(v_np, xs[0]))
    return tuple(result)


def grouped_u8_transposed_tmix_fused(
    blobs: tuple[bytes | memoryview, ...],
    xs: tuple[torch.Tensor, ...],
    shapes: tuple[tuple[int, int], ...],
    a0: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    """Optional C fast path for the two-sweep TMix adapter pipeline."""
    raw = os.environ.get("RWKV_LUT_FUSED_ADAPTERS", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return None
    kernel = _lut_kernel_pref()
    if kernel not in ("c", "auto"):
        return None
    try:
        return _scale_u8_grouped_transposed_tmix_fused_native(
            blobs, xs, shapes, a0
        )
    except Exception:
        if kernel == "c":
            raise
        return None


def _scale_u8_grouped_sparse_gemv_native(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    active_indices: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Exact grouped-U8 GEMV over a sorted list of nonzero input columns."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_sparse_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 sparse native GEMV not loaded")
    y = np.empty(out_features, dtype=np.float32)
    x_f32 = x_np.astype(np.float32, copy=False)
    active = np.ascontiguousarray(active_indices, dtype=np.int32)
    blob_view, blob_ptr = _native_blob_view(weight_blob)
    y_f = (ctypes.c_float * out_features).from_buffer(y)
    x_f = (ctypes.c_float * in_features).from_buffer(memoryview(x_f32))
    active_i = (ctypes.c_int32 * int(active.size)).from_buffer(memoryview(active))
    fn(
        y_f,
        blob_ptr,
        x_f,
        active_i,
        int(active.size),
        int(out_features),
        int(in_features),
    )
    # Keep every NumPy exporter alive until ctypes returns from the call.
    del blob_view, active_i
    return y


def _grouped_lut2_arrays(
    weight_blob: bytes | memoryview,
    *,
    out_features: int,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None, np.ndarray | None]:
    """Parse grouped LUT metadata while leaving the 2-bit payload packed."""
    magic = bytes(weight_blob[:4])
    if not is_grouped_lut2_blob(weight_blob):
        raise ValueError("not a grouped LUT2 blob")
    if len(weight_blob) < 8:
        raise ValueError("grouped LUT2 blob is too short")
    group_size = int.from_bytes(bytes(weight_blob[4:8]), "little")
    if group_size <= 0:
        raise ValueError("invalid grouped LUT2 group size")
    numel = out_features * in_features
    groups = (numel + group_size - 1) // group_size
    residual = magic in {b"TR2\x05", b"TR2\x07"}
    fp32 = magic == b"TR2\x03"
    record_bytes = 16 if residual or fp32 else 8
    codebook_end = 8 + groups * record_bytes
    need = codebook_end + (numel + 3) // 4
    if len(weight_blob) < need:
        raise ValueError("grouped LUT2 payload is too short")
    if fp32:
        codebooks = np.frombuffer(
            weight_blob, dtype=np.float32, offset=8, count=groups * 4
        ).reshape(groups, 4)
    else:
        codebooks = np.ndarray(
            (groups, 4), dtype=np.float16, buffer=weight_blob, offset=8,
            strides=(record_bytes, 2),
        ).astype(np.float32)
    packed = np.frombuffer(weight_blob, dtype=np.uint8, offset=codebook_end)
    positions = deltas = None
    if residual:
        positions = np.ndarray(
            (groups, 2), dtype=np.uint8, buffer=weight_blob, offset=16,
            strides=(record_bytes, 1),
        )
        deltas = np.ndarray(
            (groups, 2), dtype=np.float16, buffer=weight_blob, offset=18,
            strides=(record_bytes, 2),
        ).astype(np.float32)
    input_layout = magic in {b"TR2\x06", b"TR2\x07"}
    return codebooks, packed, group_size, input_layout, positions, deltas


def _grouped_lut2_gemv_numpy(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    (
        codebooks,
        packed,
        group_size,
        input_layout,
        positions,
        deltas,
    ) = _grouped_lut2_arrays(
        weight_blob, out_features=out_features, in_features=in_features
    )
    y = np.empty(out_features, dtype=np.float32)
    for row in range(out_features):
        acc = 0.0
        for col in range(in_features):
            logical = (
                col * out_features + row
                if input_layout
                else row * in_features + col
            )
            index = (int(packed[logical // 4]) >> ((logical % 4) * 2)) & 3
            weight = float(codebooks[logical // group_size, index])
            if positions is not None and deltas is not None:
                local = logical % group_size
                if local == int(positions[logical // group_size, 0]):
                    weight += float(deltas[logical // group_size, 0])
                elif local == int(positions[logical // group_size, 1]):
                    weight += float(deltas[logical // group_size, 1])
            acc += weight * float(x_np[col])
        y[row] = acc
    return y


def _scale_u8_grouped_gemv_numpy(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    if bytes(weight_blob[:4]) != _SCALE_U8_GROUPED_MAGIC or len(weight_blob) < 8:
        raise ValueError("not a grouped U8 blob")
    group_size = int.from_bytes(bytes(weight_blob[4:8]), "little")
    if group_size <= 0:
        raise ValueError("invalid grouped U8 group size")
    numel = out_features * in_features
    groups = (numel + group_size - 1) // group_size
    scales_end = 8 + groups * 8
    if len(weight_blob) < scales_end + numel:
        raise ValueError("grouped U8 payload is too short")
    scales = np.frombuffer(
        weight_blob, dtype=np.float32, offset=8, count=groups * 2
    ).reshape(groups, 2)
    quant = np.frombuffer(weight_blob, dtype=np.uint8, offset=scales_end, count=numel)
    y = np.empty(out_features, dtype=np.float32)
    for row in range(out_features):
        acc = 0.0
        for col in range(in_features):
            logical = row * in_features + col
            group = logical // group_size
            weight = float(
                scales[group, 0]
                + (scales[group, 1] - scales[group, 0])
                * (float(quant[logical]) / 255.0)
            )
            acc += weight * float(x_np[col])
        y[row] = acc
    return y


def _scale_u8_grouped_transposed_gemv_numpy(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Portable fallback for row-major SG8 ``x @ W`` projections."""
    if bytes(weight_blob[:4]) != _SCALE_U8_GROUPED_MAGIC or len(weight_blob) < 8:
        raise ValueError("not a grouped U8 blob")
    group_size = int.from_bytes(bytes(weight_blob[4:8]), "little")
    if group_size <= 0:
        raise ValueError("invalid grouped U8 group size")
    numel = out_features * in_features
    groups = (numel + group_size - 1) // group_size
    scales_end = 8 + groups * 8
    if len(weight_blob) < scales_end + numel:
        raise ValueError("grouped U8 payload is too short")
    scales = np.frombuffer(
        weight_blob, dtype=np.float32, offset=8, count=groups * 2
    ).reshape(groups, 2)
    quant = np.frombuffer(weight_blob, dtype=np.uint8, offset=scales_end, count=numel)
    y = np.empty(out_features, dtype=np.float32)
    inv_255 = 1.0 / 255.0
    for col in range(out_features):
        acc = 0.0
        for row in range(in_features):
            logical = row * out_features + col
            group = logical // group_size
            mn = scales[group, 0]
            mx = scales[group, 1]
            acc += (mn + (mx - mn) * (float(quant[logical]) * inv_255)) * float(x_np[row])
        y[col] = acc
    return y


def _grouped_lut2_gemv_numba(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    global _NUMBA_GROUPED_LUT2_GEMV
    if _NUMBA_GROUPED_LUT2_GEMV is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _gemv(
            codebooks,
            packed,
            x,
            out_f,
            in_f,
            group_size,
            input_layout,
            positions,
            deltas,
            residual,
        ):
            y = np.empty(out_f, dtype=np.float32)
            for row in prange(out_f):
                acc = 0.0
                for col in range(in_f):
                    logical = (
                        col * out_f + row
                        if input_layout
                        else row * in_f + col
                    )
                    group = logical // group_size
                    local = logical % group_size
                    b = packed[logical // 4]
                    index = (b >> ((logical % 4) * 2)) & 3
                    weight = codebooks[group, index]
                    if residual:
                        if local == positions[group, 0]:
                            weight += deltas[group, 0]
                        elif local == positions[group, 1]:
                            weight += deltas[group, 1]
                    acc += weight * x[col]
                y[row] = acc
            return y

        _NUMBA_GROUPED_LUT2_GEMV = _gemv
    (
        codebooks,
        packed,
        group_size,
        input_layout,
        positions,
        deltas,
    ) = _grouped_lut2_arrays(
        weight_blob, out_features=out_features, in_features=in_features
    )
    groups = codebooks.shape[0]
    if positions is None:
        positions = np.zeros((groups, 2), dtype=np.uint8)
        deltas = np.zeros((groups, 2), dtype=np.float32)
        residual = False
    else:
        assert deltas is not None
        residual = True
    return _NUMBA_GROUPED_LUT2_GEMV(
        codebooks,
        packed,
        x_np.astype(np.float32, copy=False),
        int(out_features),
        int(in_features),
        int(group_size),
        bool(input_layout),
        positions,
        deltas,
        residual,
    )


def _scale_u8_grouped_gemv_numba(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    global _NUMBA_GROUPED_U8_GEMV
    if _NUMBA_GROUPED_U8_GEMV is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _gemv(scales, quant, x, out_f, in_f, group_size):
            y = np.empty(out_f, dtype=np.float32)
            for row in prange(out_f):
                acc = 0.0
                row_base = row * in_f
                group = row_base // group_size
                group_end = (group + 1) * group_size
                cached_mn = scales[group, 0]
                cached_mx = scales[group, 1]
                for col in range(in_f):
                    logical = row_base + col
                    if logical >= group_end:
                        group += 1
                        group_end += group_size
                        cached_mn = scales[group, 0]
                        cached_mx = scales[group, 1]
                    weight = cached_mn + (cached_mx - cached_mn) * (quant[logical] / 255.0)
                    acc += weight * x[col]
                y[row] = acc
            return y

        _NUMBA_GROUPED_U8_GEMV = _gemv
    if bytes(weight_blob[:4]) != _SCALE_U8_GROUPED_MAGIC or len(weight_blob) < 8:
        raise ValueError("not a grouped U8 blob")
    group_size = int.from_bytes(bytes(weight_blob[4:8]), "little")
    if group_size <= 0:
        raise ValueError("invalid grouped U8 group size")
    numel = out_features * in_features
    groups = (numel + group_size - 1) // group_size
    scales_end = 8 + groups * 8
    if len(weight_blob) < scales_end + numel:
        raise ValueError("grouped U8 payload is too short")
    scales = np.frombuffer(
        weight_blob, dtype=np.float32, offset=8, count=groups * 2
    ).reshape(groups, 2)
    quant = np.frombuffer(weight_blob, dtype=np.uint8, offset=scales_end, count=numel)
    return _NUMBA_GROUPED_U8_GEMV(
        scales,
        quant,
        x_np.astype(np.float32, copy=False),
        int(out_features),
        int(in_features),
        int(group_size),
    )


def _lut2_gemv_numba(
    weight_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    global _NUMBA_GEMV
    if _NUMBA_GEMV is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _gemv(cb, packed, x, out_f, in_f):
            y = np.empty(out_f, dtype=np.float32)
            for row in prange(out_f):
                acc = 0.0
                row_base = row * in_f
                for col in range(in_f):
                    i = row_base + col
                    b = packed[i // 4]
                    shift = (i % 4) * 2
                    acc += cb[(b >> shift) & 3] * x[col]
                y[row] = acc
            return y

        _NUMBA_GEMV = _gemv

    cb, packed = _lut2_arrays(
        weight_blob, out_features=out_features, in_features=in_features
    )
    x_f = x_np.astype(np.float32, copy=False)
    return _NUMBA_GEMV(cb, packed, x_f, int(out_features), int(in_features))


def lut2_gemv(
    weight_blob: bytes | memoryview,
    x: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    activation_int8: bool | None = None,
) -> torch.Tensor:
    """``y = W @ x`` with Trinity LUT2 ``W`` packed in ``weight_blob``."""
    x_np = _tensor_to_f32_np(x)
    if x_np.size != in_features:
        raise ValueError(f"x size {x_np.size} != in_features {in_features}")
    kernel = _lut_kernel_pref()
    magic = bytes(weight_blob[:4])
    grouped = is_grouped_lut2_blob(weight_blob)
    grouped_u8 = magic == _SCALE_U8_GROUPED_MAGIC
    if activation_int8 is None:
        activation_int8 = activation_int8_enabled()
    if grouped or grouped_u8:
        native_fn = (
            _grouped_lut2_gemv_native if grouped else _scale_u8_grouped_gemv_native
        )
        numba_fn = (
            _grouped_lut2_gemv_numba if grouped else _scale_u8_grouped_gemv_numba
        )
        numpy_fn = (
            _grouped_lut2_gemv_numpy if grouped else _scale_u8_grouped_gemv_numpy
        )
        if kernel in ("c", "auto"):
            try:
                if grouped_u8 and activation_int8:
                    y_np = _scale_u8_grouped_i8_gemv_native(
                        weight_blob,
                        x_np,
                        out_features=out_features,
                        in_features=in_features,
                    )
                else:
                    y_np = native_fn(
                        weight_blob, x_np,
                        out_features=out_features, in_features=in_features,
                    )
                return _lut2_torch_from_f32(y_np, x)
            except Exception:
                if kernel == "c":
                    raise
        if kernel in ("numba", "auto"):
            try:
                y_np = numba_fn(
                    weight_blob, x_np,
                    out_features=out_features, in_features=in_features,
                )
                return _lut2_torch_from_f32(y_np, x)
            except Exception:
                if kernel == "numba":
                    raise
        # The direct Python fallback still reads compressed metadata and the
        # packed payload; it never constructs a dense weight matrix.
        y_np = numpy_fn(
            weight_blob, x_np,
            out_features=out_features, in_features=in_features,
        )
        return _lut2_torch_from_f32(y_np, x)
    if kernel in ("c", "auto"):
        try:
            from rwkv_ssd.native.lut2_gather_loader import lib

            if lib() is None:
                raise RuntimeError("no native")
            y_np = _lut2_gemv_native(
                weight_blob,
                x_np,
                out_features=out_features,
                in_features=in_features,
            )
            return _lut2_torch_from_f32(y_np, x)
        except Exception:
            if kernel == "c":
                raise
    if kernel in ("numba", "auto"):
        try:
            y_np = _lut2_gemv_numba(
                weight_blob,
                x_np,
                out_features=out_features,
                in_features=in_features,
            )
            return _lut2_torch_from_f32(y_np, x)
        except Exception:
            if kernel == "numba":
                raise
    y_np = _lut2_gemv_numpy(
        weight_blob,
        x_np,
        out_features=out_features,
        in_features=in_features,
    )
    return _lut2_torch_from_f32(y_np, x)


def grouped_u8_transposed_gemv(
    weight_blob: bytes | memoryview,
    x: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Compute ``x @ W`` from a row-major grouped-U8 ``W`` blob.

    The ordinary grouped-U8 ABI computes ``W @ x`` for the large RWKV maps.
    The compact TMix projections are stored in their natural checkpoint
    layout and are consumed as ``x @ W``; this path walks that layout without
    materializing a BF16 matrix.
    """
    if bytes(weight_blob[:4]) != _SCALE_U8_GROUPED_MAGIC:
        raise ValueError("grouped_u8_transposed_gemv requires an SG8 blob")
    x_np = _tensor_to_f32_np(x)
    if x_np.size != in_features:
        raise ValueError(f"x size {x_np.size} != in_features {in_features}")
    kernel = _lut_kernel_pref()
    if kernel in ("c", "auto"):
        try:
            y_np = _scale_u8_grouped_transposed_gemv_native(
                weight_blob,
                x_np,
                out_features=out_features,
                in_features=in_features,
            )
            return _lut2_torch_from_f32(y_np, x)
        except Exception:
            if kernel == "c":
                raise
    y_np = _scale_u8_grouped_transposed_gemv_numpy(
        weight_blob,
        x_np,
        out_features=out_features,
        in_features=in_features,
    )
    return _lut2_torch_from_f32(y_np, x)


def grouped_u8_transposed_tmix_gemv(
    blobs: tuple[bytes | memoryview, ...],
    xs: tuple[torch.Tensor, ...],
    shapes: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, ...]:
    """Batch one to four independent grouped-U8 TMix adapter GEMVs."""
    if not all(bytes(blob[:4]) == _SCALE_U8_GROUPED_MAGIC for blob in blobs):
        raise ValueError("grouped_u8_transposed_tmix_gemv requires SG8 blobs")
    kernel = _lut_kernel_pref()
    if kernel in ("c", "auto"):
        try:
            values = _scale_u8_grouped_transposed_tmix_gemv_native(
                blobs, xs, shapes
            )
            return tuple(value.to(dtype=xs[0].dtype) for value in values)
        except Exception:
            if kernel == "c":
                raise
    return tuple(
        grouped_u8_transposed_gemv(
            blob,
            x,
            out_features=int(shape[1]),
            in_features=int(shape[0]),
        )
        for blob, x, shape in zip(blobs, xs, shapes)
    )


def grouped_u8_sparse_gemv(
    weight_blob: bytes | memoryview,
    x: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    active_fraction_threshold: float | None = None,
) -> torch.Tensor:
    """Run an exact sparse value GEMV when post-ReLU activations are sparse.

    The sparse ABI is an optimization for ``scale_u8_grouped`` CMix value
    matrices.  It skips only entries where ``x == 0`` and uses the same
    per-group affine reduction as the dense native kernel, so this does not
    introduce an additional approximation.  Dense dispatch is retained when
    the active fraction is high, when a legacy native DLL lacks the optional
    symbol, or when a non-native kernel is explicitly selected.
    """
    if bytes(weight_blob[:4]) != _SCALE_U8_GROUPED_MAGIC:
        return lut2_gemv(
            weight_blob,
            x,
            out_features=out_features,
            in_features=in_features,
        )
    x_np = _tensor_to_f32_np(x)
    if x_np.size != in_features:
        raise ValueError(f"x size {x_np.size} != in_features {in_features}")
    if active_fraction_threshold is None:
        try:
            active_fraction_threshold = float(
                os.environ.get("RWKV_LUT_SPARSE_THRESHOLD", "0.05")
            )
        except ValueError:
            active_fraction_threshold = 0.05
    threshold = min(1.0, max(0.0, float(active_fraction_threshold)))
    active = np.flatnonzero(x_np != 0).astype(np.int32, copy=False)
    if active.size == 0:
        return torch.zeros(
            out_features,
            dtype=x.dtype,
            device=x.device,
        )
    if active.size >= int(np.ceil(x_np.size * threshold)):
        return lut2_gemv(
            weight_blob,
            x,
            out_features=out_features,
            in_features=in_features,
        )
    if _lut_kernel_pref() in ("c", "auto"):
        try:
            y_np = _scale_u8_grouped_sparse_gemv_native(
                weight_blob,
                x_np,
                active,
                out_features=out_features,
                in_features=in_features,
            )
            return _lut2_torch_from_f32(y_np, x)
        except Exception:
            # Sparse support is optional in older/native bundles; the dense
            # grouped path remains the correctness-preserving fallback.
            pass
    return lut2_gemv(
        weight_blob,
        x,
        out_features=out_features,
        in_features=in_features,
    )


def _scale_u8_grouped_cmix_gemv_native(
    key_blob: bytes | memoryview,
    value_blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    key_out_features: int,
    key_in_features: int,
    value_out_features: int,
    value_in_features: int,
) -> np.ndarray:
    """Run native grouped-U8 CMix key/ReLU²/value fusion when available."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "scale_u8_grouped_cmix_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("grouped U8 fused CMix native GEMV not loaded")
    key_view, key_ptr = _native_blob_view(key_blob)
    value_view, value_ptr = _native_blob_view(value_blob)
    x_f32 = np.ascontiguousarray(x_np, dtype=np.float32)
    y = np.empty(value_out_features, dtype=np.float32)
    x_f = (ctypes.c_float * key_in_features).from_buffer(memoryview(x_f32))
    y_f = (ctypes.c_float * value_out_features).from_buffer(y)
    fn(
        y_f,
        key_ptr,
        value_ptr,
        x_f,
        int(key_out_features),
        int(key_in_features),
        int(value_out_features),
        int(value_in_features),
    )
    # Keep the blob exporters and input alive until ctypes returns.
    del key_view, value_view, x_f, y_f
    return y


def grouped_u8_cmix_gemv(
    key_blob: bytes | memoryview,
    value_blob: bytes | memoryview,
    x: torch.Tensor,
    *,
    key_out_features: int,
    key_in_features: int,
    value_out_features: int,
    value_in_features: int,
) -> torch.Tensor:
    """Compute grouped-U8 ``value @ relu(key @ x)^2`` without Torch staging."""
    if bytes(key_blob[:4]) != _SCALE_U8_GROUPED_MAGIC or bytes(
        value_blob[:4]
    ) != _SCALE_U8_GROUPED_MAGIC:
        raise ValueError("grouped_u8_cmix_gemv requires SG8 blobs")
    x_np = _tensor_to_f32_np(x)
    if x_np.size != key_in_features:
        raise ValueError(f"x size {x_np.size} != key_in_features {key_in_features}")
    kernel = _lut_kernel_pref()
    if kernel in ("c", "auto"):
        try:
            y_np = _scale_u8_grouped_cmix_gemv_native(
                key_blob,
                value_blob,
                x_np,
                key_out_features=key_out_features,
                key_in_features=key_in_features,
                value_out_features=value_out_features,
                value_in_features=value_in_features,
            )
            return _lut2_torch_from_f32(y_np, x)
        except Exception:
            if kernel == "c":
                raise
    key = lut2_gemv(
        key_blob,
        x,
        out_features=key_out_features,
        in_features=key_in_features,
    )
    key = torch.relu(key) ** 2
    return lut2_gemv(
        value_blob,
        key,
        out_features=value_out_features,
        in_features=value_in_features,
    )


def _lut2_torch_from_f32(y_np: np.ndarray, x: torch.Tensor) -> torch.Tensor:
    out = torch.from_numpy(y_np)
    if out.dtype != x.dtype:
        out = out.to(dtype=x.dtype)
    if out.device != x.device:
        out = out.to(device=x.device)
    return out


def lut2_gemv_cpu(
    weight_blob: bytes | memoryview,
    x: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Back-compat alias for tests."""
    return lut2_gemv(weight_blob, x, out_features=out_features, in_features=in_features)


def _lut2_tmix_gemv_native(
    blobs: tuple[bytes, bytes, bytes, bytes],
    xs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    out_features: int,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import ctypes

    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    if native is None or not hasattr(native, "lut2_tmix_gemv_f32_export"):
        raise RuntimeError("native lut2_tmix_gemv not loaded")
    codebooks = np.empty(16, dtype=np.float32)
    packed_arrays: list[ctypes.Array] = []
    x_arrays: list[ctypes.Array] = []
    for i, blob in enumerate(blobs):
        cb, pk = _lut2_arrays(blob, out_features=out_features, in_features=in_features)
        codebooks[i * 4 : (i + 1) * 4] = cb
        packed_arrays.append(
            _lut2_native_packed_view(
                blob, pk, out_features=out_features, in_features=in_features
            )
        )
        x_f = xs[i].astype(np.float32, copy=False)
        x_arrays.append((ctypes.c_float * in_features).from_buffer(memoryview(x_f)))
    y = np.empty(4 * out_features, dtype=np.float32)
    cb_f = (ctypes.c_float * 16).from_buffer(memoryview(codebooks))
    y_f = (ctypes.c_float * (4 * out_features)).from_buffer(y)
    pk_ptrs = (ctypes.POINTER(ctypes.c_uint8) * 4)(*packed_arrays)
    x_ptrs = (ctypes.POINTER(ctypes.c_float) * 4)(*x_arrays)
    native.lut2_tmix_gemv_f32_export(
        y_f, cb_f, pk_ptrs, x_ptrs, int(out_features), int(in_features)
    )
    del packed_arrays, x_arrays, pk_ptrs, x_ptrs
    return (
        y[:out_features],
        y[out_features : 2 * out_features],
        y[2 * out_features : 3 * out_features],
        y[3 * out_features :],
    )


def _scale_u8_grouped_tmix_gemv_native(
    blobs: tuple[bytes | memoryview, bytes | memoryview, bytes | memoryview, bytes | memoryview],
    xs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    out_features: int,
    in_features: int,
    n_mats: int = 4,
    activation_int8: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run SG8 attention GEMVs in one native/OpenMP region."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    if n_mats not in (3, 4):
        raise ValueError(f"unsupported grouped TMix batch width: {n_mats}")
    symbol = (
        "scale_u8_grouped_tmix_qkv_i8_gemv_f32_export"
        if n_mats == 3 and activation_int8
        else (
            "scale_u8_grouped_tmix_qkv_gemv_f32_export"
            if n_mats == 3
            else "scale_u8_grouped_tmix_gemv_f32_export"
        )
    )
    fn = getattr(native, symbol, None)
    if fn is None:
        raise RuntimeError("grouped U8 batched native GEMV not loaded")

    # Keep every exporter alive until ctypes returns.  A layer-span read can
    # provide read-only memoryviews, so the native ABI receives pointers into
    # NumPy views rather than requiring a per-matrix copy.
    blob_views: list[np.ndarray] = []
    blob_ptrs: list[ctypes.POINTER(ctypes.c_uint8)] = []
    for blob in blobs[:n_mats]:
        blob_view, blob_ptr = _native_blob_view(blob)
        blob_views.append(blob_view)
        blob_ptrs.append(blob_ptr)

    x_arrays: list[ctypes.Array] = []
    x_views: list[np.ndarray] = []
    for x_np in xs[:n_mats]:
        x_view = np.ascontiguousarray(x_np, dtype=np.float32)
        x_views.append(x_view)
        x_arrays.append(
            (ctypes.c_float * in_features).from_buffer(memoryview(x_view))
        )

    y = np.empty(n_mats * out_features, dtype=np.float32)
    y_f = (ctypes.c_float * (n_mats * out_features)).from_buffer(y)
    blob_ptr_array = (
        ctypes.POINTER(ctypes.c_uint8) * n_mats
    )(*blob_ptrs)
    x_ptr_array = (
        ctypes.POINTER(ctypes.c_float) * n_mats
    )(*x_arrays)
    fn(
        y_f,
        blob_ptr_array,
        x_ptr_array,
        int(out_features),
        int(in_features),
    )
    return tuple(
        y[index * out_features : (index + 1) * out_features]
        for index in range(n_mats)
    )  # type: ignore[return-value]


def _lut2_tmix_qkv_gemv_native(
    blobs: tuple[bytes, bytes, bytes],
    xs: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    out_features: int,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy LUT2 QKV batch without computing the later output projection."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    fn = getattr(native, "lut2_tmix_qkv_gemv_f32_export", None)
    if fn is None:
        raise RuntimeError("native lut2 QKV GEMV not loaded")
    codebooks = np.empty(12, dtype=np.float32)
    packed_arrays: list[ctypes.Array] = []
    x_arrays: list[ctypes.Array] = []
    x_views: list[np.ndarray] = []
    for i, blob in enumerate(blobs):
        cb, pk = _lut2_arrays(blob, out_features=out_features, in_features=in_features)
        codebooks[i * 4 : (i + 1) * 4] = cb
        packed_arrays.append(
            _lut2_native_packed_view(
                blob, pk, out_features=out_features, in_features=in_features
            )
        )
        x_view = np.ascontiguousarray(xs[i], dtype=np.float32)
        x_views.append(x_view)
        x_arrays.append((ctypes.c_float * in_features).from_buffer(memoryview(x_view)))
    y = np.empty(3 * out_features, dtype=np.float32)
    cb_f = (ctypes.c_float * 12).from_buffer(memoryview(codebooks))
    y_f = (ctypes.c_float * (3 * out_features)).from_buffer(y)
    pk_ptrs = (ctypes.POINTER(ctypes.c_uint8) * 3)(*packed_arrays)
    x_ptrs = (ctypes.POINTER(ctypes.c_float) * 3)(*x_arrays)
    fn(y_f, cb_f, pk_ptrs, x_ptrs, int(out_features), int(in_features))
    return (
        y[:out_features],
        y[out_features : 2 * out_features],
        y[2 * out_features :],
    )


def _lut2_gemv_fallback_one(
    blob: bytes | memoryview,
    x_np: np.ndarray,
    *,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """Dispatch one packed blob to the matching native/Numba/NumPy path."""
    magic = bytes(blob[:4])
    if magic == _SCALE_U8_GROUPED_MAGIC:
        native_fn = _scale_u8_grouped_gemv_native
        numba_fn = _scale_u8_grouped_gemv_numba
        numpy_fn = _scale_u8_grouped_gemv_numpy
    elif is_grouped_lut2_blob(blob):
        native_fn = _grouped_lut2_gemv_native
        numba_fn = _grouped_lut2_gemv_numba
        numpy_fn = _grouped_lut2_gemv_numpy
    else:
        native_fn = _lut2_gemv_native
        numba_fn = _lut2_gemv_numba
        numpy_fn = _lut2_gemv_numpy
    try:
        return native_fn(
            blob, x_np, out_features=out_features, in_features=in_features
        )
    except Exception:
        try:
            return numba_fn(
                blob, x_np, out_features=out_features, in_features=in_features
            )
        except Exception:
            return numpy_fn(
                blob, x_np, out_features=out_features, in_features=in_features
            )


def _lut2_tmix_gemv_fallback(
    blobs: tuple[bytes, bytes, bytes, bytes],
    xs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    out_features: int,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outs = [
        _lut2_gemv_fallback_one(
            blob, x_np, out_features=out_features, in_features=in_features
        )
        for blob, x_np in zip(blobs, xs)
    ]
    return outs[0], outs[1], outs[2], outs[3]


def lut2_tmix_gemv_batched(
    blobs: tuple[bytes, bytes, bytes, bytes],
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four att linear maps (r/k/v/o) in one native kernel when available."""
    if isinstance(xs, torch.Tensor):
        xs_t = (xs, xs, xs, xs)
    else:
        xs_t = xs
    x_nps = tuple(_tensor_to_f32_np(x) for x in xs_t)
    for x_np in x_nps:
        if x_np.size != in_features:
            raise ValueError(f"x size {x_np.size} != in_features {in_features}")
    kernel = _lut_kernel_pref()
    grouped_u8 = all(bytes(blob[:4]) == _SCALE_U8_GROUPED_MAGIC for blob in blobs)
    if kernel in ("c", "auto"):
        try:
            if grouped_u8:
                ys = _scale_u8_grouped_tmix_gemv_native(
                    blobs,
                    x_nps,  # type: ignore[arg-type]
                    out_features=out_features,
                    in_features=in_features,
                )
            else:
                ys = _lut2_tmix_gemv_native(
                    blobs,
                    x_nps,
                    out_features=out_features,
                    in_features=in_features,  # type: ignore[arg-type]
                )
            return tuple(_lut2_torch_from_f32(y, xs_t[0]) for y in ys)
        except Exception:
            if kernel == "c":
                raise
    ys = _lut2_tmix_gemv_fallback(
        blobs,
        x_nps,
        out_features=out_features,
        in_features=in_features,  # type: ignore[arg-type]
    )
    return tuple(_lut2_torch_from_f32(y, xs_t[0]) for y in ys)


def lut2_tmix_qkv_gemv_batched(
    blobs: tuple[bytes, bytes, bytes],
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three TMix attention projections in one native region (no wasted O-map)."""
    if isinstance(xs, torch.Tensor):
        xs_t = (xs, xs, xs)
    else:
        xs_t = xs
    x_nps = tuple(_tensor_to_f32_np(x) for x in xs_t)
    for x_np in x_nps:
        if x_np.size != in_features:
            raise ValueError(f"x size {x_np.size} != in_features {in_features}")
    kernel = _lut_kernel_pref()
    grouped_u8 = all(bytes(blob[:4]) == _SCALE_U8_GROUPED_MAGIC for blob in blobs)
    if kernel in ("c", "auto"):
        try:
            if grouped_u8:
                ys = _scale_u8_grouped_tmix_gemv_native(
                    blobs + (blobs[-1],),
                    x_nps + (x_nps[-1],),
                    out_features=out_features,
                    in_features=in_features,
                    n_mats=3,
                    activation_int8=activation_int8_enabled(),
                )
            else:
                ys = _lut2_tmix_qkv_gemv_native(
                    blobs, x_nps, out_features=out_features, in_features=in_features
                )
            return tuple(_lut2_torch_from_f32(y, xs_t[0]) for y in ys)
        except Exception:
            if kernel == "c":
                raise
    ys = tuple(
        _lut2_gemv_fallback_one(
            blob,
            x_np,
            out_features=out_features,
            in_features=in_features,
        )
        for blob, x_np in zip(blobs, x_nps)
    )
    return tuple(_lut2_torch_from_f32(y, xs_t[0]) for y in ys)


def lut2_linear_forward(
    weight_blob: bytes | memoryview,
    x: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Fused LUT2 linear when enabled; otherwise materialize via standard decode."""
    if not fused_lut2_enabled() or len(shape) != 2:
        from rwkv_ssd.runtime.trinity_codec import _gather_lut2
        from rwkv_ssd.runtime.manifest import TensorEntry

        entry = TensorEntry(
            name="w",
            layer_id=0,
            dtype="bfloat16",
            shape=list(shape),
            offset=0,
            length=len(weight_blob),
            alignment=4096,
            residency="streamed",
            dequant="trinity_lut2",
        )
        if is_grouped_lut2_blob(weight_blob):
            from rwkv_ssd.runtime.trinity_codec import decode_trinity_lut2_to_tensor

            w = decode_trinity_lut2_to_tensor(weight_blob, entry, torch.device("cpu"))
        else:
            codebook, idx_view, _row_stride = _lut2_inner_view(weight_blob, entry)
            packed = np.frombuffer(idx_view, dtype=np.uint8)
            w = _gather_lut2(codebook, packed, entry)
        return x @ w.T if x.dim() == 1 else torch.nn.functional.linear(x, w)
    out_f, in_f = int(shape[0]), int(shape[1])
    return lut2_gemv(weight_blob, x, out_features=out_f, in_features=in_f)
