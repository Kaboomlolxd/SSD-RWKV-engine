"""Trinity LUT decode on Intel XPU / CUDA (gather), separate from ChatRWKV CPU inject."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from rwkv_ssd.runtime.device import (
    is_accelerator_device,
    xpu_compute_available,
    xpu_runtime_available,
)
from rwkv_ssd.runtime.tensor_loader import dtype_from_entry

if TYPE_CHECKING:
    from rwkv_ssd.runtime.manifest import TensorEntry

logger = logging.getLogger(__name__)


def xpu_available() -> bool:
    """Compatibility alias for the public Intel XPU probe."""
    return xpu_runtime_available()


def probe_intel_xpu() -> dict:
    """Runtime probe for bench / setup scripts."""
    info: dict = {"torch": torch.__version__, "xpu_attr": hasattr(torch, "xpu")}
    try:
        info["xpu_available"] = xpu_available()
        info["xpu_compute_available"] = xpu_compute_available()
        if info["xpu_available"]:
            info["xpu_name"] = torch.xpu.get_device_name(0)
            info["xpu_count"] = torch.xpu.device_count()
    except Exception as exc:
        info["xpu_available"] = False
        info["xpu_error"] = str(exc)
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401

        info["ipex"] = ipex.__version__
    except Exception:
        info["ipex"] = None
    return info


def resolve_trinity_decode_device(
    requested: str | None,
    *,
    fallback: torch.device | None = None,
) -> torch.device:
    """
    Pick device for Trinity LUT gather.

    ``auto`` / empty: use XPU when available, else CPU.
    ``xpu``: require Intel GPU stack; fall back to CPU with warning.
    """
    raw = (requested or os.environ.get("RWKV_TRINITY_DECODE_DEVICE") or "auto").strip().lower()
    if raw in ("", "cpu", "host"):
        return torch.device("cpu")
    if raw == "auto":
        # XPU helps on large gathers only; default auto stays CPU unless opted in.
        if os.environ.get("RWKV_TRINITY_XPU_AUTO", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            if xpu_available():
                return torch.device("xpu")
        if fallback is not None and is_accelerator_device(fallback):
            if fallback.type != "xpu" or xpu_available():
                return fallback
        return torch.device("cpu")
    if raw.startswith("xpu"):
        if xpu_available():
            try:
                device = torch.device(raw if ":" in raw else "xpu")
                if ":" in raw:
                    index = int(raw.rsplit(":", 1)[1])
                    count = int(torch.xpu.device_count())
                    if index < 0 or index >= count:
                        raise ValueError(
                            f"XPU device index {index} is outside device count {count}"
                        )
                return device
            except Exception as exc:
                logger.warning(
                    "Trinity decode device %s could not be initialized (%s); "
                    "using CPU.",
                    raw,
                    exc,
                )
        logger.warning(
            "Trinity decode device xpu requested but torch.xpu is not available — "
            "use Python 3.11/3.12 and: pip install torch --index-url "
            "https://download.pytorch.org/whl/xpu (Intel GPU driver required). "
            "Falling back to CPU LUT decode."
        )
        return torch.device("cpu")
    if raw.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(raw if ":" in raw else "cuda")
        logger.warning("CUDA Trinity decode requested but unavailable — using CPU.")
        return torch.device("cpu")
    return torch.device(raw)


def _xpu_min_gather_numel() -> int:
    """Skip XPU for tiny tensors (launch + PCIe overhead dominates)."""
    try:
        return max(0, int(os.environ.get("RWKV_TRINITY_XPU_MIN_NUMEL", "262144")))
    except ValueError:
        logger.warning("Ignoring invalid RWKV_TRINITY_XPU_MIN_NUMEL; using 262144")
        return 262144


def _record_decode_timing(timing: object | None, field: str, value_ms: float) -> None:
    """Add an accelerator timing component without coupling this module to metrics."""
    if timing is None or value_ms <= 0.0:
        return
    try:
        setattr(timing, field, getattr(timing, field, 0.0) + float(value_ms))
    except (AttributeError, TypeError):
        # Decode helpers are also used by standalone codec tools, which pass
        # no metrics object (or occasionally a lightweight test double).
        return


def _unpack_2bit_packed_on_device(
    packed: bytes | memoryview | np.ndarray,
    numel: int,
    decode_device: torch.device,
) -> torch.Tensor:
    """Transfer packed 2-bit bytes once, then unpack to XPU/CUDA indices.

    The old path unpacked to host ``uint8`` and immediately widened to an
    accelerator ``int64`` tensor.  That multiplies transfer volume by up to
    32x for LUT2.  Keeping the packed bytes compressed until they reach the
    device avoids that avoidable host-side expansion.
    """
    if numel <= 0:
        return torch.empty(0, dtype=torch.long, device=decode_device)
    need = (numel + 3) // 4
    if isinstance(packed, np.ndarray):
        source = np.asarray(packed, dtype=np.uint8).reshape(-1)
    else:
        source = np.frombuffer(packed, dtype=np.uint8)
    if source.size < need:
        raise ValueError(f"trinity_lut2 indices short: {source.size} vs {need}")
    # frombuffer views over mmap/bytes are frequently read-only.  A compact
    # writable copy also gives Torch a contiguous, stable transfer source.
    packed_np = np.array(source[:need], dtype=np.uint8, copy=True, order="C")
    packed_t = torch.from_numpy(packed_np).to(
        device=decode_device, dtype=torch.uint8, non_blocking=True
    )
    wide = packed_t.to(dtype=torch.long)
    indices = torch.stack(
        (
            wide & 3,
            (wide >> 2) & 3,
            (wide >> 4) & 3,
            (wide >> 6) & 3,
        ),
        dim=1,
    ).reshape(-1)
    return indices[:numel]


def gather_lut2_on_device(
    codebook: np.ndarray,
    indices: np.ndarray,
    entry: TensorEntry,
    decode_device: torch.device,
) -> torch.Tensor:
    cb = torch.as_tensor(
        np.array(codebook, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32,
        device=decode_device,
    )
    idx = torch.as_tensor(indices, dtype=torch.long, device=decode_device)
    values = cb[idx]
    return values.reshape(entry.shape).to(dtype=dtype_from_entry(entry))


def gather_lut2_packed_on_device(
    codebook: np.ndarray,
    packed: bytes | memoryview | np.ndarray,
    entry: TensorEntry,
    decode_device: torch.device,
) -> torch.Tensor:
    """Decode a regular LUT2 tensor without expanding indices on the host."""
    indices = _unpack_2bit_packed_on_device(
        packed, entry.numel, torch.device(decode_device)
    )
    cb = torch.as_tensor(
        np.array(codebook, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32,
        device=decode_device,
    )
    return cb[indices].reshape(entry.shape).to(dtype=dtype_from_entry(entry))


def _gather_grouped_indices_on_device(
    codebooks: np.ndarray,
    indices: np.ndarray | torch.Tensor,
    entry: TensorEntry,
    decode_device: torch.device,
    *,
    group_size: int,
    residual_positions: np.ndarray | None = None,
    residual_deltas: np.ndarray | None = None,
    input_layout: bool = False,
) -> torch.Tensor:
    device = torch.device(decode_device)
    cb = torch.as_tensor(
        np.array(codebooks, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32,
        device=device,
    )
    if isinstance(indices, torch.Tensor):
        idx = indices.to(device=device, dtype=torch.long)
    else:
        idx = torch.as_tensor(
            np.array(indices, dtype=np.int64, order="C", copy=True),
            dtype=torch.long,
            device=device,
        )
    group_ids = torch.arange(entry.numel, dtype=torch.long, device=device)
    group_ids = torch.div(group_ids, int(group_size), rounding_mode="floor")
    # Flatten the [group, code] table before indexing so we do not expand a
    # four-column codebook to an intermediate [numel, 4] tensor.
    group_ids.mul_(4).add_(idx)
    selected = cb.reshape(-1).index_select(0, group_ids)

    if residual_positions is not None or residual_deltas is not None:
        if residual_positions is None or residual_deltas is None:
            raise ValueError("grouped LUT2 residual metadata must pair")
        positions = torch.as_tensor(
            np.array(residual_positions, dtype=np.int64, order="C", copy=True),
            dtype=torch.long,
            device=device,
        )
        deltas = torch.as_tensor(
            np.array(residual_deltas, dtype=np.float32, order="C", copy=True),
            dtype=torch.float32,
            device=device,
        )
        if positions.shape != deltas.shape or positions.ndim != 2:
            raise ValueError("grouped LUT2 residual metadata has an invalid shape")
        starts = (
            torch.arange(positions.shape[0], dtype=torch.long, device=device)
            * int(group_size)
        )
        flat_positions = starts[:, None] + positions
        valid = flat_positions < entry.numel
        selected.index_add_(0, flat_positions[valid], deltas[valid])

    if input_layout and len(entry.shape) == 2:
        shaped = selected.reshape(entry.shape[1], entry.shape[0]).transpose(0, 1)
    else:
        shaped = selected.reshape(entry.shape)
    return shaped.to(dtype=dtype_from_entry(entry))


def gather_grouped_lut2_on_device(
    codebooks: np.ndarray,
    indices: np.ndarray,
    entry: TensorEntry,
    decode_device: torch.device,
    *,
    group_size: int,
    residual_positions: np.ndarray | None = None,
    residual_deltas: np.ndarray | None = None,
    input_layout: bool = False,
) -> torch.Tensor:
    """Decode a group-local LUT2 blob without materializing it on the CPU."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if codebooks.ndim != 2 or codebooks.shape[1] != 4:
        raise ValueError("grouped LUT2 codebooks must have shape [groups, 4]")
    if indices.size != entry.numel:
        raise ValueError(
            f"grouped LUT2 index count {indices.size} != tensor numel {entry.numel}"
        )

    return _gather_grouped_indices_on_device(
        codebooks,
        indices,
        entry,
        decode_device,
        group_size=group_size,
        residual_positions=residual_positions,
        residual_deltas=residual_deltas,
        input_layout=input_layout,
    )


def gather_grouped_lut2_packed_on_device(
    codebooks: np.ndarray,
    packed: bytes | memoryview | np.ndarray,
    entry: TensorEntry,
    decode_device: torch.device,
    *,
    group_size: int,
    residual_positions: np.ndarray | None = None,
    residual_deltas: np.ndarray | None = None,
    input_layout: bool = False,
) -> torch.Tensor:
    """Grouped LUT2 decode with packed-byte transfer to the accelerator."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if codebooks.ndim != 2 or codebooks.shape[1] != 4:
        raise ValueError("grouped LUT2 codebooks must have shape [groups, 4]")
    if entry.numel <= 0:
        return torch.empty(entry.shape, dtype=dtype_from_entry(entry), device=decode_device)
    indices = _unpack_2bit_packed_on_device(
        packed, entry.numel, torch.device(decode_device)
    )
    return _gather_grouped_indices_on_device(
        codebooks,
        indices,
        entry,
        decode_device,
        group_size=group_size,
        residual_positions=residual_positions,
        residual_deltas=residual_deltas,
        input_layout=input_layout,
    )


def decode_lut2_layer_on_accel(
    *,
    blobs: list[tuple[bytes | memoryview, TensorEntry]],
    decode_device: torch.device,
    output_device: torch.device,
    timing: object | None = None,
) -> dict[str, torch.Tensor]:
    """
    Layer-batched LUT2 gather on XPU/CUDA; zlib already stripped by caller.

    Each item is (lut2_inner_blob, entry) with correct inner offsets applied.
    """
    from rwkv_ssd.runtime.lut_gather_kernel import gather_lut2_into, unpack_2bit_into
    from rwkv_ssd.runtime.trinity_codec import _lut2_inner_view

    if not blobs:
        return {}
    from rwkv_ssd.runtime.trinity_codec import (
        decode_trinity_lut2_to_tensor,
        is_grouped_lut2_blob,
    )
    grouped = [(blob, entry) for blob, entry in blobs if is_grouped_lut2_blob(blob)]
    ordered = sorted(
        [(blob, entry) for blob, entry in blobs if not is_grouped_lut2_blob(blob)],
        key=lambda item: item[1].offset,
    )
    out: dict[str, torch.Tensor] = {
        entry.name: decode_trinity_lut2_to_tensor(
            blob,
            entry,
            output_device,
            decode_device=decode_device,
        )
        for blob, entry in grouped
    }
    if not ordered:
        return out
    total = sum(e.numel for _, e in ordered)
    dt = dtype_from_entry(ordered[0][1])
    meta: list[tuple[str, tuple[int, ...], int, int]] = []
    min_xpu = _xpu_min_gather_numel()
    want_accel = is_accelerator_device(decode_device) and total >= min_xpu
    device_dt = (
        dt
        if dt in (torch.float16, torch.bfloat16, torch.float32)
        else torch.float32
    )
    if want_accel:
        # Gather directly into the manifest dtype.  The previous float32 slab
        # doubled device memory and doubled the bytes copied back for bf16.
        flat_t = torch.empty(total, dtype=device_dt, device=decode_device)
    else:
        flat_np = np.empty(total, dtype=np.float32)
    flat_host: np.ndarray | None = None
    host_ranges: list[tuple[int, int]] = []

    accel_start = time.perf_counter() if want_accel else 0.0
    off = 0
    for blob, entry in ordered:
        codebook, idx_view, _row_stride = _lut2_inner_view(blob, entry)
        n = entry.numel
        packed = np.frombuffer(idx_view, dtype=np.uint8)
        if want_accel and n >= min_xpu and _row_stride is None:
            # Keep the 2-bit representation compressed during H2D.  The
            # unpacked int64 indices are produced on the accelerator, avoiding
            # the old CPU uint8 -> accelerator int64 transfer.
            idx = _unpack_2bit_packed_on_device(
                packed, n, torch.device(decode_device)
            )
            cb = torch.from_numpy(
                np.array(codebook, dtype=np.float32, order="C", copy=True)
            ).to(device=decode_device, dtype=torch.float32, non_blocking=True)
            flat_t[off : off + n] = cb[idx]
        else:
            indices_np = np.empty(n, dtype=np.uint8)
            unpack_2bit_into(indices_np, packed)
            if indices_np.max(initial=0) >= 4:
                raise ValueError(f"trinity_lut2 index out of range for {entry.name}")
            if want_accel:
                if flat_host is None:
                    flat_host = np.empty(total, dtype=np.float32)
                gather_lut2_into(flat_host, off, codebook, indices_np)
                host_ranges.append((off, off + n))
            else:
                gather_lut2_into(flat_np, off, codebook, indices_np)
        meta.append((entry.name, tuple(entry.shape), off, n))
        off += n

    if want_accel:
        if flat_host is not None and host_ranges:
            # Small/FLUTE tensors are decoded on CPU, but transfer contiguous
            # runs in one operation instead of launching one H2D copy per
            # tensor.  Merge adjacent ranges while preserving the device
            # gather results already written into the other regions.
            merged: list[list[int]] = []
            for start, end in host_ranges:
                if merged and merged[-1][1] == start:
                    merged[-1][1] = end
                else:
                    merged.append([start, end])
            for start, end in merged:
                host_t = torch.from_numpy(flat_host[start:end]).to(
                    device=decode_device, dtype=device_dt, non_blocking=True
                )
                flat_t[start:end] = host_t
        if decode_device.type == "xpu":
            torch.xpu.synchronize()
        elif decode_device.type == "cuda":
            torch.cuda.synchronize(decode_device)
        _record_decode_timing(
            timing, "decode_device_ms", (time.perf_counter() - accel_start) * 1000.0
        )
        d2h_start = time.perf_counter()
        t_flat = flat_t if flat_t.dtype == dt else flat_t.to(dtype=dt)
        if t_flat.device != output_device:
            t_flat = t_flat.to(device=output_device)
        _record_decode_timing(
            timing, "d2h_ms", (time.perf_counter() - d2h_start) * 1000.0
        )
    else:
        t_flat = torch.from_numpy(flat_np).to(dtype=dt)
        if output_device.type != "cpu":
            t_flat = t_flat.to(device=output_device)
    for name, shape, start, numel in meta:
        out[name] = t_flat[start : start + numel].reshape(shape)
    return out
