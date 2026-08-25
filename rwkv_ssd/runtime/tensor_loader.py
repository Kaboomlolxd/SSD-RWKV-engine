"""Load tensors from packed byte blobs."""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.manifest import TensorEntry

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
}

# Precomputed element sizes for each supported dtype (avoids creating a
# 0-element tensor on every ``tensor_from_bytes`` call just to read
# ``element_size()`` — that was a hot-path allocation).
_DTYPE_ELEM_SIZE: dict[torch.dtype, int] = {
    dt: torch.empty(0, dtype=dt).element_size() for dt in _DTYPE_MAP.values()
}


def dtype_from_entry(entry: TensorEntry) -> torch.dtype:
    dt = _DTYPE_MAP.get(entry.dtype)
    if dt is None:
        raise ValueError(f"unsupported tensor dtype in manifest: {entry.dtype}")
    return dt


def tensor_from_bytes(
    data: bytes, entry: TensorEntry, device: torch.device
) -> torch.Tensor:
    dt = dtype_from_entry(entry)
    elem_size = _DTYPE_ELEM_SIZE[dt]
    expected = entry.numel * elem_size
    if len(data) != expected:
        raise ValueError(
            f"byte length mismatch for {entry.name}: got {len(data)}, expected {expected}"
        )
    # ``bytes`` / mmap views are often non-writable; PyTorch warns and
    # forbids in-place ops on such tensors. Materialize a writable
    # ``bytearray`` then clone so the tensor owns its storage.
    if isinstance(data, bytearray):
        buf: bytearray | memoryview = data
    elif isinstance(data, memoryview) and not data.readonly:
        buf = data
    else:
        buf = bytearray(data)
    t = torch.frombuffer(buf, dtype=dt).reshape(entry.shape).clone()
    if device.type == "cpu":
        return t
    return t.to(device=device)
