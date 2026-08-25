"""Packed weight dequantization (M5 + Trinity codec ladder)."""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_to_tensor,
    decode_scale_u8_grouped_to_bytes,
    decode_scale_u8_grouped_to_tensor,
    decode_scale_u4_to_bytes,
    decode_scale_u8_to_bytes,
)
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes
from rwkv_ssd.runtime.trinity_codec import (
    LayerZlibCache,
    decode_trinity_layer_to_tensor,
    decode_trinity_lut2_to_bytes,
    decode_trinity_lut2_to_tensor,
    decode_trinity_to_bytes,
    decode_trinity_to_tensor,
)

SUPPORTED_DEQUANT = frozenset(
    {"none", "scale_u8", "scale_u8_grouped", "scale_u4", "trinity_lut2", "trinity", "trinity_layer"}
)


def _codec_name(entry: TensorEntry) -> str:
    return (entry.dequant or "none").strip().lower()


def decode_weight_to_tensor(
    data: bytes | memoryview,
    entry: TensorEntry,
    device: torch.device,
    *,
    trinity_layer_cache: LayerZlibCache | None = None,
    decode_device: torch.device | None = None,
) -> torch.Tensor:
    """Fast path: packed blob -> tensor (engine hot path)."""
    codec = _codec_name(entry)
    if codec == "none":
        return tensor_from_bytes(bytes(data), entry, device)
    if codec == "scale_u8":
        return decode_scale_to_tensor(data, entry, device, bits=8)
    if codec == "scale_u8_grouped":
        return decode_scale_u8_grouped_to_tensor(data, entry, device)
    if codec == "scale_u4":
        return decode_scale_to_tensor(data, entry, device, bits=4)
    if codec == "trinity_lut2":
        return decode_trinity_lut2_to_tensor(
            data, entry, device, decode_device=decode_device
        )
    if codec == "trinity_layer":
        cache = trinity_layer_cache or LayerZlibCache()
        return decode_trinity_layer_to_tensor(
            data, entry, cache, device, decode_device=decode_device
        )
    if codec == "trinity":
        return decode_trinity_to_tensor(
            data, entry, device, decode_device=decode_device
        )
    raise ValueError(
        f"unsupported dequant codec {entry.dequant!r} for {entry.name}. "
        f"Supported: {sorted(SUPPORTED_DEQUANT)}."
    )


def decode_weight_blob(data: bytes, entry: TensorEntry) -> bytes:
    """Decode packed bytes to raw tensor bytes (skeleton / legacy callers)."""
    codec = _codec_name(entry)
    if codec == "none":
        return data
    if codec == "scale_u8":
        return decode_scale_u8_to_bytes(data, entry)
    if codec == "scale_u8_grouped":
        return decode_scale_u8_grouped_to_bytes(data, entry)
    if codec == "scale_u4":
        return decode_scale_u4_to_bytes(data, entry)
    if codec == "trinity_lut2":
        return decode_trinity_lut2_to_bytes(data, entry)
    if codec == "trinity_layer":
        cache = LayerZlibCache()
        t = decode_trinity_layer_to_tensor(data, entry, cache, torch.device("cpu"))
        dt = t.dtype
        if dt == torch.bfloat16:
            return t.contiguous().view(torch.uint16).numpy().tobytes()
        return t.contiguous().numpy().tobytes()
    if codec == "trinity":
        return decode_trinity_to_bytes(data, entry)
    raise ValueError(
        f"unsupported dequant codec {entry.dequant!r} for {entry.name}. "
        f"Supported: {sorted(SUPPORTED_DEQUANT)}."
    )
