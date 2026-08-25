"""
Prepare packed checkpoint tensors for injection into ChatRWKV RWKV-7 ``model.z``.

ChatRWKV loads weights into a flat dict ``z`` (not ``state_dict``). The transforms
here mirror the RWKV-7 path in ``rwkv.model.RWKV`` after load from ``.pth``.
"""

from __future__ import annotations

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.manifest import TensorEntry


_WEIGHT_TRANSPOSE_SUFFIXES = (
    "key.weight",
    "value.weight",
    "receptance.weight",
    "output.weight",
    "head.weight",
)


def prepare_rwkv7_tensor_for_z(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply the same layout transforms ChatRWKV uses when populating ``z``."""
    t = tensor.contiguous()
    if any(suffix in name for suffix in _WEIGHT_TRANSPOSE_SUFFIXES):
        t = t.t().contiguous()
    t = t.squeeze()
    if name.endswith("att.r_k"):
        t = t.flatten()
    return t


def tensors_from_entries(
    raw_by_name: dict[str, bytes],
    entries: list[TensorEntry],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for entry in entries:
        t = decode_weight_to_tensor(raw_by_name[entry.name], entry, device)
        out[entry.name] = prepare_rwkv7_tensor_for_z(entry.name, t)
    return out


def layer_weights_in_z(z: dict[str, torch.Tensor], layer_id: int) -> bool:
    """True when block weight tensors for ``layer_id`` are already in ``z``."""
    return f"blocks.{layer_id}.att.receptance.weight" in z


def layer_skeleton_in_z(z: dict[str, torch.Tensor], layer_id: int) -> bool:
    """True when fused-stream skeleton vectors for ``layer_id`` are in ``z`` (no weight slabs)."""
    return f"blocks.{layer_id}.att.x_r" in z


def layer_ready_in_z(z: dict[str, torch.Tensor], layer_id: int) -> bool:
    """Weights or fused skeleton resident — skip pack load / provider prepared tensors."""
    return layer_weights_in_z(z, layer_id) or layer_skeleton_in_z(z, layer_id)


def all_block_layers_in_z(
    z: dict[str, torch.Tensor],
    layer_ids: list[int],
    by_layer: dict[int, list[TensorEntry]],
) -> bool:
    """True when every layer with pack entries has weights resident in ``z``."""
    for layer_id in layer_ids:
        if by_layer.get(layer_id) and not layer_weights_in_z(z, layer_id):
            return False
    return True


def all_stream_layers_prepared_in_provider(
    provider: object,
    layer_ids: list[int],
    by_layer: dict[int, list[TensorEntry]],
) -> bool:
    """True when every streamed block layer is in the provider prepared cache."""
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    if not isinstance(provider, ManifestWeightProvider):
        return False
    for layer_id in layer_ids:
        entries = by_layer.get(layer_id, [])
        if not entries:
            continue
        if not provider.layer_has_streamed_tensors(entries):
            continue
        prepared = provider._prepared_layers.get(layer_id)
        # Empty ``{}`` markers are not usable (lean-z non-fused path, or a
        # release that wiped skeleton). Treat as not prepared.
        if not prepared:
            return False
    return True


def materialize_prepared_layers_into_z(
    z: dict[str, torch.Tensor],
    provider: object,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
    metrics: object | None = None,
    *,
    free_provider: bool = False,
) -> int:
    """
    Inject provider-prepared layers into ``z`` (no disk read / decode).

    When ``free_provider``, drop provider copies so weights live only in ``z``.
    """
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    if not isinstance(provider, ManifestWeightProvider):
        return 0
    act_dtype = z["emb.weight"].dtype
    loaded = 0
    for layer_id in layer_ids:
        entries = by_layer.get(layer_id, [])
        if not entries or layer_weights_in_z(z, layer_id):
            continue
        prepared = provider._prepared_layers.get(layer_id)
        if prepared is None:
            continue
        if metrics is not None:
            row = metrics.start_layer(layer_id)
            row.layer_cache_hits += len(prepared)
        for key, tensor in prepared.items():
            t = tensor
            if t.dtype != act_dtype:
                t = t.to(dtype=act_dtype)
            z[key] = t
        loaded += 1
        if free_provider:
            provider.evict_streamed_layer(layer_id)
    return loaded


def promote_full_z_enabled(n_block_layers: int | None = None) -> bool:
    """
    Lift all block layers into ``z`` after provider warm.

    ``auto`` promotes only tiny models (``n_block_layers < 8``). On 0.1B-class packs
    use ``RWKV_PROMOTE_FULL_Z=1`` for max tok/s (~382 MB ``z``).
    """
    import os

    raw = os.environ.get("RWKV_PROMOTE_FULL_Z", "auto").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    if n_block_layers is not None and n_block_layers >= 8:
        return False
    if n_block_layers is not None:
        return True
    return False


def promote_stream_cache_to_full_z(
    z: dict[str, torch.Tensor],
    provider: object,
    layer_ids: list[int],
    by_layer: dict[int, list[TensorEntry]],
    metrics: object | None = None,
) -> bool:
    """
    When the provider has decoded every streamed layer, lift ``z`` retention to
    hold all blocks and inject prepared tensors once.

    Keeps the per-layer TMix/CMix path (faster than native ``forward`` on some
    ChatRWKV builds) while skipping per-token disk decode and inject copies.
    """
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    if not promote_full_z_enabled(len(layer_ids)):
        return all_block_layers_in_z(z, layer_ids, by_layer)
    if all_block_layers_in_z(z, layer_ids, by_layer):
        return True
    if not all_stream_layers_prepared_in_provider(provider, layer_ids, by_layer):
        return False
    if isinstance(provider, ManifestWeightProvider):
        streamed = [
            lid
            for lid in layer_ids
            if by_layer.get(lid) and provider.layer_has_streamed_tensors(by_layer[lid])
        ]
        if streamed:
            provider._z_retention.max_layers_in_z = max(
                provider._z_retention.max_layers_in_z, len(streamed)
            )
    if all_block_layers_in_z(z, layer_ids, by_layer):
        return True
    materialize_prepared_layers_into_z(
        z,
        provider,
        by_layer,
        layer_ids,
        metrics,
        free_provider=True,
    )
    if isinstance(provider, ManifestWeightProvider):
        provider.release_all_streamed_layers()
    return all_block_layers_in_z(z, layer_ids, by_layer)


def warm_stream_cache_layers_into_z(
    z: dict[str, torch.Tensor],
    provider: object,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
    metrics: object | None = None,
) -> int:
    """
    Preload all streamed block layers into ``z`` so decode uses native ``forward``.

    Returns the number of layers loaded.

    Passes ``force_materialize=True`` to ``provider.prepare_layer_for_z``
    so the fused-inject filter does not drop the bf16 att/FFN slabs — the
    native ``model.forward`` path reads them from ``z`` directly. If the
    provider's prepared cache already has the full set of weight slabs
    (from a prior ``force_materialize=True`` call), the cache is reused;
    if the cache is the filtered version (from a prior F-1 packed step
    with ``force_materialize=False``), the entry is dropped and the
    layer is re-decoded with the full bf16 slabs.
    """
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

    act_dtype = z["emb.weight"].dtype
    loaded = 0
    for layer_id in layer_ids:
        entries = by_layer.get(layer_id, [])
        if not entries or layer_weights_in_z(z, layer_id):
            continue
        row = metrics.start_layer(layer_id) if metrics else None
        # Reuse the provider's prepared cache only when it has the full
        # set of weight slabs (force_materialize=True was honored). A
        # prior F-1 packed-block step may have populated the cache with
        # the filtered version (force_materialize=False), missing the
        # att/FFN slabs that ``model.forward`` reads from ``z``. Detect
        # this by checking for the att receptance weight — if it's
        # missing, fall through and re-decode with force_materialize=True.
        cached_full = False
        if (
            isinstance(provider, ManifestWeightProvider)
            and layer_id in provider._prepared_layers
        ):
            prepared = provider._prepared_layers[layer_id]
            att_rk = f"blocks.{layer_id}.att.receptance.weight"
            if att_rk in prepared:
                cached_full = True
            if not cached_full:
                # Drop the partial cache so the re-decode below is
                # guaranteed to run (and the cache replaces it).
                provider._prepared_layers.pop(layer_id, None)
        if cached_full:
            if row is not None:
                row.layer_cache_hits += len(prepared)
        else:
            layer_tensors = provider.load_layer_tensors(entries)  # type: ignore[attr-defined]
            if isinstance(provider, ManifestWeightProvider):
                prepared = provider.prepare_layer_for_z(
                    layer_id, layer_tensors, row, force_materialize=True
                )
            else:
                prepared = {
                    name: prepare_rwkv7_tensor_for_z(name, tensor)
                    for name, tensor in layer_tensors.items()
                }
        inject_layer_into_z(z, prepared)
        for key in prepared:
            if z[key].dtype != act_dtype:
                z[key] = z[key].to(dtype=act_dtype)
        loaded += 1
    return loaded


def inject_layer_into_z(
    z: dict[str, torch.Tensor],
    layer_tensors: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Overwrite or create ``z`` keys for one layer (in-place).

    When ``device`` or ``dtype`` is provided, the tensor is cast to those
    values before the copy; otherwise the existing ``z[name]`` target's
    device and dtype are used (so the in-place copy always matches the
    surrounding state).
    """
    for name, tensor in layer_tensors.items():
        if name not in z:
            t = tensor
            if device is not None or dtype is not None:
                t = t.to(device=device or t.device, dtype=dtype or t.dtype)
            z[name] = t.clone().detach()
            continue
        target = z[name]
        if target.numel() == 0 and tensor.numel() > 0:
            continue
        if (
            tensor.data_ptr() == target.data_ptr()
            and tensor.dtype == target.dtype
            and tensor.device == target.device
        ):
            continue
        t = tensor
        if device is not None or dtype is not None:
            t = t.to(device=device or target.device, dtype=dtype or target.dtype)
        else:
            t = t.to(device=target.device, dtype=target.dtype)
        target.copy_(t)
