"""
P1.4 — one RWKV-7 block forward on packed LUT2 weights (no bf16 att/FFN inject).

Skeleton vectors (x_*, w*, ln*) live in ``z`` or in the provider's prepared
``_prepared_layers``; att/FFN weight matrices are read via fused LUT blobs on
``provider`` during TMix/CMix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.lut_gemm_fused import fused_lut2_enabled
from rwkv_ssd.runtime.rwkv7_linear import (
    cmix_one_fused,
    tmix_one_fused,
    tmix_uses_fused,
)

if TYPE_CHECKING:
    from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider


def packed_block_forward_enabled(
    provider: object | None,
    att_prefix: str,
    *,
    mode: str = "streaming",
    stream_layer_cache: bool = False,
    pack_uses_quant: bool | None = None,
) -> bool:
    if provider is None:
        return False
    # The CPU fused GEMV is disabled for XPU. Keep this gate local as well as
    # in the provider so a global RWKV_LUT_GEMM_FUSED=1 cannot route an XPU
    # layer back into the NumPy/Numba host kernel.
    use_fused = getattr(provider, "_use_fused_lut_matmul", None)
    if callable(use_fused) and not use_fused():
        return False
    if not fused_lut2_enabled(
        mode, stream_layer_cache, pack_uses_quant=pack_uses_quant
    ):
        return False
    return tmix_uses_fused(provider, att_prefix)


def _resolve_skeleton(
    name: str,
    z: dict[str, torch.Tensor],
    provider: "ManifestWeightProvider | None",
    layer_id: int,
) -> torch.Tensor:
    """Resolve a small skeleton tensor (``x_*``, ``w*``, ``ln*``, ``r_k``) from
    ``z`` first, then from the provider's prepared layer cache.

    Lets the packed forward path work on partial packs where the small
    skeleton is held by the provider rather than pinned in ``z``.
    """
    if name in z:
        return z[name]
    if provider is not None:
        prepared = getattr(provider, "_prepared_layers", None)
        if prepared is not None:
            layer = prepared.get(layer_id)
            if layer is not None and name in layer:
                return layer[name]
    raise KeyError(
        f"packed_block_forward: skeleton tensor {name!r} not in z and not "
        f"in provider._prepared_layers[{layer_id}]"
    )


def forward_block_packed(
    layer_id: int,
    x: torch.Tensor,
    state: list[torch.Tensor],
    v_first: torch.Tensor,
    z: dict[str, torch.Tensor],
    provider: ManifestWeightProvider,
    *,
    n_head: int,
    head_size: int,
    n_embd: int,
    act_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run ln1 → TMix → ln2 → CMix for one block without att/FFN weight slabs in ``z``."""
    bbb = f"blocks.{layer_id}."
    att = bbb + "att."
    ffn = bbb + "ffn."

    def _ln(x_in: torch.Tensor, wk: str, bk: str) -> torch.Tensor:
        w = _resolve_skeleton(wk, z, provider, layer_id)
        b = _resolve_skeleton(bk, z, provider, layer_id)
        x_in = x_in.to(dtype=w.dtype)
        return F.layer_norm(x_in, (n_embd,), weight=w, bias=b).to(dtype=act_dtype)

    def _adapter(name: str) -> torch.Tensor:
        """Resolve a dense adapter or provide a packed-only placeholder.

        ``tmix_one_fused`` receives the historical dense ``w/a/v/g``
        arguments for its fallback path, but its grouped-U8 adapter path does
        not read them.  Avoid forcing a dense BF16 decode merely to satisfy
        that call signature when the native packed blobs are registered.
        """
        try:
            return _resolve_skeleton(name, z, provider, layer_id)
        except KeyError:
            get_blob = getattr(provider, "get_fused_lut_blob", None)
            if callable(get_blob) and get_blob(name) is not None:
                return torch.empty(0, dtype=act_dtype, device=x.device)
            raise

    xx = _ln(x, bbb + "ln1.weight", bbb + "ln1.bias")
    x_prev = state[layer_id * 3 + 0].to(dtype=act_dtype)
    xx, state[layer_id * 3 + 0], state[layer_id * 3 + 1], v_first = tmix_one_fused(
        layer_id,
        n_head,
        head_size,
        xx,
        x_prev,
        v_first,
        state[layer_id * 3 + 1],
        _resolve_skeleton(att + "x_r", z, provider, layer_id),
        _resolve_skeleton(att + "x_w", z, provider, layer_id),
        _resolve_skeleton(att + "x_k", z, provider, layer_id),
        _resolve_skeleton(att + "x_v", z, provider, layer_id),
        _resolve_skeleton(att + "x_a", z, provider, layer_id),
        _resolve_skeleton(att + "x_g", z, provider, layer_id),
        _resolve_skeleton(att + "w0", z, provider, layer_id),
        _adapter(att + "w1"),
        _adapter(att + "w2"),
        _resolve_skeleton(att + "a0", z, provider, layer_id),
        _adapter(att + "a1"),
        _adapter(att + "a2"),
        _resolve_skeleton(att + "v0", z, provider, layer_id),
        _adapter(att + "v1"),
        _adapter(att + "v2"),
        _adapter(att + "g1"),
        _adapter(att + "g2"),
        _resolve_skeleton(att + "k_k", z, provider, layer_id),
        _resolve_skeleton(att + "k_a", z, provider, layer_id),
        _resolve_skeleton(att + "r_k", z, provider, layer_id),
        att,
        z,
        provider,
        act_dtype,
    )
    x = (x + xx).to(dtype=act_dtype)

    xx = _ln(x, bbb + "ln2.weight", bbb + "ln2.bias")
    ffn_prev = state[layer_id * 3 + 2].to(dtype=act_dtype)
    xx, state[layer_id * 3 + 2] = cmix_one_fused(
        xx,
        ffn_prev,
        _resolve_skeleton(ffn + "x_k", z, provider, layer_id),
        ffn + "key.weight",
        ffn + "value.weight",
        z,
        provider,
        act_dtype,
    )
    x = (x + xx).to(dtype=act_dtype)
    return x, v_first, state
