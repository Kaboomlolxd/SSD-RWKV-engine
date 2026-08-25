"""Fused Trinity LUT2 matvec for streaming compute (skip full weight materialization)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.lut_gemm_fused import (
    activation_int8_enabled,
    grouped_u8_sparse_gemv,
    grouped_u8_cmix_fused_enabled,
    grouped_u8_cmix_gemv,
    grouped_u8_transposed_gemv,
    grouped_u8_transposed_tmix_fused,
    grouped_u8_transposed_tmix_gemv,
    lut2_gemv,
    lut2_tmix_qkv_gemv_batched,
    small_transposed_lut_enabled,
)


def matvec_z_or_lut(
    z: dict[str, torch.Tensor],
    key: str,
    x: torch.Tensor,
    provider: object | None,
    *,
    act_dtype: torch.dtype,
) -> torch.Tensor:
    """
    ``y = x @ W`` with ``W`` in ``z`` (transposed layout) or LUT2-packed on ``provider``.

    ChatRWKV stores weight matrices transposed in ``z``; LUT2 blobs use ``[out, in]``.
    """
    fused = None
    use_fused = getattr(provider, "_use_fused_lut_matmul", None)
    if (
        provider is not None
        and hasattr(provider, "get_fused_lut_blob")
        and (not callable(use_fused) or use_fused())
    ):
        fused = provider.get_fused_lut_blob(key)
    if fused is not None:
        blob, out_f, in_f = fused
        y = lut2_gemv(
            blob,
            x,
            out_features=out_f,
            in_features=in_f,
            # The head has a separate opt-in because greedy argmax is much
            # more sensitive to a small final-logit perturbation than an
            # intermediate recurrent projection.
            activation_int8=activation_int8_enabled(head=key == "head.weight"),
        )
        return y.to(dtype=act_dtype)
    if provider is not None and hasattr(provider, "get_prepared_tensor"):
        prepared = provider.get_prepared_tensor(key)
        if prepared is not None:
            w = prepared
            return (x.to(dtype=w.dtype) @ w).to(dtype=act_dtype)
    if key not in z:
        raise KeyError(key)
    w = z[key]
    x = x.to(dtype=w.dtype)
    return (x @ w).to(dtype=act_dtype)


def matvec_x_times_weight_or_lut(
    z: dict[str, torch.Tensor],
    key: str,
    x: torch.Tensor,
    provider: object | None,
    *,
    act_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``x @ W`` using an optional row-major grouped-U8 blob.

    The normal fused helper above is for RWKV's transposed z-layout weights
    and computes ``W @ x``.  The small RWKV-7 TMix adapters (w/a/v/g) stay in
    checkpoint layout and are consumed as ``x @ W``; grouped-U8 can traverse
    that layout directly without making a dense temporary matrix.
    """
    fused = None
    use_fused = getattr(provider, "_use_fused_lut_matmul", None)
    if (
        provider is not None
        and hasattr(provider, "get_fused_lut_blob")
        and (not callable(use_fused) or use_fused())
    ):
        fused = provider.get_fused_lut_blob(key)
    if (
        small_transposed_lut_enabled()
        and fused is not None
        and bytes(fused[0][:4]) == b"SG8\x01"
    ):
        blob, _rows, _cols = fused
        return grouped_u8_transposed_gemv(
            blob,
            x,
            out_features=int(_cols),
            in_features=int(_rows),
        ).to(dtype=act_dtype)
    if key in z:
        return (x.to(dtype=z[key].dtype) @ z[key]).to(dtype=act_dtype)
    if provider is not None and hasattr(provider, "get_prepared_tensor"):
        prepared = provider.get_prepared_tensor(key)
        if prepared is not None:
            return (x.to(dtype=prepared.dtype) @ prepared).to(dtype=act_dtype)
    raise KeyError(key)


def tmix_uses_fused(provider: object | None, att_prefix: str) -> bool:
    if provider is None:
        return False
    use_fused = getattr(provider, "_use_fused_lut_matmul", None)
    if callable(use_fused) and not use_fused():
        return False
    if hasattr(provider, "get_fused_tmix_blobs") and provider.get_fused_tmix_blobs(att_prefix):
        return True
    if hasattr(provider, "get_fused_lut_blob"):
        return provider.get_fused_lut_blob(att_prefix + "receptance.weight") is not None
    return False


def tmix_one_fused(
    layer_id: int,
    H: int,
    N: int,
    x: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor,
    state: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    att_prefix: str,
    z: dict[str, torch.Tensor],
    provider: object | None,
    act_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RWKV-7 TMix_one with fused LUT2 for att linear maps (Track D att extension)."""

    def mv(suffix: str, vec: torch.Tensor) -> torch.Tensor:
        return matvec_z_or_lut(
            z, att_prefix + suffix, vec, provider, act_dtype=act_dtype
        )

    xx = x_prev - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    tmix_batch = None
    if provider is not None and hasattr(provider, "get_fused_tmix_blobs"):
        tmix_batch = provider.get_fused_tmix_blobs(att_prefix)
    if tmix_batch is not None:
        blobs, out_f, in_f = tmix_batch
        r, k, v = lut2_tmix_qkv_gemv_batched(
            blobs[:3],
            (xr, xk, xv),
            out_features=out_f,
            in_features=in_f,
        )
        r = r.to(dtype=act_dtype)
        k = k.to(dtype=act_dtype)
        v = v.to(dtype=act_dtype)
    else:
        r = mv("receptance.weight", xr)
        k = mv("key.weight", xk)
        v = mv("value.weight", xv)

    small_batch = None
    if small_transposed_lut_enabled() and provider is not None:
        get_blob = getattr(provider, "get_fused_lut_blob", None)
        if callable(get_blob):
            names = ("w1", "w2", "a1", "a2", "g1", "g2")
            if layer_id != 0:
                names += ("v1", "v2")
            candidate = [get_blob(att_prefix + suffix) for suffix in names]
            if all(
                item is not None and bytes(item[0][:4]) == b"SG8\x01"
                for item in candidate
            ):
                small_batch = dict(zip(names, candidate))

    if small_batch is not None:
        adapter_fused = grouped_u8_transposed_tmix_fused(
            tuple(small_batch[suffix][0] for suffix in names),
            tuple((xw, xa, xg, xv) if layer_id != 0 else (xw, xa, xg)),
            tuple(
                (
                    int(small_batch[suffix][1]),
                    int(small_batch[suffix][2]),
                )
                for suffix in names
            ),
            a0,
        )
    else:
        adapter_fused = None

    if adapter_fused is not None:
        w = adapter_fused[0]
        a = adapter_fused[1]
        g = adapter_fused[2]
        v_gate = adapter_fused[3] if layer_id != 0 else None
    elif small_batch is not None:
        first_names = ["w1", "a1", "g1"]
        first_xs = [xw, xa, xg]
        if layer_id != 0:
            first_names.insert(2, "v1")
            first_xs.insert(2, xv)
        first_values = grouped_u8_transposed_tmix_gemv(
            tuple(small_batch[suffix][0] for suffix in first_names),
            tuple(first_xs),
            tuple(
                (
                    int(small_batch[suffix][1]),
                    int(small_batch[suffix][2]),
                )
                for suffix in first_names
            ),
        )
        first = dict(zip(first_names, first_values))
        w_mid = torch.tanh(first["w1"])
        a_mid = first["a1"]
        g_mid = torch.sigmoid(first["g1"])
        second_names = ["w2", "a2", "g2"]
        second_xs = [w_mid, a_mid, g_mid]
        if layer_id != 0:
            second_names.insert(2, "v2")
            second_xs.insert(2, first["v1"])
        second_values = grouped_u8_transposed_tmix_gemv(
            tuple(small_batch[suffix][0] for suffix in second_names),
            tuple(second_xs),
            tuple(
                (
                    int(small_batch[suffix][1]),
                    int(small_batch[suffix][2]),
                )
                for suffix in second_names
            ),
        )
        second = dict(zip(second_names, second_values))
        w = second["w2"]
        a = torch.sigmoid(a0 + second["a2"])
        g = second["g2"]
        v_gate = second.get("v2") if layer_id != 0 else None
    else:
        w = torch.tanh(
            matvec_x_times_weight_or_lut(
                z, att_prefix + "w1", xw, provider, act_dtype=act_dtype
            )
        )
        w = matvec_x_times_weight_or_lut(
            z, att_prefix + "w2", w, provider, act_dtype=act_dtype
        )
        a = matvec_x_times_weight_or_lut(
            z,
            att_prefix + "a1",
            xa,
            provider,
            act_dtype=act_dtype,
        )
        a = torch.sigmoid(
            a0
            + matvec_x_times_weight_or_lut(
                z, att_prefix + "a2", a, provider, act_dtype=act_dtype
            )
        )
        g = torch.sigmoid(
            matvec_x_times_weight_or_lut(
                z, att_prefix + "g1", xg, provider, act_dtype=act_dtype
            )
        )
        g = matvec_x_times_weight_or_lut(
            z, att_prefix + "g2", g, provider, act_dtype=act_dtype
        )
        v_gate = None

    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        if v_gate is None:
            v_gate = matvec_x_times_weight_or_lut(
                z, att_prefix + "v1", xv, provider, act_dtype=act_dtype
            )
            v_gate = matvec_x_times_weight_or_lut(
                z, att_prefix + "v2", v_gate, provider, act_dtype=act_dtype
            )
        v = v + (v_first - v) * torch.sigmoid(v0 + v_gate)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))

    # ``ab`` is an outer product, so materializing it and multiplying the
    # recurrent state by a dense N x N matrix wastes almost all of the work:
    #
    #   state @ ((-kk)[:, :, None] @ (kk*a)[:, None, :])
    #       == (state @ -kk[:, :, None]) * (kk*a)[:, None, :]
    #
    # Keep the factors in the state dtype before the batched matvec.  This
    # preserves the model's recurrent precision while avoiding the old
    # O(H*N^3) state update (about 10.5M multiply-adds per layer at N=64).
    kk_state = kk.view(H, N).to(dtype=state.dtype)
    ka_state = (kk * a).view(H, N).to(dtype=state.dtype)
    state_ab = torch.bmm(
        state,
        (-kk_state).unsqueeze(-1),
    ).squeeze(-1)
    vk = v.view(H, N, 1) * k.view(H, 1, N)
    state = (
        state * w.view(H, 1, N)
        + state_ab.unsqueeze(-1) * ka_state.unsqueeze(1)
        + vk.to(dtype=state.dtype)
    )
    xx = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    xx = F.group_norm(
        xx.view(1, H * N),
        num_groups=H,
        weight=z[att_prefix + "ln_x.weight"],
        bias=z[att_prefix + "ln_x.bias"],
        eps=64e-5,
    ).view(H * N)
    xx = xx + (
        (r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)
    ).view(H * N)
    if tmix_batch is not None:
        blobs, out_f, in_f = tmix_batch
        out = lut2_gemv(
            blobs[3], xx * g, out_features=out_f, in_features=in_f
        ).to(dtype=act_dtype)
    else:
        out = mv("output.weight", xx * g)
    return out, x, state, v_first


def cmix_one_fused(
    x: torch.Tensor,
    x_prev: torch.Tensor,
    x_k: torch.Tensor,
    key_name: str,
    value_name: str,
    z: dict[str, torch.Tensor],
    provider: object | None,
    act_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RWKV-7 CMix_one with optional fused LUT2 FFN matrices."""
    xx = x_prev - x
    k = x + xx * x_k
    prefetch_begin = getattr(provider, "begin_cmix_tile_prefetch", None)
    prefetch_ticket = prefetch_begin(value_name) if callable(prefetch_begin) else None
    use_fused = (
        provider is not None
        and hasattr(provider, "_use_fused_lut_matmul")
        and provider._use_fused_lut_matmul()
    )
    tiled_getter = getattr(provider, "get_cmix_tiled_value_matrix", None)
    tiled = tiled_getter(value_name) if callable(tiled_getter) else None
    get_blob = getattr(provider, "get_fused_lut_blob", None)
    key_fused = get_blob(key_name) if use_fused and callable(get_blob) else None
    value_fused = get_blob(value_name) if use_fused and callable(get_blob) else None
    # The native grouped-U8 path can keep the 10,240-wide CMix intermediate
    # out of Python/Torch entirely.  Skip it when telemetry or a tiled value
    # sidecar is explicitly requested, because those paths need the
    # intermediate activation for their accounting/read selection.
    if (
        use_fused
        and grouped_u8_cmix_fused_enabled()
        and tiled is None
        and not bool(getattr(provider, "_cmix_sparsity_enabled", False))
        and key_fused is not None
        and value_fused is not None
        and bytes(key_fused[0][:4]) == b"SG8\x01"
        and bytes(value_fused[0][:4]) == b"SG8\x01"
    ):
        out = grouped_u8_cmix_gemv(
            key_fused[0],
            value_fused[0],
            k,
            key_out_features=int(key_fused[1]),
            key_in_features=int(key_fused[2]),
            value_out_features=int(value_fused[1]),
            value_in_features=int(value_fused[2]),
        ).to(dtype=act_dtype)
        return out, x
    if use_fused and hasattr(provider, "get_fused_lut_blob"):
        k = torch.relu(matvec_z_or_lut(z, key_name, k, provider, act_dtype=act_dtype)) ** 2
    else:
        k = torch.relu(k @ z[key_name]) ** 2
    record = getattr(provider, "record_cmix_sparsity", None)
    if callable(record):
        record(k)
    if tiled is not None:
        out, selective_stats = tiled.matmul(k, prefetch_ticket=prefetch_ticket)
        selective_record = getattr(provider, "record_cmix_selective_stats", None)
        if callable(selective_record):
            selective_record(selective_stats)
        out = out.to(dtype=act_dtype)
    elif use_fused and hasattr(provider, "get_fused_lut_blob"):
        fused_value = value_fused
        if fused_value is not None and bytes(fused_value[0][:4]) == b"SG8\x01":
            blob, out_features, in_features = fused_value
            out = grouped_u8_sparse_gemv(
                blob,
                k,
                out_features=out_features,
                in_features=in_features,
            ).to(dtype=act_dtype)
        else:
            out = matvec_z_or_lut(z, value_name, k, provider, act_dtype=act_dtype)
    else:
        out = k @ z[value_name]
    return out, x
