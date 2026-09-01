"""
Pack-driven RWKV-7 forward_one — inject streamed layer weights then run TMix/CMix.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.metrics import Timer
from rwkv_ssd.runtime.power import throttle_after_work
from rwkv_ssd.runtime.sampling import sample_torch
from rwkv_ssd.runtime.provider_factory import prefetch_ahead, release_layer
from rwkv_ssd.runtime.rwkv7_weights import (
    all_block_layers_in_z,
    inject_layer_into_z,
    layer_ready_in_z,
    layer_weights_in_z,
    promote_stream_cache_to_full_z,
    warm_stream_cache_layers_into_z,
)
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

if TYPE_CHECKING:
    from rwkv_ssd.runtime.metrics import MetricsCollector


def _remember_rwkv7_state(
    model: Any,
    state: list[torch.Tensor],
    last_token_id: int,
    logits: torch.Tensor | None = None,
) -> None:
    """Store the last recurrent state on the ChatRWKV model for snapshots."""
    setattr(model, "_rwkv_ssd_last_state", [t.clone() for t in state])
    setattr(model, "_rwkv_ssd_last_token_id", int(last_token_id))
    if logits is not None:
        observed = logits.detach().clone()
        # Sequence prefill kernels return [tokens, vocab], while the
        # generation contract and diagnostic probe represent the distribution
        # for the next token as one vocabulary vector.  Sampling used to pick
        # row zero implicitly, which both corrupted multi-token prompts and
        # made the conformance harness see a prompt-length-dependent shape.
        if observed.ndim > 1:
            observed = observed.reshape(-1, observed.shape[-1])[-1]
        setattr(model, "_rwkv_ssd_last_logits", observed)


def _model_dtype(z: dict[str, torch.Tensor]) -> torch.dtype:
    return z["emb.weight"].dtype


def _is_rwkv7a_deepembed_v1(model: Any) -> bool:
    return bool(getattr(model, "_rwkv7a_deepembed_v1", False))


def _rwkv7a_v1_s_emb_rows(
    model: Any,
    z: dict[str, torch.Tensor],
    ffn: str,
    token_ids: int | torch.Tensor,
) -> torch.Tensor:
    """Return ChatRWKV DeepEmbed-v1 CMix rows for one or more token ids.

    Resident ChatRWKV merges ``emb @ s_emb_x.T`` during model construction;
    a streaming skeleton keeps both source tensors so the pack only reads the
    requested rows.  ``_rwkv_ssd_s_emb_merged`` tells the two cases apart.
    """
    ids = (
        token_ids.to(dtype=torch.long, device=z["emb.weight"].device)
        if isinstance(token_ids, torch.Tensor)
        else torch.tensor([int(token_ids)], dtype=torch.long, device=z["emb.weight"].device)
    )
    base = z[ffn + "s_emb.weight"].index_select(0, ids)
    if not bool(getattr(model, "_rwkv_ssd_s_emb_merged", False)):
        projection = z[ffn + "s_emb_x.weight"]
        emb = z["emb.weight"].index_select(0, ids)
        base = base + emb @ projection.transpose(0, 1)
    return base[0] if not isinstance(token_ids, torch.Tensor) else base


def _layer_norm(
    x: torch.Tensor,
    z: dict[str, torch.Tensor],
    weight_key: str,
    bias_key: str,
    n_embd: int,
    act_dtype: torch.dtype,
) -> torch.Tensor:
    w = z[weight_key]
    b = z[bias_key]
    x = x.to(dtype=w.dtype)
    # CPU layer_norm often returns float32; kernels expect activation dtype (bf16/fp16).
    return F.layer_norm(x, (n_embd,), weight=w, bias=b).to(dtype=act_dtype)


def _can_use_packed_block(
    provider: ManifestWeightProvider,
    z: dict[str, torch.Tensor],
    layer_id: int,
    by_layer: dict[int, list[TensorEntry]],
    model: Any | None = None,
) -> bool:
    """True when the packed block forward path can run for ``layer_id`` without
    per-tensor ``inject_layer_into_z`` — the F1-F3 staging-reduction gate.

    Conditions:
      * the provider has a fused LUT matmul registered (LUT2 / shadow),
      * the att weights are NOT in ``z`` (so the fused blob will provide them),
      * the layer is not already fully in ``z`` (otherwise native ``forward`` runs).
    """
    from rwkv_ssd.runtime.packed_block_forward import packed_block_forward_enabled

    if not provider._use_fused_lut_matmul():
        return False
    if model is not None and _is_rwkv7a_deepembed_v1(model):
        return False
    att = f"blocks.{layer_id}.att."
    # If the layer's att weights are already in z (warm-z promote / resident),
    # the native forward path is faster — no need for the packed block path.
    if z.get(att + "receptance.weight") is not None:
        return False
    if z.get(att + "key.weight") is not None:
        return False
    packed_enabled = packed_block_forward_enabled(
        provider,
        att,
        mode=provider.mode,
        stream_layer_cache=provider._stream_layer_cache,
        pack_uses_quant=provider._pack_uses_quant,
    )
    if packed_enabled:
        return True

    # On the first visit to a layer the provider has not read the layer span
    # yet, so ``tmix_uses_fused`` cannot see the four registered blobs.  Auto
    # streaming used to fall through to the legacy dense branch at that point;
    # the subsequent fused decode filtered those matrices out of ``z`` and the
    # branch then failed with a missing ``att.receptance.weight``.  The
    # manifest is enough to make the safe decision before the read: if all four
    # attention projections are eligible fused entries, let
    # ``_packed_block_step`` load/register them and run the packed block.
    try:
        from rwkv_ssd.runtime.lut_gemm_fused import (
            att_fused_suffixes,
            fused_lut2_enabled,
        )

        if not fused_lut2_enabled(
            provider.mode,
            provider._stream_layer_cache,
            pack_uses_quant=provider._pack_uses_quant,
        ):
            return False
        by_name = {entry.name: entry for entry in by_layer.get(layer_id, [])}
        return all(
            (entry := by_name.get(att + suffix)) is not None
            and provider._is_fused_lut_entry(entry)
            for suffix in att_fused_suffixes()
        )
    except (ImportError, KeyError, TypeError, ValueError):
        return False


def _packed_block_step(
    layer_id: int,
    x: torch.Tensor,
    state: list[torch.Tensor],
    v_first: torch.Tensor,
    z: dict[str, torch.Tensor],
    provider: ManifestWeightProvider,
    entries: list[TensorEntry],
    *,
    n_head: int,
    head_size: int,
    n_embd: int,
    act_dtype: torch.dtype,
    metrics: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Load + prepare just the small skeleton, then run the packed block forward.

    Fused LUT blob holds the att/FFN weights; the bf16 inject step is
    unnecessary. Caches the prepared skeleton in
    ``provider._prepared_layers[layer_id]`` for re-use on the next token
    (skips a re-decode on the F2/F5 paths where the layer revisits).

    Returns updated ``(x, v_first, state)`` — callers must assign these;
    ``forward_block_packed`` rebinds ``x``/``v_first`` locally.
    """
    from rwkv_ssd.runtime.packed_block_forward import forward_block_packed

    cached = provider._prepared_layers.get(layer_id)
    # Empty ``{}`` markers are not usable for packed forward (need ln1/x_*).
    if cached:
        prepared = cached
        # Cache-hit path never calls begin_layer, so open the metrics row here.
        if metrics is not None:
            if metrics.layers and metrics.layers[-1].layer_id == layer_id:
                row = metrics.layers[-1]
            else:
                row = metrics.start_layer(layer_id)
            row.layer_cache_hits += len(prepared)
    else:
        # Note: do NOT call ``metrics.start_layer(layer_id)`` here — the
        # subsequent ``provider.load_layer_tensors`` call already creates
        # a row via ``begin_layer``. Starting a second row would double-
        # count per-layer timings in the metrics CSV (the P0.6 inflation
        # bug: 12 layers × 12 tokens = 144 expected, got 374).
        with Timer() as t_read:
            layer_tensors = provider.load_layer_tensors(entries)
        row = None
        if metrics is not None and metrics.layers and metrics.layers[-1].layer_id == layer_id:
            row = metrics.layers[-1]
            # begin_layer already stamped read_ms; don't double-add t_read.
            _ = t_read
        with Timer() as t_prep:
            prepared = provider.prepare_layer_for_z(layer_id, layer_tensors, None)
        if row is not None:
            row.staging_ms += t_prep.elapsed_ms
        # Cache so the next token reuses the skeleton (avoids a re-decode).
        if (
            provider._stream_layer_cache
            or provider._strict_fused_retain_layers()
            or layer_id in provider._z_retention.pinned_layer_ids
        ):
            provider._prepared_layers[layer_id] = prepared
    # ``tmix_one_fused`` looks up ``z[att + 'ln_x.weight']`` directly (not via
    # _resolve_skeleton). For streamed layers the small skeleton lives in
    # provider._prepared_layers; mirror ln_x into z so the call works
    # without the per-tensor inject step. The mirror is a single
    # weight + bias per layer (small). We only mirror when missing
    # (don't overwrite) — the LRU eviction in the per-tensor path
    # handles stale ln_x by removing the whole layer. A stale ln_x
    # would silently break the fused matmul on the next call.
    att = f"blocks.{layer_id}.att."
    for k in ("ln_x.weight", "ln_x.bias"):
        zk = att + k
        if zk not in z and zk in prepared:
            z[zk] = prepared[zk]
    return forward_block_packed(
        layer_id,
        x,
        state,
        v_first,
        z,
        provider,
        n_head=n_head,
        head_size=head_size,
        n_embd=n_embd,
        act_dtype=act_dtype,
    )


def _evict_layer_from_z(
    z: dict[str, torch.Tensor],
    layer_id: int,
    provider: ManifestWeightProvider | None = None,
) -> None:
    """Evict block tensors from ``z``; strict fused retain keeps skeleton vectors in ``z``."""
    from rwkv_ssd.runtime.lut_gemm_fused import is_fused_lut_tensor_name

    strict_skeleton = (
        provider is not None
        and provider._strict_fused_retain_layers()
        and not provider._strict_fused_lean_z()
    )
    prefix = f"blocks.{layer_id}."
    for key in list(z.keys()):
        if not key.startswith(prefix):
            continue
        if strict_skeleton and not is_fused_lut_tensor_name(key):
            continue
        del z[key]


def forward_one_streaming(
    model: Any,
    token_id: int,
    state: list[torch.Tensor],
    provider: ManifestWeightProvider,
    by_layer: dict[int, list[TensorEntry]],
    *,
    layer_ids: list[int] | None = None,
    prefetch: bool = True,
    metrics: MetricsCollector | None = None,
    evict_layers: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    One decode step using ChatRWKV kernels with per-layer weights from ``provider``.

    Global tensors (``emb.weight``, ``ln_out``, ``head.weight``) must already live in
    ``model.z`` (resident preload or full checkpoint load).
    """
    from rwkv.model import RWKV_x070_CMix_one, RWKV_x070_TMix_one  # type: ignore[import-untyped]

    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd
    layers = layer_ids if layer_ids is not None else list(range(n_layer))
    act_dtype = _model_dtype(z)
    # Native GEMV receives FP32 NumPy inputs.  On AVX2 hosts, keeping the
    # fused block's surrounding activations in FP32 avoids converting each
    # projection result back to BF16 only to widen it again for the next
    # native call.  The helper is hardware-gated in ``auto`` mode and still
    # supports explicit overrides for A/B testing.
    if (
        isinstance(provider, ManifestWeightProvider)
        and provider._device.type == "cpu"
        and provider._use_fused_lut_matmul()
        and not any(layer_weights_in_z(z, layer_id) for layer_id in layers)
    ):
        from rwkv_ssd.runtime.lut_gemm_fused import activation_fp32_enabled

        if activation_fp32_enabled():
            act_dtype = torch.float32
    all_in_z = all_block_layers_in_z(z, layers, by_layer)
    from rwkv_ssd.runtime.rwkv7_weights import all_stream_layers_prepared_in_provider

    stream_prepared = (
        isinstance(provider, ManifestWeightProvider)
        and provider._strict_fused_retain_layers()
        and all_stream_layers_prepared_in_provider(provider, layers, by_layer)
    )
    if stream_prepared:
        prefetch = False

    if all_in_z and hasattr(model, "forward") and not _is_rwkv7a_deepembed_v1(model):
        with torch.no_grad():
            with Timer() as t_native:
                logits, state = model.forward([token_id], state)
            if metrics:
                row = metrics.start_layer(-1)
                row.compute_ms += t_native.elapsed_ms
            return logits, state

    with torch.no_grad():
        x = z["emb.weight"][token_id].to(dtype=act_dtype)
        v_first = torch.empty_like(x)

        for idx, layer_id in enumerate(layers):
            entries = by_layer.get(layer_id, [])
            if (
                prefetch
                and isinstance(provider, ManifestWeightProvider)
                and not all_in_z
                # Prefetch is only useful when the decoded tensors will be
                # reused on the next call (stream_layer_cache or warm-z).
                # For the strict-fused / no-cache path, the prefetch thread
                # blocks the main thread on ``begin_layer`` for the I/O time
                # with no compute to overlap against — it just adds
                # ``prefetch_wait_ms`` and a ``ThreadPoolExecutor.submit``
                # round-trip per layer per token. The native forward / F-1
                # path consumes the data via the LUT blob / z skeleton, so
                # the prefetch is a net loss there too. Skip it.
                and provider._stream_layer_cache
            ):
                if not (
                    provider._stream_layer_cache and layer_weights_in_z(z, layer_id)
                ):
                    prefetch_ahead(
                        provider,
                        provider.planner,
                        layers,
                        idx,
                        by_layer,
                        z=z if provider._stream_layer_cache else None,
                    )

            # P1.4 — early-exit packed block forward (F1-F3 strict / bounded).
            # When the fused LUT provider is active and this layer's att/FFN
            # weights are not in z, the packed path loads + prepares just the
            # small skeleton (x_*, w*, ln*, r_k, ...) and runs TMix/CMix via
            # tmix_one_fused / cmix_one_fused. Skips the per-tensor bf16
            # ``inject_layer_into_z`` step which is the dominant staging cost
            # on F1-F3 (23-45 ms/tok pre-fix, target ~10 ms/tok).
            if (
                entries
                and isinstance(provider, ManifestWeightProvider)
                and _can_use_packed_block(provider, z, layer_id, by_layer)
                and not _is_rwkv7a_deepembed_v1(model)
            ):
                with Timer() as t_packed:
                    x, v_first, state = _packed_block_step(
                        layer_id,
                        x,
                        state,
                        v_first,
                        z,
                        provider,
                        entries,
                        n_head=n_head,
                        head_size=head_size,
                        n_embd=n_embd,
                        act_dtype=act_dtype,
                        metrics=metrics,
                    )
                if (
                    metrics
                    and metrics.layers
                    and metrics.layers[-1].layer_id == layer_id
                ):
                    # Subtract prep/read already on the row so compute_ms is
                    # the packed forward itself (not wall of the whole step).
                    row = metrics.layers[-1]
                    accounted = row.read_ms + row.staging_ms
                    row.compute_ms += max(0.0, t_packed.elapsed_ms - accounted)
                # Apply per-layer eviction for streamed layers.
                # Two modes:
                #   1. ``_strict_fused_lean_z()`` — always evict the
                #      streamed layer after the step (lean-z mode); also
                #      drop the provider prepared entry unless we want
                #      to retain shadow weights for the next token.
                #   2. ``should_keep_after_step()`` — general LRU-style
                #      eviction when the layer shouldn't be retained.
                # Both are combined here so the eviction is done at most
                # once per step.
                if (
                    evict_layers
                    and provider.layer_has_streamed_tensors(entries)
                    and layer_id not in provider._z_retention.pinned_layer_ids
                ):
                    should_evict = False
                    if provider._strict_fused_lean_z():
                        should_evict = True
                    else:
                        keep = provider._z_retention.should_keep_after_step(
                            z,
                            layer_id,
                            stream_layer_cache=(
                                provider._stream_layer_cache
                                and provider.layer_has_streamed_tensors(entries)
                            ),
                        )
                        should_evict = not keep
                    if should_evict:
                        _evict_layer_from_z(z, layer_id, provider)
                        if provider._strict_fused_lean_z():
                            # Keep the prepared skeleton (not ``{}``) so
                            # packed forward can reuse ln1/x_* and LUT
                            # blobs without a full pack re-read. Popping
                            # the key caused the F1/F3 collapse.
                            provider.release_prepared_tensors(layer_id)
                        release_layer(provider, entries)
                continue

            prepared: dict[str, torch.Tensor] = {}
            if entries:
                needs_weights = (
                    provider is None
                    or not isinstance(provider, ManifestWeightProvider)
                    or not provider._use_fused_lut_matmul()
                    or _is_rwkv7a_deepembed_v1(model)
                )
                layer_z_ok = (not needs_weights and layer_ready_in_z(z, layer_id)) or (
                    needs_weights and all_block_layers_in_z(z, [layer_id], by_layer)
                )
                if not layer_z_ok:
                    if isinstance(
                        provider, ManifestWeightProvider
                    ) and layer_id in provider._prepared_layers:
                        prepared = provider._prepared_layers[layer_id]
                        if metrics and entries:
                            row = metrics.start_layer(layer_id)
                            row.layer_cache_hits += len(prepared) if prepared else 1
                        from rwkv_ssd.runtime.lut_gemm_fused import (
                            filter_tensors_for_fused_inject,
                        )

                        if not _is_rwkv7a_deepembed_v1(model):
                            prepared = filter_tensors_for_fused_inject(provider, prepared)
                        if provider._strict_fused_retain_layers():
                            missing = {k: v for k, v in prepared.items() if k not in z}
                            if missing:
                                inject_layer_into_z(z, missing)
                                for key in missing:
                                    if z[key].dtype != act_dtype:
                                        z[key] = z[key].to(dtype=act_dtype)
                                if not provider._strict_fused_lean_z():
                                    provider.release_prepared_tensors(layer_id)
                        else:
                            # Only inject keys not already in z (avoids
                            # clone/copy_ on every stream-cache hit).
                            missing = {
                                k: v for k, v in prepared.items() if k not in z
                            }
                            if missing:
                                inject_layer_into_z(z, missing)
                                for key in missing:
                                    if z[key].dtype != act_dtype:
                                        z[key] = z[key].to(dtype=act_dtype)
                    else:
                        row = metrics.layers[-1] if metrics and metrics.layers else None
                        layer_tensors = provider.load_layer_tensors(entries)
                        if (
                            metrics
                            and metrics.layers
                            and metrics.layers[-1].layer_id != layer_id
                        ):
                            row = metrics.start_layer(layer_id)
                        elif metrics and metrics.layers:
                            row = metrics.layers[-1]
                        prepared = provider.prepare_layer_for_z(
                            layer_id, layer_tensors, row
                        )
                        from rwkv_ssd.runtime.lut_gemm_fused import (
                            filter_tensors_for_fused_inject,
                        )

                        if not _is_rwkv7a_deepembed_v1(model):
                            prepared = filter_tensors_for_fused_inject(provider, prepared)
                        if provider._strict_fused_retain_layers():
                            missing = {k: v for k, v in prepared.items() if k not in z}
                            if missing:
                                inject_layer_into_z(z, missing)
                                for key in missing:
                                    if z[key].dtype != act_dtype:
                                        z[key] = z[key].to(dtype=act_dtype)
                                if not provider._strict_fused_lean_z():
                                    provider.release_prepared_tensors(layer_id)
                        else:
                            # Only inject keys not already in z (avoids
                            # clone/copy_ on every stream-cache hit).
                            missing = {
                                k: v for k, v in prepared.items() if k not in z
                            }
                            if missing:
                                inject_layer_into_z(z, missing)
                                for key in missing:
                                    if z[key].dtype != act_dtype:
                                        z[key] = z[key].to(dtype=act_dtype)
                elif metrics and entries:
                    if not metrics.layers or metrics.layers[-1].layer_id != layer_id:
                        row = metrics.start_layer(layer_id)
                        if (
                            isinstance(provider, ManifestWeightProvider)
                            and provider._stream_layer_cache
                            and provider.layer_has_streamed_tensors(entries)
                        ):
                            row.layer_cache_hits += len(entries)

            bbb = f"blocks.{layer_id}."
            att = f"blocks.{layer_id}.att."
            ffn = f"blocks.{layer_id}.ffn."

            from rwkv_ssd.runtime.packed_block_forward import (
                forward_block_packed,
                packed_block_forward_enabled,
            )

            skeleton_in_z = z.get(att + "x_r") is not None
            skeleton_in_provider = (
                isinstance(provider, ManifestWeightProvider)
                and layer_id in getattr(provider, "_prepared_layers", {})
                and att + "x_r" in provider._prepared_layers[layer_id]
            )
            use_packed_block = (
                isinstance(provider, ManifestWeightProvider)
                and not _is_rwkv7a_deepembed_v1(model)
                and packed_block_forward_enabled(
                    provider,
                    att,
                    mode=provider.mode,
                    stream_layer_cache=provider._stream_layer_cache,
                    pack_uses_quant=provider._pack_uses_quant,
                )
                and (skeleton_in_z or skeleton_in_provider)
                and z.get(att + "receptance.weight") is None
            )
            if use_packed_block:
                with Timer() as t_compute:
                    x, v_first, state = forward_block_packed(
                        layer_id,
                        x,
                        state,
                        v_first,
                        z,
                        provider,
                        n_head=n_head,
                        head_size=head_size,
                        n_embd=n_embd,
                        act_dtype=act_dtype,
                    )
                if (
                    metrics
                    and metrics.layers
                    and metrics.layers[-1].layer_id == layer_id
                ):
                    metrics.layers[-1].compute_ms += t_compute.elapsed_ms
            else:
                with Timer() as t_compute:
                    xx = _layer_norm(
                        x, z, bbb + "ln1.weight", bbb + "ln1.bias", n_embd, act_dtype
                    )

                    x_prev = state[layer_id * 3 + 0].to(dtype=act_dtype)
                    from rwkv_ssd.runtime.rwkv7_linear import (
                        cmix_one_fused,
                        matvec_z_or_lut,
                        tmix_one_fused,
                        tmix_uses_fused,
                    )

                    fused_provider = (
                        provider
                        if isinstance(provider, ManifestWeightProvider)
                        and not _is_rwkv7a_deepembed_v1(model)
                        and provider._use_fused_lut_matmul()
                        else None
                    )
                    if fused_provider is not None and tmix_uses_fused(
                        fused_provider, att
                    ):
                        (
                            xx,
                            state[layer_id * 3 + 0],
                            state[layer_id * 3 + 1],
                            v_first,
                        ) = tmix_one_fused(
                            layer_id,
                            n_head,
                            head_size,
                            xx,
                            x_prev,
                            v_first,
                            state[layer_id * 3 + 1],
                            z[att + "x_r"],
                            z[att + "x_w"],
                            z[att + "x_k"],
                            z[att + "x_v"],
                            z[att + "x_a"],
                            z[att + "x_g"],
                            z[att + "w0"],
                            z[att + "w1"],
                            z[att + "w2"],
                            z[att + "a0"],
                            z[att + "a1"],
                            z[att + "a2"],
                            z[att + "v0"],
                            z[att + "v1"],
                            z[att + "v2"],
                            z[att + "g1"],
                            z[att + "g2"],
                            z[att + "k_k"],
                            z[att + "k_a"],
                            z[att + "r_k"],
                            att,
                            z,
                            fused_provider,
                            act_dtype,
                        )
                    else:
                        (
                            xx,
                            state[layer_id * 3 + 0],
                            state[layer_id * 3 + 1],
                            v_first,
                        ) = RWKV_x070_TMix_one(
                            layer_id,
                            n_head,
                            head_size,
                            xx,
                            x_prev,
                            v_first,
                            state[layer_id * 3 + 1],
                            z[att + "x_r"],
                            z[att + "x_w"],
                            z[att + "x_k"],
                            z[att + "x_v"],
                            z[att + "x_a"],
                            z[att + "x_g"],
                            z[att + "w0"],
                            z[att + "w1"],
                            z[att + "w2"],
                            z[att + "a0"],
                            z[att + "a1"],
                            z[att + "a2"],
                            z[att + "v0"],
                            z[att + "v1"],
                            z[att + "v2"],
                            z[att + "g1"],
                            z[att + "g2"],
                            z[att + "k_k"],
                            z[att + "k_a"],
                            z[att + "r_k"],
                            z[att + "receptance.weight"],
                            z[att + "key.weight"],
                            z[att + "value.weight"],
                            z[att + "output.weight"],
                            z[att + "ln_x.weight"],
                            z[att + "ln_x.bias"],
                        )
                    x = (x + xx).to(dtype=act_dtype)

                    xx = _layer_norm(
                        x, z, bbb + "ln2.weight", bbb + "ln2.bias", n_embd, act_dtype
                    )
                    ffn_prev = state[layer_id * 3 + 2].to(dtype=act_dtype)
                    if fused_provider is not None:
                        xx, state[layer_id * 3 + 2] = cmix_one_fused(
                            xx,
                            ffn_prev,
                            z[ffn + "x_k"],
                            ffn + "key.weight",
                            ffn + "value.weight",
                            z,
                            fused_provider,
                            act_dtype,
                        )
                    elif _is_rwkv7a_deepembed_v1(model):
                        s_emb_row = _rwkv7a_v1_s_emb_rows(
                            model, z, ffn, token_id
                        )
                        (
                            xx,
                            state[layer_id * 3 + 2],
                        ) = RWKV_x070_CMix_one(
                            xx,
                            ffn_prev,
                            z[ffn + "x_k"],
                            z[ffn + "key.weight"],
                            z[ffn + "value.weight"],
                            s_emb_row,
                            z[ffn + "s1"],
                            z[ffn + "s2"],
                            z[ffn + "s0"],
                        )
                    else:
                        xx, state[layer_id * 3 + 2] = RWKV_x070_CMix_one(
                            xx,
                            ffn_prev,
                            z[ffn + "x_k"],
                            z[ffn + "key.weight"],
                            z[ffn + "value.weight"],
                        )
                    x = (x + xx).to(dtype=act_dtype)

                if (
                    metrics
                    and metrics.layers
                    and metrics.layers[-1].layer_id == layer_id
                ):
                    metrics.layers[-1].compute_ms += t_compute.elapsed_ms

            if (
                evict_layers
                and isinstance(provider, ManifestWeightProvider)
                and provider._strict_fused_lean_z()
                and entries
                and provider.layer_has_streamed_tensors(entries)
                and layer_id not in provider._z_retention.pinned_layer_ids
            ):
                _evict_layer_from_z(z, layer_id, None)
                # Always leave an empty prepared marker so the next token
                # reuses retained LUT blobs instead of re-reading the pack.
                provider.release_prepared_tensors(layer_id)

            if evict_layers and entries:
                streamed = (
                    provider.layer_has_streamed_tensors(entries)
                    if isinstance(provider, ManifestWeightProvider)
                    else True
                )
                touched_layer = bool(prepared) or (
                    isinstance(provider, ManifestWeightProvider)
                    and provider._stream_layer_cache
                    and streamed
                    and any(k.startswith(f"blocks.{layer_id}.") for k in z)
                )
                if touched_layer and isinstance(provider, ManifestWeightProvider):
                    keep = provider._z_retention.should_keep_after_step(
                        z,
                        layer_id,
                        stream_layer_cache=provider._stream_layer_cache and streamed,
                    )
                elif touched_layer:
                    keep = False
                else:
                    keep = True
                if not keep:
                    _evict_layer_from_z(z, layer_id, provider)
                    if isinstance(provider, ManifestWeightProvider):
                        release_layer(provider, entries)

        x = _layer_norm(x, z, "ln_out.weight", "ln_out.bias", n_embd, act_dtype)
        from rwkv_ssd.runtime.rwkv7_linear import matvec_z_or_lut

        fused_provider = (
            provider
            if isinstance(provider, ManifestWeightProvider)
            and provider._use_fused_lut_matmul()
            else None
        )
        # Use the same dtype-safe resolver for both packed and dense heads.
        # The AVX2 FP32 activation path can leave ``x`` in FP32 while a
        # legacy/mixed pack still keeps ``head.weight`` in BF16; a direct
        # ``x @ z[...]`` then raises instead of widening/narrowing at the
        # matmul boundary.  The helper also routes a packed head through the
        # native GEMV when its blob is available.
        x = matvec_z_or_lut(
            z, "head.weight", x, fused_provider, act_dtype=act_dtype
        )
        return x, state


def forward_one(
    model: Any,
    token_id: int,
    state: list[torch.Tensor],
    provider: ManifestWeightProvider,
    by_layer: dict[int, list[TensorEntry]],
    *,
    layer_ids: list[int] | None = None,
    prefetch: bool = True,
    metrics: MetricsCollector | None = None,
    evict_layers: bool = True,
    use_native_if_complete: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    One decode step: native ChatRWKV ``forward`` when all layers are in ``z``,
    otherwise pack-driven per-layer inject + TMix/CMix.
    """
    layers = layer_ids if layer_ids is not None else list(range(model.n_layer))
    z = model.z
    if isinstance(provider, ManifestWeightProvider) and provider._stream_layer_cache:
        promote_stream_cache_to_full_z(z, provider, layers, by_layer, metrics)
    if (
        use_native_if_complete
        and isinstance(provider, ManifestWeightProvider)
        and all_block_layers_in_z(z, layers, by_layer)
        and not _is_rwkv7a_deepembed_v1(model)
        and hasattr(model, "forward")
    ):
        with torch.no_grad():
            with Timer() as t_native:
                logits, state = model.forward([token_id], state)
            if metrics:
                row = metrics.start_layer(-1)
                row.compute_ms += t_native.elapsed_ms
            return logits, state
    return forward_one_streaming(
        model,
        token_id,
        state,
        provider,
        by_layer,
        layer_ids=layer_ids,
        prefetch=prefetch,
        metrics=metrics,
        evict_layers=evict_layers,
    )


def greedy_token_ids_native(
    model: Any,
    pipeline: Any,
    prompt: str,
    max_tokens: int,
    metrics: MetricsCollector | None = None,
    power_percent: int = 100,
    *,
    temperature: float = 1.0,
    greedy: bool = True,
    token_callback=None,
    cancel_event=None,
    deadline: float | None = None,
) -> list[int]:
    """Greedy decode via native ``model.forward`` (no pipeline sampling overhead)."""
    token_ids = pipeline.encode(prompt)
    state = model.generate_zero_state()
    control = make_generation_control(
        token_callback=token_callback,
        cancel_event=cancel_event,
        deadline=deadline,
    )

    with torch.no_grad():
        t_prefill0 = time.perf_counter()
        if token_ids:
            with Timer() as t_prefill:
                logits, state = model.forward(token_ids, state)
            context_last = int(token_ids[-1])
            if metrics:
                row = metrics.start_layer(-1)
                row.compute_ms += t_prefill.elapsed_ms
        else:
            # Consume the explicit zero token only when there is no prompt;
            # otherwise prefill already produced the distribution to sample.
            context_last = 0
            with Timer() as t_prefill:
                logits, state = model.forward([context_last], state)
            if metrics:
                row = metrics.start_layer(-1)
                row.compute_ms += t_prefill.elapsed_ms
        if metrics is not None:
            metrics.prefill_wall_s = time.perf_counter() - t_prefill0

        out: list[int] = []
        for _ in range(max_tokens):
            if control is not None:
                control.check()
            t_token = time.perf_counter()
            # Prefill (or the preceding decode step) has already produced the
            # logits for the next token.  Sampling them here avoids consuming
            # the final prompt token twice and keeps this path aligned with
            # rwkv.cpp's eval-sequence contract.
            token = sample_torch(logits, temperature=temperature, greedy=greedy)
            out.append(token)
            # Publish the exact state/logits used for this choice before
            # advancing the recurrent state with the sampled token.
            if control is not None:
                # A controlled request may be cancelled from the callback or
                # deadline before the sampled token is applied.  Preserve the
                # pre-token snapshot so callers can resume at that boundary.
                _remember_rwkv7_state(model, state, context_last, logits)
                control.emit(token)
            with Timer() as t_decode:
                logits, state = model.forward([token], state)
            if metrics:
                # Decode tokens use layer_id=-2 so benches can separate them
                # from prefill (-1) without subtracting decode time.
                row = metrics.start_layer(-2)
                row.compute_ms += t_decode.elapsed_ms
            context_last = int(token)
            throttle_after_work(t_token, power_percent)
            if metrics is not None:
                token_ms = (time.perf_counter() - t_token) * 1000.0
                metrics.token_latencies_ms.append(token_ms)
                if len(metrics.token_latencies_ms) == 1 and metrics.ttft_s <= 0.0:
                    metrics.ttft_s = token_ms / 1000.0
    _remember_rwkv7_state(model, state, context_last, logits)
    return out


def decode_greedy_from_state(
    model: Any,
    state: list[torch.Tensor],
    last_token_id: int,
    max_tokens: int,
    *,
    metrics: MetricsCollector | None = None,
    power_percent: int = 100,
    provider: ManifestWeightProvider | None = None,
    by_layer: dict[int, list[TensorEntry]] | None = None,
    layer_ids: list[int] | None = None,
    temperature: float = 1.0,
    greedy: bool = True,
    token_callback=None,
    cancel_event=None,
    deadline: float | None = None,
) -> list[int]:
    """Greedy decode from an existing RWKV-7 recurrent state."""
    state = [t.clone() for t in state]
    last = int(last_token_id)
    out: list[int] = []
    control = make_generation_control(
        token_callback=token_callback,
        cancel_event=cancel_event,
        deadline=deadline,
    )
    streaming = (
        provider is not None
        and by_layer is not None
        and layer_ids is not None
    )
    with torch.no_grad():
        for _ in range(max_tokens):
            if control is not None:
                control.check()
            t_token = time.perf_counter()
            input_token = int(last)
            with Timer() as t_decode:
                if streaming:
                    logits, state = forward_one(
                        model,
                        last,
                        state,
                        provider,
                        by_layer,
                        layer_ids=layer_ids,
                        metrics=metrics,
                    )
                else:
                    logits, state = model.forward([last], state)
            if metrics:
                # Use -2 to match greedy_token_ids_native decode rows so
                # benches can split prefill vs decode cleanly.
                row = metrics.start_layer(-2)
                row.compute_ms += t_decode.elapsed_ms
            last = sample_torch(logits, temperature=temperature, greedy=greedy)
            out.append(last)
            if control is not None:
                # Keep the resumable pre-token boundary only when generation
                # control can observe/cancel this iteration.  Uncontrolled
                # decode publishes the final state once below and avoids a
                # full recurrent-state clone per token.
                _remember_rwkv7_state(model, state, input_token, logits)
                control.emit(last)
            throttle_after_work(t_token, power_percent)
            if metrics is not None:
                token_ms = (time.perf_counter() - t_token) * 1000.0
                metrics.token_latencies_ms.append(token_ms)
                if len(metrics.token_latencies_ms) == 1 and metrics.ttft_s <= 0.0:
                    metrics.ttft_s = token_ms / 1000.0
    _remember_rwkv7_state(model, state, last, logits)
    return out


def prefill_text_streaming(
    model: Any,
    pipeline: Any,
    text: str,
    state: list[torch.Tensor],
    provider: ManifestWeightProvider,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
    metrics: MetricsCollector | None = None,
    *,
    cancel_event=None,
    deadline: float | None = None,
) -> tuple[list[torch.Tensor], int]:
    """Prefill ``text`` onto ``state`` using pack-driven streaming forward."""
    token_ids = pipeline.encode(text)
    if not token_ids:
        return state, 0
    state = _prefill_streaming_tokens(
        model,
        token_ids,
        state,
        provider,
        by_layer,
        layer_ids,
        metrics,
        cancel_event=cancel_event,
        deadline=deadline,
    )
    return state, int(token_ids[-1])


def _prefill_streaming_tokens(
    model: Any,
    token_ids: list[int],
    state: list[torch.Tensor],
    provider: ManifestWeightProvider,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
    metrics: MetricsCollector | None,
    *,
    cancel_event=None,
    deadline: float | None = None,
) -> list[torch.Tensor]:
    # A known prompt can amortize streamed weights across several recurrent
    # updates.  Keep a conservative chunk boundary so long prompts do not
    # create an unbounded activation tensor.  DeepEmbed falls back to the
    # token path until its DEA-aware layer scheduler exists.
    import os

    control = make_generation_control(
        cancel_event=cancel_event,
        deadline=deadline,
    )
    if control is not None:
        control.check()

    deepembed = any(
        key.startswith("blocks.") and ".qkv." in key for key in model.z
    )
    try:
        chunk_size = max(1, int(os.environ.get("RWKV_STREAM_PREFILL_CHUNK", "256")))
    except ValueError:
        chunk_size = 256
    last_logits: torch.Tensor | None = None
    if not deepembed:
        from rwkv_ssd.backends.rwkv7_batch import forward_sequence_one_dense

        # The layer-outer sequence kernel deliberately materializes every
        # matrix as BF16 and then uses Torch's dense matmul.  That is useful
        # when a long prompt amortizes one layer sweep, but it is a severe
        # regression for the common one-to-few-token prompt on a quantized
        # CPU pack: the fused GEMV path already keeps the packed blobs (and
        # the small skeleton) available and is orders of magnitude cheaper.
        # Keep the old sequence behavior for dense/non-fused providers and
        # for explicitly long prompts, while making short fused prefill use
        # the same path as decode.  The threshold is deliberately tunable so
        # a machine with a particularly fast BLAS can choose a different
        # crossover without changing pack or numerical semantics.
        use_fused = getattr(provider, "_use_fused_lut_matmul", None)
        fused_cpu = bool(callable(use_fused) and use_fused())
        try:
            fused_sequence_min_tokens = max(
                1,
                int(os.environ.get("RWKV_STREAM_FUSED_PREFILL_MIN_TOKENS", "8")),
            )
        except ValueError:
            fused_sequence_min_tokens = 8
        if fused_cpu and len(token_ids) < fused_sequence_min_tokens:
            for tid in token_ids:
                if control is not None:
                    control.check()
                last_logits, state = forward_one(
                    model,
                    tid,
                    state,
                    provider,
                    by_layer,
                    layer_ids=layer_ids,
                    metrics=metrics,
                )
        else:
            for start in range(0, len(token_ids), chunk_size):
                if control is not None:
                    control.check()
                chunk = token_ids[start : start + chunk_size]
                try:
                    last_logits, state = forward_sequence_one_dense(
                        model,
                        chunk,
                        state,
                        provider,
                        by_layer,
                        layer_ids,
                        metrics,
                    )
                except RuntimeError as exc:
                    # Keep compatibility with older ChatRWKV installations
                    # that do not export the sequence kernels.  The fallback
                    # is exact but gives up layer-outer prefill optimization.
                    if "RWKV_x070_TMix_seq" not in str(exc) and "only implements ordinary" not in str(exc):
                        raise
                    for tid in chunk:
                        if control is not None:
                            control.check()
                        last_logits, state = forward_one(
                            model,
                            tid,
                            state,
                            provider,
                            by_layer,
                            layer_ids=layer_ids,
                            metrics=metrics,
                        )
    else:
        for tid in token_ids:
            if control is not None:
                control.check()
            last_logits, state = forward_one(
                model,
                tid,
                state,
                provider,
                by_layer,
                layer_ids=layer_ids,
                metrics=metrics,
            )
    # Keep the distribution produced by the final prefill token available to
    # the generation loop.  Re-evaluating that token before sampling would
    # advance an RWKV recurrent state twice and breaks resident/streaming
    # parity.
    if last_logits is not None:
        # ``forward_sequence_one_dense`` returns one row per prompt token;
        # only the final row is the next-token distribution after prefill.
        observed = last_logits.detach().clone()
        if observed.ndim > 1:
            observed = observed.reshape(-1, observed.shape[-1])[-1]
        setattr(model, "_rwkv_ssd_last_logits", observed)
        setattr(model, "_rwkv_ssd_last_token_id", int(token_ids[-1]))
    return state


def _prefill_transcript_cache(
    model: Any,
    pipeline: Any,
    prompt: str,
    provider: ManifestWeightProvider,
    by_layer: dict[int, list[TensorEntry]],
    layer_ids: list[int],
    prefix_cache: Any,
    metrics: MetricsCollector | None,
    *,
    cancel_event=None,
    deadline: float | None = None,
) -> tuple[list[torch.Tensor], int, bool]:
    """Prefill using content-addressed transcript prefix cache."""
    from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState
    from rwkv_ssd.runtime.transcript_cache import longest_cached_prefix, split_transcript

    if not isinstance(prefix_cache, PrefixStateCache):
        return model.generate_zero_state(), 0, False

    cache_key, stable, suffix = longest_cached_prefix(
        prompt,
        has_entry=prefix_cache.contains,
    )
    if not (cache_key or stable or suffix):
        return model.generate_zero_state(), 0, False

    state = model.generate_zero_state()
    t_prefill = time.perf_counter()
    cached = prefix_cache.get(cache_key) if cache_key else None
    if cached is not None and cached.rwkv7_state is not None:
        if metrics is not None:
            metrics.state_cache_hit = True
        state = [t.clone() for t in cached.rwkv7_state]
        if suffix:
            state, _ = prefill_text_streaming(
                model,
                pipeline,
                suffix,
                state,
                provider,
                by_layer,
                layer_ids,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
    else:
        if metrics is not None:
            metrics.state_cache_hit = False
        if stable:
            state, _ = prefill_text_streaming(
                model,
                pipeline,
                stable,
                state,
                provider,
                by_layer,
                layer_ids,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            if cache_key:
                stable_ids = pipeline.encode(stable)
                last_id = int(stable_ids[-1]) if stable_ids else 0
                prefix_cache.put(
                    cache_key,
                    RecurrentState(
                        last_token_id=last_id,
                        rwkv7_state=[t.clone() for t in state],
                    ),
                )
        if suffix:
            state, _ = prefill_text_streaming(
                model,
                pipeline,
                suffix,
                state,
                provider,
                by_layer,
                layer_ids,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
    if metrics is not None:
        metrics.prefill_wall_s = time.perf_counter() - t_prefill
        # Drop per-layer prefill rows so decode-only benches don't fold
        # transcript-cache prefill into compute_ms / layer_steps.
        metrics.layers.clear()
    all_ids = pipeline.encode(prompt)
    last = int(all_ids[-1]) if all_ids else 0
    return state, last, True


def greedy_token_ids_streaming(
    model: Any,
    pipeline: Any,
    prompt: str,
    max_tokens: int,
    provider: ManifestWeightProvider,
    manifest_layers: list[int],
    by_layer: dict[int, list[TensorEntry]],
    metrics: MetricsCollector | None = None,
    *,
    system_prefix: str | None = None,
    prefix_cache: Any | None = None,
    prefix_cache_mode: str = "system",
    power_percent: int = 100,
    temperature: float = 1.0,
    greedy: bool = True,
    token_callback=None,
    cancel_event=None,
    deadline: float | None = None,
) -> list[int]:
    """Greedy decode with pack-driven layer injection each token."""
    import time

    from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState

    layer_ids = manifest_layers or list(range(model.n_layer))
    prefill_ids: list[int] = []
    prefix_prefill_done = False
    last = 0

    if prefix_cache_mode == "transcript" and prefix_cache is not None:
        state, last, prefix_prefill_done = _prefill_transcript_cache(
            model,
            pipeline,
            prompt,
            provider,
            by_layer,
            layer_ids,
            prefix_cache,
            metrics,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if not prefix_prefill_done:
            prefill_ids = pipeline.encode(prompt)
            state = model.generate_zero_state()
    elif system_prefix and prefix_cache is not None:
        if not prompt.startswith(system_prefix):
            prompt = system_prefix + prompt
        user_text = prompt[len(system_prefix) :]
        system_ids = pipeline.encode(system_prefix)
        user_ids = pipeline.encode(user_text) if user_text else []
        state = model.generate_zero_state()
        t_prefill = time.perf_counter()
        cached = (
            prefix_cache.get(system_prefix)
            if isinstance(prefix_cache, PrefixStateCache)
            else None
        )
        if cached is not None and cached.rwkv7_state is not None:
            if metrics is not None:
                metrics.state_cache_hit = True
            state = [t.clone() for t in cached.rwkv7_state]
            if user_ids:
                state = _prefill_streaming_tokens(
                    model,
                    user_ids,
                    state,
                    provider,
                    by_layer,
                    layer_ids,
                    metrics,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
        else:
            if metrics is not None:
                metrics.state_cache_hit = False
            if system_ids:
                state = _prefill_streaming_tokens(
                    model,
                    system_ids,
                    state,
                    provider,
                    by_layer,
                    layer_ids,
                    metrics,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
            last_id = system_ids[-1] if system_ids else 0
            prefix_cache.put(
                system_prefix,
                RecurrentState(
                    last_token_id=last_id,
                    rwkv7_state=[t.clone() for t in state],
                ),
            )
            if user_ids:
                state = _prefill_streaming_tokens(
                    model,
                    user_ids,
                    state,
                    provider,
                    by_layer,
                    layer_ids,
                    metrics,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
        if metrics is not None:
            metrics.prefill_wall_s = time.perf_counter() - t_prefill
            # Drop per-layer prefill rows so decode-only benches don't fold
            # system-prefix prefill into compute_ms / layer_steps.
            metrics.layers.clear()
        prefix_prefill_done = True
    else:
        prefill_ids = pipeline.encode(prompt)
        state = model.generate_zero_state()
    use_warm_z = getattr(provider, "_warm_z", False)

    def _all_layers_in_z() -> bool:
        return all_block_layers_in_z(model.z, layer_ids, by_layer)

    if use_warm_z and isinstance(provider, ManifestWeightProvider):
        # Skip when load-time warm already filled z (avoids a second full
        # materialize pass on F5). Cheap check: receptance present on all.
        if not all_block_layers_in_z(model.z, layer_ids, by_layer):
            warm_stream_cache_layers_into_z(
                model.z, provider, by_layer, layer_ids, metrics=metrics
            )

    # Reuse the prefill tokenization when available; the prefix path
    # leaves ``prefill_ids`` empty because the system+user tokens were
    # processed in the prefix block, so we encode the full prompt once
    # here. Avoids a second ``pipeline.encode`` call inside the decode
    # loop and outside the no_grad block (the encode is pure CPU).
    if not prefill_ids:
        all_ids = pipeline.encode(prompt)
    else:
        all_ids = prefill_ids
    if not prefix_prefill_done or not last:
        last = all_ids[-1] if all_ids else 0

    with torch.no_grad():
        use_native = (
            not prefix_prefill_done and _all_layers_in_z() and hasattr(model, "forward")
        )
        if use_native:
            return greedy_token_ids_native(
                model,
                pipeline,
                prompt,
                max_tokens,
                metrics=metrics,
                power_percent=power_percent,
                temperature=temperature,
                greedy=greedy,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )

        if prefill_ids and not prefix_prefill_done:
            t_prefill = time.perf_counter()
            state = _prefill_streaming_tokens(
                model,
                prefill_ids,
                state,
                provider,
                by_layer,
                layer_ids,
                metrics,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            if metrics is not None:
                metrics.prefill_wall_s = time.perf_counter() - t_prefill
                # Drop per-layer prefill rows so decode-only benches don't
                # fold prompt sweeps into compute_ms / layer_steps.
                metrics.layers.clear()
            prefix_prefill_done = True

        out: list[int] = []
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        # Normal prefill already returned the logits for the next token.  A
        # prefix-cache hit from an older cache format may not carry logits;
        # retain a conservative compatibility fallback for that case.
        logits = getattr(model, "_rwkv_ssd_last_logits", None)
        if logits is None:
            logits, state = forward_one(
                model,
                last,
                state,
                provider,
                by_layer,
                layer_ids=layer_ids,
                metrics=metrics,
            )
        for _ in range(max_tokens):
            if control is not None:
                control.check()
            t_token = time.perf_counter()
            input_token = int(last)
            last = sample_torch(logits, temperature=temperature, greedy=greedy)
            out.append(last)
            if control is not None:
                _remember_rwkv7_state(model, state, input_token, logits)
                control.emit(last)
            # Advance once with the sampled token so the next loop iteration
            # can sample the logits it produces without consuming the same
            # context token twice.
            logits, state = forward_one(
                model,
                last,
                state,
                provider,
                by_layer,
                layer_ids=layer_ids,
                metrics=metrics,
            )
            throttle_after_work(t_token, power_percent)
            if metrics is not None:
                token_ms = (time.perf_counter() - t_token) * 1000.0
                metrics.token_latencies_ms.append(token_ms)
                if len(metrics.token_latencies_ms) == 1 and metrics.ttft_s <= 0.0:
                    metrics.ttft_s = token_ms / 1000.0
    _remember_rwkv7_state(model, state, last, logits)
    return out
