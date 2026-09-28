"""Dense ChatRWKV layer-outer/session-inner decode."""
from __future__ import annotations
from typing import Any
import torch
from rwkv_ssd.backends.rwkv7_forward import (
    _evict_layer_from_z,
    _is_rwkv7a_deepembed_v1,
    _layer_norm,
    _rwkv7a_v1_s_emb_rows,
)
from rwkv_ssd.runtime.metrics import MetricsCollector, Timer
from rwkv_ssd.runtime.rwkv7_weights import inject_layer_into_z, layer_weights_in_z


def forward_batch_one_dense(model: Any, token_ids: list[int], states: list[list[torch.Tensor]], provider: Any, by_layer: dict, layer_ids: list[int], metrics: MetricsCollector):
    """Advance B independent states while each dense layer is loaded once."""
    if not token_ids or len(token_ids) != len(states):
        raise ValueError("token_ids and states must be equally sized and non-empty")
    from rwkv.model import RWKV_x070_CMix_one, RWKV_x070_TMix_one  # type: ignore
    z = model.z
    dtype = z["emb.weight"].dtype
    xs = [z["emb.weight"][token].to(dtype=dtype) for token in token_ids]
    v_first = [torch.empty_like(x) for x in xs]
    metrics.batch_size = len(xs)
    metrics.weight_sweeps += 1
    with torch.no_grad():
        for lid in layer_ids:
            entries = by_layer.get(lid, [])
            resident = layer_weights_in_z(z, lid)
            row = None
            if entries and not resident:
                # ManifestWeightProvider.begin_layer() normally creates this
                # timing row inside load_layer_tensors_materialized().  Keep
                # the batch path correct for lightweight/custom providers that
                # implement the materialized contract without doing so, and do
                # not use metrics.layers[-1] (which may belong to another
                # layer or may not exist yet).
                layer_count = len(metrics.layers)
                tensors = provider.load_layer_tensors_materialized(entries)
                row = (
                    metrics.layers[-1]
                    if len(metrics.layers) > layer_count
                    else metrics.start_layer(lid)
                )
                prepared = provider.prepare_layer_for_z(lid, tensors, row, force_materialize=True)
                inject_layer_into_z(z, prepared)
                for key in prepared:
                    if z[key].dtype != dtype:
                        z[key] = z[key].to(dtype=dtype)
                metrics.weight_layer_loads += 1
            elif entries:
                row = metrics.start_layer(lid)
                row.layer_cache_hits += len(entries)
            b = f"blocks.{lid}."; att = b + "att."; ffn = b + "ffn."
            with Timer() as timer:
                for i in range(len(xs)):
                    x, state = xs[i], states[i]
                    xx = _layer_norm(x, z, b+"ln1.weight", b+"ln1.bias", model.n_embd, dtype)
                    xx, state[lid*3], state[lid*3+1], v_first[i] = RWKV_x070_TMix_one(
                        lid, model.n_head, model.head_size, xx, state[lid*3].to(dtype=dtype), v_first[i], state[lid*3+1],
                        z[att+"x_r"], z[att+"x_w"], z[att+"x_k"], z[att+"x_v"], z[att+"x_a"], z[att+"x_g"],
                        z[att+"w0"], z[att+"w1"], z[att+"w2"], z[att+"a0"], z[att+"a1"], z[att+"a2"],
                        z[att+"v0"], z[att+"v1"], z[att+"v2"], z[att+"g1"], z[att+"g2"], z[att+"k_k"], z[att+"k_a"], z[att+"r_k"],
                        z[att+"receptance.weight"], z[att+"key.weight"], z[att+"value.weight"], z[att+"output.weight"], z[att+"ln_x.weight"], z[att+"ln_x.bias"])
                    x = (x + xx).to(dtype=dtype)
                    xx = _layer_norm(x, z, b+"ln2.weight", b+"ln2.bias", model.n_embd, dtype)
                    if _is_rwkv7a_deepembed_v1(model):
                        xx, state[lid * 3 + 2] = RWKV_x070_CMix_one(
                            xx,
                            state[lid * 3 + 2].to(dtype=dtype),
                            z[ffn + "x_k"],
                            z[ffn + "key.weight"],
                            z[ffn + "value.weight"],
                            _rwkv7a_v1_s_emb_rows(model, z, ffn, token_ids[i]),
                            z[ffn + "s1"],
                            z[ffn + "s2"],
                            z[ffn + "s0"],
                        )
                    else:
                        xx, state[lid*3+2] = RWKV_x070_CMix_one(xx, state[lid*3+2].to(dtype=dtype), z[ffn+"x_k"], z[ffn+"key.weight"], z[ffn+"value.weight"])
                    xs[i] = (x + xx).to(dtype=dtype)
            row = row or metrics.start_layer(lid)
            row.compute_ms += timer.elapsed_ms
            if entries and not resident:
                _evict_layer_from_z(z, lid, provider)
                provider.evict_streamed_layer(lid, force=True)
        logits = []
        for x in xs:
            x = _layer_norm(x, z, "ln_out.weight", "ln_out.bias", model.n_embd, dtype)
            logits.append(x @ z["head.weight"])
    return logits, states


def forward_sequence_one_dense(
    model: Any,
    token_ids: list[int],
    state: list[torch.Tensor],
    provider: Any,
    by_layer: dict,
    layer_ids: list[int],
    metrics: MetricsCollector | None,
):
    """Process a known prompt chunk with one streamed weight sweep.

    RWKV's sequence kernels still update the recurrent state token by token,
    but the surrounding Python/I/O schedule is layer-outer: a layer is
    materialized once, then consumes the complete chunk.  This is the exact
    prefill counterpart of ``forward_batch_one_dense`` and is the main
    mechanism for amortizing streamed weights over prompt tokens.

    The qkv/DEA DeepEmbed variant is intentionally rejected here.  RWKV7a
    DeepEmbed-v1 is handled by the same layer-outer schedule with derived
    per-token CMix rows.
    """
    if not token_ids:
        return model.z["head.weight"].new_empty((0, model.z["head.weight"].shape[-1])), state
    if any(key.startswith("blocks.") and ".qkv." in key for key in model.z):
        raise RuntimeError(
            "forward_sequence_one_dense only implements ordinary RWKV-7; "
            "DeepEmbed requires its DEA/lookup-aware prefill path"
        )
    from rwkv.model import RWKV_x070_CMix_seq, RWKV_x070_TMix_seq  # type: ignore[import-untyped]

    z = model.z
    dtype = z["emb.weight"].dtype
    ids = torch.tensor(token_ids, dtype=torch.long, device=z["emb.weight"].device)
    x = z["emb.weight"].index_select(0, ids).to(dtype=dtype)
    v_first = torch.empty_like(x)
    with torch.no_grad():
        for lid in layer_ids:
            entries = by_layer.get(lid, [])
            resident = layer_weights_in_z(z, lid)
            row = metrics.start_layer(lid) if metrics is not None else None
            if entries and not resident:
                tensors = provider.load_layer_tensors_materialized(entries)
                prepared = provider.prepare_layer_for_z(
                    lid, tensors, row, force_materialize=True
                )
                inject_layer_into_z(z, prepared)
                for key in prepared:
                    if z[key].dtype != dtype:
                        z[key] = z[key].to(dtype=dtype)
            elif entries and row is not None:
                row.layer_cache_hits += len(entries)
            b = f"blocks.{lid}."
            att = b + "att."
            ffn = b + "ffn."
            with Timer() as timer:
                xx = _layer_norm(x, z, b + "ln1.weight", b + "ln1.bias", model.n_embd, dtype)
                xx, next_prev, next_att, v_first = RWKV_x070_TMix_seq(
                    lid,
                    model.n_head,
                    model.head_size,
                    xx,
                    state[lid * 3].to(dtype=dtype),
                    v_first,
                    state[lid * 3 + 1],
                    z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"],
                    z[att + "x_a"], z[att + "x_g"], z[att + "w0"], z[att + "w1"], z[att + "w2"],
                    z[att + "a0"], z[att + "a1"], z[att + "a2"], z[att + "v0"], z[att + "v1"], z[att + "v2"],
                    z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
                    z[att + "receptance.weight"], z[att + "key.weight"], z[att + "value.weight"],
                    z[att + "output.weight"], z[att + "ln_x.weight"], z[att + "ln_x.bias"],
                )
                state[lid * 3] = next_prev
                state[lid * 3 + 1] = next_att
                x = (x + xx).to(dtype=dtype)
                xx = _layer_norm(x, z, b + "ln2.weight", b + "ln2.bias", model.n_embd, dtype)
                if _is_rwkv7a_deepembed_v1(model):
                    xx, state[lid * 3 + 2] = RWKV_x070_CMix_seq(
                        xx,
                        state[lid * 3 + 2].to(dtype=dtype),
                        z[ffn + "x_k"],
                        z[ffn + "key.weight"],
                        z[ffn + "value.weight"],
                        _rwkv7a_v1_s_emb_rows(model, z, ffn, ids),
                        z[ffn + "s1"],
                        z[ffn + "s2"],
                        z[ffn + "s0"],
                    )
                else:
                    xx, state[lid * 3 + 2] = RWKV_x070_CMix_seq(
                        xx,
                        state[lid * 3 + 2].to(dtype=dtype),
                        z[ffn + "x_k"],
                        z[ffn + "key.weight"],
                        z[ffn + "value.weight"],
                    )
                x = (x + xx).to(dtype=dtype)
            if row is not None:
                row.compute_ms += timer.elapsed_ms
            if entries and not resident:
                _evict_layer_from_z(z, lid, provider)
                # Sequence prefill asks the provider for fully materialized
                # tensors, but strict fused/stream-cache modes still need the
                # retained packed blobs (and prepared skeleton) for the next
                # decode token.  ``force=True`` used to erase those caches at
                # the prefill boundary, making the first generated token pay
                # a complete layer read/stage pass again.
                strict_fused = bool(
                    getattr(provider, "_strict_fused_retain_layers", lambda: False)()
                )
                keep_provider = strict_fused or bool(
                    getattr(provider, "_stream_layer_cache", False)
                )
                if keep_provider:
                    retain = getattr(provider, "_retain_provider_layer", None)
                    if callable(retain):
                        retain(lid)
                    if strict_fused:
                        release_prepared = getattr(
                            provider, "release_prepared_tensors", None
                        )
                        if callable(release_prepared):
                            release_prepared(lid)
                else:
                    provider.evict_streamed_layer(lid, force=True)
        x = _layer_norm(x, z, "ln_out.weight", "ln_out.bias", model.n_embd, dtype)
        logits = x @ z["head.weight"]
    return logits, state

__all__ = ["forward_batch_one_dense", "forward_sequence_one_dense"]
