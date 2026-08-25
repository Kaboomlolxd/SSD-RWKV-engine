"""
Pack-driven RWKV-7 forward_one — inject streamed layer weights then run TMix/CMix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.metrics import Timer
from rwkv_ssd.runtime.provider_factory import prefetch_ahead, release_layer
from rwkv_ssd.runtime.rwkv7_weights import (
    all_block_layers_in_z,
    inject_layer_into_z,
    layer_weights_in_z,
    warm_stream_cache_layers_into_z,
)
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider

if TYPE_CHECKING:
    from rwkv_ssd.runtime.metrics import MetricsCollector


def _model_dtype(z: dict[str, torch.Tensor]) -> torch.dtype:
    return z["emb.weight"].dtype


def _layer_norm(
    x: torch.Tensor,
    z: dict[str, torch.Tensor],
    weight_key: str,
    bias_key: str,
    n_embd: int,
) -> torch.Tensor:
    w = z[weight_key]
    b = z[bias_key]
    x = x.to(dtype=w.dtype)
    return F.layer_norm(x, (n_embd,), weight=w, bias=b)


def _evict_layer_from_z(z: dict[str, torch.Tensor], layer_id: int) -> None:
    prefix = f"blocks.{layer_id}."
    for key in list(z.keys()):
        if key.startswith(prefix):
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

    with torch.no_grad():
        x = z["emb.weight"][token_id].to(dtype=act_dtype)
        v_first = torch.empty_like(x)

        for idx, layer_id in enumerate(layers):
            entries = by_layer.get(layer_id, [])
            if prefetch and isinstance(provider, ManifestWeightProvider):
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

            prepared: dict[str, torch.Tensor] = {}
            if entries:
                if not layer_weights_in_z(z, layer_id):
                    layer_tensors = provider.load_layer_tensors(entries)
                    row = metrics.layers[-1] if metrics and metrics.layers else None
                    prepared = provider.prepare_layer_for_z(layer_id, layer_tensors, row)
                    inject_layer_into_z(z, prepared)
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

            with Timer() as t_compute:
                xx = _layer_norm(x, z, bbb + "ln1.weight", bbb + "ln1.bias", n_embd)

                xx, state[layer_id * 3 + 0], state[layer_id * 3 + 1], v_first = (
                    RWKV_x070_TMix_one(
                        layer_id,
                        n_head,
                        head_size,
                        xx,
                        state[layer_id * 3 + 0],
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
                )
                x = x + xx

                xx = _layer_norm(x, z, bbb + "ln2.weight", bbb + "ln2.bias", n_embd)
                xx = xx.to(dtype=z[ffn + "x_k"].dtype)
                xx, state[layer_id * 3 + 2] = RWKV_x070_CMix_one(
                    xx,
                    state[layer_id * 3 + 2],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            if metrics and metrics.layers and metrics.layers[-1].layer_id == layer_id:
                metrics.layers[-1].compute_ms += t_compute.elapsed_ms

            if evict_layers and prepared:
                streamed = (
                    provider.layer_has_streamed_tensors(entries)
                    if isinstance(provider, ManifestWeightProvider)
                    else True
                )
                keep = (
                    isinstance(provider, ManifestWeightProvider)
                    and provider._stream_layer_cache
                    and streamed
                )
                if not keep:
                    _evict_layer_from_z(z, layer_id)
                    if isinstance(provider, ManifestWeightProvider):
                        release_layer(provider, entries)

        x = _layer_norm(x, z, "ln_out.weight", "ln_out.bias", n_embd)
        x = x @ z["head.weight"]
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
    native_when_warm: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    layers = layer_ids if layer_ids is not None else list(range(model.n_layer))
    z = model.z
    if (
        native_when_warm
        and isinstance(provider, ManifestWeightProvider)
        and provider._stream_layer_cache
        and all_block_layers_in_z(z, layers, by_layer)
        and hasattr(model, "forward_one")
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


def greedy_token_ids_streaming(
    model: Any,
    pipeline: Any,
    prompt: str,
    max_tokens: int,
    provider: ManifestWeightProvider,
    manifest_layers: list[int],
    by_layer: dict[int, list[TensorEntry]],
    metrics: MetricsCollector | None = None,
) -> list[int]:
    """Greedy decode with pack-driven layer injection each token."""
    token_ids = pipeline.encode(prompt)
    state = model.generate_zero_state()
    layer_ids = manifest_layers or list(range(model.n_layer))
    native_when_warm = getattr(provider, "_stream_layer_cache", False)

    if native_when_warm and isinstance(provider, ManifestWeightProvider):
        warm_stream_cache_layers_into_z(
            model.z, provider, by_layer, layer_ids, metrics=metrics
        )

    with torch.no_grad():
        use_native = (
            native_when_warm
            and isinstance(provider, ManifestWeightProvider)
            and provider._stream_layer_cache
            and all_block_layers_in_z(model.z, layer_ids, by_layer)
            and hasattr(model, "forward")
        )
        if use_native and token_ids:
            with Timer() as t_prefill:
                if len(token_ids) > 1:
                    _, state = model.forward(token_ids, state)
                else:
                    _, state = model.forward([token_ids[0]], state)
            if metrics:
                row = metrics.start_layer(-1)
                row.compute_ms += t_prefill.elapsed_ms
        else:
            for tid in token_ids:
                _, state = forward_one(
                    model,
                    tid,
                    state,
                    provider,
                    by_layer,
                    layer_ids=layer_ids,
                    metrics=metrics,
                    native_when_warm=native_when_warm,
                )

        last = token_ids[-1] if token_ids else 0
        out: list[int] = []
        for _ in range(max_tokens):
            logits, state = forward_one(
                model,
                last,
                state,
                provider,
                by_layer,
                layer_ids=layer_ids,
                metrics=metrics,
                native_when_warm=native_when_warm,
            )
            last = int(logits.argmax().item())
            out.append(last)
    return out
