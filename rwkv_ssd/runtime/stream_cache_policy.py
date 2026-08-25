"""Resolve bounded ``z`` / provider cache caps from pack shape."""

from __future__ import annotations


def resolve_max_layers_in_z(
    max_layers_in_z: int,
    n_block_layers: int,
    *,
    stream_layer_cache: bool,
    warm_z: bool,
) -> int:
    """
    Keep at most ``max_layers_in_z`` streamed block layers unless the model is tiny.

    For ≤2 block layers, retain all blocks so native ``forward`` can run without
    ``--warm-z`` while RAM stays near skeleton + full blocks (~few MB on 0.01B).
    """
    cap = max(0, int(max_layers_in_z))
    if warm_z or not stream_layer_cache or n_block_layers <= 0:
        return cap
    # Explicit positive cap always wins (allows a true 1-layer window).
    if cap > 0:
        return cap
    # Auto (cap==0): tiny packs keep all blocks; larger packs default to 2.
    if n_block_layers <= 4:
        return n_block_layers
    if n_block_layers >= 8:
        return min(2, n_block_layers)
    return cap


def resolve_max_provider_cache_layers(
    configured: int,
    n_block_layers: int,
    *,
    max_layers_in_z: int,
    stream_layer_cache: bool,
    decouple_provider_cache: bool,
    warm_z: bool,
    budget_limited: bool = False,
) -> int:
    """
    Cap decoded tensors retained in the provider LRU (independent of ``z`` when decoupled).

    Default (throughput): all block layers when decoupled on large models.
    ``budget_limited`` / explicit ``configured``: small sliding window; reuse
    ``.decode_cache/`` on SSD for cross-token decode.
    """
    if warm_z or not stream_layer_cache:
        return max_layers_in_z
    if not decouple_provider_cache:
        return max_layers_in_z
    cap = max(0, int(configured))
    if cap > 0:
        return cap
    if budget_limited:
        return max(max_layers_in_z, min(4, n_block_layers or 4))
    if n_block_layers <= 0:
        return max_layers_in_z
    if n_block_layers <= 4:
        return max(max_layers_in_z, n_block_layers)
    if n_block_layers >= 8:
        return n_block_layers
    return max(max_layers_in_z, min(4, n_block_layers))
