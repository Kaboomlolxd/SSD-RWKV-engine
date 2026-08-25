"""Stream cache cap resolution for tiny models."""

from __future__ import annotations

from rwkv_ssd.runtime.stream_cache_policy import (
    resolve_max_layers_in_z,
    resolve_max_provider_cache_layers,
)


def test_tiny_model_retains_all_block_layers() -> None:
    # Auto (cap=0) keeps all blocks on tiny packs.
    assert (
        resolve_max_layers_in_z(0, 2, stream_layer_cache=True, warm_z=False) == 2
    )


def test_four_layer_model_retains_all() -> None:
    assert resolve_max_layers_in_z(0, 4, stream_layer_cache=True, warm_z=False) == 4


def test_large_model_default_two_layer_window() -> None:
    # Auto (cap=0) → 2-layer window on n≥8.
    assert resolve_max_layers_in_z(0, 12, stream_layer_cache=True, warm_z=False) == 2


def test_large_model_respects_explicit_cap() -> None:
    assert resolve_max_layers_in_z(4, 12, stream_layer_cache=True, warm_z=False) == 4


def test_explicit_one_layer_cap_honored() -> None:
    """Explicit max_layers_in_z=1 must not be bumped to 2 on large models."""
    assert resolve_max_layers_in_z(1, 12, stream_layer_cache=True, warm_z=False) == 1


def test_warm_z_does_not_auto_bump() -> None:
    assert resolve_max_layers_in_z(1, 2, stream_layer_cache=True, warm_z=True) == 1


def test_provider_cache_auto_full_when_decoupled() -> None:
    assert (
        resolve_max_provider_cache_layers(
            0,
            12,
            max_layers_in_z=2,
            stream_layer_cache=True,
            decouple_provider_cache=True,
            warm_z=False,
        )
        == 12
    )


def test_provider_cache_budget_limited() -> None:
    assert (
        resolve_max_provider_cache_layers(
            0,
            100,
            max_layers_in_z=2,
            stream_layer_cache=True,
            decouple_provider_cache=True,
            warm_z=False,
            budget_limited=True,
        )
        == 4
    )
