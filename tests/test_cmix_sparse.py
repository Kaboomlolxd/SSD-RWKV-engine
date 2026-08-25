from __future__ import annotations

import torch

from rwkv_ssd.runtime.cmix_sparse import (
    TiledValueMatrix,
    pack_value_matrix,
    plan_coactivation_tile_order,
)
from rwkv_ssd.runtime.rwkv7_linear import cmix_one_fused


def test_tiled_value_matrix_matches_dense_and_skips_zero_tiles(tmp_path) -> None:
    matrix = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 10
    index = pack_value_matrix(matrix, tmp_path, tile_rows=4)
    tiled = TiledValueMatrix(index)
    try:
        activation = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        actual, stats = tiled.matmul(activation)
        torch.testing.assert_close(actual, activation @ matrix)
        assert stats.tiles_read == 2
        assert stats.tiles_skipped == 1
        assert stats.bytes_read == 2 * 4 * 4 * matrix.element_size()
        assert stats.skipped_read_fraction == 1 / 3
    finally:
        tiled.close()


def test_tiled_value_matrix_all_zero_reads_no_tiles(tmp_path) -> None:
    matrix = torch.randn(8, 3)
    tiled = TiledValueMatrix(pack_value_matrix(matrix, tmp_path, tile_rows=2))
    try:
        activation = torch.zeros(8)
        actual, stats = tiled.matmul(activation)
        torch.testing.assert_close(actual, activation @ matrix)
        assert stats.tiles_read == 0
        assert stats.tiles_skipped == 4
        assert stats.bytes_read == 0
    finally:
        tiled.close()


def test_temporal_prefetch_hits_preserve_exact_result(tmp_path) -> None:
    matrix = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 10
    tiled = TiledValueMatrix(pack_value_matrix(matrix, tmp_path, tile_rows=4))
    activation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    try:
        tiled.matmul(activation)
        ticket = tiled.begin_temporal_prefetch()
        actual, stats = tiled.matmul(activation, prefetch_ticket=ticket)
        torch.testing.assert_close(actual, activation @ matrix)
        assert stats.prefetched_tiles == 2
        assert stats.prefetch_hits == 2
        assert stats.prefetch_misses == 0
        assert stats.prefetch_wasted_tiles == 0
    finally:
        tiled.close()


def test_temporal_prefetch_miss_falls_back_to_exact_read(tmp_path) -> None:
    matrix = torch.randn(12, 5)
    tiled = TiledValueMatrix(pack_value_matrix(matrix, tmp_path, tile_rows=4))
    previous = torch.tensor([1.0] + [0.0] * 11)
    current = torch.tensor([0.0] * 8 + [2.0, 0.0, 0.0, 0.0])
    try:
        tiled.matmul(previous)
        ticket = tiled.begin_temporal_prefetch()
        actual, stats = tiled.matmul(current, prefetch_ticket=ticket)
        torch.testing.assert_close(actual, current @ matrix)
        assert stats.prefetch_hits == 0
        assert stats.prefetch_misses == 1
        assert stats.prefetch_wasted_tiles == 1
        assert stats.tiles_read == 1
    finally:
        tiled.close()


def test_hot_tile_cache_eliminates_repeat_read(tmp_path) -> None:
    matrix = torch.randn(8, 3)
    tiled = TiledValueMatrix(pack_value_matrix(matrix, tmp_path, tile_rows=2))
    tiled.set_hot_cache_limit(2 * 3 * matrix.element_size())
    activation = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    try:
        first, first_stats = tiled.matmul(activation)
        second, second_stats = tiled.matmul(activation)
        torch.testing.assert_close(first, activation @ matrix)
        torch.testing.assert_close(second, activation @ matrix)
        assert first_stats.hot_cache_hits == 0
        assert second_stats.hot_cache_hits == 1
        assert second_stats.bytes_read == 0
        assert second_stats.physical_reads == 0
    finally:
        tiled.close()


def test_coactivation_order_makes_common_tiles_physically_adjacent(tmp_path) -> None:
    matrix = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    samples = [
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    order = plan_coactivation_tile_order(samples, in_features=12, tile_rows=4)
    assert order[:2] == [0, 2]
    tiled = TiledValueMatrix(
        pack_value_matrix(matrix, tmp_path, tile_rows=4, tile_order=order)
    )
    try:
        actual, stats = tiled.matmul(samples[0])
        torch.testing.assert_close(actual, samples[0] @ matrix)
        assert stats.tiles_read == 2
        assert stats.physical_reads == 1
    finally:
        tiled.close()


def test_cmix_fused_uses_registered_tiled_value_matrix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RWKV_CMIX_SELECTIVE_READS", "1")
    value = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    tiled = TiledValueMatrix(pack_value_matrix(value, tmp_path, tile_rows=2))

    class Provider:
        def __init__(self) -> None:
            self.matrix = tiled
            self.stats = None

        def _use_fused_lut_matmul(self) -> bool:
            return True

        def get_fused_lut_blob(self, name: str):
            return None

        def get_cmix_tiled_value_matrix(self, name: str):
            return self.matrix if name == "value" else None

        def record_cmix_selective_stats(self, stats) -> None:
            self.stats = stats

    provider = Provider()
    x = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    x_prev = torch.zeros_like(x)
    x_k = torch.zeros_like(x)
    z = {"key": torch.eye(8), "value": value}
    try:
        actual, _ = cmix_one_fused(
            x, x_prev, x_k, "key", "value", z, provider, torch.float32
        )
        dense_activation = torch.relu(x @ z["key"]) ** 2
        torch.testing.assert_close(actual, dense_activation @ value)
        assert provider.stats is not None
        assert provider.stats.tiles_read == 2
    finally:
        tiled.close()
