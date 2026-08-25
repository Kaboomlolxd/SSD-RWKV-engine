"""CPU-safe selective reads for repacked CMix value matrices.

The ordinary CMix value matrix is laid out for dense matmul.  Activation
sparsity by itself therefore cannot save I/O.  This module defines a small,
explicit sidecar format that tiles rows of a value matrix so an opt-in reader
can skip tiles containing no active input rows.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from rwkv_ssd.runtime.io_pread import PreadWeightStore


_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class SelectiveReadStats:
    total_activation_elements: int
    active_activation_elements: int
    tiles_read: int
    tiles_skipped: int
    bytes_read: int
    prefetched_tiles: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_wasted_tiles: int = 0
    prefetch_bytes_submitted: int = 0
    hot_cache_hits: int = 0
    hot_cache_bytes: int = 0
    physical_reads: int = 0

    @property
    def skipped_read_fraction(self) -> float:
        total = self.tiles_read + self.tiles_skipped
        return self.tiles_skipped / total if total else 0.0


@dataclass
class TilePrefetchTicket:
    """In-flight speculative tile reads; predictions never affect correctness."""

    futures: dict[int, Future[bytes]]
    predicted_tiles: frozenset[int]
    bytes_submitted: int

    def cancel_unused(self, active_tiles: set[int]) -> None:
        for tile_index, future in self.futures.items():
            if tile_index not in active_tiles:
                future.cancel()


def _dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPES.items():
        if dtype == candidate:
            return name
    raise TypeError(f"unsupported tiled CMix dtype: {dtype}")


def pack_value_matrix(
    matrix: torch.Tensor,
    output_dir: str | Path,
    *,
    name: str = "value",
    tile_rows: int = 32,
    tile_order: list[int] | None = None,
) -> Path:
    """Write a row-tiled ``[in_features, out_features]`` value matrix.

    Returns the JSON index path.  The matrix is copied into a flat byte
    representation without changing dtype, and every tile starts at a byte
    offset recorded in the index.
    """
    if matrix.ndim != 2:
        raise ValueError("CMix value matrix must be rank 2")
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    dtype_name = _dtype_name(matrix.dtype)
    matrix = matrix.detach().contiguous().cpu()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{name}.bin"
    index_path = output / f"{name}.json"
    tiles: list[dict[str, int]] = []
    offset = 0
    row_bytes = matrix.shape[1] * matrix.element_size()
    n_tiles = (int(matrix.shape[0]) + tile_rows - 1) // tile_rows
    physical_order = list(range(n_tiles)) if tile_order is None else list(tile_order)
    if sorted(physical_order) != list(range(n_tiles)):
        raise ValueError("tile_order must be a permutation of all logical tile indices")
    with data_path.open("wb") as fh:
        for tile_index in physical_order:
            row_start = tile_index * tile_rows
            rows = min(tile_rows, matrix.shape[0] - row_start)
            payload = matrix[row_start : row_start + rows].view(torch.uint8).numpy().tobytes()
            fh.write(payload)
            tiles.append(
                {
                    "tile_index": tile_index,
                    "row_start": row_start,
                    "rows": rows,
                    "offset": offset,
                    "length": len(payload),
                }
            )
            offset += len(payload)
    tiles.sort(key=lambda tile: int(tile["tile_index"]))
    index = {
        "version": 1,
        "data_file": data_path.name,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": dtype_name,
        "tile_rows": int(tile_rows),
        "row_bytes": int(row_bytes),
        "tiles": tiles,
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def plan_coactivation_tile_order(
    activation_samples: Iterable[torch.Tensor],
    *,
    in_features: int,
    tile_rows: int,
) -> list[int]:
    """Greedily place frequently co-active tiles next to each other."""
    if in_features <= 0 or tile_rows <= 0:
        raise ValueError("in_features and tile_rows must be positive")
    n_tiles = (int(in_features) + tile_rows - 1) // tile_rows
    frequency = [0] * n_tiles
    pairs: dict[tuple[int, int], int] = {}
    for activation in activation_samples:
        flat = activation.detach().reshape(-1, in_features)
        active_rows = torch.any(flat != 0, dim=0)
        active = sorted(
            {
                int(row) // tile_rows
                for row in torch.nonzero(active_rows, as_tuple=False).reshape(-1).tolist()
            }
        )
        for tile in active:
            frequency[tile] += 1
        for left_pos, left in enumerate(active):
            for right in active[left_pos + 1 :]:
                pairs[(left, right)] = pairs.get((left, right), 0) + 1
    remaining = set(range(n_tiles))
    start = max(remaining, key=lambda tile: (frequency[tile], -tile))
    order = [start]
    remaining.remove(start)
    while remaining:
        previous = order[-1]
        next_tile = max(
            remaining,
            key=lambda tile: (
                pairs.get(tuple(sorted((previous, tile))), 0),
                frequency[tile],
                -tile,
            ),
        )
        order.append(next_tile)
        remaining.remove(next_tile)
    return order


class TiledValueMatrix:
    """Read a row-tiled value matrix through CPU pread."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        raw: Any = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
            raise ValueError("unsupported CMix tiled matrix index")
        shape = raw.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("CMix tiled matrix index has invalid shape")
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = _DTYPES.get(str(raw.get("dtype")))
        if self.dtype is None:
            raise ValueError("CMix tiled matrix index has unsupported dtype")
        self.tile_rows = int(raw.get("tile_rows", 0))
        self.row_bytes = int(raw.get("row_bytes", 0))
        if self.tile_rows <= 0 or self.row_bytes <= 0:
            raise ValueError("CMix tiled matrix index has invalid tile geometry")
        self.tiles = tuple(dict(tile) for tile in raw.get("tiles", []))
        if not self.tiles:
            raise ValueError("CMix tiled matrix index has no tiles")
        expected_row = 0
        for tile in self.tiles:
            row_start = int(tile["row_start"])
            rows = int(tile["rows"])
            length = int(tile["length"])
            if row_start != expected_row or rows <= 0 or length != rows * self.row_bytes:
                raise ValueError("CMix tiled matrix index has non-contiguous tiles")
            expected_row += rows
        if expected_row != self.shape[0]:
            raise ValueError("CMix tiled matrix index does not cover all matrix rows")
        data_path = self.index_path.parent / str(raw.get("data_file", ""))
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        self._store = PreadWeightStore(data_path)
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cmix-tile-prefetch"
        )
        self._last_active_tiles: frozenset[int] = frozenset()
        self._hot_cache_limit_bytes = 0
        self._hot_cache: dict[int, bytes] = {}
        self._tile_frequency = [0] * len(self.tiles)
        self.last_stats = SelectiveReadStats(0, 0, 0, 0, 0)

    def _tile_for_row(self, row: int) -> int:
        return min(len(self.tiles) - 1, row // self.tile_rows)

    def begin_prefetch(self, tile_indices: set[int] | frozenset[int]) -> TilePrefetchTicket:
        valid = frozenset(
            int(index)
            for index in tile_indices
            if 0 <= int(index) < len(self.tiles) and int(index) not in self._hot_cache
        )
        futures = {
            index: self._prefetch_executor.submit(
                self._store.read_bytes_span,
                int(self.tiles[index]["offset"]),
                int(self.tiles[index]["length"]),
            )
            for index in valid
        }
        return TilePrefetchTicket(
            futures=futures,
            predicted_tiles=valid,
            bytes_submitted=sum(int(self.tiles[index]["length"]) for index in valid),
        )

    def begin_temporal_prefetch(self) -> TilePrefetchTicket:
        """Predict the next active set from the previous exact activation."""
        return self.begin_prefetch(self._last_active_tiles)

    def set_hot_cache_limit(self, max_bytes: int) -> None:
        self._hot_cache_limit_bytes = max(0, int(max_bytes))
        self._rebalance_hot_cache()

    @property
    def hot_cache_bytes(self) -> int:
        return sum(len(payload) for payload in self._hot_cache.values())

    def _rebalance_hot_cache(self) -> None:
        if self._hot_cache_limit_bytes <= 0:
            self._hot_cache.clear()
            return
        while self.hot_cache_bytes > self._hot_cache_limit_bytes and self._hot_cache:
            victim = min(
                self._hot_cache,
                key=lambda index: (self._tile_frequency[index], -len(self._hot_cache[index])),
            )
            del self._hot_cache[victim]

    def _admit_hot_payload(self, tile_index: int, payload: bytes) -> None:
        if self._hot_cache_limit_bytes <= 0 or len(payload) > self._hot_cache_limit_bytes:
            return
        if tile_index in self._hot_cache:
            return
        while self._hot_cache and self.hot_cache_bytes + len(payload) > self._hot_cache_limit_bytes:
            victim = min(self._hot_cache, key=lambda index: self._tile_frequency[index])
            if self._tile_frequency[victim] > self._tile_frequency[tile_index]:
                return
            del self._hot_cache[victim]
        if self.hot_cache_bytes + len(payload) <= self._hot_cache_limit_bytes:
            self._hot_cache[tile_index] = payload

    def _read_tiles_coalesced(self, tile_indices: set[int]) -> tuple[dict[int, bytes], int]:
        if not tile_indices:
            return {}, 0
        ordered = sorted(tile_indices, key=lambda index: int(self.tiles[index]["offset"]))
        groups: list[list[int]] = []
        for index in ordered:
            if not groups:
                groups.append([index])
                continue
            previous = groups[-1][-1]
            previous_end = int(self.tiles[previous]["offset"]) + int(
                self.tiles[previous]["length"]
            )
            if int(self.tiles[index]["offset"]) == previous_end:
                groups[-1].append(index)
            else:
                groups.append([index])
        payloads: dict[int, bytes] = {}
        for group in groups:
            start = int(self.tiles[group[0]]["offset"])
            end = int(self.tiles[group[-1]]["offset"]) + int(
                self.tiles[group[-1]]["length"]
            )
            slab = self._store.read_bytes_span(start, end - start)
            for index in group:
                rel = int(self.tiles[index]["offset"]) - start
                length = int(self.tiles[index]["length"])
                payloads[index] = slab[rel : rel + length]
        return payloads, len(groups)

    def matmul(
        self,
        activation: torch.Tensor,
        *,
        prefetch_ticket: TilePrefetchTicket | None = None,
    ) -> tuple[torch.Tensor, SelectiveReadStats]:
        """Compute ``activation @ value`` while reading active row tiles only."""
        if activation.ndim not in (1, 2) or activation.shape[-1] != self.shape[0]:
            raise ValueError(
                f"activation shape must end in {self.shape[0]}, got {tuple(activation.shape)}"
            )
        flat = activation.reshape(-1, self.shape[0])
        active_rows = torch.any(flat != 0, dim=0).cpu()
        active_elements = int(torch.count_nonzero(flat).item())
        active_tiles = {
            self._tile_for_row(row)
            for row in torch.nonzero(active_rows, as_tuple=False).reshape(-1).tolist()
        }
        result = torch.zeros(
            (flat.shape[0], self.shape[1]),
            dtype=torch.promote_types(flat.dtype, self.dtype),
            device=activation.device,
        )
        bytes_read = 0
        hot_hits = len(active_tiles & set(self._hot_cache))
        sync_needed: set[int] = set()
        payload_by_tile: dict[int, bytes] = {
            index: self._hot_cache[index] for index in active_tiles if index in self._hot_cache
        }
        prefetched_reads = 0
        for tile_index in active_tiles:
            if tile_index in payload_by_tile:
                continue
            future = prefetch_ticket.futures.get(tile_index) if prefetch_ticket else None
            if future is not None:
                payload_by_tile[tile_index] = future.result()
                prefetched_reads += 1
            else:
                sync_needed.add(tile_index)
        sync_payloads, sync_physical_reads = self._read_tiles_coalesced(sync_needed)
        payload_by_tile.update(sync_payloads)
        for tile_index, tile in enumerate(self.tiles):
            if tile_index not in active_tiles:
                continue
            payload = payload_by_tile[tile_index]
            if tile_index not in self._hot_cache:
                bytes_read += len(payload)
            self._tile_frequency[tile_index] += 1
            self._admit_hot_payload(tile_index, payload)
            tile_tensor = torch.frombuffer(bytearray(payload), dtype=self.dtype).reshape(
                int(tile["rows"]), self.shape[1]
            )
            start = int(tile["row_start"])
            end = start + int(tile["rows"])
            result += flat[:, start:end].to(dtype=result.dtype) @ tile_tensor.to(
                dtype=result.dtype, device=activation.device
            )
        predicted = (
            set(prefetch_ticket.predicted_tiles)
            if prefetch_ticket is not None
            else set()
        )
        if prefetch_ticket is not None:
            prefetch_ticket.cancel_unused(active_tiles)
        self._last_active_tiles = frozenset(active_tiles)
        stats = SelectiveReadStats(
            total_activation_elements=int(flat.numel()),
            active_activation_elements=active_elements,
            tiles_read=len(active_tiles),
            tiles_skipped=len(self.tiles) - len(active_tiles),
            bytes_read=bytes_read,
            prefetched_tiles=len(predicted),
            prefetch_hits=len(predicted & active_tiles),
            prefetch_misses=len(active_tiles - predicted),
            prefetch_wasted_tiles=len(predicted - active_tiles),
            prefetch_bytes_submitted=(
                prefetch_ticket.bytes_submitted if prefetch_ticket is not None else 0
            ),
            hot_cache_hits=hot_hits,
            hot_cache_bytes=self.hot_cache_bytes,
            physical_reads=sync_physical_reads + prefetched_reads,
        )
        self.last_stats = stats
        return result.reshape(*activation.shape[:-1], self.shape[1]), stats

    def close(self) -> None:
        self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        self._store.close()


__all__ = [
    "SelectiveReadStats",
    "TilePrefetchTicket",
    "TiledValueMatrix",
    "pack_value_matrix",
    "plan_coactivation_tile_order",
]
