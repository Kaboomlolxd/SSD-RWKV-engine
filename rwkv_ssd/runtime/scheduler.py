"""Layer-ordered access helpers (layer ordering and payload queries)."""

from __future__ import annotations

from rwkv_ssd.runtime.manifest import TensorEntry


class LayerScheduler:
    """Queries on a layer→entries map: sorted ids, payload bytes, next-layer.

    This is *not* an I/O scheduler — it provides layer ordering and payload
    lookups consumed by the weight provider and prefetch planner. Actual I/O
    scheduling happens in ``weight_provider.py`` (prefetch, cache, LRU).
    """

    def __init__(self, layers: dict[int, list[TensorEntry]]) -> None:
        self.layer_ids = sorted(layers.keys())
        self.layers = layers

    def payload_bytes(self, layer_id: int) -> int:
        return sum(t.length for t in self.layers.get(layer_id, []))

    def entries_for_layer(self, layer_id: int) -> list[TensorEntry]:
        return self.layers.get(layer_id, [])

    def next_layer(self, current: int) -> int | None:
        try:
            i = self.layer_ids.index(current)
        except ValueError:
            return None
        if i + 1 < len(self.layer_ids):
            return self.layer_ids[i + 1]
        return None
