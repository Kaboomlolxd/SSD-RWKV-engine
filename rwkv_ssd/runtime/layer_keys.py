"""Group manifest tensor entries by RWKV layer id."""

from __future__ import annotations

from rwkv_ssd.runtime.manifest import Manifest, TensorEntry


def entries_for_layer(entries: list[TensorEntry], layer_id: int) -> list[TensorEntry]:
    return [e for e in entries if e.layer_id == layer_id]


def layer_ids(entries: list[TensorEntry]) -> list[int]:
    ids = {e.layer_id for e in entries if e.layer_id >= 0 and e.layer_id < 9000}
    return sorted(ids)


def global_tensor_entries(entries: list[TensorEntry]) -> list[TensorEntry]:
    """Embed, head, ln_out — not tied to a block layer id."""
    return [e for e in entries if e.layer_id in (-1, 9999) or "emb." in e.name or "head." in e.name]


def manifest_block_layers(manifest: Manifest) -> list[int]:
    return layer_ids(manifest.tensors)
