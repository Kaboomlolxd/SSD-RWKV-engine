"""P2.c heterogeneous chunks and P2.e ngram cache."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.chunk_schedule import chunk_bytes_for_entry
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry


def test_layer_size_chunk_policy() -> None:
    small = TensorEntry("a", 0, "float32", [8, 8], 0, 256, 4096, "streamed")
    large = TensorEntry("b", 1, "float32", [512, 512], 0, 512 * 512 * 4, 4096, "streamed")
    assert chunk_bytes_for_entry(small, "layer_size") == 64 * 1024
    assert chunk_bytes_for_entry(large, "layer_size") == 128 * 1024


def test_ngram_cache_hits_on_repeat_prompt(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=8,
        ngram_weight_cache=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("repeat")
        engine.generate("repeat")
        assert sum(L.ngram_hits for L in engine.metrics.layers) > 0
    finally:
        engine.close()
