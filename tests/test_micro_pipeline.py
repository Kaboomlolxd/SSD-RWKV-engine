"""Within-layer micro-pipelining (chunked reads)."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def test_chunked_reads_recorded(synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=synthetic_pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        max_tokens=8,
        io_chunk_bytes=512,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        engine.generate("chunked io test")
        assert any(L.chunk_reads >= 1 for L in engine.metrics.layers)
    finally:
        engine.close()
