"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.backends.synthetic import SyntheticBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_store import open_weight_store


def greedy_token_ids(
    pack_dir: Path,
    prompt: str,
    *,
    mode: str = "streaming",
    max_tokens: int = 8,
    io_backend: str = "mmap",
    prefetch_policy: str = "layer",
    io_chunk_bytes: int = 0,
    io_chunk_policy: str = "uniform",
    stream_layer_cache: bool = False,
    warm_z: bool = False,
    max_layers_in_z: int = 1,
    cache_format: str = "auto",
) -> list[int]:
    """Run synthetic greedy decode and return raw token ids."""
    cfg = EngineConfig(
        pack_dir=pack_dir,
        backend="synthetic",
        mode=mode,
        device="cpu",
        max_tokens=max_tokens,
        io_backend=io_backend,
        io_chunk_bytes=io_chunk_bytes,
        io_chunk_policy=io_chunk_policy,
        prefetch_policy=prefetch_policy,
        stream_layer_cache=stream_layer_cache,
        warm_z=warm_z,
        max_layers_in_z=max_layers_in_z,
        cache_format=cache_format,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        assert isinstance(engine.backend, SyntheticBackend)
        assert engine.manifest is not None
        store = open_weight_store(
            engine.manifest.weights_path, backend=io_backend
        )
        provider = create_weight_provider(
            cfg,
            store,
            engine.manifest.tensors,
            engine.device,
            engine.metrics,
        )
        try:
            return engine.backend.generate_greedy(
                prompt, provider, max_tokens, engine.metrics
            )
        finally:
            provider.close()
            store.close()
    finally:
        engine.close()
