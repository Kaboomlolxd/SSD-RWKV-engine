"""ChatRWKV greedy helpers with optional module-scoped engines."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.provider_factory import create_weight_provider
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider


def greedy_token_ids(
    pack_dir: Path,
    ckpt: Path,
    *,
    mode: str,
    prompt: str = "Hello",
    max_tokens: int = 8,
    engine: InferenceEngine | None = None,
    stream_layer_cache: bool = False,
    warm_z: bool = False,
    max_layers_in_z: int = 1,
) -> list[int]:
    own_engine = engine is None
    if own_engine:
        cfg = EngineConfig(
            pack_dir=pack_dir,
            backend="chatrwkv",
            mode=mode,
            device="cpu",
            max_tokens=max_tokens,
            checkpoint_path=str(ckpt),
            strategy="cpu bf16",
            greedy=True,
            verify_hash=False,
            stream_layer_cache=stream_layer_cache,
            warm_z=warm_z,
            max_layers_in_z=max_layers_in_z,
        )
        engine = InferenceEngine(cfg)
        old_allow = os.environ.get("RWKV_ALLOW_UNCERTIFIED_TRINITY")
        os.environ["RWKV_ALLOW_UNCERTIFIED_TRINITY"] = "1"
        try:
            engine.load()
        finally:
            if old_allow is None:
                os.environ.pop("RWKV_ALLOW_UNCERTIFIED_TRINITY", None)
            else:
                os.environ["RWKV_ALLOW_UNCERTIFIED_TRINITY"] = old_allow

    assert engine is not None
    try:
        if mode == "resident":
            backend = engine.backend
            model = backend._model
            ids, state = backend.prefill(prompt)
            if state is None:
                state = model.generate_zero_state()
            with torch.no_grad():
                logits = None
                for tid in ids:
                    logits, state = model.forward([tid], state)
                if not ids:
                    logits, state = model.forward([0], state)
                out: list[int] = []
                for _ in range(max_tokens):
                    assert logits is not None
                    next_id = int(logits.argmax().item())
                    out.append(next_id)
                    logits, state = model.forward([next_id], state)
            return out

        assert engine.manifest and engine.store and engine.scheduler
        provider = create_weight_provider(
            engine.config,
            engine.store,
            engine.manifest.tensors,
            engine.device,
            engine.metrics,
            model_z=getattr(engine.backend._model, "z", None),
        )
        try:
            return engine.backend.generate_greedy_pack_streaming(
                prompt,
                max_tokens,
                provider,
                engine.scheduler.layers,
                manifest_block_layers(engine.manifest),
            )
        finally:
            provider.close()
    finally:
        if own_engine:
            engine.close()


def make_chatrwkv_engine(
    pack_dir: Path,
    ckpt: Path,
    *,
    mode: str,
    max_tokens: int = 8,
    skeleton_load: bool = True,
) -> InferenceEngine:
    cfg = EngineConfig(
        pack_dir=pack_dir,
        backend="chatrwkv",
        mode=mode,
        device="cpu",
        max_tokens=max_tokens,
        checkpoint_path=str(ckpt),
        strategy="cpu bf16",
        greedy=True,
        verify_hash=False,
        skeleton_load=skeleton_load,
    )
    engine = InferenceEngine(cfg)
    old_allow = os.environ.get("RWKV_ALLOW_UNCERTIFIED_TRINITY")
    os.environ["RWKV_ALLOW_UNCERTIFIED_TRINITY"] = "1"
    try:
        engine.load()
    finally:
        if old_allow is None:
            os.environ.pop("RWKV_ALLOW_UNCERTIFIED_TRINITY", None)
        else:
            os.environ["RWKV_ALLOW_UNCERTIFIED_TRINITY"] = old_allow
    return engine
