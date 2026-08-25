"""Pack an HF-style safetensors model and verify the resulting pack.

Exercises the full pipeline the user asked us to support:

  Hugging Face model dir  ->  weights.bin + manifest.json

with the same verification as the existing RWKV pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.safetensors_loader import (
    is_safetensors_dir,
    load_safetensors_dir,
)
from rwkv_ssd.tools.pack_runtime import (
    _layer_id_from_name,
    detect_model_family_from_state,
    pack,
)
from rwkv_ssd.tools.verify_pack import verify


# --- helpers -------------------------------------------------------------------


def _build_llama_like_state(
    n_layer: int = 4,
    n_embd: int = 32,
    vocab: int = 64,
    intermediate: int = 64,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, n_embd, generator=g),
        "model.norm.weight": torch.randn(n_embd, generator=g),
        "lm_head.weight": torch.randn(vocab, n_embd, generator=g),
    }
    for i in range(n_layer):
        state.update(
            {
                f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(
                    n_embd, n_embd, generator=g
                ),
                f"model.layers.{i}.self_attn.k_proj.weight": torch.randn(
                    n_embd, n_embd, generator=g
                ),
                f"model.layers.{i}.self_attn.v_proj.weight": torch.randn(
                    n_embd, n_embd, generator=g
                ),
                f"model.layers.{i}.self_attn.o_proj.weight": torch.randn(
                    n_embd, n_embd, generator=g
                ),
                f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(
                    intermediate, n_embd, generator=g
                ),
                f"model.layers.{i}.mlp.up_proj.weight": torch.randn(
                    intermediate, n_embd, generator=g
                ),
                f"model.layers.{i}.mlp.down_proj.weight": torch.randn(
                    n_embd, intermediate, generator=g
                ),
                f"model.layers.{i}.input_layernorm.weight": torch.randn(
                    n_embd, generator=g
                ),
                f"model.layers.{i}.post_attention_layernorm.weight": torch.randn(
                    n_embd, generator=g
                ),
            }
        )
    return state


def _write_config(path: Path, *, model_type: str = "llama", n_layer: int = 4) -> None:
    cfg = {
        "model_type": model_type,
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": n_layer,
        "vocab_size": 64,
        "max_position_embeddings": 2048,
    }
    (path / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# --- layer-id detection --------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        # HF Llama/Qwen2 style.
        ("model.embed_tokens.weight", 0),
        ("model.norm.weight", 0),
        ("model.layers.0.self_attn.q_proj.weight", 0),
        ("model.layers.5.mlp.gate_proj.weight", 5),
        ("model.layers.31.input_layernorm.weight", 31),
        ("lm_head.weight", 9999),
        # RWKV legacy.
        ("emb.weight", 0),
        ("blocks.0.att.x_r", 0),
        ("blocks.7.att.x_w", 7),
        ("ln_out.weight", 0),
        ("head.weight", 9999),
        # Misc.
        ("total_garbage.weight", -1),
    ],
)
def test_layer_id_from_name(name: str, expected: int) -> None:
    assert _layer_id_from_name(name) == expected


def test_layer_id_handles_double_prefix() -> None:
    # Some HF models (Qwen2.5, etc.) double the prefix when the
    # checkpoint is the result of a state-dict round-trip.
    assert _layer_id_from_name("model.model.layers.2.self_attn.q_proj.weight") == 2


# --- family detection ----------------------------------------------------------


def test_detect_family_llama(tmp_path: Path) -> None:
    state = _build_llama_like_state(n_layer=2)
    assert detect_model_family_from_state(state) == "llama"


def test_detect_family_qwen2() -> None:
    g = torch.Generator().manual_seed(0)
    state = {
        "model.embed_tokens.weight": torch.randn(8, 8, generator=g),
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(24, 8, generator=g),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 8, generator=g),
        "lm_head.weight": torch.randn(8, 8, generator=g),
    }
    assert detect_model_family_from_state(state) == "qwen2"


def test_detect_family_rwkv7() -> None:
    g = torch.Generator().manual_seed(0)
    state = {
        "emb.weight": torch.randn(8, 8, generator=g),
        "blocks.0.att.x_r": torch.randn(8, 8, generator=g),
        "blocks.0.att.x_w": torch.randn(8, 8, generator=g),
        "ln_out.weight": torch.randn(8, generator=g),
        "head.weight": torch.randn(8, 8, generator=g),
    }
    assert detect_model_family_from_state(state) == "rwkv7"


# --- end-to-end pack -----------------------------------------------------------


def test_pack_hf_single_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    state = _build_llama_like_state(n_layer=3, n_embd=32)
    model_dir.mkdir()
    save_file(state, str(model_dir / "model.safetensors"))
    _write_config(model_dir, model_type="llama", n_layer=3)

    pack_dir = tmp_path / "pack"
    pack(model_dir, pack_dir, model_family="llama", hash_weights=False, quiet=True)
    assert verify(pack_dir, check_hash=False, quiet=True)

    m = Manifest.load(pack_dir)
    assert m.model_family == "llama"
    assert m.meta.get("n_layer") == 3
    assert m.meta.get("vocab_size") == 64
    assert m.meta.get("n_embd") == 32
    # 2 (embed+norm) + 1 (head) + 3 layers * 9 tensors = 30 tensors
    assert len(m.tensors) == 30
    # Sanity: all layer ids are sane.
    layer_ids = sorted({t.layer_id for t in m.tensors})
    assert layer_ids == [0, 1, 2, 9999]


def test_pack_auto_detects_hf_family(tmp_path: Path) -> None:
    """The public importer should not require a family flag for HF models."""
    model_dir = tmp_path / "model"
    state = _build_llama_like_state(n_layer=1)
    model_dir.mkdir()
    save_file(state, str(model_dir / "model.safetensors"))
    _write_config(model_dir, model_type="llama", n_layer=1)

    pack_dir = tmp_path / "pack"
    pack(model_dir, pack_dir, model_family="auto", hash_weights=False, quiet=True)

    manifest = Manifest.load(pack_dir)
    assert manifest.model_family == "llama"


def test_pack_hf_sharded_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    state = _build_llama_like_state(n_layer=3, n_embd=32)
    model_dir.mkdir()
    # Manually write sharded safetensors with an index file.
    keys = sorted(state.keys())
    n_shards = 2
    weight_map: dict[str, str] = {}
    for i, k in enumerate(keys):
        shard_name = f"model-{i % n_shards + 1:05d}-of-{n_shards:05d}.safetensors"
        weight_map[k] = shard_name
    for shard_idx in range(n_shards):
        group = {
            k: state[k]
            for k, name in weight_map.items()
            if name == f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.safetensors"
        }
        save_file(group, str(model_dir / f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map, "metadata": {"total_size": 0}}, indent=2)
    )
    _write_config(model_dir, model_type="llama", n_layer=3)

    pack_dir = tmp_path / "pack"
    pack(model_dir, pack_dir, model_family="llama", hash_weights=False, quiet=True)
    assert verify(pack_dir, check_hash=False, quiet=True)

    m = Manifest.load(pack_dir)
    assert m.model_family == "llama"
    assert m.meta.get("n_layer") == 3
    assert len(m.tensors) == len(state)


def test_pack_hf_uses_layer_grouped_layout(tmp_path: Path) -> None:
    """When pack_layout=layer_grouped, all tensors in the same layer
    are written contiguously. We assert this by checking the offsets in
    the manifest are non-decreasing within each layer.
    """
    model_dir = tmp_path / "model"
    state = _build_llama_like_state(n_layer=2, n_embd=32)
    model_dir.mkdir()
    save_file(state, str(model_dir / "model.safetensors"))
    _write_config(model_dir, model_type="llama", n_layer=2)

    pack_dir = tmp_path / "pack"
    pack(
        model_dir,
        pack_dir,
        model_family="llama",
        hash_weights=False,
        pack_layout="layer_grouped",
        quiet=True,
    )
    assert verify(pack_dir, check_hash=False, quiet=True)
    m = Manifest.load(pack_dir)
    # Group by layer_id and check offsets are non-decreasing.
    by_layer: dict[int, list] = {}
    for t in m.tensors:
        by_layer.setdefault(t.layer_id, []).append(t)
    for layer_id, entries in by_layer.items():
        offsets = [e.offset for e in entries]
        assert offsets == sorted(offsets), f"layer {layer_id} offsets not sorted"


def test_pack_rwkv_pth_still_works(tmp_path: Path) -> None:
    """Backwards-compat: the existing RWKV .pth path must not regress."""
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack_dir = create_synthetic_pack(tmp_path / "syn", quiet=True)
    assert verify(pack_dir, check_hash=False, quiet=True)


def test_pack_source_checkpoint_provenance_is_portable(tmp_path: Path) -> None:
    """Pack metadata must not leak the machine-local checkpoint path."""
    checkpoint = tmp_path / "inputs" / "rwkv-model.pth"
    checkpoint.parent.mkdir()
    torch.save({"emb.weight": torch.ones(4, 4)}, checkpoint)

    pack_dir = tmp_path / "pack"
    pack(checkpoint, pack_dir, model_family="rwkv7", hash_weights=False, quiet=True)

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    assert manifest["meta"]["source_checkpoint"] == checkpoint.name
    assert meta["source_checkpoint"] == checkpoint.name
    assert str(checkpoint.parent) not in json.dumps(manifest)
    assert str(checkpoint.parent) not in json.dumps(meta)


def test_pack_directory_no_safetensors_rejected(tmp_path: Path) -> None:
    """A directory that has neither safetensors nor pytorch_model.* raises."""
    bad = tmp_path / "empty"
    bad.mkdir()
    with pytest.raises(FileNotFoundError):
        pack(bad, tmp_path / "out", quiet=True)
