"""Unit tests for the safetensors loader.

Exercises the three real layouts the engine must support:

  1. A single ``model.safetensors`` file (smallest HF layout).
  2. A sharded HF directory with ``model.safetensors.index.json`` +
     multiple ``model-NNNNN-of-MMMMM.safetensors`` shards.
  3. A "no index, multiple shards" directory — must **refuse** to
     silently merge (the loader raises with a clear message).
  4. Round-trip: save_file → load_file → equal tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rwkv_ssd.runtime.safetensors_loader import (
    is_safetensors_dir,
    is_safetensors_file,
    load_safetensors,
    load_safetensors_dir,
    load_safetensors_file,
    looks_like_hf_dir,
)


# --- fixtures ------------------------------------------------------------------


def _build_llama_like_state(
    n_layer: int = 3,
    n_embd: int = 32,
    n_head: int = 4,
    vocab: int = 64,
    intermediate: int = 64,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a Llama-architecture state_dict for testing."""
    g = torch.Generator().manual_seed(seed)
    head_dim = n_embd // n_head
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
    # Add a per-head rotary buffer for realism.
    for i in range(n_layer):
        state[f"model.layers.{i}.self_attn.rotary_emb.inv_freq"] = torch.randn(
            head_dim // 2, generator=g
        )
    return state


def _save_hf_single(path: Path, state: dict[str, torch.Tensor]) -> None:
    """Save a single model.safetensors file."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"))


def _save_hf_sharded(
    path: Path,
    state: dict[str, torch.Tensor],
    n_shards: int = 2,
    prefix: str = "model",
) -> None:
    """Save a sharded model.safetensors.index.json + N shard files.

    Splits ``state`` into ``n_shards`` roughly-equal groups (by sorted
    key) and writes each group to its own safetensors file. The
    resulting index file matches the Hugging Face format.
    """
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted(state.keys())
    n_shards = max(1, min(n_shards, len(keys)))
    width = max(5, len(str(n_shards)))  # HF uses 5-digit zero-padded
    weight_map: dict[str, str] = {}
    shards: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(n_shards)]
    for i, k in enumerate(keys):
        shard_idx = i * n_shards // len(keys)
        shards[shard_idx].append((k, state[k]))
        # HF convention: model-NNNNN-of-MMMMM.safetensors, both
        # numbers zero-padded to 5 digits.
        shard_name = f"{prefix}-{shard_idx + 1:0{width}d}-of-{n_shards:0{width}d}.safetensors"
        weight_map[k] = shard_name
    for shard_idx, group in enumerate(shards):
        shard_name = f"{prefix}-{shard_idx + 1:0{width}d}-of-{n_shards:0{width}d}.safetensors"
        save_file(dict(group), str(path / shard_name))
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": weight_map,
    }
    (path / f"{prefix}.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )


def _write_minimal_config(path: Path) -> None:
    """Drop a minimal config.json so a directory looks HF."""
    (path / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 32}), encoding="utf-8"
    )


# --- detection -----------------------------------------------------------------


def test_is_safetensors_file(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"")
    assert is_safetensors_file(p)
    assert not is_safetensors_file(tmp_path / "nope.safetensors")
    assert not is_safetensors_file(tmp_path / "x.bin")


def test_is_safetensors_dir_single(tmp_path: Path) -> None:
    d = tmp_path / "single"
    _save_hf_single(d, _build_llama_like_state(n_layer=1))
    _write_minimal_config(d)
    assert is_safetensors_dir(d)
    assert looks_like_hf_dir(d)


def test_is_safetensors_dir_sharded(tmp_path: Path) -> None:
    d = tmp_path / "sharded"
    _save_hf_sharded(d, _build_llama_like_state(n_layer=2), n_shards=2)
    _write_minimal_config(d)
    assert is_safetensors_dir(d)
    assert looks_like_hf_dir(d)


def test_is_safetensors_dir_no_safetensors(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert looks_like_hf_dir(d)  # has config.json
    assert not is_safetensors_dir(d)  # no safetensors


# --- single file ---------------------------------------------------------------


def test_load_single_safetensors_roundtrip(tmp_path: Path) -> None:
    state = _build_llama_like_state(n_layer=2, n_embd=16)
    f = tmp_path / "m.safetensors"
    save_file(state, str(f))
    loaded = load_safetensors_file(f)
    assert set(loaded.keys()) == set(state.keys())
    for k in state:
        assert torch.equal(loaded[k], state[k]), f"mismatch for {k}"


# --- sharded directory ---------------------------------------------------------


def test_load_sharded_safetensors_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "sharded"
    state = _build_llama_like_state(n_layer=4, n_embd=16)
    _save_hf_sharded(d, state, n_shards=3)
    loaded = load_safetensors_dir(d)
    assert set(loaded.keys()) == set(state.keys())
    for k in state:
        assert torch.equal(loaded[k], state[k]), f"mismatch for {k}"  # type: ignore[arg-type]


def test_load_sharded_detects_missing_shard(tmp_path: Path) -> None:
    """If the index references a shard that doesn't exist, raise cleanly."""
    d = tmp_path / "broken"
    d.mkdir()
    state = _build_llama_like_state(n_layer=1, n_embd=16)
    # Write the first shard only, but lie in the index.
    save_file(state, str(d / "model-00001-of-00002.safetensors"))
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    k: "model-00001-of-00002.safetensors"
                    for k in sorted(state.keys())
                },
            }
        )
    )
    # Add a fake reference to a non-existent shard.
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    k: "model-00001-of-00002.safetensors"
                    if i % 2 == 0
                    else "model-00002-of-00002.safetensors"
                    for i, k in enumerate(sorted(state.keys()))
                },
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing shards"):
        load_safetensors_dir(d)


def test_load_multi_shard_no_index_refuses(tmp_path: Path) -> None:
    """Two .safetensors files with no index → refuse to guess."""
    d = tmp_path / "no_index"
    d.mkdir()
    s = _build_llama_like_state(n_layer=2, n_embd=16)
    keys = sorted(s.keys())
    half = len(keys) // 2
    save_file({k: s[k] for k in keys[:half]}, str(d / "a.safetensors"))
    save_file({k: s[k] for k in keys[half:]}, str(d / "b.safetensors"))
    with pytest.raises(RuntimeError, match="refusing to guess"):
        load_safetensors_dir(d)


def test_load_no_safetensors_dir(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="no safetensors files found"):
        load_safetensors_dir(d)


# --- top-level dispatch --------------------------------------------------------


def test_load_safetensors_dispatches(tmp_path: Path) -> None:
    state = _build_llama_like_state(n_layer=2, n_embd=16)
    f = tmp_path / "x.safetensors"
    save_file(state, str(f))
    out = load_safetensors(f)
    assert set(out.keys()) == set(state.keys())


def test_load_safetensors_rejects_non_safetensors_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="expected .safetensors"):
        load_safetensors(p)


def test_load_safetensors_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_safetensors(tmp_path / "nope")
