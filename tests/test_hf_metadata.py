"""Tests for the HF metadata bundle (config + tokenizer + generation).

The pack output must be self-contained: if you can load the weights
and the config, you can run the model without re-downloading from
Hugging Face. This module proves that:

  * ``copy_hf_metadata`` preserves the standard HF files byte-for-byte.
  * SHA-256 and size are recorded in the manifest's ``hf_metadata`` field.
  * Re-packing the same dir is idempotent (does not re-copy on the
    second call unless ``overwrite=True``).
  * Files absent from the source are silently skipped.
  * The pack still verifies after metadata is copied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.safetensors_loader import (
    HF_METADATA_FILES,
    copy_hf_metadata,
    find_hf_metadata_files,
    load_hf_metadata,
)
from rwkv_ssd.tools.pack_runtime import pack
from rwkv_ssd.tools.verify_pack import verify


# --- helpers -------------------------------------------------------------------


def _build_llama_like_state(
    n_layer: int = 2, n_embd: int = 32, vocab: int = 64, seed: int = 0
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, n_embd, generator=g),
        "model.norm.weight": torch.randn(n_embd, generator=g),
        "lm_head.weight": torch.randn(vocab, n_embd, generator=g),
    }
    for i in range(n_layer):
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(
            n_embd, n_embd, generator=g
        )
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(
            n_embd, n_embd, generator=g
        )
    return state


def _write_full_hf_dir(model_dir: Path, *, n_layer: int = 2) -> dict[str, Path]:
    """Create a HF-style directory with a full metadata bundle.

    Returns a {filename: path} map of what was written, for the
    tests to compare against.  Content is intentionally non-trivial
    (not just empty / placeholder) so byte equality checks are real.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    state = _build_llama_like_state(n_layer=n_layer)
    save_file(state, str(model_dir / "model.safetensors"))
    written["model.safetensors"] = model_dir / "model.safetensors"

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": n_layer,
        "vocab_size": 64,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    written["config.json"] = model_dir / "config.json"

    gen_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "temperature": 1.0,
        "top_p": 0.9,
    }
    (model_dir / "generation_config.json").write_text(
        json.dumps(gen_config, indent=2), encoding="utf-8"
    )
    written["generation_config.json"] = model_dir / "generation_config.json"

    # A minimal but valid-looking tokenizer.json (the real one is huge;
    # we just need the right shape and content for round-trip equality).
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": False},
            {"id": 1, "content": "<bos>", "single_word": False},
        ],
        "normalizer": {"type": "BertNormalizer", "lowercase": True},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {"<pad>": 0, "<bos>": 1, "hello": 2, "world": 3},
            "merges": ["h e", "l o"],
        },
    }
    (model_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2), encoding="utf-8"
    )
    written["tokenizer.json"] = model_dir / "tokenizer.json"

    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["tokenizer_config.json"] = model_dir / "tokenizer_config.json"

    (model_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"},
            indent=2,
        ),
        encoding="utf-8",
    )
    written["special_tokens_map.json"] = model_dir / "special_tokens_map.json"

    (model_dir / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "<bos>": 1, "hello": 2, "world": 3}, indent=2),
        encoding="utf-8",
    )
    written["vocab.json"] = model_dir / "vocab.json"

    (model_dir / "merges.txt").write_text(
        "#version: 0.2\nh e\nl o\n", encoding="utf-8"
    )
    written["merges.txt"] = model_dir / "merges.txt"

    return written


# --- copy_hf_metadata ----------------------------------------------------------


def test_copy_hf_metadata_preserves_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    written = _write_full_hf_dir(src, n_layer=2)

    info = copy_hf_metadata(src, dst)
    # All HF_METADATA_FILES we wrote should appear in info.
    expected = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    }
    assert set(info.keys()) == expected
    # All files copied cleanly.
    for name, src_path in written.items():
        if name == "model.safetensors":
            continue  # weights are not in the metadata bundle
        dst_path = dst / name
        assert dst_path.is_file()
        assert dst_path.read_bytes() == src_path.read_bytes()


def test_copy_hf_metadata_records_sha256_and_size(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_full_hf_dir(src, n_layer=1)
    info = copy_hf_metadata(src, dst)
    for name, meta in info.items():
        assert int(meta["size"]) == (dst / name).stat().st_size
        assert isinstance(meta["sha256"], str)
        assert len(meta["sha256"]) == 64  # sha256 hex
        assert meta["copied"] is True
        assert meta["kind"] in {"config", "tokenizer"}


def test_copy_hf_metadata_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_full_hf_dir(src, n_layer=1)

    info1 = copy_hf_metadata(src, dst)
    # All "copied" the first time.
    assert all(meta["copied"] for meta in info1.values())

    # Second call: same files, "copied=False" for all.
    info2 = copy_hf_metadata(src, dst)
    assert all(meta["copied"] is False for meta in info2.values())
    # SHA-256 must be identical.
    for name in info1:
        assert info1[name]["sha256"] == info2[name]["sha256"]


def test_copy_hf_metadata_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_full_hf_dir(src, n_layer=1)
    copy_hf_metadata(src, dst)

    # Mutate the source and re-copy with overwrite.
    (src / "config.json").write_text(
        json.dumps({"model_type": "llama", "mutated": True}, indent=2),
        encoding="utf-8",
    )
    info = copy_hf_metadata(src, dst, overwrite=True)
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("mutated") is True
    assert all(meta["copied"] for meta in info.values())


def test_copy_hf_metadata_skips_missing(tmp_path: Path) -> None:
    """A directory with only some of the HF files still works."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    info = copy_hf_metadata(src, dst)
    assert set(info.keys()) == {"config.json"}


def test_copy_hf_metadata_empty_dir(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    dst = tmp_path / "dst"
    info = copy_hf_metadata(src, dst)
    assert info == {}
    # No files were created.
    assert list(dst.iterdir()) == []


def test_find_hf_metadata_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_full_hf_dir(src, n_layer=1)
    found = find_hf_metadata_files(src)
    # Only files that exist are returned; weights (model.safetensors)
    # are not in HF_METADATA_FILES so they are excluded.
    assert "model.safetensors" not in found
    assert "config.json" in found
    assert "tokenizer.json" in found
    # A file that doesn't exist in src is silently absent.
    assert "added_tokens.json" not in found


def test_load_hf_metadata_is_inverse(tmp_path: Path) -> None:
    """load_hf_metadata should find the files we just copied."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_full_hf_dir(src, n_layer=1)
    copy_hf_metadata(src, dst)
    loaded = load_hf_metadata(dst)
    assert set(loaded.keys()) >= {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    }
    # All paths are inside dst.
    for p in loaded.values():
        assert p.parent == dst


# --- end-to-end: pack with --copy-hf-metadata --------------------------------


def test_pack_hf_dir_copies_metadata_into_pack(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_full_hf_dir(src, n_layer=2)
    pack_dir = tmp_path / "pack"

    pack(
        src,
        pack_dir,
        model_family="llama",
        hash_weights=False,
        copy_hf_metadata=True,
        quiet=True,
    )
    # Pack verifies.
    assert verify(pack_dir, check_hash=False, quiet=True)
    # Metadata files were copied next to weights.bin.
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        assert (pack_dir / name).is_file()
    # Manifest records them.
    m = Manifest.load(pack_dir)
    hf_meta = m.meta.get("hf_metadata")
    assert isinstance(hf_meta, dict)
    assert "config.json" in hf_meta
    assert hf_meta["config.json"]["kind"] == "config"
    assert hf_meta["tokenizer.json"]["kind"] == "tokenizer"
    assert "size" in hf_meta["config.json"]
    assert "sha256" in hf_meta["config.json"]


def test_pack_hf_dir_metadata_bytes_match_source(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_full_hf_dir(src, n_layer=1)
    pack_dir = tmp_path / "pack"
    pack(
        src,
        pack_dir,
        model_family="llama",
        hash_weights=False,
        copy_hf_metadata=True,
        quiet=True,
    )
    for name in HF_METADATA_FILES:
        src_p = src / name
        dst_p = pack_dir / name
        if src_p.is_file():
            assert dst_p.is_file()
            assert dst_p.read_bytes() == src_p.read_bytes()


def test_pack_pth_does_not_try_to_copy_metadata(tmp_path: Path) -> None:
    """A bare .pth has no sibling metadata; the pack path stays clean."""
    ckpt = tmp_path / "m.pth"
    torch.save(
        {
            "emb.weight": torch.randn(8, 8),
            "blocks.0.att.x_r": torch.randn(8, 8),
            "blocks.0.att.x_w": torch.randn(8, 8),
            "ln_out.weight": torch.randn(8),
            "head.weight": torch.randn(8, 8),
        },
        ckpt,
    )
    pack_dir = tmp_path / "pack"
    pack(
        ckpt,
        pack_dir,
        model_family="rwkv7",
        hash_weights=False,
        copy_hf_metadata=True,  # explicitly requested; should be ignored
        quiet=True,
    )
    # No config.json appears in the pack.
    assert not (pack_dir / "config.json").exists()
    m = Manifest.load(pack_dir)
    assert "hf_metadata" not in m.meta
