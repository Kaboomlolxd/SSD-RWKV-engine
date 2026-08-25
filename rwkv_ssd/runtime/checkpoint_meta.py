"""Inspect model checkpoint files (RWKV, HF safetensors) for packing."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rwkv_ssd.runtime.deepembed import infer_deepembed_meta


def load_checkpoint_tensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def is_rwkv7_checkpoint(tensors: dict[str, torch.Tensor]) -> bool:
    return any("att.x_r" in k or "att.x_w" in k for k in tensors)


def _collect_layer_ids(tensors: dict[str, torch.Tensor]) -> set[int]:
    """Collect numeric layer ids from tensor names across common naming schemes.

    Recognizes:
      * HF: ``model.layers.N.*``, ``model.model.layers.N.*``, and
        ``backbone.layers.N.*``
      * RWKV: ``blocks.N.*``
      * GGML: ``blk.N.*`` / ``token_embd.weight``
    """
    out: set[int] = set()
    for k in tensors:
        for prefix in (
            "model.layers.",
            "model.model.layers.",
            "backbone.layers.",
            "blocks.",
            "blk.",
        ):
            if prefix in k:
                tail = k.split(prefix, 1)[1]
                head = tail.split(".", 1)[0]
                if head.isdigit():
                    out.add(int(head))
                break
    return out


def infer_checkpoint_meta(
    tensors: dict[str, torch.Tensor],
    *,
    config: dict | None = None,
) -> dict[str, object]:
    """Pull a few high-signal facts out of a state_dict (+ optional HF config).

    The keys here are advisory / for manifest ``meta`` only — the pack
    itself stores full tensor info, so this is just a convenience for
    logs and downstream heuristics.
    """
    meta: dict[str, object] = {}

    # DeepEmbed is a model-contract change: it adds a lookup-backed qkv/DEA
    # path and three derived vocabulary tables.  Keep that fact in metadata so
    # backends cannot accidentally treat the checkpoint as ordinary RWKV-7.
    meta.update(infer_deepembed_meta(tensors))

    # Family detection.
    if meta.get("deepembed"):
        meta["rwkv_version"] = 7
        meta["rwkv_variant"] = str(meta.get("deepembed_variant", "deepembed"))
    elif is_rwkv7_checkpoint(tensors):
        meta["rwkv_version"] = 7
    elif any(k.startswith("blocks.") for k in tensors):
        meta["rwkv_version"] = 5
    else:
        meta["rwkv_version"] = 0  # 0 = "not RWKV"

    # Vocab / embedding size.
    emb = tensors.get("emb.weight")
    if emb is None:
        emb = tensors.get("model.embed_tokens.weight")
    if emb is None:
        emb = tensors.get("model.model.embed_tokens.weight")
    if emb is None:
        emb = tensors.get("token_embd.weight")
    if emb is None:
        emb = tensors.get("backbone.embeddings.weight")
    if emb is not None:
        meta["vocab_size"] = int(emb.shape[0])
        meta["n_embd"] = int(emb.shape[1])
    elif config:
        vs = config.get("vocab_size")
        hs = (
            config.get("hidden_size")
            or config.get("n_embd")
            or config.get("d_model")
        )
        if vs is not None:
            meta["vocab_size"] = int(vs)
        if hs is not None:
            meta["n_embd"] = int(hs)

    blocks = _collect_layer_ids(tensors)
    if blocks:
        meta["n_layer"] = len(blocks)
        meta["layer_id_min"] = min(blocks)
        meta["layer_id_max"] = max(blocks)
    elif config:
        nl = config.get("num_hidden_layers") or config.get("n_layer")
        if nl is not None:
            meta["n_layer"] = int(nl)

    if tensors:
        sample = next(iter(tensors.values()))
        meta["primary_dtype"] = str(sample.dtype).replace("torch.", "")

    # HF arch: prefer an explicit config field if present.
    if config:
        arch = (
            config.get("model_type")
            or config.get("architectures", [None])[0]
        )
        if arch:
            meta["hf_architectures"] = (
                arch if isinstance(arch, str) else arch
            )
        # Common attention/head hints.
        for k_src, k_dst in (
            ("num_attention_heads", "n_head"),
            ("num_key_value_heads", "n_kv_head"),
            ("intermediate_size", "n_intermediate"),
            ("max_position_embeddings", "max_seq_len"),
        ):
            if k_src in config:
                meta[k_dst] = int(config[k_src])

    return meta


def load_hf_config(path: Path) -> dict | None:
    """Load a Hugging Face ``config.json`` if present.

    Returns the parsed dict, or None if the file is missing. Used to
    enrich ``infer_checkpoint_meta`` when the user points ``pack`` at
    a HF model directory.
    """
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def default_rwkv_strategy(
    tensors: dict[str, torch.Tensor],
    device: str,
) -> str:
    """Pick a ChatRWKV / rwkv pip strategy string for this checkpoint."""
    dev = device.lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        dev = "cuda"
    elif dev.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = "xpu"
    else:
        dev = "cpu"
    dt = infer_checkpoint_meta(tensors).get("primary_dtype", "float32")
    if dt == "bfloat16":
        return f"{dev} bf16"
    if dt == "float16":
        return f"{dev} fp16"
    return f"{dev} fp32"


def model_path_base(checkpoint: str | Path) -> str:
    p = Path(checkpoint)
    return str(p.with_suffix("")) if p.suffix == ".pth" else str(p)
