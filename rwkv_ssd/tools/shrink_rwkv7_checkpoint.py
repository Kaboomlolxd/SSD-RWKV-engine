#!/usr/bin/env python3
"""
Shrink an RWKV-7 checkpoint for fast bench / smoke tests (output quality not preserved).

Drops upper block layers and truncates vocab / n_embd / FFN width from the source tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rwkv_ssd.runtime.checkpoint_meta import load_checkpoint_tensors
from rwkv_ssd.tools.pack_runtime import pack

HEAD_SIZE = 64


def _layer_id(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    part = name.split(".")[1]
    return int(part) if part.isdigit() else None


def shrink_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    old_embd: int,
    new_embd: int,
    old_vocab: int,
    new_vocab: int,
    old_ffn: int,
    new_ffn: int,
    old_n_head: int,
    new_n_head: int,
) -> torch.Tensor:
    shape = tensor.shape
    slices: list[slice | int] = []
    for dim in shape:
        if dim == old_vocab:
            slices.append(slice(0, new_vocab))
        elif dim == old_embd:
            slices.append(slice(0, new_embd))
        elif dim == old_ffn:
            slices.append(slice(0, new_ffn))
        elif dim == old_n_head:
            slices.append(slice(0, new_n_head))
        else:
            slices.append(slice(None))
    return tensor[tuple(slices)].contiguous()


def shrink_rwkv7_state(
    tensors: dict[str, torch.Tensor],
    *,
    n_layer: int,
    n_embd: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    emb = tensors.get("emb.weight")
    if emb is None:
        raise ValueError("checkpoint missing emb.weight")
    old_vocab, old_embd = int(emb.shape[0]), int(emb.shape[1])
    if n_embd % HEAD_SIZE != 0:
        raise ValueError(f"n_embd must be divisible by {HEAD_SIZE}")
    old_ffn = old_embd * 4
    new_ffn = n_embd * 4
    old_n_head = old_embd // HEAD_SIZE
    new_n_head = n_embd // HEAD_SIZE

    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        lid = _layer_id(name)
        if lid is not None and lid >= n_layer:
            continue
        out[name] = shrink_tensor(
            name,
            tensor,
            old_embd=old_embd,
            new_embd=n_embd,
            old_vocab=old_vocab,
            new_vocab=vocab_size,
            old_ffn=old_ffn,
            new_ffn=new_ffn,
            old_n_head=old_n_head,
            new_n_head=new_n_head,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Shrink RWKV-7 checkpoint for bench (~0.01B)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help=".pth output path")
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help="Also write runtime pack to this directory",
    )
    args = p.parse_args()

    tensors = load_checkpoint_tensors(args.input)
    shrunk = shrink_rwkv7_state(
        tensors,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shrunk, args.output)
    params = sum(t.numel() for t in shrunk.values())
    print(
        f"Wrote {args.output}  layers={args.n_layer} n_embd={args.n_embd} "
        f"vocab={args.vocab_size}  params={params / 1e6:.2f}M"
    )

    if args.pack_dir:
        pack(args.output, args.pack_dir, model_family="rwkv7")
        print(f"Packed -> {args.pack_dir}")


if __name__ == "__main__":
    main()
