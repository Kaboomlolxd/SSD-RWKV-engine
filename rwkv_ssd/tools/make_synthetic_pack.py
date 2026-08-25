"""Build a tiny synthetic pack for tests and CPU demos."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rwkv_ssd.runtime.manifest import ALIGNMENT
from rwkv_ssd.tools.pack_runtime import pack


def build_synthetic_state(
    n_layer: int = 4,
    n_embd: int = 32,
    vocab_size: int = 256,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(vocab_size, n_embd, generator=g) * 0.02,
        "head.weight": torch.randn(n_embd, vocab_size, generator=g) * 0.02,
    }
    for i in range(n_layer):
        state[f"blocks.{i}.weight"] = torch.randn(n_embd, n_embd, generator=g) * 0.02
    return state


def write_synthetic_checkpoint(path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_synthetic_state(**kwargs), path)  # type: ignore[arg-type]


def create_synthetic_pack(
    output_dir: Path,
    n_layer: int = 4,
    n_embd: int = 32,
    vocab_size: int = 256,
    seed: int = 42,
    *,
    pack_codec: str = "none",
    pack_layout: str = "default",
    sector_bytes: int = 0,
    bf16_shadow: bool = False,
    shadow_min_numel: int = 0,
    trinity_codebook: str = "kmeans",
    quiet: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "_synthetic_source.pt"
    write_synthetic_checkpoint(
        ckpt, n_layer=n_layer, n_embd=n_embd, vocab_size=vocab_size, seed=seed
    )
    pack(
        ckpt,
        output_dir,
        model_family="synthetic_rwkv",
        pack_codec=pack_codec,
        pack_layout=pack_layout,
        sector_bytes=sector_bytes,
        bf16_shadow=bf16_shadow,
        shadow_min_numel=shadow_min_numel,
        trinity_codebook=trinity_codebook,
        quiet=quiet,
    )
    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "model_type": "synthetic_rwkv",
            "n_layer": n_layer,
            "n_embd": n_embd,
            "vocab_size": vocab_size,
            "seed": seed,
        }
    )
    from rwkv_ssd.runtime.pack_bench import pack_full_stats

    meta["pack_composition"] = pack_full_stats(output_dir)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["meta"].update(meta)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    return output_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Create a tiny synthetic runtime pack")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=32)
    args = p.parse_args()
    out = create_synthetic_pack(args.output, n_layer=args.n_layer, n_embd=args.n_embd)
    print(f"Created synthetic pack at {out}")


if __name__ == "__main__":
    main()
