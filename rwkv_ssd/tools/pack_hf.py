#!/usr/bin/env python3
"""Pack a Hugging Face model directory into the engine's pack format.

The full-featured CLI lives in :mod:`rwkv_ssd.tools.pack_runtime`. This
script is a thin convenience wrapper that pre-fills sensible defaults
for an HF model directory and surfaces a few HF-specific flags
(``--prefer-safetensors``, ``--include-config``).

Usage:

    python -m rwkv_ssd.tools.pack_hf \\
        --input  ./models/Qwen2-0.5B-Instruct \\
        --output ./hf_pack

    python -m rwkv_ssd.tools.pack_hf \\
        --hf-repo Qwen/Qwen2-0.5B-Instruct \\
        --output ./hf_pack \\
        --pack-codec trinity_lut2 \\
        --pack-layout layer_grouped

The output pack is the same ``weights.bin`` + ``manifest.json`` as
any other engine pack — you can pass it to the CLI / serve / bench
commands unchanged. ``manifest.model_family`` is auto-detected from
the tensor names if not passed explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rwkv_ssd.runtime.checkpoint_meta import load_hf_config
from rwkv_ssd.runtime.safetensors_loader import (
    is_safetensors_dir,
    is_safetensors_file,
    looks_like_hf_dir,
)
from rwkv_ssd.tools.pack_runtime import _load_state_dict, pack


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pack a Hugging Face model (safetensors) into a "
        "weights.bin/manifest.json pack for the engine."
    )
    p.add_argument(
        "--input",
        help="Path to a HF model directory (must contain safetensors or "
        "pytorch_model.bin) OR a single .safetensors/.pth file",
    )
    p.add_argument(
        "--hf-repo",
        help="Hugging Face repo ID; the script will download the safetensors "
        "weights into the local cache and pack from there.",
    )
    p.add_argument("--output", required=True, help="Output pack directory")
    p.add_argument(
        "--model-family",
        default=None,
        help="Override auto-detected family (e.g. llama, qwen2, mistral). "
        "If omitted, the family is detected from the tensor names.",
    )
    p.add_argument(
        "--pack-codec",
        default="none",
        choices=["none", "scale_u8", "scale_u4", "trinity_lut2", "trinity"],
    )
    p.add_argument(
        "--pack-layout",
        default="layer_grouped",
        choices=["default", "layer_grouped"],
        help="layer_grouped is the right choice for SSD streaming.",
    )
    p.add_argument(
        "--sector-bytes",
        type=int,
        default=262144,
        help="Pad between layer groups in weights.bin (0=off; "
        "262144 = 256 KiB is the recommended default for NVMe).",
    )
    p.add_argument(
        "--bf16-shadow",
        action="store_true",
        help="Also write shadow.bin (raw bf16 per tensor) for fast decode.",
    )
    p.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip computing the weights.bin SHA-256 (faster on big models).",
    )
    p.add_argument(
        "--hf-prefer",
        choices=["safetensors", "pth"],
        default="safetensors",
        help="With --hf-repo: which weight format to download.",
    )
    p.add_argument(
        "--copy-hf-metadata",
        action="store_true",
        default=True,
        help="Copy config.json, tokenizer files, and generation_config.json "
        "from the source HF directory into the pack dir so the pack is "
        "self-contained. Default: on for HF inputs.",
    )
    p.add_argument(
        "--no-copy-hf-metadata",
        dest="copy_hf_metadata",
        action="store_false",
        help="Disable --copy-hf-metadata (weights only).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
    )
    args = p.parse_args()

    if not args.input and not args.hf_repo:
        print("error: provide --input or --hf-repo", file=sys.stderr)
        sys.exit(2)

    # Resolve to a local path.
    if args.hf_repo:
        from rwkv_ssd.runtime.safetensors_loader import download_hf_repo

        path = download_hf_repo(args.hf_repo, prefer=args.hf_prefer)
    else:
        path = Path(args.input)

    # Sanity: refuse obviously wrong input before doing heavy I/O.
    if path.is_dir() and not (
        is_safetensors_dir(path) or looks_like_hf_dir(path)
    ):
        print(
            f"error: {path} doesn't look like a HF model dir (no safetensors "
            "or config.json). For a single .pth file pass the file path "
            "directly.",
            file=sys.stderr,
        )
        sys.exit(2)
    if path.is_file() and not (
        is_safetensors_file(path)
        or path.suffix in (".pth", ".pt", ".bin")
    ):
        print(
            f"error: {path} is not a .safetensors, .pth, .pt, or .bin file",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load the state dict just to detect family / fill config-derived meta.
    # The full pack() below reloads it (one extra pass is fine; we don't
    # keep the in-memory dict around because the model can be many GB).
    state = _load_state_dict(path)
    hf_config = load_hf_config(path if path.is_dir() else path.parent)

    if args.model_family is not None:
        family = args.model_family
    else:
        from rwkv_ssd.tools.pack_runtime import detect_model_family_from_state

        family = detect_model_family_from_state(state)

    if not args.quiet:
        from rwkv_ssd.runtime.safetensors_loader import count_tensors

        info = count_tensors(state)
        print(f"Detected model family: {family}")
        print(
            f"Loaded {info['tensor_count']} tensors "
            f"({info['total_bytes'] / (1024**2):.1f} MiB; "
            f"dtypes={info['dtype_counts']})"
        )
        if hf_config:
            arch = hf_config.get("model_type") or (
                hf_config.get("architectures") or [None]
            )[0]
            if arch:
                print(f"HF config.json model_type={arch}")

    # Free the temporary state to keep peak memory low for big HF models.
    del state

    # Defer the actual pack() to the existing implementation so we
    # don't fork the codec / layout / shadow logic.
    from rwkv_ssd.tools.pack_runtime import _resolve_input

    inp, hf = _resolve_input(
        Path(args.input) if args.input else None,
        args.hf_repo,
        hf_prefer=args.hf_prefer,
    )

    pack(
        inp,
        Path(args.output),
        model_family=family,
        hf_repo=hf,
        hash_weights=not args.no_hash,
        pack_codec=args.pack_codec,
        pack_layout=args.pack_layout,
        sector_bytes=args.sector_bytes,
        bf16_shadow=args.bf16_shadow,
        quiet=args.quiet,
        copy_hf_metadata=args.copy_hf_metadata and inp.is_dir(),
    )


if __name__ == "__main__":
    main()
