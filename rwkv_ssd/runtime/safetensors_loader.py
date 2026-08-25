"""Safetensors loaders — single file, sharded HF directory, and HF repo helper.

The engine's pack pipeline (``rwkv_ssd.tools.pack_runtime``) is the
single entry point that turns any model checkpoint into the engine's
``weights.bin`` + ``manifest.json`` format. Historically it consumed
only PyTorch ``.pth``/``.pt`` (a single file) and a single
``.safetensors`` file. This module extends the input surface to:

  1. A single ``*.safetensors`` file (in-place of the .pth path).
  2. A Hugging Face model *directory* containing either:
       a) one ``model.safetensors`` file, or
       b) a sharded ``model.safetensors.index.json`` pointing at
          ``model-00001-of-00003.safetensors``-style shards.
  3. A directory containing any ``*.safetensors`` files (single or
     sharded, no index file) — useful for trimmed/test fixtures.

The returned dict is a plain ``{tensor_name: torch.Tensor}`` map
identical in shape to what ``torch.load(...)`` returns, so the rest of
the pack pipeline does not need to know it came from safetensors.

This module is intentionally **format-only**: it does not know about
the engine's manifest, codecs, or layer grouping. That stays in
``pack_runtime`` and friends.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)


# ---- detection ----------------------------------------------------------------


def is_safetensors_file(path: Path) -> bool:
    """True if ``path`` is a ``.safetensors`` file that exists."""
    return path.is_file() and path.suffix == ".safetensors"


def is_safetensors_dir(path: Path) -> bool:
    """True if ``path`` is a directory containing safetensors we can load.

    Either a HF layout (``model.safetensors`` or
    ``model.safetensors.index.json``) or any directory with at least
    one ``*.safetensors`` file. The check is intentionally cheap
    (one stat / one glob) so it can be used to decide whether to
    dispatch into the safetensors path.
    """
    if not path.is_dir():
        return False
    if (path / "model.safetensors").is_file():
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    return any(path.glob("*.safetensors"))


def looks_like_hf_dir(path: Path) -> bool:
    """True if ``path`` looks like a Hugging Face model directory.

    Heuristic: has ``config.json``, or has the HF sharded layout
    (``model.safetensors.index.json``). The check is intentionally
    permissive — false positives are fine because the loader is
    read-only and a "no tensors found" error is a clean failure.
    """
    if not path.is_dir():
        return False
    if (path / "config.json").is_file():
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    return False


# ---- single file --------------------------------------------------------------


def load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    """Load a single ``.safetensors`` file as a name->Tensor map.

    Memory-efficient for big files: the safetensors library is mmap'd
    on disk, so a 7B model does not need to be fully resident on load.
    Tensors are returned on CPU; downstream code decides device/dtype.
    """
    from safetensors.torch import load_file  # local import: keep cold-start cheap

    if not is_safetensors_file(path):
        raise FileNotFoundError(f"not a safetensors file: {path}")
    logger.info("loading safetensors file: %s", path)
    return dict(load_file(str(path)))


# ---- sharded HF directory -----------------------------------------------------


_HF_INDEX_FILENAMES = (
    "model.safetensors.index.json",
)
_HF_SINGLE_FILENAMES = (
    "model.safetensors",
)


def _resolve_hf_shard_files(
    index_path: Path, root: Path
) -> tuple[list[Path], dict[str, str]]:
    """Return (shard_files, weight_map) for a HF sharded safetensors index.

    ``weight_map`` maps every tensor name to the relative shard file
    (relative to ``root``). Raises if the index is malformed.
    """
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"bad HF safetensors index (not a JSON object): {index_path}")
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"bad HF safetensors index (missing 'weight_map'): {index_path}"
        )
    # Order of shard files does not matter for correctness, but we
    # preserve the index's order-of-first-appearance so logging is
    # stable across runs.
    seen: set[str] = set()
    ordered: list[str] = []
    for name, shard_name in weight_map.items():
        shard_name = str(shard_name)
        if shard_name not in seen:
            seen.add(shard_name)
            ordered.append(shard_name)
    shard_files = [root / rel for rel in ordered]
    missing = [p for p in shard_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"HF safetensors index references missing shards: {missing} "
            f"(root={root})"
        )
    return shard_files, {str(k): str(v) for k, v in weight_map.items()}


def _load_shards(
    shard_files: Iterable[Path],
    *,
    prefix_filter: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Load tensors from one or more safetensors shards (mmap'd, CPU)."""
    from safetensors import safe_open  # mmap-backed reader

    out: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        logger.info("loading shard: %s", shard)
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if prefix_filter and not key.startswith(prefix_filter):
                    continue
                # Lazy so a 7B model does not all hit RAM at once if
                # downstream only touches a few tensors at a time.
                out[key] = f.get_tensor(key)
    return out


def load_safetensors_dir(
    path: Path,
    *,
    single_name: str | None = None,
    index_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load tensors from a directory of safetensors files.

    Resolution order:

      1. ``<path>/<index_name>``  (default ``model.safetensors.index.json``).
      2. ``<path>/<single_name>`` (default ``model.safetensors``).
      3. Any ``*.safetensors`` under ``<path>`` (single or sharded, no
         index file). If more than one shard is present and no index
         file is present, we **refuse** to silently merge — too easy
         to silently drop a tensor. Raise with a clear message.

    All tensors are loaded on CPU. Names match the file keys verbatim
    (HF convention: ``model.layers.0.self_attn.q_proj.weight`` etc.).
    """
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")

    index_path = path / (index_name or _HF_INDEX_FILENAMES[0])
    single_path = path / (single_name or _HF_SINGLE_FILENAMES[0])

    if index_path.is_file():
        shard_files, weight_map = _resolve_hf_shard_files(index_path, path)
        tensors = _load_shards(shard_files)
        # Sanity: every tensor promised by the index must be present.
        missing = [k for k in weight_map if k not in tensors]
        if missing:
            raise RuntimeError(
                f"HF safetensors index lists {len(missing)} tensors not found "
                f"in shards (e.g. {missing[:3]})"
            )
        # Surface any extra tensors in the shards (not in the index) — a
        # common symptom of a half-finished repack.
        extras = [k for k in tensors if k not in weight_map]
        if extras:
            logger.warning(
                "shards contain %d tensors not listed in %s (e.g. %s); "
                "loading them anyway",
                len(extras),
                index_path.name,
                extras[:3],
            )
        return tensors

    if single_path.is_file():
        return load_safetensors_file(single_path)

    # No HF layout — fall back to globbing.
    shards = sorted(p for p in path.glob("*.safetensors") if p.is_file())
    if not shards:
        raise FileNotFoundError(
            f"no safetensors files found in {path} "
            f"(looked for {index_path.name}, {single_path.name}, and *.safetensors)"
        )
    if len(shards) > 1:
        raise RuntimeError(
            f"{path} contains {len(shards)} .safetensors files but no "
            "model.safetensors.index.json — refusing to guess the "
            "shard->tensor mapping. Either drop a HF index file, or "
            "pass a single .safetensors file."
        )
    return load_safetensors_file(shards[0])


# ---- public dispatch ----------------------------------------------------------


def load_safetensors(
    path: str | Path,
    *,
    single_name: str | None = None,
    index_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Top-level dispatch: file -> directory -> NotFound.

    Accepts:
      * a ``*.safetensors`` file
      * a directory containing safetensors (HF sharded, HF single, or
        any glob-matching layout)
    """
    p = Path(path)
    if p.is_file():
        if p.suffix != ".safetensors":
            raise ValueError(
                f"load_safetensors: expected .safetensors, got {p.name}"
            )
        return load_safetensors_file(p)
    if p.is_dir():
        return load_safetensors_dir(
            p, single_name=single_name, index_name=index_name
        )
    raise FileNotFoundError(f"load_safetensors: path does not exist: {p}")


# ---- HF repo helper -----------------------------------------------------------


def download_hf_repo(
    repo_id: str,
    *,
    prefer: str = "safetensors",
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a Hugging Face model repo's weight files to local cache.

    Returns the directory containing the safetensors (or .pth) files.
    The default ``prefer='safetensors'`` is the new behavior: HF repos
    that ship both ``pytorch_model.bin`` and ``model.safetensors`` get
    the latter. Pass ``prefer='pth'`` for the legacy path.

    This intentionally avoids the full ``snapshot_download`` (which
    grabs configs/tokenizers we don't need for the pack step). We
    fetch just the weight files, then re-use :func:`load_safetensors_dir`.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "pip install 'rwkv-ssd[hf]' (huggingface_hub) to use --hf-repo"
        ) from e

    def _download_optional_metadata(root: Path) -> None:
        """Populate the cached snapshot with config/tokenizer sidecars.

        ``hf_hub_download`` normally returns a single weight file. Returning
        its snapshot directory after fetching the small metadata bundle lets
        the pack command make an HF import self-contained when
        ``--copy-hf-metadata`` is requested. Metadata is best-effort here:
        packing weights should still work for repositories that intentionally
        publish no tokenizer or config.
        """
        for filename in HF_METADATA_FILES:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            except Exception:
                continue

    if prefer == "safetensors":
        # Try the HF index first (most common modern layout), then a
        # single safetensors, then a single pth as a last resort.
        for filename in (
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin",
        ):
            try:
                p = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
                # For sharded index we need the *directory*, not the file
                root = Path(p).parent
                _download_optional_metadata(root)
                # A directory is needed for sharded weights and for copying
                # config/tokenizer files into the output pack. A single-file
                # safetensors import also works through the directory loader.
                return root
            except Exception:
                continue
        raise FileNotFoundError(
            f"no safetensors or .bin weights found in HF repo {repo_id!r}"
        )

    # Legacy .pth path (kept for backwards-compat with the existing
    # pack_runtime CLI; behaviour matches the old code).
    for pattern in ("*.pth", "RWKV*.pth", "*.pt"):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=pattern.split("/")[-1],
                    token=token,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
        except Exception:
            continue
    raise FileNotFoundError(f"no .pth found in Hugging Face repo {repo_id!r}")


# ---- misc ---------------------------------------------------------------------


def count_tensors(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Small helper for logging / manifests."""
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    for t in state.values():
        n = t.numel() * t.element_size()
        total_bytes += n
        dt = str(t.dtype).replace("torch.", "")
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    return {
        "tensor_count": len(state),
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
    }


# ---- HF metadata bundle (config / tokenizer / generation) -------------------
#
# A "pack" that only has weights.bin is useless — the downstream
# transformer backend needs config.json to instantiate the model
# architecture, the tokenizer to encode/decode text, and the
# generation_config for sane defaults. This block finds those files
# in a HF directory, copies them next to weights.bin, and reports
# what it preserved so the manifest can record it.
#
# We **copy** rather than symlink so the pack directory is self-
# contained: you can move it, zip it, or serve it without breaking
# relative paths. The list of files below is the standard HF set;
# anything else in the source dir is ignored by default.

HF_METADATA_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.spm",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
)

# Per-kind grouping, for nicer manifest output.
HF_METADATA_KIND: dict[str, str] = {
    "config.json": "config",
    "generation_config.json": "config",
    "tokenizer.json": "tokenizer",
    "tokenizer.model": "tokenizer",
    "spiece.model": "tokenizer",
    "vocab.spm": "tokenizer",
    "tokenizer_config.json": "tokenizer",
    "vocab.json": "tokenizer",
    "merges.txt": "tokenizer",
    "special_tokens_map.json": "tokenizer",
    "added_tokens.json": "tokenizer",
    "chat_template.json": "tokenizer",
}


def find_hf_metadata_files(
    source_dir: Path,
    *,
    names: tuple[str, ...] = HF_METADATA_FILES,
) -> dict[str, Path]:
    """Return a name->absolute-path map for every metadata file present.

    Files that are absent are silently omitted; we only return ones
    that exist. The returned keys are always basenames so callers
    can write them under a stable layout in the pack dir.
    """
    if not source_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for name in names:
        p = source_dir / name
        if p.is_file():
            out[name] = p
    return out


def copy_hf_metadata(
    source_dir: Path,
    pack_dir: Path,
    *,
    names: tuple[str, ...] = HF_METADATA_FILES,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    """Copy the HF metadata bundle from ``source_dir`` into ``pack_dir``.

    Returns a name->info dict suitable for embedding in the
    manifest's ``meta.hf_metadata`` field. The info dict has keys:

      * ``path``      — relative path inside ``pack_dir`` (always the
        basename, so the pack stays self-contained)
      * ``size``      — file size in bytes
      * ``sha256``    — hex digest of the file contents
      * ``kind``      — ``"config"`` / ``"tokenizer"`` (group label)
      * ``copied``    — bool, false if the file already existed and
        ``overwrite=False``

    Skips a file silently if it doesn't exist in ``source_dir``;
    raises if a copy fails (e.g. permission denied, disk full).
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, object]] = {}
    found = find_hf_metadata_files(source_dir, names=names)
    for name, src in found.items():
        dst = pack_dir / name
        copied = True
        if dst.exists() and not overwrite:
            # Re-emit metadata for the existing file; do not clobber.
            copied = False
        else:
            dst.write_bytes(src.read_bytes())
        h = hashlib.sha256()
        h.update(dst.read_bytes())
        out[name] = {
            "path": name,
            "size": int(dst.stat().st_size),
            "sha256": h.hexdigest(),
            "kind": HF_METADATA_KIND.get(name, "other"),
            "copied": copied,
        }
    return out


def load_hf_metadata(pack_dir: Path) -> dict[str, Path]:
    """Find the HF metadata files that were copied next to weights.bin.

    Inverse of :func:`copy_hf_metadata`. Used by downstream code that
    wants to instantiate the model + tokenizer from a pack directory
    without re-downloading from Hugging Face.
    """
    return find_hf_metadata_files(pack_dir)
