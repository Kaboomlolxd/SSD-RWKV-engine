"""Runtime pack integrity checks (no dependency on CLI tools)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rwkv_ssd.runtime.device import SUPPORTED_MANIFEST_VERSION
from rwkv_ssd.runtime.errors import ManifestVersionError, PackError
from rwkv_ssd.runtime.manifest import Manifest


def verify_pack(pack_dir: Path, *, check_hash: bool = True) -> tuple[bool, list[str]]:
    """
    Verify manifest offsets and optional SHA-256.

    Returns (ok, messages) where messages contains FAIL/WARN lines.
    """
    messages: list[str] = []
    ok = True
    try:
        manifest = Manifest.load(pack_dir, require_quality_certificate=True)
    except (FileNotFoundError, OSError, ValueError, TypeError) as e:
        return False, [f"FAIL: {e}"]

    if manifest.version != SUPPORTED_MANIFEST_VERSION:
        messages.append(
            f"FAIL: manifest version {manifest.version} != {SUPPORTED_MANIFEST_VERSION}"
        )
        ok = False

    file_sizes: dict[Path, int] = {}
    logical_sizes: dict[Path, int] = {}
    weight_paths: list[Path] = [manifest.weights_path]
    weight_paths.extend(
        path for path in manifest.shard_files if path not in weight_paths
    )
    weight_paths.extend(
        path
        for path in (manifest.shard_path_for(tensor) for tensor in manifest.tensors)
        if path not in weight_paths
    )
    for path in weight_paths:
        try:
            file_sizes[path] = path.stat().st_size
        except OSError as exc:
            messages.append(f"FAIL: cannot stat weights file {path}: {exc}")
            ok = False
    # Whole-pack compression preserves manifest offsets relative to the
    # decompressed image, so bounds checks use the logical size while the
    # physical file size above remains useful for the final report/hash.
    if (
        not manifest.is_sharded()
        and str(manifest.meta.get("weights_compression", "")).lower() == "zstd"
    ):
        try:
            logical_sizes[manifest.weights_path] = int(
                manifest.meta["weights_uncompressed_bytes"]
            )
        except (KeyError, TypeError, ValueError):
            messages.append(
                "FAIL: zstd pack is missing a valid weights_uncompressed_bytes value"
            )
            ok = False

    for tensor in manifest.tensors:
        path = manifest.shard_path_for(tensor)
        size = logical_sizes.get(path, file_sizes.get(path))
        if size is None:
            messages.append(f"FAIL {tensor.name}: backing file not found: {path}")
            ok = False
            continue
        if tensor.offset < 0 or tensor.length < 0:
            messages.append(
                f"FAIL {tensor.name}: negative offset/length "
                f"({tensor.offset}, {tensor.length})"
            )
            ok = False
            continue
        end = tensor.offset + tensor.length
        if end > size:
            messages.append(
                f"FAIL {tensor.name}: end {end} > file size {size} ({path.name})"
            )
            ok = False
        if tensor.alignment <= 0:
            messages.append(
                f"FAIL {tensor.name}: alignment must be positive, got {tensor.alignment}"
            )
            ok = False
        elif tensor.offset % tensor.alignment != 0:
            messages.append(
                f"WARN {tensor.name}: offset {tensor.offset} not aligned to {tensor.alignment}"
            )

    if check_hash:
        expected_by_file = manifest.meta.get("weights_sha256_by_file")
        if manifest.is_sharded() and isinstance(expected_by_file, dict):
            for path in weight_paths:
                expected = expected_by_file.get(_relative_file_key(manifest, path))
                if expected is None:
                    expected = expected_by_file.get(path.name)
                if not expected:
                    messages.append(f"WARN: no SHA-256 recorded for {path.name}")
                    continue
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    messages.append(f"FAIL: cannot hash {path}: {exc}")
                    ok = False
                    continue
                if digest != expected:
                    messages.append(f"FAIL: SHA-256 mismatch for {path.name}")
                    ok = False
        elif not manifest.is_sharded() and manifest.meta.get("weights_sha256"):
            digest = hashlib.sha256(manifest.weights_path.read_bytes()).hexdigest()
            expected = manifest.meta["weights_sha256"]
            if digest != expected:
                messages.append("FAIL: weights_sha256 mismatch")
                ok = False
        elif manifest.is_sharded() and manifest.meta.get("weights_sha256"):
            # A legacy source hash describes the pre-sharding weights.bin and
            # cannot validate any one shard. Do not compare it to shard 0.
            messages.append(
                "WARN: legacy weights_sha256 ignored for sharded pack; "
                "rebuild shard hashes with shard_pack"
            )

    if ok:
        total_size = sum(file_sizes.values())
        messages.append(
            f"OK: {len(manifest.tensors)} tensors, {total_size} bytes, "
            f"version={manifest.version}"
        )
    return ok, messages


def _relative_file_key(manifest: Manifest, path: Path) -> str:
    """Return the manifest's normalized relative name for a weights file."""
    try:
        root = manifest.pack_dir or manifest.weights_path.parent
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def verify_pack_or_raise(pack_dir: Path, *, check_hash: bool = True) -> None:
    ok, messages = verify_pack(pack_dir, check_hash=check_hash)
    if ok:
        return
    detail = "; ".join(m for m in messages if m.startswith("FAIL"))
    raise PackError(detail or f"pack verification failed: {pack_dir}")


def load_manifest_checked(pack_dir: Path, *, check_hash: bool = True) -> Manifest:
    """Load manifest after version and integrity checks."""
    try:
        manifest = Manifest.load(pack_dir, require_quality_certificate=True)
    except (FileNotFoundError, OSError, ValueError, TypeError) as e:
        raise PackError(f"pack not found or incomplete: {pack_dir}") from e

    if manifest.version != SUPPORTED_MANIFEST_VERSION:
        raise ManifestVersionError(
            f"manifest version {manifest.version} != supported {SUPPORTED_MANIFEST_VERSION}"
        )

    verify_pack_or_raise(pack_dir, check_hash=check_hash)
    return manifest
