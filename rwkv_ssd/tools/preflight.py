"""Validate a released pack and its optional backend assets before serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.backends.rwkvcpp import (
    _resolve_ggml_path,
    find_rwkvcpp_dll,
    find_rwkvcpp_root,
)
from rwkv_ssd.native.lut2_gather_loader import native_lib_path
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_profiles import resolve_trinity_pack
from rwkv_ssd.runtime.pack_verify import verify_pack
from rwkv_ssd.runtime.state_envelope import model_fingerprint, tokenizer_fingerprint


def _sha256_path(path: Path) -> str:
    """Hash a checkpoint file or a deterministic directory of shards."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            return "missing"
        root = path.resolve()
        for item in files:
            digest.update(item.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    return "missing"


def run_preflight(
    pack_dir: str | Path,
    *,
    backend: str = "synthetic",
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    requested_root = Path(pack_dir)
    root = resolve_trinity_pack(requested_root)
    checks: list[dict[str, Any]] = []
    identity: dict[str, Any] = {
        "pack_identity_sha256": None,
        "codec": None,
        "pack_layout": None,
        "checkpoint_sha256": None,
        "checkpoint_sha256_expected": None,
        "tokenizer_sha256": None,
        "tokenizer_version": None,
        "native_abi": None,
        "native_library_sha256": None,
    }

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if root != requested_root:
        check("profile", True, f"pack profile redirected to {root}")

    try:
        manifest = Manifest.load(root, require_quality_certificate=True)
        identity["codec"] = manifest.meta.get("pack_codec", "none")
        identity["pack_layout"] = manifest.meta.get("pack_layout", "default")
        identity["checkpoint_sha256_expected"] = manifest.meta.get(
            "checkpoint_sha256"
        )
        identity["checkpoint_sha256"] = identity["checkpoint_sha256_expected"]
        check(
            "pack",
            True,
            f"{manifest.model_family} / {manifest.meta.get('pack_codec', 'none')}",
        )
    except Exception as exc:  # noqa: BLE001 - preflight must report all failures
        check("pack", False, str(exc))
        manifest = None

    try:
        identity["pack_identity_sha256"] = model_fingerprint(root)
        check("pack_identity", True, identity["pack_identity_sha256"])
    except (FileNotFoundError, OSError, ValueError) as exc:
        check("pack_identity", False, str(exc))

    if checkpoint is not None:
        path = Path(checkpoint)
        check("checkpoint", path.is_file() or path.is_dir(), str(path))
        if path.is_file() or path.is_dir():
            identity["checkpoint_sha256"] = _sha256_path(path)
            check("checkpoint_identity", True, identity["checkpoint_sha256"])
            expected = identity["checkpoint_sha256_expected"]
            if expected:
                check(
                    "checkpoint_compatibility",
                    identity["checkpoint_sha256"] == expected,
                    f"expected {expected}, got {identity['checkpoint_sha256']}",
                )

    normalized = backend.strip().lower()
    backend_aliases = {
        "synthetic": "synthetic",
        "reference": "synthetic",
        "mock": "synthetic",
        "chatrwkv": "chatrwkv",
        "chat": "chatrwkv",
        "rwkv7": "chatrwkv",
        "rwkvcpp": "rwkvcpp",
        "cpp": "rwkvcpp",
    }
    canonical_backend = backend_aliases.get(normalized)
    if canonical_backend is None:
        check(
            "backend",
            False,
            f"unsupported runtime backend {backend!r}; choose synthetic, "
            "chatrwkv, or rwkvcpp",
        )
    elif canonical_backend == "chatrwkv":
        root_path = find_chatrwkv_root()
        check("chatrwkv", root_path is not None, str(root_path or "not found"))
    elif canonical_backend == "rwkvcpp":
        identity["native_abi"] = "ggml-weight-bridge-v2 + native-layer-streaming"
        cpp_root = find_rwkvcpp_root()
        cpp_dll = find_rwkvcpp_dll(cpp_root) if cpp_root is not None else None
        check("rwkvcpp_root", cpp_root is not None, str(cpp_root or "not found"))
        check("rwkvcpp_dll", cpp_dll is not None, str(cpp_dll or "not found"))
        if cpp_dll is not None:
            identity["native_library_sha256"] = _sha256_path(cpp_dll)
            check(
                "rwkvcpp_dll_sha256",
                True,
                identity["native_library_sha256"],
            )
        if checkpoint is None:
            check("ggml", False, "rwkvcpp requires --checkpoint or RWKVCPP_GGML_PATH")
        else:
            try:
                ggml = _resolve_ggml_path(str(Path(checkpoint)))
            except (FileNotFoundError, OSError, ValueError) as exc:
                check("ggml", False, str(exc))
            else:
                check("ggml", ggml.is_file(), str(ggml))

    if manifest is not None:
        identity["tokenizer_sha256"] = tokenizer_fingerprint(root)
        identity["tokenizer_version"] = (
            manifest.meta.get("tokenizer_version")
            or manifest.meta.get("tokenizer_revision")
            or manifest.meta.get("tokenizer_name")
        )
        if identity["tokenizer_sha256"] == "missing":
            check("tokenizer_identity", True, "no tokenizer assets declared in pack")
        else:
            check("tokenizer_identity", True, identity["tokenizer_sha256"])
        valid, messages = verify_pack(root, check_hash=True)
        check(
            "pack_integrity",
            valid,
            "; ".join(messages[-3:]) if messages else "verified",
        )
        codec = str(manifest.meta.get("pack_codec", "none")).lower()
        if codec in {"trinity", "trinity_lut2", "trinity_layer"}:
            native = native_lib_path()
            check("lut2_native", native is not None, str(native or "not found"))
        else:
            check("lut2_native", True, "not required by pack codec")
        for key in ("tokenizer_file", "tokenizer_path", "ggml_file", "ggml_path"):
            declared = manifest.meta.get(key)
            if not declared:
                continue
            path = Path(str(declared))
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                check(key, False, f"declared path escapes pack root: {declared}")
            else:
                check(key, path.is_file(), str(path))

    passed = all(bool(item["ok"]) for item in checks)
    return {
        "passed": passed,
        "pack": str(root),
        "backend": normalized,
        "identity": identity,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["synthetic", "chatrwkv", "rwkvcpp"],
        default="synthetic",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_preflight(args.pack, backend=args.backend, checkpoint=args.checkpoint)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result["checks"]:
            print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
