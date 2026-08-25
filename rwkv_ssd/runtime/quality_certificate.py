"""Manifest-bound quality certificates checked whenever a pack is loaded."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


_PATH_KEYS = (
    "tokenizer_file",
    "tokenizer_path",
    "tokenizer_json",
    "tokenizer_config",
    "ggml_file",
    "ggml_path",
    "native_library",
    "native_library_path",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_identity(raw: dict[str, Any]) -> str:
    normalized = deepcopy(raw)
    meta = normalized.get("meta")
    if isinstance(meta, dict):
        meta.pop("quality_certificate_file", None)
        meta.pop("quality_certificate_sha256", None)
    return _canonical_hash(normalized)


def _safe_path(root: Path, name: str, field: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"certificate {field} must be a non-empty path")
    path = (root / name.replace("\\", "/")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"certificate {field} escapes pack root: {name!r}") from exc
    if path == root.resolve():
        raise ValueError(f"certificate {field} must name a file")
    return path


def _declared_artifact_names(raw: dict[str, Any]) -> list[str]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    names: list[str] = []

    def add(value: Any, field: str) -> None:
        if isinstance(value, str) and value:
            names.append(value.replace("\\", "/"))
        elif value is not None:
            raise ValueError(f"certificate {field} must contain path strings")

    weights = raw.get("weights_files")
    if isinstance(weights, list):
        for value in weights:
            add(value, "weights_files")
    else:
        add(raw.get("weights_file", "weights.bin"), "weights_file")
    shadow = meta.get("shadow_files")
    if isinstance(shadow, list):
        for value in shadow:
            add(value, "shadow_files")
    elif meta.get("shadow_file"):
        add(meta.get("shadow_file"), "shadow_file")
    for key in _PATH_KEYS:
        value = meta.get(key)
        if isinstance(value, list):
            for item in value:
                add(item, key)
        elif value:
            add(value, key)
    # Preserve declaration order while rejecting duplicate identities.
    unique = list(dict.fromkeys(names))
    return unique


def _with_sidecar_meta(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Return an identity view with sidecar metadata resolved like Manifest."""
    merged = deepcopy(raw)
    meta = merged.setdefault("meta", {})
    sidecar = root / "meta.json"
    if sidecar.is_file():
        sidecar_raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(sidecar_raw, dict):
            raise ValueError("meta.json must be a JSON object")
        for key, value in sidecar_raw.items():
            if key != "pack_composition":
                meta.setdefault(key, value)
    return merged


def _artifact_hashes(root: Path, raw: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _declared_artifact_names(raw):
        path = _safe_path(root, name, "artifact path")
        if not path.is_file():
            raise FileNotFoundError(f"certificate artifact not found: {path}")
        hashes[name] = _sha256(path)
    return hashes


def _evaluate(metrics: dict[str, float], gates: dict[str, dict]) -> tuple[bool, dict]:
    results = {}
    passed = True
    for name, gate in gates.items():
        metric = str(gate.get("metric", name))
        value = float(metrics[metric])
        if "max" in gate:
            ok = value <= float(gate["max"])
        elif "min" in gate:
            ok = value >= float(gate["min"])
        else:
            raise ValueError(f"quality gate {name!r} needs min or max")
        results[name] = {"metric": metric, "value": value, **gate, "passed": ok}
        passed = passed and ok
    return passed, results


def issue_quality_certificate(
    pack_dir: str | Path,
    metrics: dict[str, float],
    gates: dict[str, dict],
    *,
    evidence_scope: str,
    allow_failed: bool = False,
) -> Path:
    root = Path(pack_dir).resolve()
    manifest_path = root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest.json must be a JSON object")
    identity_raw = _with_sidecar_meta(root, raw)
    hashes = _artifact_hashes(root, identity_raw)
    passed, results = _evaluate(metrics, gates)
    if not passed and not allow_failed:
        raise ValueError("quality certificate gates failed")
    declared_weights = identity_raw.get("weights_files")
    if not isinstance(declared_weights, list):
        declared_weights = [identity_raw.get("weights_file", "weights.bin")]
    normalized_weight_names = {
        str(name).replace("\\", "/") for name in declared_weights
    }
    sidecar = root / "meta.json"
    certificate = {
        "version": 2,
        "passed": passed,
        "evidence_scope": str(evidence_scope),
        "manifest_sha256": _manifest_identity(raw),
        "meta_json_sha256": (
            _canonical_hash(json.loads(sidecar.read_text(encoding="utf-8")))
            if sidecar.is_file()
            else None
        ),
        "artifact_sha256_by_file": hashes,
        # Keep this stable field for tooling written against certificate v1.
        "weights_sha256_by_file": {
            name: digest
            for name, digest in hashes.items()
            if name in normalized_weight_names
        },
        "metrics": {key: float(value) for key, value in metrics.items()},
        "gates": results,
    }
    target = root / "quality_certificate.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    raw.setdefault("meta", {})
    raw["meta"]["quality_certificate_file"] = target.name
    raw["meta"]["quality_certificate_sha256"] = _sha256(target)
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    return target


def verify_quality_certificate(pack_dir: str | Path, meta: dict) -> dict | None:
    root = Path(pack_dir).resolve()
    name = meta.get("quality_certificate_file")
    if not name:
        return None
    path = _safe_path(root, str(name), "quality_certificate_file")
    if not path.is_file():
        raise FileNotFoundError(f"quality certificate not found: {path}")
    expected = str(meta.get("quality_certificate_sha256", ""))
    actual = _sha256(path)
    if not expected or actual != expected:
        raise ValueError("quality certificate checksum mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("version", 0)) != 2:
        raise ValueError("unsupported quality certificate version; regenerate with v2")
    if not bool(raw.get("passed", False)):
        raise ValueError("quality certificate records failed gates")

    manifest_path = root / "manifest.json"
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("manifest_sha256") != _manifest_identity(manifest_raw):
        raise ValueError("quality certificate manifest identity mismatch")
    sidecar = root / "meta.json"
    expected_meta = raw.get("meta_json_sha256")
    actual_meta = (
        _canonical_hash(json.loads(sidecar.read_text(encoding="utf-8")))
        if sidecar.is_file()
        else None
    )
    if expected_meta != actual_meta:
        raise ValueError("quality certificate metadata identity mismatch")

    expected_files = raw.get("artifact_sha256_by_file")
    if not isinstance(expected_files, dict):
        raise ValueError("quality certificate has no complete artifact identity")
    current_files = _declared_artifact_names(_with_sidecar_meta(root, manifest_raw))
    if set(expected_files) != set(current_files):
        raise ValueError("quality certificate artifact declaration mismatch")
    for name, digest in expected_files.items():
        file_path = _safe_path(root, str(name), "artifact path")
        if not file_path.is_file() or _sha256(file_path) != str(digest):
            raise ValueError(f"quality certificate artifact identity mismatch: {name}")
    return raw


__all__ = ["issue_quality_certificate", "verify_quality_certificate"]
