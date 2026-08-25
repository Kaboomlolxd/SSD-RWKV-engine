"""Failure paths (M4 errors baseline)."""

from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.errors import (
    BackendNotAvailableError,
    CapabilityNotSupportedError,
    ManifestVersionError,
    PackError,
    StreamingNotSupportedError,
)
from rwkv_ssd.backends.factory import ensure_v0_backend


def test_missing_pack_fails(tmp_path: Path) -> None:
    cfg = EngineConfig(pack_dir=tmp_path / "nope", backend="synthetic")
    engine = InferenceEngine(cfg)
    with pytest.raises(PackError, match="pack not found or incomplete"):
        engine.load()


def test_bad_manifest_version(mutable_synthetic_pack: Path) -> None:
    import json

    manifest_path = mutable_synthetic_pack / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["version"] = 999
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    cfg = EngineConfig(pack_dir=mutable_synthetic_pack, backend="synthetic")
    engine = InferenceEngine(cfg)
    with pytest.raises(ManifestVersionError):
        engine.load()


def test_corrupt_weights_rejected(tmp_path: Path) -> None:
    from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

    pack = create_synthetic_pack(tmp_path / "pack", quiet=True)
    weights = pack / "weights.bin"
    weights.write_bytes(weights.read_bytes()[:100])

    engine = InferenceEngine(EngineConfig(pack_dir=pack, backend="synthetic"))
    with pytest.raises(PackError):
        engine.load()


def test_streaming_requires_supported_backend(mutable_synthetic_pack: Path) -> None:
    cfg = EngineConfig(
        pack_dir=mutable_synthetic_pack,
        backend="chatrwkv",
        mode="streaming",
        checkpoint_path="dummy.pth",
    )
    engine = InferenceEngine(cfg)
    engine.backend = type("B", (), {"generate_simple": lambda *a, **k: ""})()
    with pytest.raises(StreamingNotSupportedError):
        engine.generate("x")


def test_missing_checkpoint_on_chatrwkv_load(mutable_synthetic_pack: Path) -> None:
    import json

    meta_path = mutable_synthetic_pack / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("source_checkpoint", None)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest_path = mutable_synthetic_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("meta", {}).pop("source_checkpoint", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cfg = EngineConfig(
        pack_dir=mutable_synthetic_pack,
        backend="chatrwkv",
        mode="resident",
        checkpoint_path=None,
    )
    engine = InferenceEngine(cfg)
    with pytest.raises(ValueError, match="checkpoint"):
        engine.load()


def test_planned_backend_rejected() -> None:
    with pytest.raises(BackendNotAvailableError):
        ensure_v0_backend("albatross")


def test_rwkvcpp_rejects_deepembed_with_capability_error(mutable_synthetic_pack: Path) -> None:
    import json

    manifest_path = mutable_synthetic_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("meta", {}).update(
        {
            "deepembed": True,
            "deepembed_variant": "qkv_dea",
            "deepembed_sidecar_required": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    engine = InferenceEngine(
        EngineConfig(
            pack_dir=mutable_synthetic_pack,
            backend="rwkvcpp",
            mode="resident",
            checkpoint_path="dummy.pth",
        )
    )
    with pytest.raises(CapabilityNotSupportedError, match="deepembed"):
        engine.load()
