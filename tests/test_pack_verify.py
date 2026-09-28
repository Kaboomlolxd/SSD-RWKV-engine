"""Pack and verify tests (M0/M2)."""

from pathlib import Path
import hashlib
import json

import pytest
import torch

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.pack_profiles import resolve_trinity_pack
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack
from rwkv_ssd.tools.pack_runtime import pack
from rwkv_ssd.tools.verify_pack import verify


def test_create_synthetic_pack(synthetic_pack: Path) -> None:
    assert verify(synthetic_pack)
    m = Manifest.load(synthetic_pack)
    assert m.meta.get("model_type") == "synthetic_rwkv"
    assert m.meta.get("n_layer") == 4
    assert m.version == 1


def test_pack_roundtrip(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    torch.save({"blocks.0.weight": torch.randn(8, 8)}, ckpt)
    out = tmp_path / "packed"
    pack(ckpt, out)
    assert verify(out)
    m = Manifest.load(out)
    assert len(m.tensors) == 1
    assert m.meta.get("weights_sha256")
    assert m.meta["checkpoint_sha256"] == hashlib.sha256(
        ckpt.read_bytes()
    ).hexdigest()


def test_trinity_default_uses_native_safe_mixed_precision(tmp_path: Path) -> None:
    ckpt = tmp_path / "mixed.pt"
    torch.save(
        {
            "blocks.0.att.x_k": torch.randn(64),
            "blocks.0.att.key.weight": torch.randn(64, 64),
            "emb.weight": torch.randn(64, 64),
        },
        ckpt,
    )
    out = tmp_path / "mixed"
    pack(ckpt, out, pack_codec="trinity_lut2", hash_weights=False, quiet=True)
    manifest = Manifest.load(out)
    codecs = {entry.name: entry.dequant for entry in manifest.tensors}
    assert codecs["blocks.0.att.x_k"] == "none"
    assert codecs["blocks.0.att.key.weight"] == "scale_u8_grouped"
    assert codecs["emb.weight"] == "scale_u8_grouped"
    assert manifest.meta["trinity_quality_preset"] == "native_safe"
    assert manifest.meta["scale_group_size"] == 64
    assert manifest.meta["trinity_group_size"] == 128


def test_verify_rejects_truncated(tmp_path: Path) -> None:
    pack_dir = create_synthetic_pack(tmp_path / "p")
    weights = pack_dir / "weights.bin"
    weights.write_bytes(weights.read_bytes()[:100])
    assert not verify(pack_dir)


@pytest.mark.parametrize("field", ["weights_file", "weights_files"])
def test_manifest_rejects_paths_outside_pack_root(tmp_path: Path, field: str) -> None:
    pack_dir = create_synthetic_pack(tmp_path / "p")
    manifest_path = pack_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "weights_file":
        raw["weights_file"] = "../outside.bin"
    else:
        raw["weights_files"] = ["../outside.bin"]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes pack root"):
        Manifest.load(pack_dir)


def test_auto_trinity_profile_prefers_grouped_pack_over_safe_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "trinity_lut2_0.1b"
    grouped = tmp_path / "trinity_grouped_0.1b"
    safe = tmp_path / "trinity_safe_0.1b"
    requested.mkdir()
    grouped.mkdir()
    safe.mkdir()
    monkeypatch.delenv("RWKV_PACK_PROFILE", raising=False)

    assert resolve_trinity_pack(requested) == safe

    (grouped / "quality_certificate.json").write_text("{}", encoding="utf-8")
    assert resolve_trinity_pack(requested) == grouped

    (grouped / "quality_certificate.json").unlink()
    grouped.rmdir()
    assert resolve_trinity_pack(requested) == safe

    grouped.mkdir()
    (requested / "quality_certificate.json").write_text("{}", encoding="utf-8")
    assert resolve_trinity_pack(requested) == requested


def test_explicit_grouped_profile_selects_grouped_sibling(tmp_path: Path) -> None:
    requested = tmp_path / "trinity_lut2_0.1b"
    grouped = tmp_path / "trinity_grouped_0.1b"
    requested.mkdir()
    grouped.mkdir()

    assert resolve_trinity_pack(requested, profile="grouped") == grouped


def test_explicit_lut2_profile_does_not_silently_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "trinity_lut2_0.1b"
    grouped = tmp_path / "trinity_grouped_0.1b"
    safe = tmp_path / "trinity_safe_0.1b"
    requested.mkdir()
    grouped.mkdir()
    safe.mkdir()
    monkeypatch.setenv("RWKV_PACK_PROFILE", "lut2")

    assert resolve_trinity_pack(requested) == requested


def test_auto_2_9b_profile_promotes_only_certified_grouped_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "runtime_pack_2.9b"
    grouped = tmp_path / "runtime_pack_2.9b_grouped_quality"
    requested.mkdir()
    grouped.mkdir()
    monkeypatch.delenv("RWKV_PACK_PROFILE", raising=False)

    assert resolve_trinity_pack(requested) == requested

    (grouped / "quality_certificate.json").write_text("{}", encoding="utf-8")
    assert resolve_trinity_pack(requested) == grouped


def test_explicit_2_9b_grouped_profile_selects_sibling(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "runtime_pack_2.9b"
    grouped = tmp_path / "runtime_pack_2.9b_grouped_quality"
    requested.mkdir()
    grouped.mkdir()

    assert resolve_trinity_pack(requested, profile="grouped") == grouped
