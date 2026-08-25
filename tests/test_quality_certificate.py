from __future__ import annotations

import json

import pytest

from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.quality_certificate import issue_quality_certificate


def test_passing_quality_certificate_is_enforced_at_manifest_load(
    mutable_synthetic_pack
) -> None:
    path = issue_quality_certificate(
        mutable_synthetic_pack,
        {"max_kl": 0.01, "top_k_overlap": 0.9},
        {
            "kl": {"metric": "max_kl", "max": 0.02},
            "overlap": {"metric": "top_k_overlap", "min": 0.8},
        },
        evidence_scope="synthetic unit",
    )
    assert path.is_file()
    manifest = Manifest.load(mutable_synthetic_pack)
    assert manifest.meta["quality_certificate_file"] == path.name


def test_failed_quality_certificate_prevents_pack_load(mutable_synthetic_pack) -> None:
    issue_quality_certificate(
        mutable_synthetic_pack,
        {"max_kl": 1.0},
        {"kl": {"metric": "max_kl", "max": 0.1}},
        evidence_scope="negative test",
        allow_failed=True,
    )
    with pytest.raises(ValueError, match="failed gates"):
        Manifest.load(mutable_synthetic_pack)


def test_certificate_or_weight_corruption_is_detected(mutable_synthetic_pack) -> None:
    cert = issue_quality_certificate(
        mutable_synthetic_pack,
        {"drift": 0.01},
        {"drift": {"max": 0.02}},
        evidence_scope="synthetic unit",
    )
    raw = json.loads(cert.read_text(encoding="utf-8"))
    raw["metrics"]["drift"] = 99
    cert.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        Manifest.load(mutable_synthetic_pack)


def test_certificate_binds_sidecar_metadata(mutable_synthetic_pack) -> None:
    cert = issue_quality_certificate(
        mutable_synthetic_pack,
        {"drift": 0.01},
        {"drift": {"max": 0.02}},
        evidence_scope="synthetic unit",
    )
    sidecar = mutable_synthetic_pack / "meta.json"
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["model_family"] = "mutated"
    sidecar.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata identity"):
        Manifest.load(mutable_synthetic_pack)


def test_certificate_normalizes_windows_artifact_paths(mutable_synthetic_pack) -> None:
    manifest_path = mutable_synthetic_pack / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_dir = mutable_synthetic_pack / "payload"
    payload_dir.mkdir()
    (mutable_synthetic_pack / "weights.bin").rename(payload_dir / "weights.bin")
    raw["weights_file"] = r"payload\weights.bin"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    cert = issue_quality_certificate(
        mutable_synthetic_pack,
        {"drift": 0.01},
        {"drift": {"max": 0.02}},
        evidence_scope="synthetic unit",
    )
    certificate = json.loads(cert.read_text(encoding="utf-8"))
    assert set(certificate["weights_sha256_by_file"]) == {"payload/weights.bin"}
    assert Manifest.load(mutable_synthetic_pack).weights_path.name == "weights.bin"


def test_runtime_certificate_policy_is_only_for_real_rwkv7_trinity(
    mutable_synthetic_pack
) -> None:
    manifest_path = mutable_synthetic_pack / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["model_family"] = "rwkv7"
    raw["meta"].update({"pack_codec": "trinity_lut2", "n_layer": 8, "vocab_size": 4096})
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="quality_certificate"):
        Manifest.load(mutable_synthetic_pack, require_quality_certificate=True)


@pytest.mark.parametrize("codec", ["scale_u8", "scale_u8_grouped", "scale_u4"])
def test_runtime_certificate_policy_covers_all_lossy_real_rwkv7_codecs(
    mutable_synthetic_pack, codec: str
) -> None:
    manifest_path = mutable_synthetic_pack / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["model_family"] = "rwkv7"
    raw["meta"].update({"pack_codec": codec, "n_layer": 8, "vocab_size": 4096})
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="quality_certificate"):
        Manifest.load(mutable_synthetic_pack, require_quality_certificate=True)
