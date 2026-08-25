"""Release preflight contract tests."""

from __future__ import annotations

import json

from rwkv_ssd.tools.preflight import run_preflight


def test_synthetic_preflight_checks_pack_integrity(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="synthetic")
    assert result["passed"] is True
    assert {item["name"] for item in result["checks"]} >= {
        "pack",
        "pack_integrity",
        "lut2_native",
    }


def test_preflight_reports_manifest_escape(mutable_synthetic_pack) -> None:
    path = mutable_synthetic_pack / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["weights_file"] = "../outside.bin"
    path.write_text(json.dumps(raw), encoding="utf-8")
    result = run_preflight(mutable_synthetic_pack, backend="synthetic")
    assert result["passed"] is False
    assert any(
        item["name"] == "pack" and "escapes pack root" in item["detail"]
        for item in result["checks"]
    )


def test_preflight_rejects_archived_or_unknown_runtime_backend(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="mamba2")
    assert result["passed"] is False
    assert any(
        item["name"] == "backend" and item["ok"] is False
        for item in result["checks"]
    )


def test_preflight_accepts_rwkv_backend_aliases(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="cpp")
    assert result["backend"] == "cpp"
    assert any(item["name"] == "rwkvcpp_root" for item in result["checks"])
