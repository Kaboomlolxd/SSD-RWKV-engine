"""Release preflight contract tests."""

from __future__ import annotations

import json
import hashlib

from rwkv_ssd.tools.preflight import main as preflight_main, run_preflight


def test_synthetic_preflight_checks_pack_integrity(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="synthetic")
    assert result["passed"] is True
    assert {item["name"] for item in result["checks"]} >= {
        "pack",
        "pack_integrity",
        "lut2_native",
    }


def test_preflight_reports_reproducibility_identity(synthetic_pack, tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-for-preflight")
    result = run_preflight(
        synthetic_pack,
        backend="synthetic",
        checkpoint=checkpoint,
    )

    identity = result["identity"]
    assert identity["pack_identity_sha256"]
    assert identity["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert identity["native_abi"] is None
    assert any(item["name"] == "checkpoint_identity" for item in result["checks"])


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


def test_preflight_rejects_unknown_runtime_backend(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="unsupported")
    assert result["passed"] is False
    assert any(
        item["name"] == "backend" and item["ok"] is False
        for item in result["checks"]
    )


def test_preflight_text_output_has_remediation_for_missing_backend(
    synthetic_pack, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["rwkv-ssd-preflight", "--pack", str(synthetic_pack), "--backend", "rwkvcpp"],
    )
    try:
        preflight_main()
    except SystemExit as exc:
        assert exc.code == 1
    output = capsys.readouterr().out
    assert "CPU preflight" in output
    assert "Convert the matching checkpoint to GGML" in output
    assert "Next steps:" in output


def test_preflight_accepts_rwkv_backend_aliases(synthetic_pack) -> None:
    result = run_preflight(synthetic_pack, backend="cpp")
    assert result["backend"] == "cpp"
    assert any(item["name"] == "rwkvcpp_root" for item in result["checks"])
