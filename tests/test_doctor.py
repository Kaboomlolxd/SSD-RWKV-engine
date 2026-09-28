from __future__ import annotations

from pathlib import Path

from rwkv_ssd.tools import doctor


def test_doctor_recommends_native_when_native_files_are_available(
    tmp_path: Path, monkeypatch
) -> None:
    cpp_root = tmp_path / "cpp"
    dll = tmp_path / "rwkv.dll"
    ggml = tmp_path / "model.bin"
    checkpoint = tmp_path / "model.pth"
    for path in (cpp_root, dll, ggml, checkpoint):
        if path.suffix:
            path.write_bytes(b"fixture")
        else:
            path.mkdir()
    monkeypatch.setattr(doctor, "find_chatrwkv_root", lambda: tmp_path / "ChatRWKV")
    monkeypatch.setattr(doctor, "find_rwkvcpp_root", lambda: cpp_root)
    monkeypatch.setattr(doctor, "find_rwkvcpp_dll", lambda _root: dll)
    monkeypatch.setattr(doctor, "_find_matching_ggml", lambda _path: ggml)

    report = doctor.build_report(checkpoint=checkpoint)

    assert report["recommended_backend"] == "rwkvcpp"
    assert "Fast native CPU" in report["backends"]["rwkvcpp"]["speed"]


def test_doctor_falls_back_to_reference_when_native_files_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    chat_root = tmp_path / "ChatRWKV"
    chat_root.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(doctor, "find_chatrwkv_root", lambda: chat_root)
    monkeypatch.setattr(doctor, "find_rwkvcpp_root", lambda: None)
    monkeypatch.setattr(doctor, "find_rwkvcpp_dll", lambda _root: None)
    monkeypatch.setattr(doctor, "_find_matching_ggml", lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing")))

    report = doctor.build_report(checkpoint=checkpoint)

    assert report["recommended_backend"] == "chatrwkv"
    assert "compatibility path" in report["recommendation_reason"]


def test_doctor_without_local_backend_files_gives_setup_guidance(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "find_chatrwkv_root", lambda: None)
    monkeypatch.setattr(doctor, "find_rwkvcpp_root", lambda: None)
    monkeypatch.setattr(doctor, "find_rwkvcpp_dll", lambda _root: None)

    report = doctor.build_report()

    assert report["recommended_backend"] is None
    assert report["system"]["logical_cpus"] >= 1
    assert "supply the missing artifacts" in report["recommendation_reason"]
