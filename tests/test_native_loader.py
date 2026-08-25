"""Native LUT2 loader safety tests."""

from __future__ import annotations


def test_windows_dependency_search_does_not_walk_process_path(tmp_path, monkeypatch) -> None:
    from rwkv_ssd.native.lut2_gather_loader import _windows_dll_dependency_dirs

    lib_path = tmp_path / "lut2_gather.dll"
    lib_path.write_bytes(b"")
    inaccessible = tmp_path / "inaccessible"
    monkeypatch.setenv("PATH", str(inaccessible))
    monkeypatch.delenv("RWKV_LUT2_DLL_DIRS", raising=False)

    assert _windows_dll_dependency_dirs(lib_path) == [tmp_path]


def test_windows_dependency_search_accepts_explicit_directory(tmp_path, monkeypatch) -> None:
    from rwkv_ssd.native.lut2_gather_loader import _windows_dll_dependency_dirs

    lib_path = tmp_path / "native" / "lut2_gather.dll"
    lib_path.parent.mkdir()
    lib_path.write_bytes(b"")
    dependency = tmp_path / "compiler" / "bin"
    dependency.mkdir(parents=True)
    (dependency / "libgomp-1.dll").write_bytes(b"")
    monkeypatch.setenv("RWKV_LUT2_DLL_DIRS", str(dependency))

    dirs = _windows_dll_dependency_dirs(lib_path)
    assert dirs == [lib_path.parent, dependency]
