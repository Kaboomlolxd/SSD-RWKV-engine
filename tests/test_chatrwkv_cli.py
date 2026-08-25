"""ChatRWKV CLI usability tests (M1/M0)."""

import subprocess
import sys
import time
from pathlib import Path


def test_chatrwkv_missing_exits_fast(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(
        '{"version":1,"tensors":[],"meta":{}}', encoding="utf-8"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "CHATRWKV_ROOT"}
    env["RWKV_SSD_SKIP_BUNDLED_CHATRWKV"] = "1"
    t0 = time.perf_counter()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "--model",
            str(pack),
            "--backend",
            "chatrwkv",
            "--checkpoint",
            "dummy.pth",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
    )
    elapsed = time.perf_counter() - t0
    assert r.returncode != 0
    assert "ChatRWKV not found" in r.stderr
    assert elapsed < 15.0
