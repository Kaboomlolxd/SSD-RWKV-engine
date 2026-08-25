"""I/O store bench smoke (M2)."""

import subprocess
import sys
from pathlib import Path


def test_bench_io(synthetic_pack: Path) -> None:
    r = subprocess.run(
        [
            sys.executable,
            "archive/bench/bench_io.py",
            "--model",
            str(synthetic_pack),
            "--trials",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stderr
    assert "bw_GB_s" in r.stdout
