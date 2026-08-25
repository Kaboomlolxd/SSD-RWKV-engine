"""Integration tests with the bundled RWKV-7 0.1B checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.tools.verify_pack import verify
from tests.conftest import REPO

# Back-compat for tests that import these names.
DEFAULT_CKPT = REPO / "test_model" / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
DEFAULT_PACK = REPO / "test_model" / "runtime_pack"


def _require_checkpoint() -> Path:
    from tests.conftest import require_checkpoint

    return require_checkpoint()


def test_pack_real_rwkv7(real_pack: Path) -> None:
    assert verify(real_pack)
    import json

    meta = json.loads((real_pack / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("n_layer") == 12
    assert meta.get("rwkv_version") == 7
    assert meta.get("primary_dtype") == "bfloat16"
    manifest = json.loads((real_pack / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == 402


@pytest.mark.integration
def test_bench_io_real_pack(real_pack: Path) -> None:
    r = subprocess.run(
        [sys.executable, "archive/bench/bench_io.py", "--model", str(real_pack), "--trials", "1"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    assert "bw_GB_s" in r.stdout


@pytest.mark.chatrwkv
@pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not available")
def test_chatrwkv_resident_real_model(real_pack: Path, test_checkpoint: Path) -> None:
    cfg = EngineConfig(
        pack_dir=real_pack,
        backend="chatrwkv",
        mode="resident",
        device="cpu",
        max_tokens=8,
        checkpoint_path=str(test_checkpoint),
        strategy="cpu bf16",
        greedy=True,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        out = engine.generate("Hello")
        assert isinstance(out, str)
        assert len(out.strip()) > 0
        assert engine.metrics.tokens_generated == 8
    finally:
        engine.close()


@pytest.mark.chatrwkv
@pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not available")
def test_chatrwkv_resident_repeatability(real_pack: Path, test_checkpoint: Path) -> None:
    cfg = EngineConfig(
        pack_dir=real_pack,
        backend="chatrwkv",
        mode="resident",
        device="cpu",
        max_tokens=4,
        checkpoint_path=str(test_checkpoint),
        strategy="cpu bf16",
    )
    engine = InferenceEngine(cfg)
    engine.load()
    try:
        outputs = [engine.generate("Hi") for _ in range(3)]
    finally:
        engine.close()
    assert len(set(outputs)) == 1
