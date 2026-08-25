"""Shared fixtures — session-scoped packs and fast default selection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack

# Stable imports before ChatRWKV loads (matches engine).
os.environ.setdefault("RWKV_JIT_ON", "0")
os.environ.setdefault("RWKV_V7_ON", "1")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "test_model" / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
DEFAULT_PACK = REPO / "test_model" / "runtime_pack"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "chatrwkv: needs ChatRWKV + RWKV-7 checkpoint (~30s+)",
    )
    config.addinivalue_line(
        "markers",
        "integration: subprocess or heavy I/O smoke",
    )


def require_checkpoint() -> Path:
    ckpt = Path(os.environ.get("RWKV_SSD_TEST_CKPT", DEFAULT_CKPT))
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not found: {ckpt}")
    return ckpt


@pytest.fixture(scope="session")
def synthetic_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn")
    return create_synthetic_pack(base / "pack", quiet=True)


@pytest.fixture
def mutable_synthetic_pack(tmp_path: Path) -> Path:
    """Per-test pack for cases that rewrite manifest/meta (do not use session pack)."""
    return create_synthetic_pack(tmp_path / "pack", quiet=True)


@pytest.fixture(scope="session")
def synthetic_pack_deep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn_deep")
    return create_synthetic_pack(
        base / "pack", n_layer=8, n_embd=64, seed=7, quiet=True
    )


@pytest.fixture(scope="session")
def synthetic_pack_u8(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn_u8")
    return create_synthetic_pack(
        base / "u8", pack_codec="scale_u8", pack_layout="layer_grouped", quiet=True
    )


@pytest.fixture(scope="session")
def synthetic_pack_u4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn_u4")
    return create_synthetic_pack(
        base / "u4", pack_codec="scale_u4", pack_layout="layer_grouped", quiet=True
    )


@pytest.fixture(scope="session")
def synthetic_pack_trinity_lut2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn_tr2")
    return create_synthetic_pack(
        base / "tr2",
        pack_codec="trinity_lut2",
        pack_layout="layer_grouped",
        quiet=True,
    )


@pytest.fixture(scope="session")
def synthetic_pack_trinity(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("syn_tra")
    return create_synthetic_pack(
        base / "tra",
        pack_codec="trinity",
        pack_layout="layer_grouped",
        quiet=True,
    )


@pytest.fixture(scope="session")
def real_pack() -> Path:
    from rwkv_ssd.tools.pack_runtime import pack
    from rwkv_ssd.tools.verify_pack import verify

    ckpt = require_checkpoint()
    pack_dir = Path(os.environ.get("RWKV_SSD_TEST_PACK", DEFAULT_PACK))
    if not verify(pack_dir, quiet=True):
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack(ckpt, pack_dir, model_family="rwkv7", quiet=True)
        assert verify(pack_dir, quiet=True)
    return pack_dir


@pytest.fixture(scope="session")
def test_checkpoint() -> Path:
    return require_checkpoint()
