"""V1 phase 1: packed layer tensors match ChatRWKV model.z after RWKV-7 transforms."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.runtime.layer_keys import entries_for_layer
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.rwkv7_weights import prepare_rwkv7_tensor_for_z, tensors_from_entries
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes
from rwkv_ssd.runtime.weight_store import open_weight_store
pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not installed"),
]


def test_packed_layer_matches_model_z(real_pack: Path, test_checkpoint: Path) -> None:
    import sys

    os.environ.setdefault("RWKV_V7_ON", "1")
    os.environ.setdefault("RWKV_JIT_ON", "0")
    root = find_chatrwkv_root()
    assert root is not None
    src = root / "rwkv_pip_package" / "src"
    if not src.is_dir():
        src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from rwkv.model import RWKV  # type: ignore[import-untyped]

    ckpt = test_checkpoint
    base = str(ckpt).replace(".pth", "")
    model = RWKV(base, strategy="cpu bf16")
    z = model.z

    manifest = Manifest.load(real_pack)
    store = open_weight_store(manifest.weights_path)
    try:
        layer_id = 3
        entries = entries_for_layer(manifest.tensors, layer_id)
        assert entries, f"no tensors for layer {layer_id}"
        raw = {e.name: store.read_bytes(e) for e in entries}
        device = next(iter(z.values())).device
        prepared = tensors_from_entries(raw, entries, torch.device(device))
        for name, tensor in prepared.items():
            assert name in z, f"missing in model.z: {name}"
            ref = z[name]
            assert ref.shape == tensor.shape, f"{name} shape {tensor.shape} vs {ref.shape}"
            assert torch.allclose(ref.float(), tensor.float(), rtol=1e-2, atol=1e-2), name
    finally:
        store.close()


def test_prepare_rwkv7_transpose() -> None:
    t = torch.randn(4, 8)
    out = prepare_rwkv7_tensor_for_z("blocks.0.att.key.weight", t)
    assert out.shape == (8, 4)
