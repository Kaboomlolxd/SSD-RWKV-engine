"""Trinity stride / quant audit vs FP16 reference pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.tools.trinity_stride_audit import audit_layer
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store

FP16_0_1B = Path("test_model/runtime_pack")
TRINITY_0_1B = Path("test_model/trinity_eval/trinity_lut2_0.1b")


@pytest.mark.skipif(
    not FP16_0_1B.is_dir()
    or not TRINITY_0_1B.is_dir()
    or not (FP16_0_1B / "weights.bin").is_file()
    or not (TRINITY_0_1B / "weights.bin").is_file(),
    reason="0.1B eval packs not built",
)
def test_trinity_lut2_stride_matches_fp16_layer0() -> None:
    fp_m = Manifest.load(FP16_0_1B)
    tr_m = Manifest.load(TRINITY_0_1B)
    fp_store = open_weight_store(fp_m.weights_path)
    tr_store = open_weight_store(tr_m.weights_path)
    try:
        audit = audit_layer(fp_m, tr_m, fp_store, tr_store, 0)
    finally:
        fp_store.close()
        tr_store.close()
    assert audit.tensors
    assert all(t.stride_ok for t in audit.tensors)
    weight_diffs = [
        t.max_abs_diff
        for t in audit.tensors
        if any(
            s in t.name
            for s in (
                "key.weight",
                "value.weight",
                "receptance.weight",
                "output.weight",
            )
        )
    ]
    assert weight_diffs
    assert max(weight_diffs) < 0.35


def test_audit_rejects_pack_size_mismatch(tmp_path: Path) -> None:
    from rwkv_ssd.runtime.manifest import TensorEntry

    small = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "a.bin",
        tensors=[],
        meta={"n_embd": 256, "n_layer": 2},
    )
    large = Manifest(
        version=1,
        model_family="rwkv7",
        weights_path=tmp_path / "b.bin",
        tensors=[],
        meta={"n_embd": 768, "n_layer": 12},
    )
    class _Null:
        def read_bytes(self, entry: TensorEntry) -> bytes:
            return b""

    with pytest.raises(ValueError, match="pack size mismatch"):
        audit_layer(small, large, _Null(), _Null(), 0)
