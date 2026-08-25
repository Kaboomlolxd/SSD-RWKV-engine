"""LUT gather kernel roundtrip."""

from __future__ import annotations

import numpy as np
import torch

from rwkv_ssd.runtime.lut_gather_kernel import gather_lut2_into, resolve_lut_kernel, unpack_2bit_into
from rwkv_ssd.runtime.trinity_codec import encode_trinity_lut2


def test_numba_gather_matches_numpy() -> None:
    import os

    os.environ["RWKV_LUT_KERNEL"] = "numpy"
    t = torch.randn(64, 64, dtype=torch.bfloat16)
    raw = encode_trinity_lut2(t)
    codebook = np.frombuffer(raw, dtype=np.float32, count=4, offset=4)
    packed = np.frombuffer(raw, dtype=np.uint8, offset=20)
    n = int(t.numel())
    idx = np.empty(n, dtype=np.uint8)
    unpack_2bit_into(idx, packed)
    out_np = np.empty(n, dtype=np.float32)
    gather_lut2_into(out_np, 0, codebook, idx)

    os.environ["RWKV_LUT_KERNEL"] = "numba"
    out_nb = np.empty(n, dtype=np.float32)
    gather_lut2_into(out_nb, 0, codebook, idx)
    np.testing.assert_allclose(out_np, out_nb, rtol=0, atol=0)
    assert resolve_lut_kernel() == "numba"


def test_numba_bf16_gather_matches_torch_cast() -> None:
    import os

    os.environ["RWKV_LUT_KERNEL"] = "numba"
    t = torch.randn(64, 64, dtype=torch.bfloat16)
    raw = encode_trinity_lut2(t)
    codebook = np.frombuffer(raw, dtype=np.float32, count=4, offset=4)
    packed = np.frombuffer(raw, dtype=np.uint8, offset=20)
    n = int(t.numel())
    from rwkv_ssd.runtime.lut_gather_kernel import gather_lut2_layer

    flat_f = np.empty(n, dtype=np.float32)
    gather_lut2_layer(flat_f, [(0, codebook, packed, n)])
    ref = torch.from_numpy(flat_f).to(torch.bfloat16).view(torch.uint16).numpy()

    flat_u16 = np.empty(n, dtype=np.uint16)
    gather_lut2_layer(flat_u16, [(0, codebook, packed, n)], bf16_out=True)
    np.testing.assert_array_equal(flat_u16, ref)


def test_native_kernel_has_standalone_unpack_fallback(monkeypatch) -> None:
    """Native fused gather selection must not break callers needing indices."""
    monkeypatch.setenv("RWKV_LUT_KERNEL", "native")
    packed = np.array([0xE4, 0x1B], dtype=np.uint8)
    out = np.empty(8, dtype=np.uint8)
    unpack_2bit_into(out, packed)
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint8))


def test_packed_device_gather_matches_cpu() -> None:
    from rwkv_ssd.runtime.trinity_accel import gather_lut2_packed_on_device
    from rwkv_ssd.runtime.manifest import TensorEntry

    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    raw = encode_trinity_lut2(t)
    codebook = np.frombuffer(raw, dtype=np.float32, count=4, offset=4)
    packed = np.frombuffer(raw, dtype=np.uint8, offset=20)
    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=[2, 2],
        offset=0,
        length=len(raw),
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    out = gather_lut2_packed_on_device(codebook, packed, entry, torch.device("cpu"))
    indices = np.empty(4, dtype=np.uint8)
    unpack_2bit_into(indices, packed)
    reference = torch.from_numpy(codebook[indices]).to(torch.bfloat16).reshape(2, 2)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
