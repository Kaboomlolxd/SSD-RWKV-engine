"""Direct GEMV correctness for grouped runtime codecs."""

from __future__ import annotations

import torch
import pytest

from rwkv_ssd.runtime.lut_gemm_fused import (
    grouped_u8_transposed_gemv,
    grouped_u8_transposed_tmix_fused,
    lut2_gemv,
    lut2_tmix_gemv_batched,
    lut2_tmix_qkv_gemv_batched,
)
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_u8_grouped_to_tensor,
    encode_scale_u8_grouped,
)
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_tensor,
    encode_trinity_lut2,
)


def _entry(shape: tuple[int, int], length: int, codec: str) -> TensorEntry:
    return TensorEntry(
        name="blocks.0.att.key.weight",
        layer_id=0,
        dtype="float32",
        shape=list(shape),
        offset=0,
        length=length,
        alignment=1,
        residency="streamed",
        dequant=codec,
    )


def test_grouped_lut2_gemv_matches_decoded_transpose_and_residual(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_KERNEL", "numpy")
    torch.manual_seed(17)
    weight = torch.randn(7, 5)
    x = torch.randn(5)
    for algo in (
        "groupwise_kmeans",
        "groupwise_kmeans_residual",
        "groupwise_input_fp16",
        "groupwise_input_residual",
    ):
        blob = encode_trinity_lut2(weight, codebook=algo, group_size=4)
        entry = _entry((7, 5), len(blob), "trinity_lut2")
        got = lut2_gemv(blob, x, out_features=7, in_features=5)
        reference = decode_trinity_lut2_to_tensor(
            blob, entry, torch.device("cpu")
        ) @ x
        torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)


def test_grouped_u8_gemv_matches_decoded(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_KERNEL", "numpy")
    torch.manual_seed(19)
    weight = torch.randn(7, 5)
    x = torch.randn(5)
    blob = encode_scale_u8_grouped(weight, group_size=4)
    entry = _entry((7, 5), len(blob), "scale_u8_grouped")
    got = lut2_gemv(blob, x, out_features=7, in_features=5)
    reference = decode_scale_u8_grouped_to_tensor(
        blob, entry, torch.device("cpu")
    ) @ x
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)


def test_grouped_u8_tmix_batch_matches_four_gemvs(monkeypatch) -> None:
    """The batched path must stay equivalent to four ordinary SG8 GEMVs."""
    monkeypatch.setenv("RWKV_LUT_KERNEL", "numpy")
    torch.manual_seed(23)
    in_f, out_f = 13, 19
    xs = tuple(torch.randn(in_f) for _ in range(4))
    blobs = tuple(
        encode_scale_u8_grouped(torch.randn(out_f, in_f), group_size=7)
        for _ in range(4)
    )
    got = lut2_tmix_gemv_batched(
        blobs, xs, out_features=out_f, in_features=in_f
    )
    refs = tuple(
        lut2_gemv(blob, x, out_features=out_f, in_features=in_f)
        for blob, x in zip(blobs, xs)
    )
    for actual, expected in zip(got, refs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_grouped_u8_tmix_qkv_batch_matches_three_gemvs(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_LUT_KERNEL", "numpy")
    torch.manual_seed(29)
    in_f, out_f = 13, 19
    xs = tuple(torch.randn(in_f) for _ in range(3))
    blobs = tuple(
        encode_scale_u8_grouped(torch.randn(out_f, in_f), group_size=7)
        for _ in range(3)
    )
    got = lut2_tmix_qkv_gemv_batched(
        blobs, xs, out_features=out_f, in_features=in_f
    )
    refs = tuple(
        lut2_gemv(blob, x, out_features=out_f, in_features=in_f)
        for blob, x in zip(blobs, xs)
    )
    for actual, expected in zip(got, refs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_grouped_u8_fused_tmix_adapters_matches_torch_pipeline(monkeypatch) -> None:
    """The optional C adapter pipeline must preserve the Python reference."""
    from rwkv_ssd.native.lut2_gather_loader import lib

    native = lib()
    if native is None or not hasattr(
        native, "scale_u8_grouped_transposed_tmix_fused_f32_export"
    ):
        pytest.skip("native fused TMix adapter ABI is unavailable")
    monkeypatch.setenv("RWKV_LUT_KERNEL", "c")
    monkeypatch.setenv("RWKV_LUT_FUSED_ADAPTERS", "1")
    torch.manual_seed(31)
    n_embd, w_rank, a_rank, g_rank, v_rank = 19, 5, 4, 6, 3
    shapes = (
        (n_embd, w_rank),
        (w_rank, n_embd),
        (n_embd, a_rank),
        (a_rank, n_embd),
        (n_embd, g_rank),
        (g_rank, n_embd),
        (n_embd, v_rank),
        (v_rank, n_embd),
    )
    weights = [torch.randn(shape) for shape in shapes]
    blobs = tuple(encode_scale_u8_grouped(weight, group_size=7) for weight in weights)
    xw, xa, xg, xv = (torch.randn(n_embd) for _ in range(4))
    a0 = torch.randn(n_embd)

    got = grouped_u8_transposed_tmix_fused(
        blobs,
        (xw, xa, xg, xv),
        shapes,
        a0,
    )
    assert got is not None

    w_mid = torch.tanh(
        grouped_u8_transposed_gemv(
            blobs[0], xw, out_features=w_rank, in_features=n_embd
        )
    )
    a_mid = grouped_u8_transposed_gemv(
        blobs[2], xa, out_features=a_rank, in_features=n_embd
    )
    g_mid = torch.sigmoid(
        grouped_u8_transposed_gemv(
            blobs[4], xg, out_features=g_rank, in_features=n_embd
        )
    )
    v_mid = grouped_u8_transposed_gemv(
        blobs[6], xv, out_features=v_rank, in_features=n_embd
    )
    expected = (
        grouped_u8_transposed_gemv(
            blobs[1], w_mid, out_features=n_embd, in_features=w_rank
        ),
        torch.sigmoid(
            a0
            + grouped_u8_transposed_gemv(
                blobs[3], a_mid, out_features=n_embd, in_features=a_rank
            )
        ),
        grouped_u8_transposed_gemv(
            blobs[5], g_mid, out_features=n_embd, in_features=g_rank
        ),
        grouped_u8_transposed_gemv(
            blobs[7], v_mid, out_features=n_embd, in_features=v_rank
        ),
    )
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
