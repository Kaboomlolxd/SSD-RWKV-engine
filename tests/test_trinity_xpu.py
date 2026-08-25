"""Trinity Intel XPU decode helpers (skipped when torch.xpu unavailable)."""

from __future__ import annotations

import pytest
import torch

from app.engine_args import add_engine_args, build_engine_config
from rwkv_ssd.runtime.trinity_accel import (
    probe_intel_xpu,
    resolve_trinity_decode_device,
    xpu_available,
)
from rwkv_ssd.runtime.trinity_codec import encode_trinity_lut2
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider


def test_resolve_trinity_decode_cpu() -> None:
    assert resolve_trinity_decode_device("cpu").type == "cpu"


def test_probe_intel_xpu_dict() -> None:
    info = probe_intel_xpu()
    assert "torch" in info
    assert "xpu_available" in info


def test_xpu_runtime_probe_handles_driver_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rwkv_ssd.runtime import device as device_runtime

    class BrokenXPU:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("XPU runtime unavailable")

    monkeypatch.setattr(device_runtime.torch, "xpu", BrokenXPU(), raising=False)
    assert device_runtime.xpu_runtime_available() is False
    assert device_runtime.resolve_device("xpu").type == "cpu"


def test_xpu_compute_probe_handles_matrix_engine_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rwkv_ssd.runtime import device as device_runtime

    original_ones = torch.ones

    class FakeXPU:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None

    def fake_ones(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_ones(*args, **kwargs)

    def broken_mm(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("could not make an engine with allocator")

    monkeypatch.setattr(device_runtime.torch, "xpu", FakeXPU(), raising=False)
    monkeypatch.setattr(device_runtime.torch, "ones", fake_ones)
    monkeypatch.setattr(device_runtime.torch, "mm", broken_mm)
    assert device_runtime.xpu_runtime_available() is True
    assert device_runtime.xpu_compute_available() is False


def test_xpu_strategy_falls_back_when_matrix_compute_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rwkv_ssd.runtime import device as device_runtime

    monkeypatch.setattr(device_runtime, "xpu_compute_available", lambda *_args: False)
    assert (
        device_runtime.resolve_strategy(
            "xpu bf16", torch.device("xpu"), rwkv7=True
        )
        == "cpu fp32"
    )


def test_xpu_decode_device_cli_override() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(
        ["--model", "runtime_pack", "--trinity-decode-device", "xpu"]
    )
    assert build_engine_config(args).trinity_decode_device == "xpu"


def test_xpu_disables_cpu_fused_lut(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(ManifestWeightProvider)
    provider._device = torch.device("xpu")
    provider.mode = "streaming"
    provider._stream_layer_cache = False
    provider._pack_uses_quant = True
    monkeypatch.setenv("RWKV_LUT_GEMM_FUSED", "1")
    assert provider._use_fused_lut_matmul() is False

    from rwkv_ssd.backends.rwkv7_forward import _can_use_packed_block
    from rwkv_ssd.runtime.packed_block_forward import packed_block_forward_enabled
    from rwkv_ssd.runtime.rwkv7_linear import tmix_uses_fused

    assert packed_block_forward_enabled(
        provider, "blocks.0.att.", mode="streaming", pack_uses_quant=True
    ) is False
    assert _can_use_packed_block(provider, {}, 0, {}) is False
    assert tmix_uses_fused(provider, "blocks.0.att.") is False


def test_auto_decode_does_not_return_unavailable_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rwkv_ssd.runtime import trinity_accel

    monkeypatch.setattr(trinity_accel, "xpu_available", lambda: False)
    assert resolve_trinity_decode_device(
        "auto", fallback=torch.device("xpu")
    ).type == "cpu"


@pytest.mark.parametrize(
    "codebook",
    ["groupwise_kmeans", "groupwise_kmeans_residual", "groupwise_input_residual"],
)
def test_grouped_device_gather_cpu_reference(
    codebook: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the accelerator gather implementation without XPU hardware."""
    from rwkv_ssd.runtime import device as device_runtime
    from rwkv_ssd.runtime.trinity_codec import decode_trinity_lut2_to_tensor

    monkeypatch.setattr(
        device_runtime,
        "is_accelerator_device",
        lambda device: torch.device(device).type == "cpu",
    )
    torch.manual_seed(29)
    shape = [8, 32]
    tensor = torch.randn(*shape, dtype=torch.bfloat16)
    raw = encode_trinity_lut2(tensor, codebook=codebook, group_size=32)
    entry = TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="bfloat16",
        shape=shape,
        offset=0,
        length=len(raw),
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    out = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("cpu"), decode_device=torch.device("cpu")
    )
    # The patched dispatch exercises the device gather; compare against the
    # normal host implementation after restoring the real predicate.
    monkeypatch.setattr(
        device_runtime,
        "is_accelerator_device",
        lambda device: torch.device(device).type in ("cuda", "xpu"),
    )
    reference = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("cpu"), decode_device=torch.device("cpu")
    )
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


@pytest.mark.skipif(not xpu_available(), reason="Intel XPU not available")
def test_lut2_gather_xpu_matches_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from rwkv_ssd.runtime.trinity_codec import decode_trinity_lut2_to_tensor

    monkeypatch.setenv("RWKV_LUT_KERNEL", "native")
    monkeypatch.setenv("RWKV_TRINITY_XPU_MIN_NUMEL", "1")
    t = torch.randn(32, 32, dtype=torch.bfloat16)
    raw = encode_trinity_lut2(t)
    entry = TensorEntry(
        name="w",
        layer_id=0,
        dtype="bfloat16",
        shape=[32, 32],
        offset=0,
        length=len(raw),
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    cpu = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("cpu"), decode_device=torch.device("cpu")
    )
    xpu = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("cpu"), decode_device=torch.device("xpu")
    )
    torch.testing.assert_close(cpu, xpu, rtol=0, atol=0)


@pytest.mark.parametrize(
    "codebook",
    [
        "groupwise_kmeans",
        "groupwise_kmeans_fp16",
        "groupwise_kmeans_residual",
        "groupwise_input_fp16",
        "groupwise_input_residual",
    ],
)
@pytest.mark.skipif(not xpu_available(), reason="Intel XPU not available")
def test_grouped_lut2_gather_xpu_matches_cpu(codebook: str) -> None:
    from rwkv_ssd.runtime.trinity_codec import decode_trinity_lut2_to_tensor

    torch.manual_seed(23)
    shape = [16, 32]
    t = torch.randn(*shape, dtype=torch.bfloat16)
    raw = encode_trinity_lut2(t, codebook=codebook, group_size=32)
    entry = TensorEntry(
        name="blocks.0.weight",
        layer_id=0,
        dtype="bfloat16",
        shape=shape,
        offset=0,
        length=len(raw),
        alignment=4096,
        residency="streamed",
        dequant="trinity_lut2",
    )
    cpu = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("cpu"), decode_device=torch.device("cpu")
    )
    xpu = decode_trinity_lut2_to_tensor(
        raw, entry, torch.device("xpu"), decode_device=torch.device("xpu")
    )
    torch.testing.assert_close(cpu, xpu.cpu(), rtol=1e-3, atol=1e-3)
