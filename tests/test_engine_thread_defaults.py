"""CPU thread-policy regressions for the native grouped-U8 path."""

from __future__ import annotations

import os

import torch

from rwkv_ssd.runtime.engine import _apply_cpu_thread_defaults


def test_grouped_fused_cpu_auto_seeds_native_openmp(monkeypatch) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("RWKV_CPU_THREADS", raising=False)
    monkeypatch.delenv("RWKV_LUT2_OMP_THREADS", raising=False)
    old_threads = torch.get_num_threads()
    try:
        _apply_cpu_thread_defaults(
            torch.device("cpu"), n_embd=2560, fused_lut=True
        )
        assert os.environ["OMP_NUM_THREADS"] == str(
            max(1, min(8, os.cpu_count() or 4))
        )
        assert torch.get_num_threads() == max(1, min(8, os.cpu_count() or 4))
    finally:
        torch.set_num_threads(old_threads)


def test_explicit_native_openmp_is_respected(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("RWKV_CPU_THREADS", raising=False)
    _apply_cpu_thread_defaults(
        torch.device("cpu"), n_embd=2560, fused_lut=True
    )
    assert os.environ["OMP_NUM_THREADS"] == "3"
