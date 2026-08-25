from __future__ import annotations

import torch

from rwkv_ssd.runtime.activation_calibration import (
    ActivationStatsCollector,
    load_activation_rms,
)
from rwkv_ssd.tools.calibrate_activations import _LinearCaptureMode


def test_activation_stats_accumulate_and_roundtrip(tmp_path) -> None:
    collector = ActivationStatsCollector()
    collector.observe("blocks.0.att.key.weight", torch.tensor([[3.0, 4.0]]))
    collector.observe("blocks.0.att.key.weight", torch.tensor([[0.0, 0.0]]))
    expected = torch.tensor([(9.0 / 2.0) ** 0.5, (16.0 / 2.0) ** 0.5])
    assert torch.allclose(collector.rms("blocks.0.att.key.weight"), expected)
    path = collector.save(tmp_path / "activations.pt")
    loaded = load_activation_rms(path)
    assert torch.equal(loaded["blocks.0.att.key.weight"], expected)


def test_activation_stats_reject_width_change() -> None:
    collector = ActivationStatsCollector()
    collector.observe("w", torch.ones(4))
    try:
        collector.observe("w", torch.ones(5))
    except ValueError as exc:
        assert "width changed" in str(exc)
    else:
        raise AssertionError("expected activation width mismatch")


def test_dispatch_mode_captures_direct_matmul() -> None:
    collector = ActivationStatsCollector()
    weight = torch.randn(4, 3)
    activation = torch.randn(2, 4)
    with _LinearCaptureMode({weight.data_ptr(): "w"}, collector):
        actual = activation @ weight
    assert torch.equal(actual, activation @ weight)
    assert collector.rms("w") is not None
