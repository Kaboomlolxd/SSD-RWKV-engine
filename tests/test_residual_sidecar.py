from __future__ import annotations

import torch

from rwkv_ssd.runtime.residual_sidecar import (
    ResidualSidecar,
    build_residual_sidecar,
)


def test_residual_sidecar_meets_calibration_quality_gate(tmp_path) -> None:
    torch.manual_seed(1)
    reference = torch.randn(8, 4)
    candidate = reference.clone()
    candidate[2] += 0.5
    candidate[6] -= 0.25
    calibration = torch.randn(32, 8)
    index, report = build_residual_sidecar(
        reference,
        candidate,
        calibration,
        tmp_path,
        max_output_rmse=1e-6,
    )
    assert report.passed
    assert report.selected_rows == 2
    sidecar = ResidualSidecar(index)
    try:
        repaired = sidecar.apply(calibration, calibration @ candidate)
        torch.testing.assert_close(repaired, calibration @ reference, rtol=0, atol=1e-6)
    finally:
        sidecar.close()


def test_residual_sidecar_reports_failed_row_budget(tmp_path) -> None:
    reference = torch.eye(4)
    candidate = torch.zeros_like(reference)
    calibration = torch.eye(4)
    _index, report = build_residual_sidecar(
        reference,
        candidate,
        calibration,
        tmp_path,
        max_output_rmse=0.0,
        max_rows=1,
    )
    assert not report.passed
    assert report.selected_rows == 1
    assert report.final_output_rmse < report.baseline_output_rmse


def test_empty_residual_sidecar_is_zero_correction(tmp_path) -> None:
    matrix = torch.randn(5, 3)
    calibration = torch.randn(4, 5)
    index, report = build_residual_sidecar(
        matrix,
        matrix.clone(),
        calibration,
        tmp_path,
        max_output_rmse=0.0,
    )
    assert report.selected_rows == 0
    sidecar = ResidualSidecar(index)
    try:
        correction = sidecar.correction(calibration)
        assert torch.count_nonzero(correction) == 0
    finally:
        sidecar.close()


def test_activation_weighting_prioritizes_salient_row(tmp_path) -> None:
    reference = torch.zeros(3, 2)
    reference[0] = 1
    reference[1] = 1
    candidate = torch.zeros_like(reference)
    calibration = torch.tensor([[10.0, 0.1, 0.0], [8.0, 0.1, 0.0]])
    index, report = build_residual_sidecar(
        reference,
        candidate,
        calibration,
        tmp_path,
        max_output_rmse=0.0,
        max_rows=1,
    )
    sidecar = ResidualSidecar(index)
    try:
        assert sidecar.selected_rows == (0,)
        assert report.selected_rows == 1
    finally:
        sidecar.close()
