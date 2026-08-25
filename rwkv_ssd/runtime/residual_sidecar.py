"""Activation-weighted sparse residual sidecars for quality experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rwkv_ssd.runtime.io_pread import PreadWeightStore


@dataclass(frozen=True)
class ResidualBuildReport:
    baseline_output_rmse: float
    final_output_rmse: float
    target_output_rmse: float
    selected_rows: int
    total_rows: int
    selected_fraction: float
    passed: bool


def _rmse(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.float() ** 2)).item()) if value.numel() else 0.0


def build_residual_sidecar(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    calibration_activations: torch.Tensor,
    output_dir: str | Path,
    *,
    name: str = "residual",
    max_output_rmse: float,
    max_rows: int | None = None,
) -> tuple[Path, ResidualBuildReport]:
    """Select input rows whose residual contribution best repairs outputs."""
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must be equal rank-2 matrices")
    if calibration_activations.ndim != 2 or calibration_activations.shape[1] != reference.shape[0]:
        raise ValueError("calibration activations must have shape [samples, in_features]")
    if max_output_rmse < 0:
        raise ValueError("max_output_rmse must be non-negative")
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    activations = calibration_activations.detach().float().cpu()
    residual = ref - cand
    remaining_error = activations @ residual
    baseline = _rmse(remaining_error)
    impact = torch.sqrt(torch.mean(activations**2, dim=0)) * torch.linalg.vector_norm(
        residual, dim=1
    )
    order = torch.argsort(impact, descending=True).tolist()
    limit = len(order) if max_rows is None else max(0, min(int(max_rows), len(order)))
    selected: list[int] = []
    final_rmse = baseline
    for row in order[:limit]:
        if final_rmse <= max_output_rmse:
            break
        selected.append(int(row))
        remaining_error -= activations[:, row : row + 1] @ residual[row : row + 1, :]
        final_rmse = _rmse(remaining_error)
    selected.sort()
    rows = residual[selected].contiguous() if selected else residual[:0].contiguous()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{name}.bin"
    index_path = output / f"{name}.json"
    data_path.write_bytes(rows.numpy().tobytes())
    report = ResidualBuildReport(
        baseline_output_rmse=baseline,
        final_output_rmse=final_rmse,
        target_output_rmse=float(max_output_rmse),
        selected_rows=len(selected),
        total_rows=int(reference.shape[0]),
        selected_fraction=(len(selected) / reference.shape[0] if reference.shape[0] else 0.0),
        passed=final_rmse <= max_output_rmse,
    )
    index = {
        "version": 1,
        "data_file": data_path.name,
        "shape": [int(reference.shape[0]), int(reference.shape[1])],
        "dtype": "float32",
        "selected_row_indices": selected,
        "row_bytes": int(reference.shape[1]) * 4,
        "report": asdict(report),
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path, report


class ResidualSidecar:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if int(raw.get("version", 0)) != 1 or raw.get("dtype") != "float32":
            raise ValueError("unsupported residual sidecar index")
        self.shape = tuple(int(value) for value in raw["shape"])
        self.selected_rows = tuple(int(value) for value in raw["selected_row_indices"])
        if any(row < 0 or row >= self.shape[0] for row in self.selected_rows):
            raise ValueError("residual sidecar row index out of range")
        self._store = PreadWeightStore(
            self.index_path.parent / str(raw["data_file"])
        )
        self._payload_bytes = len(self.selected_rows) * self.shape[1] * 4
        self._rows: torch.Tensor | None = None

    def _load_rows(self) -> torch.Tensor:
        if self._rows is None:
            payload = self._store.read_bytes_span(0, self._payload_bytes)
            self._rows = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
                len(self.selected_rows), self.shape[1]
            )
        return self._rows

    def correction(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.shape[0]:
            raise ValueError("activation width does not match residual sidecar")
        output_shape = (*activation.shape[:-1], self.shape[1])
        if not self.selected_rows:
            return torch.zeros(output_shape, dtype=activation.dtype, device=activation.device)
        index = torch.tensor(self.selected_rows, dtype=torch.long, device=activation.device)
        selected_activation = activation.index_select(-1, index)
        rows = self._load_rows().to(dtype=selected_activation.dtype, device=activation.device)
        return selected_activation @ rows

    def apply(self, activation: torch.Tensor, candidate_output: torch.Tensor) -> torch.Tensor:
        return candidate_output + self.correction(activation).to(candidate_output.dtype)

    def close(self) -> None:
        self._store.close()


__all__ = ["ResidualBuildReport", "ResidualSidecar", "build_residual_sidecar"]
