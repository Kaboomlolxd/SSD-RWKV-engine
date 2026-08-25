"""Per-linear-input activation statistics for calibrated quantization."""

from __future__ import annotations

from pathlib import Path

import torch


class ActivationStatsCollector:
    """Accumulate featurewise second moments without retaining activations."""

    def __init__(self) -> None:
        self._sum_squares: dict[str, torch.Tensor] = {}
        self._samples: dict[str, int] = {}

    def observe(self, name: str, activation: torch.Tensor) -> None:
        value = activation.detach().float().cpu()
        if value.ndim == 0:
            return
        rows = value.reshape(-1, value.shape[-1])
        sums = rows.square().sum(dim=0, dtype=torch.float64)
        previous = self._sum_squares.get(name)
        if previous is not None and previous.shape != sums.shape:
            raise ValueError(f"activation width changed for {name}")
        self._sum_squares[name] = sums if previous is None else previous + sums
        self._samples[name] = self._samples.get(name, 0) + rows.shape[0]

    def rms(self, name: str) -> torch.Tensor | None:
        sums = self._sum_squares.get(name)
        count = self._samples.get(name, 0)
        if sums is None or count <= 0:
            return None
        return torch.sqrt(sums / count).float()

    def to_artifact(self) -> dict[str, object]:
        return {
            "version": 1,
            "rms": {name: self.rms(name) for name in sorted(self._sum_squares)},
            "samples": dict(sorted(self._samples.items())),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_artifact(), output)
        return output


def load_activation_rms(path: str | Path) -> dict[str, torch.Tensor]:
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
        raise ValueError("unsupported activation calibration artifact")
    values = raw.get("rms")
    if not isinstance(values, dict):
        raise ValueError("activation calibration artifact lacks rms values")
    return {
        str(name): torch.as_tensor(value).detach().float().cpu().contiguous()
        for name, value in values.items()
    }


__all__ = ["ActivationStatsCollector", "load_activation_rms"]
