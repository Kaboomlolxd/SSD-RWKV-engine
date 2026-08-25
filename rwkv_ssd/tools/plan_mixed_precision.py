"""Plan calibrated LUT2 -> grouped-U8 promotions by error reduction per byte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rwkv_ssd.runtime.activation_calibration import load_activation_rms
from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_u8_grouped_to_tensor,
    encode_scale_u8_grouped,
)
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_tensor,
    encode_trinity_lut2,
)
from rwkv_ssd.tools.pack_runtime import _load_state_dict


def _entry(name: str, tensor: torch.Tensor, length: int, codec: str) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        offset=0,
        length=length,
        alignment=1,
        residency="streamed",
        dequant=codec,
    )


def _weighted_squared_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    feature_rms: torch.Tensor | None,
) -> float:
    error = (candidate.float() - reference.float()).square()
    if feature_rms is None or reference.ndim != 2:
        return float(error.sum().item())
    rms2 = feature_rms.float().square()
    if rms2.numel() == reference.shape[1]:
        error = error * rms2.reshape(1, -1)
    elif rms2.numel() == reference.shape[0]:
        error = error * rms2.reshape(-1, 1)
    return float(error.sum().item())


def rank_codec_promotions(
    tensors: dict[str, torch.Tensor],
    activation_rms: dict[str, torch.Tensor],
    *,
    lut_group_size: int = 128,
    u8_group_size: int = 64,
    extra_budget_bytes: int,
    mandatory_suffixes: tuple[str, ...] = (
        ".ffn.value.weight",
        ".att.output.weight",
    ),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, value in sorted(tensors.items()):
        # RWKV's tiny time-mix vectors and low-rank controls are recurrently
        # amplified.  They cost little to promote but cannot safely be treated
        # as insignificant merely because they are not large matrices.
        if not name.startswith("blocks.") or value.numel() == 0:
            continue
        tensor = value.detach().cpu().contiguous()
        importance = activation_rms.get(name)
        lut = encode_trinity_lut2(
            tensor,
            codebook="groupwise_residual_activation",
            group_size=lut_group_size,
            importance=importance,
        )
        lut_tensor = decode_trinity_lut2_to_tensor(
            lut, _entry(name, tensor, len(lut), "trinity_lut2"), torch.device("cpu")
        )
        u8 = encode_scale_u8_grouped(tensor, group_size=u8_group_size)
        u8_tensor = decode_scale_u8_grouped_to_tensor(
            u8,
            _entry(name, tensor, len(u8), "scale_u8_grouped"),
            torch.device("cpu"),
        )
        lut_error = _weighted_squared_error(tensor, lut_tensor, importance)
        u8_error = _weighted_squared_error(tensor, u8_tensor, importance)
        extra = max(0, len(u8) - len(lut))
        benefit = max(0.0, lut_error - u8_error)
        rows.append(
            {
                "name": name,
                "lut2_bytes": len(lut),
                "u8_bytes": len(u8),
                "extra_bytes": extra,
                "weighted_error_reduction": benefit,
                "benefit_per_byte": benefit / max(1, extra),
                "calibrated": importance is not None,
            }
        )
    rows.sort(key=lambda row: (-row["benefit_per_byte"], row["name"]))
    selected: list[dict[str, Any]] = []
    used = 0
    mandatory = [row for row in rows if row["name"].endswith(mandatory_suffixes)]
    optional = [row for row in rows if not row["name"].endswith(mandatory_suffixes)]
    for row in mandatory + optional:
        extra = int(row["extra_bytes"])
        if extra <= 0 or used + extra > extra_budget_bytes:
            continue
        selected.append(row)
        used += extra
    codec_map = {row["name"]: "scale_u8_grouped" for row in selected}
    return {
        "version": 1,
        "lut_group_size": lut_group_size,
        "u8_group_size": u8_group_size,
        "extra_budget_bytes": int(extra_budget_bytes),
        "selected_extra_bytes": used,
        "selected_tensors": len(selected),
        "candidate_tensors": len(rows),
        "mandatory_suffixes": list(mandatory_suffixes),
        "mandatory_selected": sum(
            1 for row in selected if row["name"].endswith(mandatory_suffixes)
        ),
        "codec_map": codec_map,
        "ranking": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-budget-mb", required=True, type=float)
    parser.add_argument("--lut-group-size", type=int, default=128)
    parser.add_argument("--u8-group-size", type=int, default=64)
    parser.add_argument(
        "--mandatory-suffix",
        action="append",
        dest="mandatory_suffixes",
        help=(
            "Tensor-name suffix that must be promoted before ranked optional tensors. "
            "Repeat for multiple families. Defaults to ffn.value and att.output."
        ),
    )
    args = parser.parse_args()
    report = rank_codec_promotions(
        _load_state_dict(args.checkpoint),
        load_activation_rms(args.activation_stats),
        lut_group_size=args.lut_group_size,
        u8_group_size=args.u8_group_size,
        extra_budget_bytes=max(0, int(args.extra_budget_mb * 1024 * 1024)),
        mandatory_suffixes=(
            tuple(args.mandatory_suffixes)
            if args.mandatory_suffixes
            else (".ffn.value.weight", ".att.output.weight")
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("ranking", "codec_map")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
