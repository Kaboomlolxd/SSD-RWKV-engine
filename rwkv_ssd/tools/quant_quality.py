"""Quality harness for streamed/quantized packs.

The harness is intentionally model-agnostic.  It compares matching manifest
tensors directly and can optionally compare saved logits and recurrent-state
tensors.  This keeps codec decisions honest before adding a new layout or
residual sidecar to the engine hot path.

Examples::

    python -m rwkv_ssd.tools.quant_quality \
      --reference ./fp16_pack --candidate ./lut2_pack --json
    python -m rwkv_ssd.tools.quant_quality \
      --reference-logits ref.pt --candidate-logits lut2.pt --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.runtime.weight_store_sharded import open_sharded_weight_store


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("logits", "state", "tensor", "values"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (list, tuple)):
        value = torch.stack([torch.as_tensor(item) for item in value])
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return value.detach().to(device="cpu", dtype=torch.float32)


def _empty_tensor_metrics() -> dict[str, float | int]:
    """Metrics for two equal, empty tensors.

    Empty tensors are valid for optional/zero-width manifest entries.  Treat
    them as an exact comparison with cosine 1 rather than letting reductions
    such as ``mean`` and ``max`` produce NaNs or reduction errors.
    """
    return {
        "elements": 0,
        "rmse": 0.0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "relative_l2": 0.0,
        "cosine": 1.0,
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    ref = reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    cand = candidate.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if ref.numel() == 0:
        return _empty_tensor_metrics()
    delta = cand - ref
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cand_norm = float(torch.linalg.vector_norm(cand).item())
    denom = max(ref_norm, 1e-12)
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(ref, cand).item()) / max(
            ref_norm * cand_norm, 1e-12
        )
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "elements": int(ref.numel()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()) if ref.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / denom,
        "cosine": cosine,
    }


def compare_logits(
    reference: torch.Tensor, candidate: torch.Tensor, *, top_k: int = 10
) -> dict[str, float | int]:
    if reference.ndim < 1 or candidate.ndim < 1:
        raise ValueError("logit tensors must have at least one dimension")
    metrics = compare_tensors(reference, candidate)
    if reference.shape[-1] == 0:
        metrics.update(
            {
                "top_k": 0,
                "top_k_overlap": 1.0,
                "min_top_k_overlap": 1.0,
                "kl_candidate_to_reference": 0.0,
                "max_kl_candidate_to_reference": 0.0,
            }
        )
        return metrics
    if reference.numel() == 0:
        metrics.update(
            {
                "top_k": min(max(1, int(top_k)), reference.shape[-1]),
                "top_k_overlap": 1.0,
                "min_top_k_overlap": 1.0,
                "kl_candidate_to_reference": 0.0,
                "max_kl_candidate_to_reference": 0.0,
            }
        )
        return metrics
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
        candidate = candidate.unsqueeze(0)
    k = min(max(1, int(top_k)), reference.shape[-1])
    ref_top = torch.topk(reference, k=k, dim=-1).indices
    cand_top = torch.topk(candidate, k=k, dim=-1).indices
    overlap_rows = (
        (ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_p = torch.softmax(candidate, dim=-1)
    kl_rows = torch.sum(
        cand_p * (torch.log(cand_p.clamp_min(1e-12)) - ref_logp), dim=-1
    )
    metrics.update(
        {
            "top_k": k,
            "top_k_overlap": float(overlap_rows.mean().item()),
            "min_top_k_overlap": float(overlap_rows.min().item()),
            "kl_candidate_to_reference": float(kl_rows.mean().item()),
            "max_kl_candidate_to_reference": float(kl_rows.max().item()),
        }
    )
    return metrics


def compare_state_sequence(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | int]:
    """Measure recurrent-state drift over the leading time dimension."""
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("state sequences must have matching shape and a time dimension")
    if reference.shape[0] == 0:
        return {
            "steps": 0,
            "max_relative_l2": 0.0,
            "mean_relative_l2": 0.0,
            "final_relative_l2": 0.0,
            "final_cosine": 1.0,
        }
    rows = [
        compare_tensors(reference[index], candidate[index])
        for index in range(reference.shape[0])
    ]
    return {
        "steps": int(reference.shape[0]),
        "max_relative_l2": max(float(row["relative_l2"]) for row in rows),
        "mean_relative_l2": sum(float(row["relative_l2"]) for row in rows) / len(rows),
        "final_relative_l2": float(rows[-1]["relative_l2"]),
        "final_cosine": float(rows[-1]["cosine"]),
    }


def _store_for(manifest: Manifest):
    if manifest.is_sharded():
        return open_sharded_weight_store(manifest)
    return open_weight_store(manifest.weights_path, backend="pread")


def compare_packs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    max_elements: int = 0,
    per_layer: bool = False,
) -> dict[str, Any]:
    reference = Manifest.load(reference_dir)
    candidate = Manifest.load(candidate_dir)
    ref_by_name = {entry.name: entry for entry in reference.tensors}
    cand_by_name = {entry.name: entry for entry in candidate.tensors}
    names = sorted(set(ref_by_name) & set(cand_by_name))
    rows: list[dict[str, Any]] = []
    ref_store = _store_for(reference)
    cand_store = _store_for(candidate)
    try:
        for name in names:
            ref_entry = ref_by_name[name]
            cand_entry = cand_by_name[name]
            if ref_entry.numel != cand_entry.numel or ref_entry.shape != cand_entry.shape:
                rows.append(
                    {
                        "name": name,
                        "layer_id": ref_entry.layer_id,
                        "error": "shape mismatch",
                        "reference_shape": list(ref_entry.shape),
                        "candidate_shape": list(cand_entry.shape),
                    }
                )
                continue
            ref_raw = ref_store.read_bytes(ref_entry)
            cand_raw = cand_store.read_bytes(cand_entry)
            ref_tensor = decode_weight_to_tensor(ref_raw, ref_entry, torch.device("cpu"))
            cand_tensor = decode_weight_to_tensor(
                cand_raw, cand_entry, torch.device("cpu"), decode_device=torch.device("cpu")
            )
            if max_elements and ref_tensor.numel() > max_elements:
                ref_tensor = ref_tensor.reshape(-1)[:max_elements]
                cand_tensor = cand_tensor.reshape(-1)[:max_elements]
            rows.append(
                {
                    "name": name,
                    "layer_id": ref_entry.layer_id,
                    **compare_tensors(ref_tensor.float(), cand_tensor.float()),
                }
            )
    finally:
        ref_store.close()
        cand_store.close()
    valid = [row for row in rows if "rmse" in row]
    total = sum(int(row["elements"]) for row in valid)
    # Preserve the historical field meanings: ``missing_reference`` are
    # candidate-only names, while ``missing_candidate`` are reference-only
    # names.  The names are consumed by existing reports and callers.
    missing_reference = sorted(set(cand_by_name) - set(ref_by_name))
    missing_candidate = sorted(set(ref_by_name) - set(cand_by_name))
    weighted_rmse = (
        sum(float(row["rmse"]) ** 2 * int(row["elements"]) for row in valid) / total
    ) ** 0.5 if total else 0.0
    aggregate = {
        "matched": len(names),
        "compared": len(valid),
        "errors": len(rows) - len(valid),
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "complete": (
            not missing_reference
            and not missing_candidate
            and len(valid) == len(names)
        ),
        "weighted_rmse": weighted_rmse,
        "max_rmse": max((float(row["rmse"]) for row in valid), default=0.0),
        "max_relative_l2": max(
            (float(row["relative_l2"]) for row in valid), default=0.0
        ),
        "elements": total,
    }
    result: dict[str, Any] = {
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "aggregate": aggregate,
        "tensors": rows,
    }
    if per_layer:
        layer_ids = sorted(
            {
                int(row["layer_id"])
                for row in rows
                if "layer_id" in row
            }
            | {
                int(cand_by_name[name].layer_id)
                for name in missing_reference
                if name in cand_by_name
            }
            | {
                int(ref_by_name[name].layer_id)
                for name in missing_candidate
                if name in ref_by_name
            }
        )
        layer_rows: list[dict[str, Any]] = []
        for layer_id in layer_ids:
            layer_entries = [row for row in rows if row.get("layer_id") == layer_id]
            layer_valid = [row for row in layer_entries if "rmse" in row]
            layer_missing_reference = [
                name
                for name in missing_reference
                if cand_by_name[name].layer_id == layer_id
            ]
            layer_missing_candidate = [
                name
                for name in missing_candidate
                if ref_by_name[name].layer_id == layer_id
            ]
            layer_elements = sum(int(row["elements"]) for row in layer_valid)
            layer_weighted = (
                sum(
                    float(row["rmse"]) ** 2 * int(row["elements"])
                    for row in layer_valid
                )
                / layer_elements
            ) ** 0.5 if layer_elements else 0.0
            layer_rows.append(
                {
                    "layer_id": layer_id,
                    "matched": len(layer_entries),
                    "compared": len(layer_valid),
                    "errors": len(layer_entries) - len(layer_valid),
                    "missing_reference": layer_missing_reference,
                    "missing_candidate": layer_missing_candidate,
                    "complete": (
                        len(layer_entries) == len(layer_valid)
                        and not layer_missing_reference
                        and not layer_missing_candidate
                    ),
                    "elements": layer_elements,
                    "weighted_rmse": layer_weighted,
                    "max_rmse": max(
                        (float(row["rmse"]) for row in layer_valid), default=0.0
                    ),
                }
            )
        result["per_layer"] = layer_rows
    return result


def make_pass_fail_summary(
    result: dict[str, Any],
    gates: dict[str, bool],
) -> dict[str, Any]:
    """Return a stable, machine-readable quality summary.

    The summary keeps observed values separate from gate booleans so callers
    can compare runs without reconstructing which optional inputs were passed.
    ``None`` means that a metric was not supplied on this invocation.
    """
    packs = result.get("packs")
    aggregate = packs.get("aggregate", {}) if isinstance(packs, dict) else {}
    logits = result.get("logits") or {}
    state = result.get("state") or {}
    passed = all(gates.values())
    return {
        "pack_completeness": (
            bool(aggregate.get("complete")) if packs is not None else None
        ),
        "max_rmse": (
            float(aggregate.get("max_rmse", 0.0)) if packs is not None else None
        ),
        "weighted_rmse": (
            float(aggregate.get("weighted_rmse", 0.0)) if packs is not None else None
        ),
        "min_top_k_overlap": (
            float(logits.get("min_top_k_overlap", logits["top_k_overlap"]))
            if "top_k_overlap" in logits
            else None
        ),
        "max_kl": (
            float(
                logits.get(
                    "max_kl_candidate_to_reference",
                    logits["kl_candidate_to_reference"],
                )
            )
            if "kl_candidate_to_reference" in logits
            else None
        ),
        "max_recurrent_state_drift": (
            float(state["max_relative_l2"])
            if "max_relative_l2" in state
            else None
        ),
        "gates": dict(gates),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="FP16/reference runtime pack")
    parser.add_argument("--candidate", type=Path, help="quantized/candidate runtime pack")
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--candidate-logits", type=Path)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--candidate-state", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-elements", type=int, default=0)
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="include optional layer-level aggregate metrics for pack comparisons",
    )
    parser.add_argument("--max-rmse", type=float, default=float("inf"))
    parser.add_argument("--max-weighted-rmse", type=float, default=float("inf"))
    parser.add_argument("--min-top-k-overlap", type=float, default=0.0)
    parser.add_argument("--max-kl", type=float, default=float("inf"))
    parser.add_argument("--max-state-relative-l2", type=float, default=float("inf"))
    parser.add_argument("--strict", action="store_true", help="exit 1 when a supplied gate fails")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.candidate):
        parser.error("--reference and --candidate must be supplied together")
    if bool(args.reference_logits) != bool(args.candidate_logits):
        parser.error("logit reference and candidate must be supplied together")
    if bool(args.reference_state) != bool(args.candidate_state):
        parser.error("state reference and candidate must be supplied together")
    result: dict[str, Any] = {}
    if args.reference and args.candidate:
        result["packs"] = compare_packs(
            args.reference,
            args.candidate,
            max_elements=args.max_elements,
            per_layer=args.per_layer,
        )
    if args.reference_logits and args.candidate_logits:
        result["logits"] = compare_logits(
            _load_tensor(args.reference_logits),
            _load_tensor(args.candidate_logits),
            top_k=args.top_k,
        )
    if args.reference_state and args.candidate_state:
        result["state"] = compare_state_sequence(
            _load_tensor(args.reference_state), _load_tensor(args.candidate_state)
        )
    if not result:
        parser.error("supply pack, logits, or state reference/candidate pairs")
    gates: dict[str, bool] = {}
    if "packs" in result:
        gates["pack_complete"] = bool(result["packs"]["aggregate"]["complete"])
        gates["weighted_rmse"] = (
            float(result["packs"]["aggregate"]["weighted_rmse"])
            <= args.max_weighted_rmse
        )
        gates["max_rmse"] = (
            float(result["packs"]["aggregate"]["max_rmse"]) <= args.max_rmse
        )
    if "logits" in result:
        gates["top_k_overlap"] = (
            float(
                result["logits"].get(
                    "min_top_k_overlap", result["logits"]["top_k_overlap"]
                )
            )
            >= args.min_top_k_overlap
        )
        gates["kl"] = (
            float(
                result["logits"].get(
                    "max_kl_candidate_to_reference",
                    result["logits"]["kl_candidate_to_reference"],
                )
            )
            <= args.max_kl
        )
    if "state" in result:
        gates["state_relative_l2"] = (
            float(result["state"]["max_relative_l2"])
            <= args.max_state_relative_l2
        )
    result["gates"] = gates
    result["passes"] = all(gates.values())
    result["summary"] = make_pass_fail_summary(result, gates)
    print(json.dumps(result, indent=2))
    if args.strict and not result["passes"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
