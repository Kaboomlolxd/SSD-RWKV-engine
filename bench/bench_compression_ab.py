#!/usr/bin/env python3
"""A/B Trinity codebooks on synthetic or real RWKV checkpoint tensors.

This is a quality/size benchmark, not a generation-quality substitute.  It
measures exact tensor error and deterministic random-activation output error,
then emits explicit gates so a new codec cannot become the default merely
because it made the file smaller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_tensor,
    encode_trinity_lut2,
)
from rwkv_ssd.tools.pack_runtime import _load_state_dict
from rwkv_ssd.tools.quant_quality import compare_tensors


def _entry(name: str, tensor: torch.Tensor, length: int) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        offset=0,
        length=length,
        alignment=1,
        residency="streamed",
        dequant="trinity_lut2",
    )


def evaluate_compression_ab(
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    group_sizes: tuple[int, ...] = (64, 128, 256, 512),
    activation_samples: int = 8,
    min_compression_ratio: float = 4.0,
    seed: int = 1234,
) -> dict[str, Any]:
    specs: list[tuple[str, str, int]] = [("global_kmeans", "kmeans", 0)]
    specs.extend(
        (f"groupwise_kmeans_g{size}", "groupwise_kmeans", int(size))
        for size in group_sizes
    )
    specs.extend(
        (f"groupwise_input_fp16_g{size}", "groupwise_input_fp16", int(size))
        for size in group_sizes
    )
    specs.extend(
        (
            f"groupwise_input_residual_g{size}",
            "groupwise_input_residual",
            int(size),
        )
        for size in group_sizes
        if size <= 256
    )
    specs.extend(
        (
            f"groupwise_kmeans_salient_g{size}",
            "groupwise_kmeans_salient",
            int(size),
        )
        for size in group_sizes
        if size <= 256
    )
    specs.extend(
        (
            f"groupwise_symmetric_fp16_g{size}",
            "groupwise_symmetric_fp16",
            int(size),
        )
        for size in group_sizes
    )
    specs.extend(
        (f"groupwise_kmeans_fp16_g{size}", "groupwise_kmeans_fp16", int(size))
        for size in group_sizes
    )
    specs.extend(
        (
            f"groupwise_kmeans_residual_g{size}",
            "groupwise_kmeans_residual",
            int(size),
        )
        for size in group_sizes
        if size <= 256
    )
    accum = {
        name: {
            "name": name,
            "codebook": algo,
            "group_size": size,
            "encoded_bytes": 0,
            "dense_bytes": 0,
            "elements": 0,
            "squared_error_sum": 0.0,
            "output_squared_error_sum": 0.0,
            "output_elements": 0,
            "max_abs": 0.0,
            "tensors": 0,
        }
        for name, algo, size in specs
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor_count = 0
    for tensor_name, value in tensors:
        if not torch.is_floating_point(value) or value.numel() == 0:
            continue
        reference = value.detach().cpu().contiguous()
        tensor_count += 1
        activations = None
        reference_output = None
        if reference.ndim == 2 and activation_samples > 0:
            activations = torch.randn(
                activation_samples,
                reference.shape[0],
                generator=generator,
                dtype=torch.float32,
            )
            reference_output = activations @ reference.float()
        for result_name, algo, group_size in specs:
            kwargs: dict[str, Any] = {"codebook": algo}
            if group_size:
                kwargs["group_size"] = group_size
            blob = encode_trinity_lut2(reference, **kwargs)
            candidate = decode_trinity_lut2_to_tensor(
                blob, _entry(tensor_name, reference, len(blob)), torch.device("cpu")
            )
            metrics = compare_tensors(reference, candidate)
            row = accum[result_name]
            elements = int(metrics["elements"])
            row["encoded_bytes"] += len(blob)
            row["dense_bytes"] += reference.numel() * reference.element_size()
            row["elements"] += elements
            row["squared_error_sum"] += float(metrics["rmse"]) ** 2 * elements
            row["max_abs"] = max(row["max_abs"], float(metrics["max_abs"]))
            row["tensors"] += 1
            if activations is not None and reference_output is not None:
                delta = activations @ candidate.float() - reference_output
                row["output_squared_error_sum"] += float(delta.square().sum().item())
                row["output_elements"] += delta.numel()

    rows: list[dict[str, Any]] = []
    for name, _, _ in specs:
        raw = accum[name]
        elements = max(1, int(raw.pop("elements")))
        output_elements = int(raw.pop("output_elements"))
        squared = float(raw.pop("squared_error_sum"))
        output_squared = float(raw.pop("output_squared_error_sum"))
        dense_bytes = int(raw["dense_bytes"])
        encoded_bytes = int(raw["encoded_bytes"])
        raw["weighted_rmse"] = (squared / elements) ** 0.5
        raw["activation_output_rmse"] = (
            (output_squared / output_elements) ** 0.5 if output_elements else None
        )
        raw["compression_ratio"] = dense_bytes / max(1, encoded_bytes)
        rows.append(raw)

    baseline = rows[0]["weighted_rmse"] if rows else 0.0
    for row in rows:
        row["gates"] = {
            "min_compression_ratio": row["compression_ratio"] >= min_compression_ratio,
            "improves_weighted_rmse_vs_global": (
                row["name"] == "global_kmeans" or row["weighted_rmse"] < baseline
            ),
        }
        row["passed"] = all(row["gates"].values())
    eligible = [row for row in rows[1:] if row["passed"]]
    recommendation = min(eligible, key=lambda row: row["weighted_rmse"])["name"] if eligible else None
    return {
        "schema_version": 1,
        "tensor_count": tensor_count,
        "activation_samples": activation_samples,
        "minimum_compression_ratio": min_compression_ratio,
        "variants": rows,
        "recommended_variant": recommendation,
        "caveat": "Tensor/output probes require logit and recurrent-state gates before deployment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--group-sizes", default="64,128,256,512")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--min-numel", type=int, default=4096)
    parser.add_argument("--activation-samples", type=int, default=8)
    parser.add_argument("--min-compression-ratio", type=float, default=4.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    state = _load_state_dict(args.checkpoint)
    selected = [
        (name, tensor)
        for name, tensor in sorted(state.items())
        if torch.is_floating_point(tensor) and tensor.numel() >= args.min_numel
    ]
    if args.max_tensors > 0:
        selected = selected[: args.max_tensors]
    sizes = tuple(int(item) for item in args.group_sizes.split(",") if item.strip())
    report = evaluate_compression_ab(
        selected,
        group_sizes=sizes,
        activation_samples=args.activation_samples,
        min_compression_ratio=args.min_compression_ratio,
    )
    payload = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
