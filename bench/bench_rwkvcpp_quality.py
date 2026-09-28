#!/usr/bin/env python3
"""Compare resident FP16 rwkv.cpp outputs with a provider-backed pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.layer_keys import manifest_block_layers
from rwkv_ssd.runtime.sampling import sample_numpy
from rwkv_ssd.tools.quant_quality import compare_logits, compare_tensors


def summarize_quality_pairs(
    pairs: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    top_k: int,
    min_top_k_overlap: float,
    max_kl: float,
    max_state_relative_l2: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prompt, ref_logits, cand_logits, ref_state, cand_state in pairs:
        rows.append(
            {
                "prompt": prompt,
                "logits": compare_logits(ref_logits, cand_logits, top_k=top_k),
                "state": compare_tensors(ref_state, cand_state),
            }
        )
    observed = {
        "minimum_top_k_overlap": min(
            (float(row["logits"]["min_top_k_overlap"]) for row in rows), default=1.0
        ),
        "maximum_kl": max(
            (float(row["logits"]["max_kl_candidate_to_reference"]) for row in rows),
            default=0.0,
        ),
        "maximum_state_relative_l2": max(
            (float(row["state"]["relative_l2"]) for row in rows), default=0.0
        ),
    }
    gates = {
        "minimum_top_k_overlap": observed["minimum_top_k_overlap"] >= min_top_k_overlap,
        "maximum_kl": observed["maximum_kl"] <= max_kl,
        "maximum_state_relative_l2": (
            observed["maximum_state_relative_l2"] <= max_state_relative_l2
        ),
    }
    return {"observed": observed, "gates": gates, "passed": all(gates.values()), "prompts": rows}


def evaluate_rwkvcpp_pack(
    checkpoint: Path,
    pack_dir: Path,
    prompts: list[str],
    *,
    generation_tokens: int = 0,
    top_k: int = 10,
    min_top_k_overlap: float = 0.8,
    max_kl: float = 0.05,
    max_state_relative_l2: float = 0.1,
) -> dict[str, Any]:
    reference = RWKVCppBackend()
    reference.load(str(checkpoint), "cpu fp32", "cpu")
    config = EngineConfig(
        pack_dir=pack_dir,
        checkpoint_path=str(checkpoint),
        backend="rwkvcpp",
        mode="streaming",
        device="cpu",
        cache_format="prepared",
        residency_policy="static",
    )
    pairs = []
    resolved_pack = str(pack_dir)
    try:
        with InferenceEngine(config) as candidate_engine:
            candidate = candidate_engine.backend
            if not isinstance(candidate, RWKVCppBackend):
                raise TypeError("candidate engine did not create rwkv.cpp backend")
            resolved_pack = str(candidate_engine.config.pack_dir)
            assert candidate_engine.manifest is not None
            provider = candidate_engine._get_or_create_streaming_provider()
            by_layer = candidate_engine.manifest.by_layer()
            layer_ids = manifest_block_layers(candidate_engine.manifest)
            for prompt in prompts:
                ids = reference._encode_fn(prompt) or [0]
                ref_logits, ref_state = reference._model.eval_sequence_in_chunks(
                    ids, None, None, None, use_numpy=True
                )
                if candidate._native_layer_streaming_active():
                    # The bounded native path intentionally keeps globals in
                    # Python and uploads only block plans.  Use the same
                    # engine-facing API as production; calling the raw GGML
                    # full-graph evaluator here would reject a valid
                    # shape-only packed graph as "not uploaded".
                    candidate._load_layer_global_weights(provider, by_layer)
                    candidate._layer_reset_state(None)
                    cand_logits = candidate._layer_advance_sequence(
                        [int(token_id) for token_id in ids],
                        provider,
                        by_layer,
                        layer_ids,
                        None,
                    )
                    cand_state = candidate._layer_flat_state().copy()
                    if cand_logits is None or cand_state is None:
                        raise RuntimeError(
                            "rwkv.cpp layer-streaming quality probe produced no state/logits"
                        )
                else:
                    # The complete native graph still needs its global
                    # provider tensors loaded before packed-only evaluation.
                    candidate._load_layer_global_weights(provider, by_layer)
                    candidate._sync_provider_layers(provider, by_layer, layer_ids, None)
                    cand_logits, cand_state = candidate._model.eval_sequence_in_chunks(
                        ids, None, None, None, use_numpy=True
                    )
                pairs.append(
                    (
                        prompt,
                        torch.from_numpy(np.asarray(ref_logits).copy()).float(),
                        torch.from_numpy(np.asarray(cand_logits).copy()).float(),
                        torch.from_numpy(np.asarray(ref_state).copy()).float(),
                        torch.from_numpy(np.asarray(cand_state).copy()).float(),
                    )
                )

                # Compare a fixed, reference-model token trajectory rather
                # than allowing an early quantization mismatch to change all
                # later inputs.  This is the strongest useful CPU pack gate:
                # every row is evaluated at the same recurrent context, while
                # both implementations still advance their own state.  The
                # free-running greedy token stream is checked separately by
                # the normal generation tests and is not hidden by this
                # teacher-forced diagnostic.
                if generation_tokens > 0:
                    ref_current_state = np.asarray(ref_state, dtype=np.float32)
                    ref_next_state = np.empty_like(ref_current_state)
                    ref_current_logits = np.asarray(ref_logits, dtype=np.float32)
                    cand_current_logits = np.asarray(cand_logits, dtype=np.float32)
                    for step in range(int(generation_tokens)):
                        ref_token = sample_numpy(
                            ref_current_logits, temperature=1.0, greedy=True
                        )
                        pairs.append(
                            (
                                f"{prompt} [generation {step + 1}]",
                                torch.from_numpy(ref_current_logits.copy()).float(),
                                torch.from_numpy(cand_current_logits.copy()).float(),
                                torch.from_numpy(ref_current_state.copy()).float(),
                                torch.from_numpy(
                                    np.asarray(
                                        candidate._layer_flat_state()
                                        if candidate._native_layer_streaming_active()
                                        else candidate._state
                                    ).copy()
                                ).float(),
                            )
                        )
                        ref_current_logits, ref_next_state = reference._model.eval(
                            int(ref_token),
                            ref_current_state,
                            ref_next_state,
                            ref_current_logits,
                            use_numpy=True,
                        )
                        ref_current_state, ref_next_state = (
                            ref_next_state,
                            ref_current_state,
                        )
                        if candidate._native_layer_streaming_active():
                            cand_current_logits = candidate._layer_advance_token(
                                int(ref_token),
                                provider,
                                by_layer,
                                layer_ids,
                                None,
                            )
                        else:
                            cand_next_state = np.empty_like(cand_state)
                            cand_current_logits, cand_next_state = candidate._model.eval(
                                int(ref_token),
                                cand_state,
                                cand_next_state,
                                cand_current_logits,
                                use_numpy=True,
                            )
                            cand_state, cand_next_state = cand_next_state, cand_state
    finally:
        reference.close()
    result = summarize_quality_pairs(
        pairs,
        top_k=top_k,
        min_top_k_overlap=min_top_k_overlap,
        max_kl=max_kl,
        max_state_relative_l2=max_state_relative_l2,
    )
    result.update(
        {
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "candidate_pack": str(pack_dir),
            "resolved_candidate_pack": resolved_pack,
            "generation_tokens": max(0, int(generation_tokens)),
            "native_upload_scope": "all_manifest_layers_including_embedding_and_head",
            "resident_native_tensors": (
                "non-matrix/control tensors remain dense; grouped 2-D tensors "
                "use the native SG8 upload path"
            ),
        }
    )
    return result


def evaluate_rwkvcpp_checkpoint(
    reference_checkpoint: Path,
    candidate_checkpoint: Path,
    prompts: list[str],
    *,
    generation_tokens: int = 0,
    top_k: int = 10,
    min_top_k_overlap: float = 0.8,
    max_kl: float = 0.05,
    max_state_relative_l2: float = 0.1,
) -> dict[str, Any]:
    """Compare two resident native checkpoints (for Q4_K/Q5 baselines)."""
    reference = RWKVCppBackend()
    candidate = RWKVCppBackend()
    reference.load(str(reference_checkpoint), "cpu fp32", "cpu")
    candidate.load(str(candidate_checkpoint), "cpu fp32", "cpu")
    pairs = []
    try:
        for prompt in prompts:
            ids = reference._encode_fn(prompt) or [0]
            ref_logits, ref_state = reference._model.eval_sequence_in_chunks(
                ids, None, None, None, use_numpy=True
            )
            cand_logits, cand_state = candidate._model.eval_sequence_in_chunks(
                ids, None, None, None, use_numpy=True
            )
            pairs.append(
                (
                    prompt,
                    torch.from_numpy(np.asarray(ref_logits).copy()).float(),
                    torch.from_numpy(np.asarray(cand_logits).copy()).float(),
                    torch.from_numpy(np.asarray(ref_state).copy()).float(),
                    torch.from_numpy(np.asarray(cand_state).copy()).float(),
                )
            )
            if generation_tokens > 0:
                ref_current_state = np.asarray(ref_state, dtype=np.float32)
                cand_current_state = np.asarray(cand_state, dtype=np.float32)
                ref_next_state = np.empty_like(ref_current_state)
                cand_next_state = np.empty_like(cand_current_state)
                ref_current_logits = np.asarray(ref_logits, dtype=np.float32)
                cand_current_logits = np.asarray(cand_logits, dtype=np.float32)
                for step in range(int(generation_tokens)):
                    ref_token = sample_numpy(
                        ref_current_logits, temperature=1.0, greedy=True
                    )
                    pairs.append(
                        (
                            f"{prompt} [generation {step + 1}]",
                            torch.from_numpy(ref_current_logits.copy()).float(),
                            torch.from_numpy(cand_current_logits.copy()).float(),
                            torch.from_numpy(ref_current_state.copy()).float(),
                            torch.from_numpy(cand_current_state.copy()).float(),
                        )
                    )
                    ref_current_logits, ref_next_state = reference._model.eval(
                        int(ref_token),
                        ref_current_state,
                        ref_next_state,
                        ref_current_logits,
                        use_numpy=True,
                    )
                    ref_current_state, ref_next_state = (
                        ref_next_state,
                        ref_current_state,
                    )
                    cand_current_logits, cand_next_state = candidate._model.eval(
                        int(ref_token),
                        cand_current_state,
                        cand_next_state,
                        cand_current_logits,
                        use_numpy=True,
                    )
                    cand_current_state, cand_next_state = (
                        cand_next_state,
                        cand_current_state,
                    )
    finally:
        reference.close()
        candidate.close()
    result = summarize_quality_pairs(
        pairs,
        top_k=top_k,
        min_top_k_overlap=min_top_k_overlap,
        max_kl=max_kl,
        max_state_relative_l2=max_state_relative_l2,
    )
    result.update(
        {
            "schema_version": 1,
            "checkpoint": str(reference_checkpoint),
            "candidate_checkpoint": str(candidate_checkpoint),
            "generation_tokens": max(0, int(generation_tokens)),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-pack", type=Path)
    candidates.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--prompts", default="Hello,The future of storage is,Once upon a time")
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=0,
        help="teacher-forced autoregressive quality rows per prompt (0 keeps prompt-only mode)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-top-k-overlap", type=float, default=0.8)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--max-state-relative-l2", type=float, default=0.1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "top_k": args.top_k,
        "min_top_k_overlap": args.min_top_k_overlap,
        "max_kl": args.max_kl,
        "max_state_relative_l2": args.max_state_relative_l2,
        "generation_tokens": max(0, args.generation_tokens),
    }
    prompts = [item for item in args.prompts.split(",") if item]
    if args.candidate_checkpoint is not None:
        report = evaluate_rwkvcpp_checkpoint(
            args.checkpoint, args.candidate_checkpoint, prompts, **kwargs
        )
    else:
        report = evaluate_rwkvcpp_pack(
            args.checkpoint, args.candidate_pack, prompts, **kwargs
        )
    payload = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload)
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
