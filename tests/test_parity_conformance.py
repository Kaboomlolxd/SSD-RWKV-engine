from __future__ import annotations

import numpy as np
import pytest
import torch

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.parity import (
    BackendProbeResult,
    ParityThresholds,
    StateObservation,
    relative_state_error,
    run_backend_conformance,
    probe_engine_backend,
)


def _config(pack, mode: str) -> EngineConfig:
    return EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode=mode,
        device="cpu",
        strategy="cpu fp32",
        max_tokens=3,
        greedy=True,
        prefetch_enabled=False,
    )


def test_engine_probe_compares_resident_and_streaming_contracts(synthetic_pack) -> None:
    reference = probe_engine_backend(
        _config(synthetic_pack, "resident"), "parity", max_tokens=3
    )
    candidate = probe_engine_backend(
        _config(synthetic_pack, "streaming"), "parity", max_tokens=3
    )
    traces = run_backend_conformance(
        lambda _prompt, _count: reference,
        lambda _prompt, _count: candidate,
        ["parity"],
        max_tokens=3,
        reference_backend="synthetic-resident",
        candidate_backend="synthetic-streaming",
        thresholds=ParityThresholds(
            min_top10_overlap=0.80,
            max_kl=0.05,
            max_state_relative_error=0.10,
        ),
    )
    trace = traces[0]
    assert trace.exact_greedy_match
    assert trace.decoded_text_match is True
    assert trace.passed
    assert trace.prompt_token_ids
    assert len(trace.steps) == 3
    assert all(step.top10_overlap is not None for step in trace.steps)
    assert all(step.kl is not None for step in trace.steps)
    assert all(step.state_relative_error is not None for step in trace.steps)


def test_parity_text_decoding_is_reported_separately_from_token_ids() -> None:
    traces = run_backend_conformance(
        lambda _prompt, _count: BackendProbeResult([1, 2], text="ab"),
        lambda _prompt, _count: BackendProbeResult([1, 2], text="xy"),
        ["prompt"],
        max_tokens=2,
    )
    trace = traces[0]
    assert trace.exact_greedy_match
    assert trace.decoded_text_match is False
    assert not trace.passed


def test_parity_certification_requires_complete_guardrail_probes() -> None:
    traces = run_backend_conformance(
        lambda _prompt, _count: BackendProbeResult([1, 2], text="ab"),
        lambda _prompt, _count: BackendProbeResult([1, 2], text="ab"),
        ["prompt"],
        max_tokens=2,
        require_guardrails=True,
    )
    trace = traces[0]
    assert trace.exact_greedy_match
    assert trace.diagnostics_complete is False
    assert "logit probes" in " ".join(trace.notes)
    assert not trace.passed


def test_parity_thresholds_fail_closed_for_invalid_values() -> None:
    with pytest.raises(ValueError, match="min_top10_overlap"):
        ParityThresholds(min_top10_overlap=1.1)
    with pytest.raises(ValueError, match="finite"):
        ParityThresholds(max_kl=float("nan"))


def test_relative_state_error_flattens_backend_owned_state_shapes() -> None:
    reference = [torch.ones(2), torch.ones((2, 2))]
    candidate = [torch.ones(2), torch.ones((2, 2)) * 1.01]
    error = relative_state_error(reference, candidate)
    assert 0.0 < error < 0.02

    assert np.isfinite(error)


def test_relative_state_error_accepts_bfloat16_tensors() -> None:
    reference = [torch.ones(4, dtype=torch.bfloat16)]
    candidate = [torch.ones(4, dtype=torch.bfloat16)]

    assert relative_state_error(reference, candidate) == 0.0


def test_rwkv7_state_observation_maps_cpp_layout_to_chatrwkv_layout() -> None:
    """The guardrail compares semantic fields, not backend buffer order."""
    n_layer = 2
    n_embed = 4
    head_count = 1
    head_size = 4
    chat_state: list[torch.Tensor] = []
    cpp_layers: list[np.ndarray] = []
    for layer in range(n_layer):
        att_prev = torch.arange(n_embed, dtype=torch.float32) + layer
        att_matrix = torch.arange(head_count * head_size * head_size, dtype=torch.float32)
        att_matrix = att_matrix.reshape(head_count, head_size, head_size) + 10 * layer
        ffn_prev = torch.arange(n_embed, dtype=torch.float32) + 100 + layer
        chat_state.extend((att_prev, att_matrix, ffn_prev))
        cpp_layers.append(
            np.concatenate(
                (ffn_prev.numpy(), att_prev.numpy(), att_matrix.numpy().reshape(-1))
            )
        )

    reference = StateObservation(chat_state, layout="chatrwkv_rwkv7")
    candidate = StateObservation(
        np.concatenate(cpp_layers),
        layout="rwkv_cpp_rwkv7",
        n_layer=n_layer,
        n_embed=n_embed,
        head_count=head_count,
        head_size=head_size,
    )
    assert relative_state_error(reference, candidate) == 0.0
