"""CPU smoke/parity tests for the downloaded small HF checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.real_sequence_models import (
    ReferenceLlama,
    ReferenceMamba2,
    dspark_propose_reference,
)
from rwkv_ssd.tools.pack_runtime import _load_state_dict, detect_model_family_from_state


REPO = Path(__file__).resolve().parents[1]
MAMBA = REPO / "test_model" / "Small mamba"
TRANSFORMER = REPO / "test_model" / "Small_transformer"


@pytest.mark.integration
@pytest.mark.skipif(not (MAMBA / "model.safetensors").is_file(), reason="real Mamba-2 checkpoint not downloaded")
def test_real_mamba2_cpu_forward_and_cached_decode() -> None:
    model = ReferenceMamba2.from_pretrained_local(MAMBA)
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    full, _ = model.forward(ids)
    hidden, _ = model.forward_hidden(ids)
    projected = model.logits_from_hidden(hidden)
    torch.testing.assert_close(full, projected, rtol=0, atol=0)
    state = None
    pieces = []
    for index in range(ids.shape[1]):
        logits, state = model.forward(ids[:, index : index + 1], state)
        pieces.append(logits)
    cached = torch.cat(pieces, dim=1)
    assert list(full.shape) == [1, 4, model.vocab_size]
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full, cached, rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.skipif(not (TRANSFORMER / "model.safetensors").is_file(), reason="real Transformer checkpoint not downloaded")
def test_real_transformer_cpu_forward_and_cached_decode() -> None:
    model = ReferenceLlama.from_pretrained_local(TRANSFORMER)
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    full, _ = model.forward(ids)
    hidden, _ = model.forward_hidden(ids)
    projected = model.logits_from_hidden(hidden)
    torch.testing.assert_close(full, projected, rtol=1e-6, atol=1e-5)
    state = None
    pieces = []
    for index in range(ids.shape[1]):
        logits, state = model.forward(ids[:, index : index + 1], state)
        pieces.append(logits)
    cached = torch.cat(pieces, dim=1)
    assert list(full.shape) == [1, 4, model.vocab_size]
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full, cached, rtol=1e-6, atol=1e-5)


@pytest.mark.integration
@pytest.mark.skipif(
    not (MAMBA / "model.safetensors").is_file()
    or not (TRANSFORMER / "model.safetensors").is_file(),
    reason="real Mamba-2 and Transformer checkpoints are required",
)
def test_real_checkpoint_family_detection() -> None:
    assert detect_model_family_from_state(_load_state_dict(MAMBA)) == "mamba2"
    assert detect_model_family_from_state(_load_state_dict(TRANSFORMER)) == "llama"


@pytest.mark.integration
@pytest.mark.skipif(
    not (MAMBA / "model.safetensors").is_file()
    or not (TRANSFORMER / "model.safetensors").is_file(),
    reason="real Mamba-2 and Transformer checkpoints are required",
)
def test_real_reference_dspark_adapter_supports_both_architectures() -> None:
    for path, loader in (
        (MAMBA, ReferenceMamba2.from_pretrained_local),
        (TRANSFORMER, ReferenceLlama.from_pretrained_local),
    ):
        model = loader(path)
        proposal = dspark_propose_reference(
            model,
            torch.tensor([1]),
            proposal_length=3,
            mask_token_id=0,
            rank=8,
        )
        assert list(proposal.tokens.shape) == [1, 3]
        assert proposal.prefix_survival is not None
        assert torch.isfinite(proposal.logits).all()
