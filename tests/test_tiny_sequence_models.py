from __future__ import annotations

import torch

from research.tiny_sequence_models import (
    TinyConfig,
    TinyMambaLM,
    benchmark_model,
    build_tiny_model,
    dspark_propose_tiny,
    parameter_count,
)
from rwkv_ssd.runtime.dspark import DSparkDrafter


def _config() -> TinyConfig:
    return TinyConfig(vocab_size=97, d_model=32, n_layer=2, d_ff=64, n_head=4, d_state=4)


def test_all_research_baselines_have_language_model_logits() -> None:
    tokens = torch.randint(0, 97, (2, 8))
    for kind in ("mamba", "attention", "hybrid"):
        model = build_tiny_model(kind, _config())
        logits = model(tokens)
        assert list(logits.shape) == [2, 8, 97]
        assert parameter_count(model) > 0


def test_mamba_recurrent_step_has_state_and_logits() -> None:
    model = TinyMambaLM(_config())
    logits, states = model.decode_step(torch.tensor([1, 2]))
    assert list(logits.shape) == [2, 97]
    assert len(states) == 2
    next_logits, next_states = model.decode_step(torch.tensor([3, 4]), states)
    assert list(next_logits.shape) == [2, 97]
    assert all(left.shape == right.shape for left, right in zip(states, next_states))


def test_tiny_benchmark_is_structured() -> None:
    report = benchmark_model(build_tiny_model("attention", _config()), seq_len=8)
    assert report["architecture"] == "attention"
    assert report["parameters"] > 0
    assert report["tokens_per_second"] > 0


def test_dspark_adapts_to_attention_and_mamba_backbones() -> None:
    config = _config()
    for kind in ("attention", "mamba"):
        model = build_tiny_model(kind, config)
        drafter = DSparkDrafter(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            rank=8,
        )
        proposal = dspark_propose_tiny(
            model,
            torch.tensor([1, 2]),
            proposal_length=4,
            mask_token_id=0,
            drafter=drafter,
        )
        assert list(proposal.tokens.shape) == [2, 4]
        assert proposal.prefix_survival is not None
        assert list(proposal.prefix_survival.shape) == [2, 4]
        assert torch.isfinite(proposal.logits).all()
