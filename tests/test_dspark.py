from __future__ import annotations

import torch
import pytest

from rwkv_ssd.runtime.dspark import (
    DSparkCapacityCurve,
    DSparkDrafter,
    DSparkRNNHead,
    schedule_prefix_lengths,
    verify_speculative,
)


def test_dspark_markov_head_corrects_parallel_block_and_reports_confidence() -> None:
    torch.manual_seed(7)
    drafter = DSparkDrafter(
        vocab_size=31,
        hidden_size=12,
        rank=5,
        head="markov",
        with_confidence=True,
    )
    base_logits = torch.randn(2, 4, 31)
    hidden = torch.randn(2, 4, 12)
    proposal = drafter.propose(base_logits, hidden, torch.tensor([2, 3]), greedy=True)
    assert list(proposal.tokens.shape) == [2, 4]
    assert list(proposal.logits.shape) == [2, 4, 31]
    assert list(proposal.probabilities.shape) == [2, 4, 31]
    assert proposal.conditional_confidence is not None
    assert list(proposal.conditional_confidence.shape) == [2, 4]
    assert torch.isfinite(proposal.logits).all()
    assert torch.all((proposal.conditional_confidence > 0) & (proposal.conditional_confidence < 1))
    torch.testing.assert_close(
        proposal.probabilities.sum(dim=-1),
        torch.ones(2, 4),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dspark_rnn_head_accepts_recurrent_hidden_stream() -> None:
    torch.manual_seed(11)
    head = DSparkRNNHead(vocab_size=17, hidden_size=8, rank=4)
    state = torch.zeros(3, 4)
    previous = torch.tensor([1, 2, 3])
    for _ in range(3):
        state, bias, features = head.forward_step(
            state, previous, torch.randn(3, 8)
        )
        assert list(state.shape) == [3, 4]
        assert list(bias.shape) == [3, 17]
        assert list(features.shape) == [3, 4]
        previous = torch.argmax(bias, dim=-1)
    assert torch.isfinite(state).all()


def test_dspark_rnn_correction_state_can_continue_across_blocks() -> None:
    torch.manual_seed(13)
    drafter = DSparkDrafter(
        vocab_size=19,
        hidden_size=8,
        rank=4,
        head="rnn",
        with_confidence=False,
    )
    base_logits = torch.randn(2, 4, 19)
    hidden = torch.randn(2, 4, 8)
    anchor = torch.tensor([1, 2])
    whole = drafter.propose(base_logits, hidden, anchor)
    first = drafter.propose(base_logits[:, :2], hidden[:, :2], anchor)
    assert first.correction_state is not None
    second = drafter.propose(
        base_logits[:, 2:],
        hidden[:, 2:],
        whole.tokens[:, 1],
        correction_state=first.correction_state,
    )
    torch.testing.assert_close(second.logits, whole.logits[:, 2:])
    torch.testing.assert_close(second.tokens, whole.tokens[:, 2:])


def test_dspark_scheduler_is_prefix_closed_and_respects_capacity_cliffs() -> None:
    curve = DSparkCapacityCurve.from_mapping({2: 1.0, 3: 0.95, 4: 0.7, 5: 0.4})
    schedule = schedule_prefix_lengths(
        [[0.8, 0.9, 0.9], [0.6, 0.6]], curve, early_stop=True
    )
    assert schedule.lengths == (1, 0)
    assert schedule.batch_tokens == 3
    assert schedule.expected_accepts == 2.8
    assert schedule.expected_throughput == pytest.approx(2.66)


def test_dspark_speculative_verification_returns_bonus_after_all_accepts() -> None:
    # The draft distribution is identical to the target distribution, so both
    # draft tokens must be accepted and the final target row supplies bonus.
    target = torch.tensor(
        [
            [3.0, 0.0, -2.0],
            [0.0, 3.0, -2.0],
            [0.0, 0.0, 3.0],
        ]
    )
    probabilities = torch.softmax(target[:2], dim=-1)
    draft_tokens = torch.tensor([0, 1])
    result = verify_speculative(
        target, draft_tokens, probabilities, generator=torch.Generator().manual_seed(5)
    )
    assert result.accepted_draft_tokens == 2
    assert result.rejected_at is None
    assert result.used_bonus is True
    assert result.tokens.tolist() == [0, 1, 2]


def test_dspark_speculative_verification_stops_at_first_rejection() -> None:
    target = torch.tensor(
        [
            [0.0, 4.0],
            [4.0, 0.0],
        ]
    )
    draft = torch.tensor([[4.0, 0.0]])
    # A highly confident but wrong draft token is rejected with overwhelming
    # probability; the residual distribution then returns target token 1.
    result = verify_speculative(
        target, torch.tensor([0]), torch.softmax(draft, dim=-1), generator=torch.Generator().manual_seed(0)
    )
    assert result.accepted_draft_tokens == 0
    assert result.rejected_at == 0
    assert result.used_bonus is False
    assert result.tokens.tolist() == [1]


def test_dspark_verification_normalizes_low_precision_draft_mass() -> None:
    target = torch.tensor(
        [
            [0.0, 4.0],
            [4.0, 0.0],
        ]
    )
    # Deliberately pass unnormalized proposal mass, as can happen after a
    # low-precision softmax. The verifier must form a proper residual.
    result = verify_speculative(
        target,
        torch.tensor([0]),
        torch.tensor([[4.0, 0.0]], dtype=torch.bfloat16),
        generator=torch.Generator().manual_seed(0),
    )
    assert result.rejected_at == 0
    assert result.tokens.tolist() == [1]


def test_dspark_rejects_empty_or_invalid_research_inputs() -> None:
    drafter = DSparkDrafter(vocab_size=7, hidden_size=4, rank=3)
    with pytest.raises(ValueError, match="at least one step"):
        drafter.propose(
            torch.empty(1, 0, 7), torch.empty(1, 0, 4), torch.tensor([1])
        )
    with pytest.raises(ValueError, match="positive mass"):
        verify_speculative(
            torch.zeros(2, 3), torch.tensor([0]), torch.zeros(1, 3)
        )
