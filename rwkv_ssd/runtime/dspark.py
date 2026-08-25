"""DSpark-style speculative decoding primitives.

This module contains the model-agnostic pieces from DSpark
(arXiv:2607.05147v1): a parallel base-logit block, a cheap sequential
correction head, calibrated conditional-confidence inputs, hardware-aware
prefix scheduling, and standard lossless speculative verification.

The RWKV backends intentionally do not enable these components by default.
A trained drafter/head and a target-verification loop are required before they
can improve end-to-end latency. The research interface accepts hidden states
from an RWKV recurrent block without pretending it has parallel draft cost.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DSparkConfig:
    """Configuration for a DSpark proposal head.

    ``head`` is ``markov`` by default because the paper found its first-order
    head nearly as effective as the more expensive RNN head.  This config is
    deliberately separate from :class:`MTPConfig`: it describes a trained
    proposal mechanism, not merely the existing workload gate.
    """

    enabled: bool = False
    proposal_length: int = 4
    head: str = "markov"
    rank: int = 256
    temperature: float = 1.0
    confidence: bool = True
    early_stop: bool = True

    def validate(self) -> "DSparkConfig":
        if self.proposal_length <= 0:
            raise ValueError("DSpark proposal_length must be positive")
        if self.rank <= 0:
            raise ValueError("DSpark rank must be positive")
        if self.temperature <= 0:
            raise ValueError("DSpark temperature must be positive")
        if self.head.strip().lower() not in {"markov", "rnn"}:
            raise ValueError("DSpark head must be 'markov' or 'rnn'")
        return self


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order transition bias from DSpark Equation (5)."""

    def __init__(self, vocab_size: int, rank: int = 256) -> None:
        super().__init__()
        if vocab_size <= 0 or rank <= 0:
            raise ValueError("vocab_size and rank must be positive")
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.token_embedding = nn.Embedding(self.vocab_size, self.rank)
        self.logit_projection = nn.Linear(self.rank, self.vocab_size, bias=False)

    def token_features(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(previous_tokens.to(dtype=torch.long))

    def forward(self, previous_tokens: torch.Tensor) -> torch.Tensor:
        return self.logit_projection(self.token_features(previous_tokens))


class DSparkRNNHead(nn.Module):
    """Gated recurrent correction head from DSpark Equation (6)."""

    def __init__(self, vocab_size: int, hidden_size: int, rank: int = 256) -> None:
        super().__init__()
        if vocab_size <= 0 or hidden_size <= 0 or rank <= 0:
            raise ValueError("vocab_size, hidden_size, and rank must be positive")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.token_embedding = nn.Embedding(self.vocab_size, self.rank)
        joint_size = 2 * self.rank + self.hidden_size
        self.gated_update = nn.Linear(joint_size, 3 * self.rank)
        self.logit_projection = nn.Linear(self.rank, self.vocab_size, bias=False)

    def forward_step(
        self,
        state: torch.Tensor,
        previous_tokens: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_features = self.token_embedding(previous_tokens.to(dtype=torch.long))
        joined = torch.cat((state, previous_features, hidden), dim=-1)
        gate, candidate, output = self.gated_update(joined).chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        next_state = gate * state + (1.0 - gate) * torch.tanh(candidate)
        bias = self.logit_projection(torch.tanh(output))
        return next_state, bias, previous_features


class DSparkConfidenceHead(nn.Module):
    """Predict conditional per-position draft survival confidence."""

    def __init__(self, hidden_size: int, rank: int = 256) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        self.projection = nn.Linear(hidden_size + rank, 1)

    def forward(
        self, hidden: torch.Tensor, previous_features: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.projection(torch.cat((hidden, previous_features), dim=-1))
        ).squeeze(-1)


@dataclass
class DSparkProposal:
    """A sequentially corrected proposal block.

    ``probabilities`` are retained because exact speculative verification
    needs the draft probability of each sampled token, not only its argmax.
    ``conditional_confidence`` is a per-position quantity; callers must take
    its cumulative product before scheduling a verification prefix.
    ``correction_state`` carries the final RNN-head state into a subsequent
    proposal block. It is correction-head state only, not a target
    RWKV backbone state.
    """

    tokens: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    conditional_confidence: torch.Tensor | None = None
    correction_state: torch.Tensor | None = None

    @property
    def prefix_survival(self) -> torch.Tensor | None:
        if self.conditional_confidence is None:
            return None
        return torch.cumprod(self.conditional_confidence, dim=-1)


class DSparkDrafter(nn.Module):
    """Apply a DSpark sequential correction to parallel base logits.

    ``base_logits`` and ``hidden_states`` come from the model-specific
    parallel or state-rolled backbone and have shape ``[batch, steps, ...]``.
    The correction loop is intentionally tiny relative to a target model
    forward. Only the RWKV source of the base block is maintained here.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        rank: int = 256,
        head: str = "markov",
        with_confidence: bool = True,
    ) -> None:
        super().__init__()
        choice = head.strip().lower()
        if choice not in {"markov", "rnn"}:
            raise ValueError("DSpark head must be 'markov' or 'rnn'")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.head_kind = choice
        self.head = (
            DSparkMarkovHead(vocab_size, rank)
            if choice == "markov"
            else DSparkRNNHead(vocab_size, hidden_size, rank)
        )
        self.confidence_head = (
            DSparkConfidenceHead(hidden_size, rank) if with_confidence else None
        )

    def _validate_inputs(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
    ) -> tuple[int, int]:
        if base_logits.ndim != 3 or hidden_states.ndim != 3:
            raise ValueError("base_logits and hidden_states must have shape [batch, steps, width]")
        if base_logits.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("base_logits and hidden_states must share batch/steps")
        if base_logits.shape[1] <= 0:
            raise ValueError("DSpark proposal blocks must contain at least one step")
        if base_logits.shape[-1] != self.vocab_size:
            raise ValueError("base_logits vocabulary width does not match DSpark head")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states width does not match DSpark head")
        if anchor_tokens.ndim != 1 or anchor_tokens.shape[0] != base_logits.shape[0]:
            raise ValueError("anchor_tokens must have shape [batch]")
        return int(base_logits.shape[0]), int(base_logits.shape[1])

    def propose(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
        *,
        temperature: float = 1.0,
        greedy: bool = True,
        generator: torch.Generator | None = None,
        correction_state: torch.Tensor | None = None,
    ) -> DSparkProposal:
        """Correct and sample a block left-to-right.

        The backbone remains parallel/state-rolled; only the low-rank head and
        token choice are sequential.  ``greedy=False`` is the mode required by
        exact speculative sampling because the draft probabilities must match
        the sampled proposal distribution. For an RNN correction head,
        ``correction_state`` continues the lightweight correction context from
        an earlier block; the returned proposal exposes the updated state.
        """

        batch, steps = self._validate_inputs(base_logits, hidden_states, anchor_tokens)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        previous = anchor_tokens.to(device=base_logits.device, dtype=torch.long)
        if isinstance(self.head, DSparkRNNHead):
            if correction_state is None:
                rnn_state = hidden_states.new_zeros(batch, self.rank)
            else:
                if correction_state.ndim != 2 or tuple(correction_state.shape) != (batch, self.rank):
                    raise ValueError(
                        "correction_state must have shape [batch, DSpark rank]"
                    )
                rnn_state = correction_state.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
        else:
            if correction_state is not None:
                raise ValueError("correction_state is only supported by the RNN head")
            rnn_state = None
        corrected: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for position in range(steps):
            if isinstance(self.head, DSparkMarkovHead):
                previous_features = self.head.token_features(previous)
                bias = self.head.logit_projection(previous_features)
            else:
                rnn_state, bias, previous_features = self.head.forward_step(
                    rnn_state, previous, hidden_states[:, position, :]
                )
            logits = (base_logits[:, position, :] + bias) / float(temperature)
            probs = F.softmax(logits.float(), dim=-1).to(dtype=base_logits.dtype)
            if greedy:
                token = torch.argmax(logits, dim=-1)
            else:
                token = torch.multinomial(probs.float(), 1, generator=generator).squeeze(-1)
            if self.confidence_head is not None:
                confidence = self.confidence_head(hidden_states[:, position, :], previous_features)
                confidences.append(confidence)
            corrected.append(logits)
            probabilities.append(probs)
            tokens.append(token)
            previous = token
        return DSparkProposal(
            tokens=torch.stack(tokens, dim=1),
            logits=torch.stack(corrected, dim=1),
            probabilities=torch.stack(probabilities, dim=1),
            conditional_confidence=torch.stack(confidences, dim=1)
            if confidences
            else None,
            correction_state=rnn_state,
        )


@dataclass(frozen=True)
class DSparkCapacityCurve:
    """Profiled steps-per-second lookup used by the prefix scheduler.

    The curve is intentionally discrete.  Missing batch sizes use the nearest
    lower profiled point, preserving jagged hardware cliffs instead of
    smoothing them away with interpolation.
    """

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("DSpark capacity curve cannot be empty")
        previous = 0
        for batch, steps_per_second in self.points:
            if (
                batch <= previous
                or not math.isfinite(float(steps_per_second))
                or steps_per_second < 0
            ):
                raise ValueError(
                    "capacity points must have increasing positive batch sizes "
                    "and finite non-negative capacities"
                )
            previous = batch

    @classmethod
    def from_mapping(cls, values: Mapping[int, float]) -> "DSparkCapacityCurve":
        return cls(tuple(sorted((int(k), float(v)) for k, v in values.items())))

    def __call__(self, batch_tokens: int) -> float:
        batch_tokens = max(1, int(batch_tokens))
        batches = [point[0] for point in self.points]
        index = bisect_right(batches, batch_tokens) - 1
        if index < 0:
            index = 0
        return float(self.points[index][1])


@dataclass(frozen=True)
class DSparkSchedule:
    """Result of hardware-aware prefix admission."""

    lengths: tuple[int, ...]
    expected_accepts: float
    batch_tokens: int
    expected_throughput: float
    candidates_considered: int


def schedule_prefix_lengths(
    conditional_confidences: Sequence[Sequence[float]],
    capacity: DSparkCapacityCurve | Mapping[int, float] | Callable[[int], float],
    *,
    early_stop: bool = True,
) -> DSparkSchedule:
    """Select request-local verification prefixes using DSpark Algorithm 1.

    Confidence values are conditional per-position probabilities.  The
    scheduler multiplies them into prefix survival probabilities, globally
    admits the highest-value prefixes, and keeps only the best throughput
    point.  ``early_stop=True`` is the lossless/non-anticipating mode when
    future confidence values are not revealed until the scheduler reaches
    them.  Set it to ``False`` only when all supplied confidence values were
    computed from information available before the corresponding token was
    sampled (the asynchronous barrier described in DSpark Section 5.2).
    """

    if not conditional_confidences:
        raise ValueError("at least one request is required")
    curve: Callable[[int], float]
    curve = capacity if callable(capacity) else DSparkCapacityCurve.from_mapping(capacity)
    survival: list[tuple[float, int, int]] = []
    for request_id, row in enumerate(conditional_confidences):
        running = 1.0
        for position, value in enumerate(row, start=1):
            confidence = min(1.0, max(0.0, float(value)))
            running *= confidence
            if running > 0.0:
                survival.append((running, request_id, position))
    # Position ascending breaks ties in favor of the shorter prefix, which
    # keeps the admission path explicitly prefix-closed.
    survival.sort(key=lambda item: (-item[0], item[1], item[2]))
    request_count = len(conditional_confidences)
    lengths = [0] * request_count
    batch_tokens = request_count
    expected_accepts = float(request_count)
    best_throughput = expected_accepts * float(curve(batch_tokens))
    best_lengths = list(lengths)
    best_accepts = expected_accepts
    best_batch = batch_tokens
    considered = 0
    for probability, request_id, position in survival:
        if position <= lengths[request_id]:
            continue
        # Because survival probabilities are non-increasing per request, all
        # preceding positions have already been admitted or have an equal
        # probability tie that sorts before this candidate.
        lengths[request_id] = position
        batch_tokens += 1
        expected_accepts += probability
        considered += 1
        throughput = expected_accepts * float(curve(batch_tokens))
        if throughput > best_throughput:
            best_throughput = throughput
            best_lengths = list(lengths)
            best_accepts = expected_accepts
            best_batch = batch_tokens
        elif early_stop:
            break
    return DSparkSchedule(
        lengths=tuple(best_lengths),
        expected_accepts=float(best_accepts),
        batch_tokens=int(best_batch),
        expected_throughput=float(best_throughput),
        candidates_considered=considered,
    )


@dataclass(frozen=True)
class DSparkVerification:
    """One exact speculative-verification result."""

    tokens: torch.Tensor
    accepted_draft_tokens: int
    rejected_at: int | None
    used_bonus: bool


def verify_speculative(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> DSparkVerification:
    """Run standard lossless speculative sampling for one request.

    ``target_logits`` has ``draft_length + 1`` rows: one target distribution
    for each draft position and a final bonus-token distribution.  On the
    first rejection, the correction is sampled from ``max(p_target -
    p_draft, 0)``.  If every draft token survives, the final target row
    supplies the bonus token.  No DSpark confidence or scheduler decision is
    used here, preserving the target distribution exactly.
    """

    if target_logits.ndim != 2 or draft_tokens.ndim != 1 or draft_probabilities.ndim != 2:
        raise ValueError("target_logits=[steps+1,vocab], draft_tokens=[steps], draft_probabilities=[steps,vocab]")
    if draft_tokens.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("draft_tokens must use an integer dtype")
    steps = int(draft_tokens.shape[0])
    if target_logits.shape[0] != steps + 1 or draft_probabilities.shape[0] != steps:
        raise ValueError("target and draft sequence lengths do not match")
    if target_logits.shape[1] != draft_probabilities.shape[1]:
        raise ValueError("target and draft vocabulary widths do not match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(target_logits).all():
        raise ValueError("target_logits must be finite")
    target_probs = F.softmax(target_logits.float() / float(temperature), dim=-1)
    draft_tokens = draft_tokens.to(device=target_logits.device)
    draft_probs = draft_probabilities.to(device=target_logits.device, dtype=torch.float32)
    if not torch.isfinite(draft_probs).all() or torch.any(draft_probs < 0):
        raise ValueError("draft probabilities must be finite and non-negative")
    draft_totals = draft_probs.sum(dim=-1, keepdim=True)
    if torch.any(draft_totals <= 0):
        raise ValueError("each draft probability row must have positive mass")
    # A BF16/FP16 proposal can sum to something slightly different from one.
    # Normalize in FP32 so the residual distribution remains a valid exact
    # correction rather than inheriting the proposal dtype's rounding error.
    draft_probs = draft_probs / draft_totals
    output: list[torch.Tensor] = []
    for position in range(steps):
        token = int(draft_tokens[position].item())
        if token < 0 or token >= target_probs.shape[-1]:
            raise ValueError("draft token is outside the target vocabulary")
        probability = target_probs[position, token]
        proposal_probability = draft_probs[position, token]
        if bool(proposal_probability <= 0):
            raise ValueError("draft token must have positive draft probability")
        acceptance = torch.minimum(torch.ones_like(probability), probability / proposal_probability)
        draw = torch.rand((), device=target_probs.device, generator=generator)
        if bool(draw < acceptance):
            output.append(draft_tokens[position])
            continue
        residual = (target_probs[position] - draft_probs[position]).clamp_min(0.0)
        residual_total = residual.sum()
        if float(residual_total) <= 0.0:
            replacement = torch.argmax(target_probs[position])
        else:
            replacement = torch.multinomial(residual / residual_total, 1, generator=generator)[0]
        output.append(replacement.to(dtype=draft_tokens.dtype))
        return DSparkVerification(
            tokens=torch.stack(output),
            accepted_draft_tokens=position,
            rejected_at=position,
            used_bonus=False,
        )
    bonus = torch.multinomial(target_probs[-1], 1, generator=generator)[0]
    output.append(bonus.to(dtype=draft_tokens.dtype))
    return DSparkVerification(
        tokens=torch.stack(output),
        accepted_draft_tokens=steps,
        rejected_at=None,
        used_bonus=True,
    )


__all__ = [
    "DSparkCapacityCurve",
    "DSparkConfidenceHead",
    "DSparkConfig",
    "DSparkDrafter",
    "DSparkMarkovHead",
    "DSparkProposal",
    "DSparkRNNHead",
    "DSparkSchedule",
    "DSparkVerification",
    "schedule_prefix_lengths",
    "verify_speculative",
]
