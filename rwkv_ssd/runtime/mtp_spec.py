"""Recurrent MTP speculative decode scaffold (P2.e — prefill/decode gated)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MTPConfig:
    """Workload gate: only enable when prefill or decode is dominated."""

    enabled: bool = False
    draft_tokens: int = 4
    min_prefill_tokens: int = 32

    def should_speculate(self, *, prefill_tokens: int, decode_tokens: int) -> bool:
        if not self.enabled or self.draft_tokens <= 1:
            return False
        if prefill_tokens >= self.min_prefill_tokens and decode_tokens == 0:
            return True
        return decode_tokens >= self.min_prefill_tokens


@dataclass
class MTPStats:
    draft_steps: int = 0
    accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_steps <= 0:
            return 0.0
        return self.accepted_tokens / self.draft_steps
