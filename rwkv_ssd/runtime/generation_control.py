"""Cancellation, deadlines, and incremental token delivery for generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


class GenerationCancelled(RuntimeError):
    """Raised when a generation request is cancelled or exceeds its deadline."""


TokenCallback = Callable[[int], None]


@dataclass
class GenerationControl:
    """Small, backend-neutral control object checked at token boundaries."""

    token_callback: TokenCallback | None = None
    cancel_event: threading.Event | None = None
    deadline: float | None = None

    def check(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise GenerationCancelled("generation cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise GenerationCancelled("generation deadline exceeded")

    def emit(self, token_id: int) -> None:
        self.check()
        if self.token_callback is not None:
            self.token_callback(int(token_id))


def make_generation_control(
    *,
    token_callback: TokenCallback | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> GenerationControl | None:
    if token_callback is None and cancel_event is None and deadline is None:
        return None
    return GenerationControl(
        token_callback=token_callback,
        cancel_event=cancel_event,
        deadline=deadline,
    )


__all__ = [
    "GenerationCancelled",
    "GenerationControl",
    "TokenCallback",
    "make_generation_control",
]
