"""Shared token sampling primitives used by every backend family.

Sampling is deliberately kept outside the model backends.  A request-scoped
context supplies one RNG for the complete decode, which makes seeded sampling
deterministic even when a backend calls the sampler once per token.  The
context is also useful for backends implemented in different numeric stacks:
Torch and NumPy use their native fast paths but consume the same request
options and apply the same top-p definition.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from typing import Iterator

import numpy as np
import torch


def validate_top_p(value: float | None) -> float:
    """Validate and normalize nucleus-sampling probability.

    OpenAI-compatible APIs define top-p in the interval ``(0, 1]``.  Keeping
    the validation here means direct engine callers and HTTP callers receive
    the same error instead of silently producing an empty probability mass.
    """

    if value is None:
        return 1.0
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError(f"top_p must be a finite value in (0, 1], got {value!r}")
    return result


@dataclass
class _SamplingState:
    seed: int | None
    top_p: float
    numpy_rng: np.random.Generator = field(init=False)
    torch_generators: dict[str, torch.Generator | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.numpy_rng = np.random.default_rng(self.seed)

    def torch_generator(self, device: torch.device) -> torch.Generator | None:
        """Return a generator compatible with ``device`` when seeded.

        CPU is the production path.  Some optional Torch devices do not
        expose a constructible per-device generator; those paths return
        ``None`` and use the NumPy fallback below rather than failing a
        request merely because the sampler cannot bind a device generator.
        """

        if self.seed is None:
            return None
        key = str(device)
        if key in self.torch_generators:
            return self.torch_generators[key]
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.seed))
        except (RuntimeError, TypeError, ValueError):
            generator = None
        self.torch_generators[key] = generator
        return generator


_ACTIVE_SAMPLING: ContextVar[_SamplingState | None] = ContextVar(
    "rwkv_ssd_sampling_state", default=None
)


@contextmanager
def sampling_context(
    *, seed: int | None = None, top_p: float | None = None
) -> Iterator[None]:
    """Install request-scoped sampling options.

    Nested backend calls reuse the outer state.  This matters for the engine's
    follow-up path, where a public method can call the token API internally;
    resetting the RNG at that boundary would repeat the first sampled token.
    """

    current = _ACTIVE_SAMPLING.get()
    if current is not None:
        yield
        return
    state = _SamplingState(seed=None if seed is None else int(seed), top_p=validate_top_p(top_p))
    token = _ACTIVE_SAMPLING.set(state)
    try:
        yield
    finally:
        _ACTIVE_SAMPLING.reset(token)


def _resolve_options(
    *, seed: int | None, top_p: float | None
) -> tuple[_SamplingState | None, float]:
    state = _ACTIVE_SAMPLING.get()
    if state is not None:
        return state, state.top_p
    return None, validate_top_p(top_p)


def _top_p_torch(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    # Always retain the highest-probability token, even if top_p is below its
    # mass due to floating-point rounding.
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    normalizer = filtered.sum(dim=-1, keepdim=True)
    return filtered / normalizer.clamp_min(torch.finfo(filtered.dtype).eps)


def _top_p_numpy(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return probs
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order].copy()
    remove = np.cumsum(sorted_probs) > top_p
    remove[0] = False
    sorted_probs[remove] = 0.0
    filtered = np.zeros_like(probs)
    filtered[order] = sorted_probs
    total = float(filtered.sum())
    if not np.isfinite(total) or total <= 0.0:
        return probs
    return filtered / total


def sample_torch(
    logits: torch.Tensor,
    *,
    temperature: float,
    greedy: bool,
    top_p: float | None = None,
    seed: int | None = None,
) -> int:
    """Select one token with the same semantics for Torch-backed engines."""

    # Direct backend callers can opt in without knowing about the context
    # manager.  Engine calls normally enter the context once per request.
    if _ACTIVE_SAMPLING.get() is None and (top_p is not None or seed is not None):
        with sampling_context(seed=seed, top_p=top_p):
            return sample_torch(
                logits,
                temperature=temperature,
                greedy=greedy,
            )

    values = logits.reshape(-1, logits.shape[-1])[0]
    if greedy or temperature <= 0:
        return int(values.argmax().item())
    state, resolved_top_p = _resolve_options(seed=seed, top_p=top_p)
    probs = torch.softmax(values.float() / float(temperature), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = probs.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
        return int(values.argmax().item())
    probs = probs / total
    probs = _top_p_torch(probs, resolved_top_p)
    generator = state.torch_generator(values.device) if state is not None else None
    try:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    except (RuntimeError, TypeError):
        # A few optional device backends reject a CPU/per-device generator.
        # Sampling on the host is a correctness-preserving fallback and keeps
        # fixed-seed behavior available on those devices.
        rng = state.numpy_rng if state is not None else np.random.default_rng()
        return int(rng.choice(probs.numel(), p=probs.detach().cpu().numpy()))


def sample_numpy(
    logits: np.ndarray,
    *,
    temperature: float,
    greedy: bool,
    top_p: float | None = None,
    seed: int | None = None,
) -> int:
    """Select one token with numerically stable NumPy sampling."""

    if _ACTIVE_SAMPLING.get() is None and (top_p is not None or seed is not None):
        with sampling_context(seed=seed, top_p=top_p):
            return sample_numpy(logits, temperature=temperature, greedy=greedy)

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if greedy or temperature <= 0:
        return int(np.argmax(values))
    state, resolved_top_p = _resolve_options(seed=seed, top_p=top_p)
    scaled = values / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(values))
    probs = _top_p_numpy(probs / total, resolved_top_p)
    rng = state.numpy_rng if state is not None else np.random.default_rng()
    return int(rng.choice(values.size, p=probs))


__all__ = [
    "sample_numpy",
    "sample_torch",
    "sampling_context",
    "validate_top_p",
]
