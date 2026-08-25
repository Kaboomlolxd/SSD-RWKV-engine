"""Typed recurrent state for packed sequence-model backends.

RWKV state predates the packed Mamba/Transformer paths and is intentionally
kept in its own fields on :class:`RecurrentState`.  This module provides a
small, device-agnostic representation for the two sequence families without
making the generic state/cache code know about model math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class MambaLayerState:
    """State carried by one Mamba-2 mixer layer.

    ``conv`` has shape ``[batch, conv_dim, kernel_size - 1]`` and ``ssm`` has
    shape ``[batch, heads, state_size, head_dim]``.  The SSM tensor is kept in
    float32 by the backend even when model weights use BF16/FP16.
    """

    conv: torch.Tensor
    ssm: torch.Tensor


@dataclass
class TransformerKVState:
    """Append-only key/value cache for one Transformer layer."""

    key: torch.Tensor
    value: torch.Tensor
    # Optional backing storage keeps decode from reallocating for every token.
    # Snapshot/cache serialization intentionally writes only the active views.
    key_storage: torch.Tensor | None = None
    value_storage: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.key.shape[-2])

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape[:-2] != self.key.shape[:-2] or value.shape[:-2] != self.value.shape[:-2]:
            raise ValueError("Transformer KV batch/head shape changed during decode")
        current = self.length
        amount = int(key.shape[-2])
        # The initial state uses an empty placeholder because the activation
        # dtype is only known once the embedding layer has been loaded.  Keep
        # the first real KV tensors directly so a BF16/FP16 model does not
        # silently promote its cache to the placeholder's FP32 dtype.  This
        # also avoids an unnecessary first copy; later appends grow the
        # reusable backing storage below.
        if current == 0 and amount > 0 and self.key_storage is None:
            self.key = key.detach()
            self.value = value.detach()
            return
        capacity = (
            int(self.key_storage.shape[-2])
            if self.key_storage is not None
            else current
        )
        if capacity < current + amount:
            new_capacity = max(current + amount, max(1, capacity * 2))
            self.key_storage = torch.empty(
                (*self.key.shape[:-2], new_capacity, self.key.shape[-1]),
                dtype=self.key.dtype,
                device=self.key.device,
            )
            self.value_storage = torch.empty(
                (*self.value.shape[:-2], new_capacity, self.value.shape[-1]),
                dtype=self.value.dtype,
                device=self.value.device,
            )
            if current:
                self.key_storage[..., :current, :].copy_(self.key)
                self.value_storage[..., :current, :].copy_(self.value)
        assert self.key_storage is not None and self.value_storage is not None
        self.key_storage[..., current : current + amount, :].copy_(key)
        self.value_storage[..., current : current + amount, :].copy_(value)
        self.key = self.key_storage[..., : current + amount, :]
        self.value = self.value_storage[..., : current + amount, :]

    def compact_left(self, keep: int) -> None:
        """Discard the oldest entries while retaining reusable backing storage.

        Sliding-window attention calls this occasionally (rather than on every
        token), so the copy is amortized over a window and decode does not
        create an ever-growing KV cache.  The active views remain contiguous,
        which keeps the attention kernels on their normal fast path.
        """
        keep = max(0, int(keep))
        current = self.length
        if keep >= current:
            return
        if keep == 0:
            self.key = self.key[..., :0, :]
            self.value = self.value[..., :0, :]
            return
        if self.key_storage is None or self.value_storage is None:
            capacity = current
            self.key_storage = torch.empty(
                (*self.key.shape[:-2], capacity, self.key.shape[-1]),
                dtype=self.key.dtype,
                device=self.key.device,
            )
            self.value_storage = torch.empty(
                (*self.value.shape[:-2], capacity, self.value.shape[-1]),
                dtype=self.value.dtype,
                device=self.value.device,
            )
        drop = current - keep
        self.key_storage[..., :keep, :].copy_(self.key[..., drop:, :])
        self.value_storage[..., :keep, :].copy_(self.value[..., drop:, :])
        self.key = self.key_storage[..., :keep, :]
        self.value = self.value_storage[..., :keep, :]


@dataclass
class SequenceState:
    """Versioned model-family-neutral sequence state.

    ``layers`` contains :class:`MambaLayerState` or
    :class:`TransformerKVState` objects according to ``kind``.  Position is
    the number of tokens already consumed, not the last zero-based index.
    """

    kind: str
    position: int
    layers: list[MambaLayerState | TransformerKVState]
    batch_size: int = 1
    context_limit: int | None = None
    # Logits for the next token after the consumed prefix.  Recurrent state
    # alone is insufficient to resume a causal LM because the final hidden
    # vector is not recoverable from Mamba/attention caches.
    next_logits: torch.Tensor | None = None

    def clone(self) -> "SequenceState":
        copied: list[MambaLayerState | TransformerKVState] = []
        for layer in self.layers:
            if isinstance(layer, MambaLayerState):
                copied.append(MambaLayerState(layer.conv.clone(), layer.ssm.clone()))
            elif isinstance(layer, TransformerKVState):
                copied.append(
                    TransformerKVState(
                        layer.key.clone(),
                        layer.value.clone(),
                        layer.key_storage.clone() if layer.key_storage is not None else None,
                        layer.value_storage.clone() if layer.value_storage is not None else None,
                    )
                )
            else:  # pragma: no cover - defensive for callers constructing by hand
                raise TypeError(f"unsupported sequence layer state: {type(layer)!r}")
        return SequenceState(
            kind=str(self.kind),
            position=int(self.position),
            layers=copied,
            batch_size=int(self.batch_size),
            context_limit=(
                int(self.context_limit) if self.context_limit is not None else None
            ),
            next_logits=self.next_logits.clone() if self.next_logits is not None else None,
        )

    @property
    def context_length(self) -> int:
        if self.kind == "transformer" and self.layers:
            first = self.layers[0]
            if isinstance(first, TransformerKVState):
                return first.length
        return int(self.position)

    def nbytes(self) -> int:
        total = 0
        for layer in self.layers:
            if isinstance(layer, MambaLayerState):
                total += int(layer.conv.numel() * layer.conv.element_size())
                total += int(layer.ssm.numel() * layer.ssm.element_size())
            elif isinstance(layer, TransformerKVState):
                key = layer.key_storage if layer.key_storage is not None else layer.key
                value = layer.value_storage if layer.value_storage is not None else layer.value
                total += int(key.numel() * key.element_size())
                total += int(value.numel() * value.element_size())
        if self.next_logits is not None:
            total += int(self.next_logits.numel() * self.next_logits.element_size())
        return total


def clone_sequence_state(state: SequenceState | None) -> SequenceState | None:
    return state.clone() if state is not None else None


def sequence_state_tensors(state: SequenceState) -> Iterable[torch.Tensor]:
    """Yield tensors in a deterministic order for serialization."""
    for layer in state.layers:
        if isinstance(layer, MambaLayerState):
            yield layer.conv
            yield layer.ssm
        elif isinstance(layer, TransformerKVState):
            yield layer.key
            yield layer.value
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported sequence layer state: {type(layer)!r}")
    if state.next_logits is not None:
        yield state.next_logits


def validate_sequence_state(state: SequenceState, *, kind: str | None = None) -> None:
    expected = kind.strip().lower() if kind is not None else None
    actual = str(state.kind).strip().lower()
    if actual not in {"mamba2", "transformer"}:
        raise ValueError(f"unsupported sequence state kind {state.kind!r}")
    if expected is not None and actual != expected:
        raise ValueError(f"sequence state kind mismatch: expected {expected!r}, got {actual!r}")
    if state.position < 0 or state.batch_size <= 0:
        raise ValueError("sequence state position/batch_size must be non-negative/positive")
    if len(state.layers) == 0:
        raise ValueError("sequence state must contain at least one layer")
    for layer in state.layers:
        if actual == "mamba2" and not isinstance(layer, MambaLayerState):
            raise ValueError("Mamba sequence state contains a non-Mamba layer")
        if actual == "transformer" and not isinstance(layer, TransformerKVState):
            raise ValueError("Transformer sequence state contains a non-KV layer")
        if not isinstance(layer.conv if isinstance(layer, MambaLayerState) else layer.key, torch.Tensor):
            raise ValueError("sequence state tensors must be torch.Tensor instances")
    if state.next_logits is None or not isinstance(state.next_logits, torch.Tensor):
        raise ValueError("sequence state must carry next-token logits")


__all__ = [
    "MambaLayerState",
    "SequenceState",
    "TransformerKVState",
    "clone_sequence_state",
    "sequence_state_tensors",
    "validate_sequence_state",
]
