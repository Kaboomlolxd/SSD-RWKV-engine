"""Common sampling semantics across the Torch and native NumPy paths."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rwkv_ssd.runtime.sampling import (
    sample_numpy,
    sample_torch,
    sampling_context,
    validate_top_p,
)


def test_top_p_keeps_at_least_one_token_and_validates() -> None:
    logits = np.asarray([10.0, 9.0, 1.0], dtype=np.float32)
    with sampling_context(seed=3, top_p=0.01):
        assert sample_numpy(logits, temperature=1.0, greedy=False) == 0
    with pytest.raises(ValueError, match="top_p"):
        validate_top_p(0.0)
    with pytest.raises(ValueError, match="top_p"):
        validate_top_p(1.01)


def test_numpy_seed_is_request_deterministic_and_ids_are_valid() -> None:
    logits = np.asarray([0.1, 1.0, 0.8, 0.2, -0.4], dtype=np.float32)

    def draw() -> list[int]:
        with sampling_context(seed=1234, top_p=0.85):
            return [
                sample_numpy(logits, temperature=0.9, greedy=False)
                for _ in range(32)
            ]

    first = draw()
    second = draw()
    assert first == second
    assert all(0 <= token < logits.size for token in first)


def test_torch_seed_is_request_deterministic_and_ids_are_valid() -> None:
    logits = torch.tensor([0.1, 1.0, 0.8, 0.2, -0.4])

    def draw() -> list[int]:
        with sampling_context(seed=1234, top_p=0.85):
            return [
                sample_torch(logits, temperature=0.9, greedy=False)
                for _ in range(32)
            ]

    first = draw()
    second = draw()
    assert first == second
    assert all(0 <= token < logits.numel() for token in first)


def test_greedy_ignores_rng_but_accepts_top_p() -> None:
    torch_logits = torch.tensor([[-2.0, 4.0, 3.0]])
    numpy_logits = torch_logits.numpy()
    with sampling_context(seed=77, top_p=0.2):
        assert sample_torch(torch_logits, temperature=0.7, greedy=True) == 1
        assert sample_numpy(numpy_logits, temperature=0.7, greedy=True) == 1
