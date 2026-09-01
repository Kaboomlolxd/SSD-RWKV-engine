"""Intermediate state publication is retained only for controlled decode."""

from __future__ import annotations

import torch

from rwkv_ssd.backends import rwkv7_forward


class _Pipeline:
    def encode(self, _text: str) -> list[int]:
        return [7]


class _Model:
    def generate_zero_state(self) -> list[torch.Tensor]:
        return [torch.zeros(2)]

    def forward(
        self, _token_ids: list[int], state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Token 1 is always the greedy result; return a fresh state to mirror
        # the recurrent model contract without requiring ChatRWKV in the unit
        # test environment.
        return torch.tensor([0.0, 1.0, 0.0]), [state[0] + 1.0]


def test_uncontrolled_native_decode_publishes_only_final_snapshot(monkeypatch) -> None:
    published: list[int] = []

    def remember(_model, _state, last_token_id, _logits=None) -> None:
        published.append(int(last_token_id))

    monkeypatch.setattr(rwkv7_forward, "_remember_rwkv7_state", remember)
    model = _Model()

    out = rwkv7_forward.greedy_token_ids_native(
        model,
        _Pipeline(),
        "prompt",
        4,
    )

    assert out == [1, 1, 1, 1]
    assert published == [1]


def test_controlled_native_decode_keeps_pre_token_snapshots(monkeypatch) -> None:
    published: list[int] = []

    def remember(_model, _state, last_token_id, _logits=None) -> None:
        published.append(int(last_token_id))

    monkeypatch.setattr(rwkv7_forward, "_remember_rwkv7_state", remember)
    model = _Model()

    out = rwkv7_forward.greedy_token_ids_native(
        model,
        _Pipeline(),
        "prompt",
        4,
        token_callback=lambda _token_id: None,
    )

    assert out == [1, 1, 1, 1]
    # Four resumable pre-token boundaries plus the final post-token state.
    assert len(published) == 5


def test_uncontrolled_state_continuation_publishes_only_final_snapshot(monkeypatch) -> None:
    published: list[int] = []

    def remember(_model, _state, last_token_id, _logits=None) -> None:
        published.append(int(last_token_id))

    monkeypatch.setattr(rwkv7_forward, "_remember_rwkv7_state", remember)
    model = _Model()

    out = rwkv7_forward.decode_greedy_from_state(
        model,
        [torch.zeros(2)],
        5,
        3,
    )

    assert out == [1, 1, 1]
    assert published == [1]
