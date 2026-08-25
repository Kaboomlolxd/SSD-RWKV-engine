"""Spawned worker-pool contract tests using the tiny synthetic backend."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.state_envelope import (
    make_state_envelope,
    model_fingerprint,
    tokenizer_fingerprint,
)
from app.worker_pool import (
    InferenceWorkerPool,
    estimate_worker_memory,
    validate_worker_count,
)


def test_worker_cancel_ack_is_not_treated_as_terminal() -> None:
    """Parent cancellation waits for backend termination after IPC receipt."""
    from app.serve import _await_worker_terminal

    class Handle:
        def __init__(self) -> None:
            self.events: queue.Queue[dict] = queue.Queue()

    handle = Handle()
    handle.events.put({"kind": "cancel_ack", "request_id": "r", "accepted": True})
    handle.events.put({"kind": "cancelled", "request_id": "r"})
    terminal = _await_worker_terminal(handle, deadline=time.monotonic() + 1.0)
    assert terminal is not None
    assert terminal["kind"] == "cancelled"


def _config(pack, *, max_tokens: int = 4) -> EngineConfig:
    return EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        strategy="cpu fp32",
        max_tokens=max_tokens,
        greedy=True,
        prefetch_enabled=False,
    )


def _terminal(handle, *, timeout: float = 30.0) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    while True:
        try:
            event = handle.events.get(timeout=timeout)
        except queue.Empty as exc:
            raise AssertionError("worker did not return a terminal event") from exc
        events.append(event)
        if event.get("kind") not in {"token", "ready"}:
            return event, events


def test_worker_count_uses_pack_footprint_when_no_budget_is_configured(
    synthetic_pack, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(synthetic_pack)
    estimate, source = estimate_worker_memory(config)
    assert estimate > 0
    assert source.startswith("manifest_") or source.endswith("fallback")
    monkeypatch.setattr(
        "app.worker_pool._available_memory_bytes",
        lambda: int(estimate * 1.5),
    )
    validate_worker_count(config, 1)
    with pytest.raises(ValueError, match="estimated"):
        validate_worker_count(config, 2)


def test_raw_kimi_worker_estimate_is_conservative(tmp_path: Path) -> None:
    model_dir = tmp_path / "kimi"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "kimi_k3"}), encoding="utf-8"
    )
    checkpoint = model_dir / "model.safetensors"
    checkpoint.write_bytes(b"k" * (4 * 1024 * 1024))
    config = EngineConfig(
        pack_dir=model_dir,
        backend="kimi_k3",
        mode="resident",
        device="cpu",
    )
    estimate, source = estimate_worker_memory(config)
    assert source == "kimi_resident_safetensors_upper_bound"
    assert estimate >= checkpoint.stat().st_size + 896 * 1024 * 1024


def test_raw_hf_state_fingerprints_include_kimi_assets(tmp_path: Path) -> None:
    model_dir = tmp_path / "kimi"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "kimi_k3", "vocab_size": 8}), encoding="utf-8"
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "tiktoken.model").write_bytes(b"tokenizer-a")
    (model_dir / "encoding_k3.py").write_text("version = 1", encoding="utf-8")
    (model_dir / "tokenization_kimi.py").write_text("version = 1", encoding="utf-8")
    model_a = model_fingerprint(model_dir)
    tokenizer_a = tokenizer_fingerprint(model_dir)
    (model_dir / "tiktoken.model").write_bytes(b"tokenizer-b")
    assert model_fingerprint(model_dir) == model_a
    assert tokenizer_fingerprint(model_dir) != tokenizer_a
    (model_dir / "model.safetensors").write_bytes(b"weights-changed")
    assert model_fingerprint(model_dir) != model_a


def test_worker_ipc_message_carries_sampling_controls_without_engine_leak(
    synthetic_pack,
) -> None:
    """The parent must put all request sampling fields on the IPC envelope."""
    pool = InferenceWorkerPool(_config(synthetic_pack), workers=1)
    slot = pool._slots[0]
    slot.request_queue = queue.Queue()
    slot.healthy = True
    slot.ready = True
    try:
        handle = pool.submit(
            "sampling request",
            max_tokens=3,
            temperature=0.65,
            greedy=False,
            top_p=0.77,
            seed=314,
        )
        message = slot.request_queue.get(timeout=1.0)
        assert message["temperature"] == 0.65
        assert message["greedy"] is False
        assert message["top_p"] == 0.77
        assert message["seed"] == 314
        pool.finish(handle, {"kind": "cancelled"})
    finally:
        pool.close()


@pytest.mark.integration
def test_spawned_worker_pool_routes_streams_and_restores_sessions(
    synthetic_pack, tmp_path
) -> None:
    from rwkv_ssd.runtime.state_parking import HierarchicalStateStore

    pool = InferenceWorkerPool(
        _config(synthetic_pack),
        workers=2,
        start_timeout_s=30.0,
        state_store=HierarchicalStateStore(tmp_path / "sessions", max_ram_bytes=0),
    ).start()
    handles = []
    try:
        snapshot = pool.metrics_snapshot()
        assert snapshot["worker_count"] == 2
        assert snapshot["healthy_workers"] == 2
        assert all(row["pid"] for row in snapshot["workers"])

        first = pool.submit(
            "first",
            max_tokens=3,
            temperature=1.0,
            greedy=True,
            session_id="session-a",
        )
        handles.append(first)
        terminal, events = _terminal(first)
        assert terminal["kind"] == "done"
        assert sum(event.get("kind") == "token" for event in events) == 3
        envelope = terminal["validated_state_envelope"]
        assert envelope is not None
        pool.record_session_state("session-a", envelope)
        pool.finish(first, terminal)

        followup = pool.submit(
            "follow-up",
            max_tokens=2,
            temperature=1.0,
            greedy=True,
            session_id="session-a",
        )
        handles.append(followup)
        terminal2, events2 = _terminal(followup)
        assert terminal2["kind"] == "done"
        assert sum(event.get("kind") == "token" for event in events2) == 2
        assert followup.worker_id == first.worker_id
        pool.finish(followup, terminal2)

        stateless_a = pool.submit(
            "a", max_tokens=1, temperature=1.0, greedy=True
        )
        handles.append(stateless_a)
        done_a, _ = _terminal(stateless_a)
        pool.finish(stateless_a, done_a)
        stateless_b = pool.submit(
            "b", max_tokens=1, temperature=1.0, greedy=True
        )
        handles.append(stateless_b)
        done_b, _ = _terminal(stateless_b)
        pool.finish(stateless_b, done_b)
        assert stateless_a.worker_id != stateless_b.worker_id

        parked = pool.state_store.get(
            "session-a",
            backend=pool._backend_kind,
            model_fingerprint=pool.model_fingerprint,
            tokenizer_fingerprint=pool.tokenizer_fingerprint,
            serialization_version=1,
        )
        assert parked is not None
        assert pool.state_store.stats.disk_hits >= 1
    finally:
        for handle in handles:
            if not handle.terminal:
                pool.abort(handle, "test cleanup")
        pool.close()
    assert all(not row["healthy"] for row in pool.metrics_snapshot()["workers"])


@pytest.mark.integration
def test_worker_pool_rejects_incompatible_state_envelope(synthetic_pack) -> None:
    pool = InferenceWorkerPool(_config(synthetic_pack), workers=1, start_timeout_s=30.0).start()
    try:
        state = pool._slots[0]  # access only to keep this test independent of a model state shape
        assert state.healthy
        from rwkv_ssd.runtime.state_cache import RecurrentState

        bad = make_state_envelope(
            RecurrentState(last_token_id=0),
            backend_kind="different-backend",
            model_fingerprint=pool.model_fingerprint,
            tokenizer_fingerprint=pool.tokenizer_fingerprint,
        )
        pool._session_states["bad"] = bad
        with pytest.raises(ValueError, match="state backend mismatch"):
            pool.submit("bad", max_tokens=1, temperature=1.0, greedy=True, session_id="bad")
    finally:
        pool.close()


@pytest.mark.integration
def test_worker_pool_emits_cancel_ack_before_terminal(synthetic_pack) -> None:
    pool = InferenceWorkerPool(
        _config(synthetic_pack, max_tokens=100_000),
        workers=1,
        start_timeout_s=30.0,
    ).start()
    handle = None
    terminal = None
    try:
        handle = pool.submit(
            "cancel me",
            max_tokens=100_000,
            temperature=1.0,
            greedy=True,
        )
        handle.cancel()
        saw_ack = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and terminal is None:
            try:
                event = handle.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.get("kind") == "cancel_ack":
                saw_ack = True
            elif event.get("kind") not in {"token", "ready"}:
                terminal = event
        assert saw_ack
        assert terminal is not None
        assert terminal["kind"] in {"cancelled", "done", "error"}
    finally:
        if handle is not None and not handle.terminal:
            pool.abort(handle, "cancel acknowledgement test cleanup")
        pool.close()


@pytest.mark.integration
def test_worker_pool_restarts_a_crashed_worker(synthetic_pack) -> None:
    pool = InferenceWorkerPool(
        _config(synthetic_pack),
        workers=1,
        max_restarts=1,
        start_timeout_s=30.0,
    ).start()
    handle = None
    try:
        original_pid = pool.metrics_snapshot()["workers"][0]["pid"]
        process = pool._slots[0].process
        assert process is not None
        process.terminate()
        process.join(timeout=5.0)

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            row = pool.metrics_snapshot()["workers"][0]
            if row["healthy"] and row["pid"] and row["pid"] != original_pid:
                break
            time.sleep(0.1)
        row = pool.metrics_snapshot()["workers"][0]
        assert row["healthy"]
        assert row["restarts"] == 1
        assert row["pid"] != original_pid

        handle = pool.submit("after restart", max_tokens=1, temperature=1.0, greedy=True)
        terminal, _events = _terminal(handle, timeout=15.0)
        assert terminal["kind"] == "done"
        pool.finish(handle, terminal)
    finally:
        if handle is not None and not handle.terminal:
            pool.abort(handle, "restart test cleanup")
        pool.close()
