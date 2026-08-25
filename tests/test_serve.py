"""HTTP serve health and OpenAI endpoint tests."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from argparse import Namespace
from http.client import HTTPConnection
from pathlib import Path
from collections.abc import Iterator
import pytest

from app.serve import (
    CTX,
    RequestScheduler,
    ServerStats,
    _request_bool,
    build_engine_from_args,
)
from app.worker_pool import InferenceWorkerPool
from http.server import ThreadingHTTPServer
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.state_parking import HierarchicalStateStore


def test_request_scheduler_shutdown_rejects_waiters() -> None:
    scheduler = RequestScheduler(max_concurrent=1, max_queue=1)
    assert scheduler.acquire(0.1)
    result: list[bool] = []

    def waiter() -> None:
        result.append(scheduler.acquire(5.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    # Ensure the waiter has entered the bounded queue before shutdown.
    for _ in range(20):
        if scheduler.snapshot()["waiting"] == 1:
            break
        time.sleep(0.01)
    scheduler.close()
    thread.join(timeout=1.0)
    scheduler.release()
    assert result == [False]


def _serve_args(pack: Path, **overrides: object) -> Namespace:
    base = dict(
        model=str(pack),
        config=None,
        checkpoint=None,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        strategy="cpu fp32",
        max_tokens=4,
        temperature=None,
        greedy=None,
        io_backend="mmap",
        io_chunk_bytes=0,
        prefetch_policy="layer",
        no_prefetch=False,
        stream_layer_cache=False,
        max_layers_in_z=None,
        warm_z=False,
        low_ram=False,
        ram_budget_gb=None,
        cache_budget_gb=None,
        cache_budget_auto=False,
        power=None,
        decode_disk_cache=None,
        no_decouple_provider_cache=False,
        max_provider_cache_layers=None,
        residency_profile=None,
        mmap_sequential=False,
        no_mmap_willneed=False,
        mmap_dontneed=False,
        system_prefix=None,
        state_cache=False,
        stateless_prefix_cache=False,
        prefix_cache_mode=None,
        prefix_cache_max_entries=None,
    )
    base.update(overrides)
    return Namespace(**base)


@contextmanager
def _test_server(pack: Path, **options: object) -> Iterator[tuple[str, int]]:
    """Start an isolated local server and restore the process-global context."""
    CTX.engine = build_engine_from_args(_serve_args(pack))
    CTX.lightning_url = None
    CTX.cors = bool(options.get("cors", False))
    CTX.cors_origin = options.get("cors_origin")  # type: ignore[assignment]
    CTX.api_key = options.get("api_key")  # type: ignore[assignment]
    CTX.max_body_bytes = int(options.get("max_body_bytes", 4 * 1024 * 1024))
    CTX.request_timeout_s = float(options.get("request_timeout_s", 30.0))
    CTX.scheduler = RequestScheduler(max_queue=int(options.get("max_queue", 32)))
    CTX.stats = ServerStats()
    CTX.state_store = None
    from app.serve import InferenceHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        if CTX.engine is not None:
            CTX.engine.close()
        CTX.engine = None
        CTX.lightning_url = None
        CTX.cors = False
        CTX.cors_origin = None
        CTX.api_key = None
        CTX.max_body_bytes = 4 * 1024 * 1024
        CTX.request_timeout_s = 30.0
        CTX.scheduler = None
        CTX.stats = ServerStats()
        CTX.state_store = None
        thread.join(timeout=2)


def test_request_bool_parses_string_flags() -> None:
    assert _request_bool("false") is False
    assert _request_bool("off") is False
    assert _request_bool("true") is True
    assert _request_bool(None, default=True) is True


def test_session_id_is_bounded() -> None:
    from app.serve import _session_id

    assert _session_id("abc") == "abc"
    with pytest.raises(ValueError):
        _session_id("x" * 129)


def test_request_scheduler_bounds_waiters() -> None:
    scheduler = RequestScheduler(max_queue=0)
    assert scheduler.acquire(0)
    try:
        assert scheduler.acquire(0) is False
        assert scheduler.snapshot()["active"] == 1
    finally:
        scheduler.release()


def test_health_payload_exposes_runtime_diagnostics(synthetic_pack: Path) -> None:
    from app.serve import _health_payload

    args = _serve_args(synthetic_pack)
    CTX.engine = build_engine_from_args(args)
    try:
        body = _health_payload()
        assert body["status"] == "ok"
        assert body["ready"] is True
        assert body["backend"] == "synthetic"
        assert "scheduler" in body
        assert "counters" in body
    finally:
        CTX.engine.close()
        CTX.engine = None


def test_health_endpoint(synthetic_pack: Path) -> None:
    from app.serve import InferenceHandler

    args = _serve_args(synthetic_pack)
    CTX.engine = build_engine_from_args(args)
    CTX.lightning_url = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["status"] == "ok"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        CTX.engine.close()
        CTX.engine = None
        thread.join(timeout=2)


def test_openai_chat_completion_endpoint(synthetic_pack: Path) -> None:
    from app.serve import InferenceHandler

    args = _serve_args(
        synthetic_pack,
        max_tokens=2,
        state_cache=True,
        stateless_prefix_cache=True,
    )
    CTX.engine = build_engine_from_args(args)
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=10)
        body = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "x"},
                ],
                "max_tokens": 2,
            }
        )
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "rwkv_ssd_metrics" in data
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        CTX.engine.close()
        CTX.engine = None
        thread.join(timeout=2)


def test_openai_streaming_endpoint(synthetic_pack: Path) -> None:
    from app.serve import InferenceHandler

    args = _serve_args(synthetic_pack, max_tokens=2)
    CTX.engine = build_engine_from_args(args)
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=10)
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 2,
                "stream": True,
            }
        )
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        chunks: list[str] = []
        while True:
            line = resp.readline().decode("utf-8")
            if not line:
                break
            chunks.append(line)
            if line.strip() == "data: [DONE]":
                break
        payload = "".join(chunks)
        assert "chat.completion.chunk" in payload
        assert "[DONE]" in payload
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        CTX.engine.close()
        CTX.engine = None
        thread.join(timeout=2)


def test_http_request_sampling_controls_reach_dispatcher(
    synthetic_pack: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_worker_request(
        handler,
        path,
        prompt,
        payload,
        **kwargs,
    ) -> None:
        captured.update(kwargs)
        from app.serve import _json_response

        _json_response(handler, 200, {"ok": True})

    monkeypatch.setattr("app.serve._handle_worker_pool_request", fake_worker_request)
    try:
        with _test_server(synthetic_pack) as (host, port):
            # Selecting the worker dispatch path lets this test verify the
            # HTTP parser-to-IPC boundary without starting another process.
            CTX.worker_pool = type(
                "PoolStub", (), {"config": CTX.engine.config}  # type: ignore[union-attr]
            )()
            conn = HTTPConnection(host, port, timeout=10)
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(
                    {
                        "prompt": "sampling",
                        "max_tokens": 2,
                        "temperature": 0.55,
                        "top_p": 0.74,
                        "seed": 19,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            assert response.status == 200
            response.read()
            conn.close()
    finally:
        CTX.worker_pool = None

    assert captured["temperature"] == 0.55
    assert captured["top_p"] == 0.74
    assert captured["seed"] == 19
    assert captured["greedy"] is False


def test_http_rejects_invalid_sampling_values(synthetic_pack: Path) -> None:
    invalid_payloads = [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.01}, "top_p"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
    ]
    with _test_server(synthetic_pack) as (host, port):
        for sampling, field in invalid_payloads:
            payload = {"prompt": "invalid sampling", "max_tokens": 1}
            payload.update(sampling)
            conn = HTTPConnection(host, port, timeout=10)
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            body = json.loads(response.read().decode("utf-8"))
            conn.close()
            assert response.status == 400
            assert field in body["error"]


@contextmanager
def _test_worker_server(
    pack: Path, tmp_path: Path, *, workers: int = 2
) -> Iterator[tuple[str, int]]:
    from app.serve import InferenceHandler

    cfg = EngineConfig(
        pack_dir=pack,
        backend="synthetic",
        mode="streaming",
        device="cpu",
        strategy="cpu fp32",
        max_tokens=4,
        greedy=True,
        prefetch_enabled=False,
    )
    CTX.engine = None
    CTX.worker_pool = InferenceWorkerPool(
        cfg,
        workers=workers,
        start_timeout_s=30.0,
        state_store=HierarchicalStateStore(tmp_path / "sessions", max_ram_bytes=0),
    ).start()
    CTX.lightning_url = None
    CTX.cors = False
    CTX.cors_origin = None
    CTX.api_key = None
    CTX.max_body_bytes = 4 * 1024 * 1024
    CTX.request_timeout_s = 30.0
    CTX.scheduler = RequestScheduler(max_concurrent=workers, max_queue=8)
    CTX.stats = ServerStats()
    CTX.state_store = CTX.worker_pool.state_store
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        if CTX.worker_pool is not None:
            CTX.worker_pool.close()
        CTX.worker_pool = None
        CTX.engine = None
        CTX.lightning_url = None
        CTX.scheduler = None
        CTX.stats = ServerStats()
        CTX.state_store = None
        thread.join(timeout=2)


def test_engine_stream_is_incremental_and_records_memory(synthetic_pack: Path) -> None:
    engine = build_engine_from_args(_serve_args(synthetic_pack, max_tokens=3))
    try:
        chunks = list(engine.generate_stream("x"))
        assert len(chunks) == 3
        assert engine.metrics.tokens_generated == 3
        assert engine.metrics.process_rss_peak_bytes >= engine.metrics.process_rss_bytes >= 0
    finally:
        engine.close()


def test_engine_stream_applies_request_lifecycle_once(
    synthetic_pack: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming must delegate the complete lifecycle to generate_tokens once."""
    engine = build_engine_from_args(_serve_args(synthetic_pack, max_tokens=1))
    calls: list[str] = []
    try:
        for name in (
            "_apply_adaptive_residency_boundary",
            "_apply_session_promotion_boundary",
            "_observe_adaptive_residency",
            "_observe_session_promotion",
        ):
            monkeypatch.setattr(engine, name, lambda name=name: calls.append(name))
        assert list(engine.generate_stream("x"))
        assert calls == [
            "_apply_adaptive_residency_boundary",
            "_apply_session_promotion_boundary",
            "_observe_adaptive_residency",
            "_observe_session_promotion",
        ]
    finally:
        engine.close()


def test_http_auth_cors_and_metrics_controls(synthetic_pack: Path) -> None:
    with _test_server(
        synthetic_pack,
        api_key="secret",
        cors=True,
        cors_origin="https://client.example",
    ) as (host, port):
        conn = HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        conn.close()

        conn = HTTPConnection(host, port, timeout=10)
        conn.request(
            "GET",
            "/health",
            headers={"Origin": "https://client.example"},
        )
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == "https://client.example"
        health = json.loads(response.read().decode("utf-8"))
        assert health["authenticated"] is True
        conn.close()

        conn = HTTPConnection(host, port, timeout=10)
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps({"prompt": "x", "max_tokens": 1}),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
        )
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        conn.close()

        conn = HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/metrics", headers={"X-API-Key": "secret"})
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "rwkv_ssd_requests_total" in body
        assert "rwkv_ssd_tokens_generated_total" in body
        conn.close()


def test_http_rejects_oversized_request_body(synthetic_pack: Path) -> None:
    with _test_server(synthetic_pack, api_key="secret", max_body_bytes=16) as (host, port):
        conn = HTTPConnection(host, port, timeout=10)
        body = json.dumps({"prompt": "this request is too large"})
        conn.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={"Content-Type": "application/json", "X-API-Key": "secret"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 413
        assert "exceeds" in payload["error"]
        conn.close()


def test_http_session_id_parks_and_restores_recurrent_state(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    from app.serve import InferenceHandler
    from rwkv_ssd.runtime.state_parking import HierarchicalStateStore

    args = _serve_args(synthetic_pack, max_tokens=2)
    CTX.engine = build_engine_from_args(args)
    CTX.state_store = HierarchicalStateStore(tmp_path / "sessions", max_ram_bytes=1024 * 1024)
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for index in range(2):
            conn = HTTPConnection(host, port, timeout=10)
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(
                    {"prompt": f"turn-{index}", "max_tokens": 2, "session_id": "abc"}
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert data["rwkv_ssd_metrics"]["session_parking"]["restored"] is (
                index == 1
            )
            conn.close()
        assert CTX.state_store.contains("abc")
    finally:
        server.shutdown()
        server.server_close()
        CTX.engine.close()
        CTX.engine = None
        CTX.state_store = None
        thread.join(timeout=2)


@pytest.mark.integration
def test_http_worker_pool_openai_stream_session_and_metrics(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    with _test_worker_server(synthetic_pack, tmp_path, workers=2) as (host, port):
        conn = HTTPConnection(host, port, timeout=20)
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps({"prompt": "worker", "max_tokens": 2, "session_id": "http"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        first = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert first["usage"]["completion_tokens"] == 2
        assert first["rwkv_ssd_metrics"]["session_parking"]["restored"] is False
        conn.close()

        conn = HTTPConnection(host, port, timeout=20)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {
                    "messages": [{"role": "user", "content": "next"}],
                    "max_tokens": 2,
                    "session_id": "http",
                    "stream": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200
        payload = response.read().decode("utf-8")
        assert "chat.completion.chunk" in payload
        assert '"finish_reason": "stop"' in payload
        assert '"completion_tokens": 2' in payload
        assert "data: [DONE]" in payload
        conn.close()

        conn = HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/health")
        health = json.loads(conn.getresponse().read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["workers"]["worker_count"] == 2
        assert health["workers"]["healthy_workers"] == 2
        conn.close()

        conn = HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/metrics")
        metrics = conn.getresponse().read().decode("utf-8")
        assert "rwkv_ssd_workers 2" in metrics
        assert "rwkv_ssd_worker_rss_bytes" in metrics
        assert "rwkv_ssd_ttft_ms_p95" in metrics
        conn.close()
