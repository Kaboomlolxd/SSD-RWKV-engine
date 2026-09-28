#!/usr/bin/env python3
"""HTTP API for local inference with OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from app.engine_args import add_engine_args, build_engine_config
from app.lightning_proxy import forward_openai_request
from app.worker_pool import (
    InferenceWorkerPool,
    WorkerCrashedError,
    WorkerPoolError,
    validate_worker_count,
)
from app.web_ui import CHAT_HTML
from rwkv_ssd import __version__ as RWKV_SSD_VERSION
from rwkv_ssd.backends.factory import ensure_v0_backend, supports_streaming_mode
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from rwkv_ssd.runtime.sampling import validate_top_p
from rwkv_ssd.runtime.state_parking import StateParkingCompatibilityError

logger = logging.getLogger(__name__)


class ServeContext:
    engine: InferenceEngine | None = None
    worker_pool: InferenceWorkerPool | None = None
    lightning_url: str | None = None
    cors: bool = False
    cors_origin: str | None = None
    api_key: str | None = None
    max_body_bytes: int = 4 * 1024 * 1024
    request_timeout_s: float = 30.0
    scheduler: RequestScheduler | None = None
    stats: ServerStats | None = None
    state_store: Any = None


class ServerStats:
    """Small thread-safe counter set exposed through health/metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, int] = {
            "requests_total": 0,
            "requests_failed": 0,
            "requests_rejected": 0,
            "tokens_generated_total": 0,
            "requests_cancelled": 0,
            "worker_failures": 0,
        }
        self._queue_wait_ms: list[float] = []
        self._generation_latency_ms: list[float] = []
        self._ttft_ms: list[float] = []

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile / 100.0
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        fraction = position - low
        return ordered[low] + (ordered[high] - ordered[low]) * fraction

    def inc(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0) + int(amount)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = dict(self._values)
            result.update(
                {
                    "queue_wait_ms_p50": self._percentile(self._queue_wait_ms, 50),
                    "queue_wait_ms_p95": self._percentile(self._queue_wait_ms, 95),
                    "generation_latency_ms_p50": self._percentile(
                        self._generation_latency_ms, 50
                    ),
                    "generation_latency_ms_p95": self._percentile(
                        self._generation_latency_ms, 95
                    ),
                    "ttft_ms_p50": self._percentile(self._ttft_ms, 50),
                    "ttft_ms_p95": self._percentile(self._ttft_ms, 95),
                }
            )
            return result  # type: ignore[return-value]

    def record_generation(
        self,
        *,
        queue_wait_ms: float,
        latency_ms: float,
        ttft_ms: float = 0.0,
        tokens: int = 0,
        cancelled: bool = False,
        failed: bool = False,
    ) -> None:
        with self._lock:
            for values, value in (
                (self._queue_wait_ms, queue_wait_ms),
                (self._generation_latency_ms, latency_ms),
                (self._ttft_ms, ttft_ms),
            ):
                values.append(max(0.0, float(value)))
                if len(values) > 2048:
                    del values[: len(values) - 2048]
            self._values["tokens_generated_total"] += int(tokens)
            if cancelled:
                self._values["requests_cancelled"] += 1
            if failed:
                self._values["requests_failed"] += 1


class RequestScheduler:
    """Bounded global admission for one or more engine workers.

    The engine mutates request-scoped configuration and recurrent state, so a
    shared instance remains single-generation. This scheduler makes that
    local-only constraint explicit, bounds waiting clients, and prevents an
    unbounded ThreadingHTTPServer queue from exhausting memory.
    """

    def __init__(self, *, max_concurrent: int = 1, max_queue: int = 32) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_queue < 0:
            raise ValueError("max_queue must be >= 0")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._condition = threading.Condition()
        self._active = 0
        self._waiting = 0
        self._closed = False

    def acquire(self, timeout_s: float | None) -> bool:
        deadline = None if timeout_s is None or timeout_s <= 0 else time.monotonic() + timeout_s
        with self._condition:
            if self._closed:
                return False
            if self._active >= self.max_concurrent:
                if self._waiting >= self.max_queue:
                    return False
                self._waiting += 1
                try:
                    while self._active >= self.max_concurrent and not self._closed:
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            return False
                        self._condition.wait(remaining)
                finally:
                    self._waiting -= 1
                if self._closed:
                    return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("request scheduler released without an active request")
            self._active -= 1
            self._condition.notify()

    def close(self) -> None:
        """Stop admission and wake queued HTTP handlers during shutdown."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "max_concurrent": self.max_concurrent,
                "max_queue": self.max_queue,
            }


CTX = ServeContext()
CTX.stats = ServerStats()


def _server_stats() -> ServerStats:
    if CTX.stats is None:
        CTX.stats = ServerStats()
    return CTX.stats


def _scheduler() -> RequestScheduler:
    if CTX.scheduler is None:
        CTX.scheduler = RequestScheduler()
    return CTX.scheduler


def _add_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    if not CTX.cors or not CTX.cors_origin:
        return
    allowed = {item.strip() for item in CTX.cors_origin.split(",") if item.strip()}
    request_origin = handler.headers.get("Origin")
    if request_origin and request_origin not in allowed:
        return
    origin = request_origin if request_origin in allowed else next(iter(allowed), None)
    if origin:
        handler.send_header("Access-Control-Allow-Origin", origin)
        handler.send_header("Vary", "Origin")


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    expected = CTX.api_key
    if not expected:
        return True
    supplied = handler.headers.get("X-API-Key", "")
    if not supplied:
        authorization = handler.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
    return hmac.compare_digest(supplied, expected)


def _json_response(
    handler: BaseHTTPRequestHandler,
    code: int,
    body: dict[str, Any],
    *,
    close_connection: bool = False,
) -> None:
    data = json.dumps(body).encode("utf-8")
    if close_connection:
        # Early request rejection can happen before the declared request body
        # has been consumed.  Explicitly close the HTTP/1.x connection after
        # the response so clients (notably Windows http.client) receive the
        # 4xx response instead of observing an aborted socket while the server
        # tears down an unread request.
        handler.close_connection = True
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    if close_connection:
        handler.send_header("Connection", "close")
    _add_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _drain_rejected_request_body(
    handler: BaseHTTPRequestHandler, length: int
) -> None:
    """Consume a bounded prefix before closing an early-rejected request.

    Windows can reset a connection that closes with unread request bytes even
    after the response has been written.  Drain the body when it is already
    available, but keep the read bounded so a client cannot turn the size
    check into an unbounded slow-upload wait.
    """
    if length <= 0:
        return
    limit = max(64 * 1024, int(CTX.max_body_bytes) + 1)
    remaining = min(int(length), limit)
    connection = handler.connection
    previous_timeout = connection.gettimeout()
    try:
        connection.settimeout(0.25)
        while remaining > 0:
            chunk = handler.rfile.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
    except (OSError, ValueError):
        # The response still needs to be sent.  Any bytes that were not
        # available within the short drain window will be discarded when the
        # explicit Connection: close response is finalized.
        pass
    finally:
        try:
            connection.settimeout(previous_timeout)
        except OSError:
            pass


def _sse_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    _add_cors_headers(handler)
    handler.end_headers()


def _metrics_response(handler: BaseHTTPRequestHandler) -> None:
    stats = _server_stats().snapshot()
    scheduler = _scheduler().snapshot()
    pool = CTX.worker_pool.metrics_snapshot() if CTX.worker_pool is not None else {
        "worker_count": 1,
        "healthy_workers": 1 if CTX.engine is not None else 0,
        "degraded": CTX.engine is None,
        "active_generations": scheduler["active"],
        "workers": [],
    }
    lines = [
        "# TYPE rwkv_ssd_requests_total counter",
        f"rwkv_ssd_requests_total {stats['requests_total']}",
        "# TYPE rwkv_ssd_requests_failed counter",
        f"rwkv_ssd_requests_failed {stats['requests_failed']}",
        "# TYPE rwkv_ssd_requests_rejected counter",
        f"rwkv_ssd_requests_rejected {stats['requests_rejected']}",
        "# TYPE rwkv_ssd_tokens_generated_total counter",
        f"rwkv_ssd_tokens_generated_total {stats['tokens_generated_total']}",
        "# TYPE rwkv_ssd_scheduler_active gauge",
        f"rwkv_ssd_scheduler_active {scheduler['active']}",
        "# TYPE rwkv_ssd_scheduler_waiting gauge",
        f"rwkv_ssd_scheduler_waiting {scheduler['waiting']}",
        "# TYPE rwkv_ssd_workers gauge",
        f"rwkv_ssd_workers {pool['worker_count']}",
        "# TYPE rwkv_ssd_healthy_workers gauge",
        f"rwkv_ssd_healthy_workers {pool['healthy_workers']}",
        "# TYPE rwkv_ssd_active_generations gauge",
        f"rwkv_ssd_active_generations {pool['active_generations']}",
        "# TYPE rwkv_ssd_queue_wait_ms_p50 gauge",
        f"rwkv_ssd_queue_wait_ms_p50 {stats['queue_wait_ms_p50']:.3f}",
        "# TYPE rwkv_ssd_queue_wait_ms_p95 gauge",
        f"rwkv_ssd_queue_wait_ms_p95 {stats['queue_wait_ms_p95']:.3f}",
        "# TYPE rwkv_ssd_ttft_ms_p50 gauge",
        f"rwkv_ssd_ttft_ms_p50 {stats['ttft_ms_p50']:.3f}",
        "# TYPE rwkv_ssd_ttft_ms_p95 gauge",
        f"rwkv_ssd_ttft_ms_p95 {stats['ttft_ms_p95']:.3f}",
        "# TYPE rwkv_ssd_generation_latency_ms_p50 gauge",
        f"rwkv_ssd_generation_latency_ms_p50 {stats['generation_latency_ms_p50']:.3f}",
        "# TYPE rwkv_ssd_generation_latency_ms_p95 gauge",
        f"rwkv_ssd_generation_latency_ms_p95 {stats['generation_latency_ms_p95']:.3f}",
        "# TYPE rwkv_ssd_requests_cancelled counter",
        f"rwkv_ssd_requests_cancelled {stats['requests_cancelled']}",
        "# TYPE rwkv_ssd_worker_failures counter",
        f"rwkv_ssd_worker_failures {stats['worker_failures']}",
    ]
    for row in pool.get("workers", []):
        worker_id = row.get("worker_id", 0)
        lines.extend(
            [
                "# TYPE rwkv_ssd_worker_healthy gauge",
                f"rwkv_ssd_worker_healthy{{worker=\"{worker_id}\"}} "
                f"{1 if row.get('healthy') else 0}",
                "# TYPE rwkv_ssd_worker_rss_bytes gauge",
                f"rwkv_ssd_worker_rss_bytes{{worker=\"{worker_id}\"}} "
                f"{int(row.get('rss_bytes', 0) or 0)}",
            ]
        )
    data = ("\n".join(lines) + "\n").encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain; version=0.0.4")
    handler.send_header("Content-Length", str(len(data)))
    _add_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _health_payload() -> dict[str, Any]:
    engine = CTX.engine
    pool = CTX.worker_pool
    if CTX.lightning_url:
        status = "ok"
        ready = True
        backend = "lightning_proxy"
        mode = "proxy"
        device = None
        pack = None
    elif pool is not None:
        pool_metrics = pool.metrics_snapshot()
        ready = int(pool_metrics["healthy_workers"]) > 0
        status = "ok" if ready and not pool_metrics["degraded"] else "degraded"
        backend = pool.config.backend
        mode = pool.config.mode
        device = pool.config.device
        pack = str(pool.config.pack_dir)
    elif engine is not None:
        status = "ok"
        ready = True
        backend = engine.config.backend
        mode = engine.config.mode
        device = str(engine.device)
        pack = str(engine.config.pack_dir)
    else:
        status = "degraded"
        ready = False
        backend = None
        mode = None
        device = None
        pack = None
    return {
        "status": status,
        "ready": ready,
        "backend": backend,
        "mode": mode,
        "device": device,
        "pack": pack,
        "authenticated": bool(CTX.api_key),
        "max_body_bytes": CTX.max_body_bytes,
        "scheduler": _scheduler().snapshot(),
        "workers": pool.metrics_snapshot() if pool is not None else {
            "worker_count": 1 if engine is not None else 0,
            "healthy_workers": 1 if engine is not None else 0,
            "degraded": False,
            "active_generations": _scheduler().snapshot()["active"],
            "workers": [],
        },
        "counters": _server_stats().snapshot(),
    }


def _sse_write(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    handler.wfile.flush()


def _request_bool(value: Any, *, default: bool = False) -> bool:
    """Accept JSON booleans and common string forms from HTTP clients."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _request_temperature(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if not (result >= 0.0) or result != result or result == float("inf"):
        raise ValueError("temperature must be a finite non-negative number")
    return result


def _request_top_p(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return validate_top_p(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _request_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise ValueError("seed must be an integer")
    return result


def _session_id(value: Any) -> str:
    """Validate the bounded key used by the local recurrent-state store."""
    if value is None or value == "":
        return ""
    result = str(value)
    if len(result) > 128 or any(ord(char) < 0x20 for char in result):
        raise ValueError("session_id must be at most 128 printable characters")
    return result


def _default_max_tokens() -> int:
    if CTX.worker_pool is not None:
        return int(CTX.worker_pool.config.max_tokens)
    if CTX.engine is not None:
        return int(CTX.engine.config.max_tokens)
    return 64


class InferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        logger.info(fmt % args)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _add_cors_headers(self)
        if CTX.cors and CTX.cors_origin:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in {"/", "/index.html"}:
            data = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.rstrip("/") == "/health":
            _json_response(self, 200, _health_payload())
            return
        if self.path.rstrip("/") == "/metrics":
            if not _authorized(self):
                _json_response(self, 401, {"error": "authentication required"})
                return
            _metrics_response(self)
            return
        if self.path.rstrip("/") == "/v1/models":
            if not _authorized(self):
                _json_response(self, 401, {"error": "authentication required"})
                return
            model_id = _model_id()
            _json_response(
                self,
                200,
                {"object": "list", "data": [_model_object(model_id)]},
            )
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in {"/generate", "/v1/chat/completions", "/v1/completions"}:
            _json_response(self, 404, {"error": "unsupported path"})
            return
        _server_stats().inc("requests_total")
        if not _authorized(self):
            _server_stats().inc("requests_failed")
            _json_response(self, 401, {"error": "authentication required"})
            return
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            _server_stats().inc("requests_failed")
            _json_response(self, 411, {"error": "Content-Length is required"})
            return
        try:
            length = int(content_length)
        except (TypeError, ValueError):
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "invalid Content-Length"})
            return
        if length < 0:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "invalid Content-Length"})
            return
        if length > CTX.max_body_bytes:
            _server_stats().inc("requests_rejected")
            _drain_rejected_request_body(self, length)
            _json_response(
                self,
                413,
                {"error": f"request body exceeds {CTX.max_body_bytes} bytes"},
                close_connection=True,
            )
            return
        try:
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                _server_stats().inc("requests_failed")
                _json_response(self, 400, {"error": "incomplete request body"})
                return
            payload = json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": f"request body must be UTF-8: {exc}"})
            return
        except json.JSONDecodeError as exc:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": f"invalid json: {exc}"})
            return
        if not isinstance(payload, dict):
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "request body must be a JSON object"})
            return

        if CTX.lightning_url:
            code, body, _headers = forward_openai_request(
                CTX.lightning_url, path, payload
            )
            if isinstance(body, dict):
                _json_response(self, code, body)
            else:
                _json_response(self, code, {"text": str(body)})
            return

        try:
            if path == "/v1/chat/completions":
                prompt, system_prefix = _render_chat_prompt(payload)
            else:
                prompt = str(payload.get("prompt", ""))
                system_prefix = str(payload.get("system_prefix", "") or "")
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        if not prompt:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "prompt required"})
            return

        try:
            max_tokens = int(
                payload.get(
                    "max_tokens",
                    payload.get("max_completion_tokens", _default_max_tokens()),
                )
            )
        except (TypeError, ValueError):
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "max_tokens must be an integer"})
            return
        if max_tokens <= 0:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": "max_tokens must be positive"})
            return
        if CTX.worker_pool is None:
            from rwkv_ssd.runtime.runtime_intelligence import admit_request

            assert CTX.engine is not None
            admission = admit_request(CTX.engine, max_tokens)
            if not admission.admitted:
                _server_stats().inc("requests_rejected")
                _json_response(
                    self,
                    429,
                    {
                        "error": admission.reason,
                        "predicted_stream_bytes": admission.predicted_stream_bytes,
                        "limit_bytes": admission.limit_bytes,
                    },
                )
                return
        stream = _request_bool(payload.get("stream"), default=False)
        try:
            session_id = _session_id(payload.get("session_id"))
        except ValueError as exc:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": str(exc)})
            return
        try:
            temperature = _request_temperature(payload.get("temperature"))
            top_p = _request_top_p(payload.get("top_p"))
            seed = _request_seed(payload.get("seed"))
        except ValueError as exc:
            _server_stats().inc("requests_failed")
            _json_response(self, 400, {"error": str(exc)})
            return
        greedy = payload.get("greedy")
        if greedy is None and temperature is not None and temperature > 0.0:
            # Resolve this at the HTTP-to-dispatch boundary as well as inside
            # the worker handler.  This keeps IPC payloads self-describing and
            # prevents a worker inheriting a greedy service default when the
            # client explicitly requested sampling via temperature.
            greedy = False

        if CTX.worker_pool is not None:
            _handle_worker_pool_request(
                self,
                path,
                prompt,
                payload,
                max_tokens=max_tokens,
                stream=stream,
                session_id=session_id,
                system_prefix=system_prefix,
                temperature=temperature,
                greedy=greedy,
                top_p=top_p,
                seed=seed,
            )
            return

        scheduler = _scheduler()
        request_deadline = (
            time.monotonic() + CTX.request_timeout_s
            if CTX.request_timeout_s > 0
            else None
        )
        if not scheduler.acquire(CTX.request_timeout_s):
            _server_stats().inc("requests_rejected")
            _json_response(
                self,
                429,
                {"error": "generation queue is full or request wait timed out"},
            )
            return
        try:
            assert CTX.engine is not None
            cfg = CTX.engine.config
            old_max = cfg.max_tokens
            old_prefix = cfg.system_prefix
            old_temp = cfg.temperature
            old_greedy = cfg.greedy
            old_top_p = cfg.top_p
            old_seed = cfg.seed
            try:
                cfg.max_tokens = max_tokens
                if system_prefix and CTX.engine.prefix_cache is not None:
                    cfg.system_prefix = system_prefix
                if temperature is not None:
                    cfg.temperature = float(temperature)
                    cfg.greedy = False
                if greedy is not None:
                    cfg.greedy = _request_bool(greedy)
                if top_p is not None:
                    cfg.top_p = float(top_p)
                if seed is not None:
                    cfg.seed = int(seed)
                if stream:
                    _handle_stream(
                        self,
                        path,
                        prompt,
                        payload,
                        deadline=request_deadline,
                    )
                else:
                    text, metrics_dict, summary = _run_generate(
                        CTX.engine,
                        prompt,
                        session_id=session_id,
                        deadline=request_deadline,
                    )
                    _write_json_result(
                        self,
                        path,
                        text,
                        max_tokens,
                        cfg,
                        summary,
                        metrics_dict,
                    )
                _server_stats().inc(
                    "tokens_generated_total",
                    int(getattr(CTX.engine.metrics, "tokens_generated", 0) or 0),
                )
            finally:
                cfg.max_tokens = old_max
                cfg.system_prefix = old_prefix
                cfg.temperature = old_temp
                cfg.greedy = old_greedy
                cfg.top_p = old_top_p
                cfg.seed = old_seed
        except GenerationCancelled as exc:
            _server_stats().inc("requests_failed")
            _json_response(self, 408, {"error": str(exc)})
        except Exception as exc:
            _server_stats().inc("requests_failed")
            logger.exception("generate failed")
            _json_response(self, 500, {"error": str(exc)})
        finally:
            scheduler.release()


def _run_generate(
    engine: InferenceEngine,
    prompt: str,
    *,
    session_id: str = "",
    deadline: float | None = None,
) -> tuple[str, dict | None, str | None]:
    parked = CTX.state_store.get(session_id) if session_id and CTX.state_store else None
    if parked is not None:
        engine.backend.set_recurrent_state(parked)
        text = engine.generate_followup(prompt, deadline=deadline)
    else:
        text = engine.generate(prompt, deadline=deadline)
    if session_id and CTX.state_store is not None:
        state = engine.backend.get_recurrent_state()
        if state is not None:
            CTX.state_store.put(
                session_id,
                state,
                backend=engine.config.backend,
                model_family=(engine.manifest.model_family if engine.manifest else "rwkv7"),
            )
    summary = engine.metrics.summary() if engine.metrics.layers else None
    metrics_dict = engine.metrics.to_dict() if engine.metrics.layers else None
    if engine.prefix_cache is not None and metrics_dict is not None:
        ps = engine.prefix_cache.stats
        total = ps.hits + ps.misses
        metrics_dict["cache_stats"] = {
            "hits": ps.hits,
            "misses": ps.misses,
            "hit_rate": round(ps.hits / total, 3) if total > 0 else 0.0,
            "prefill_ms_saved": round(ps.prefill_ms_saved, 3),
        }
    if metrics_dict is not None:
        from rwkv_ssd.runtime.runtime_intelligence import diagnose_metrics

        metrics_dict["diagnosis"] = diagnose_metrics(engine.metrics)
        if session_id and CTX.state_store is not None:
            metrics_dict["session_parking"] = {
                "session_id": session_id,
                "restored": parked is not None,
                "ram_bytes": CTX.state_store.ram_bytes,
                "disk_bytes": CTX.state_store.disk_bytes(),
            }
    return text, metrics_dict, summary


def _await_worker_terminal(
    handle: Any,
    *,
    deadline: float,
) -> dict[str, Any] | None:
    """Drain token/control events until a worker acknowledges cancellation."""
    while time.monotonic() < deadline:
        try:
            event = handle.events.get(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        # ``cancel_ack`` confirms that the control message crossed the IPC
        # boundary; it is not permission to release the worker slot.  Wait
        # for the backend's terminal event so the next request cannot race
        # mutable engine state.
        if str(event.get("kind", "")) not in {"token", "ready", "cancel_ack"}:
            return event
    return None


def _handle_worker_pool_request(
    handler: BaseHTTPRequestHandler,
    path: str,
    prompt: str,
    payload: dict[str, Any],
    *,
    max_tokens: int,
    stream: bool,
    session_id: str,
    system_prefix: str,
    temperature: Any,
    greedy: Any,
    top_p: Any,
    seed: Any,
) -> None:
    """Run one request through the parent-owned worker admission/IPC path."""
    pool = CTX.worker_pool
    if pool is None:
        _json_response(handler, 503, {"error": "worker pool is unavailable"})
        return
    scheduler = _scheduler()
    started = time.monotonic()
    request_deadline = (
        started + CTX.request_timeout_s if CTX.request_timeout_s > 0 else None
    )
    if not scheduler.acquire(CTX.request_timeout_s):
        _server_stats().inc("requests_rejected")
        _json_response(
            handler,
            429,
            {"error": "generation queue is full or request wait timed out"},
        )
        return
    queue_wait_ms = (time.monotonic() - started) * 1000.0
    handle: Any | None = None
    terminal: dict[str, Any] | None = None
    try:
        requested_temperature = (
            float(temperature)
            if temperature is not None
            else float(pool.config.temperature)
        )
        if greedy is not None:
            requested_greedy = _request_bool(greedy)
        elif temperature is not None and float(temperature) > 0.0:
            # Match InferenceEngine._sampling_scope: an explicit positive
            # temperature opts into sampling even when the service default is
            # greedy.  Preserve the configured default only when the request
            # leaves both controls unspecified.
            requested_greedy = False
        else:
            requested_greedy = bool(pool.config.greedy)
        requested_top_p = (
            float(top_p) if top_p is not None else float(pool.config.top_p)
        )
        requested_seed = (
            int(seed) if seed is not None else pool.config.seed
        )
        handle = pool.submit(
            prompt,
            max_tokens=max_tokens,
            temperature=requested_temperature,
            greedy=requested_greedy,
            top_p=requested_top_p,
            seed=requested_seed,
            session_id=session_id,
            system_prefix=system_prefix,
            deadline=request_deadline,
        )
        if stream:
            _handle_worker_stream(
                handler,
                path,
                handle,
                max_tokens=max_tokens,
                queue_wait_ms=queue_wait_ms,
                started=started,
                session_id=session_id,
            )
            return

        while terminal is None:
            try:
                event = handle.events.get(timeout=0.25)
            except queue.Empty:
                if request_deadline is not None and time.monotonic() >= request_deadline:
                    handle.cancel()
                    terminal = _await_worker_terminal(
                        handle, deadline=time.monotonic() + 1.5
                    )
                    if terminal is None:
                        pool.abort(handle, "worker did not acknowledge deadline cancellation")
                        terminal = {
                            "kind": "cancelled",
                            "error": "request deadline exceeded",
                        }
                    break
                continue
            kind = str(event.get("kind", ""))
            if kind in {"token", "ready", "cancel_ack"}:
                continue
            terminal = event
        kind = str(terminal.get("kind", ""))
        if kind == "done":
            envelope = terminal.get("validated_state_envelope")
            if session_id and envelope is not None:
                pool.record_session_state(session_id, envelope)
            metrics_dict = terminal.get("metrics")
            metrics_dict = metrics_dict if isinstance(metrics_dict, dict) else None
            if session_id and metrics_dict is not None:
                metrics_dict["session_parking"] = {
                    "session_id": session_id,
                    "restored": handle.state_envelope is not None,
                    "ram_bytes": (
                        pool.state_store.ram_bytes if pool.state_store is not None else 0
                    ),
                    "disk_bytes": (
                        pool.state_store.disk_bytes() if pool.state_store is not None else 0
                    ),
                }
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                ttft_ms=float((metrics_dict or {}).get("ttft_s", 0.0) or 0.0) * 1000.0,
                tokens=int(terminal.get("tokens_generated", 0) or 0),
            )
            cfg = pool.config
            _write_json_result(
                handler,
                path,
                str(terminal.get("text", "")),
                max_tokens,
                cfg,
                terminal.get("summary"),
                metrics_dict,
            )
        elif kind == "cancelled":
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                cancelled=True,
            )
            _json_response(handler, 408, {"error": str(terminal.get("error", "cancelled"))})
        elif kind == "worker_crashed":
            _server_stats().inc("worker_failures")
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                failed=True,
            )
            _json_response(handler, 503, {"error": str(terminal.get("error", "worker crashed"))})
        elif kind == "capability_error":
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                failed=True,
            )
            _json_response(
                handler,
                501,
                {
                    "error": str(terminal.get("error", "unsupported capability")),
                    "capability": terminal.get("capability", "unsupported"),
                },
            )
        elif kind == "state_error":
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                failed=True,
            )
            _json_response(handler, 409, {"error": str(terminal.get("error", "invalid session state"))})
        else:
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                failed=True,
            )
            _json_response(handler, 500, {"error": str(terminal.get("error", "worker request failed"))})
        pool.finish(handle, terminal)
    except (WorkerPoolError, WorkerCrashedError) as exc:
        _server_stats().inc("worker_failures")
        _server_stats().inc("requests_failed")
        _json_response(handler, 503, {"error": str(exc)})
    except StateParkingCompatibilityError as exc:
        _server_stats().record_generation(
            queue_wait_ms=queue_wait_ms,
            latency_ms=(time.monotonic() - started) * 1000.0,
            failed=True,
        )
        _json_response(handler, 409, {"error": str(exc)})
    except GenerationCancelled as exc:
        _server_stats().record_generation(
            queue_wait_ms=queue_wait_ms,
            latency_ms=(time.monotonic() - started) * 1000.0,
            cancelled=True,
        )
        _json_response(handler, 408, {"error": str(exc)})
    finally:
        if handle is not None and not handle.terminal:
            if terminal is not None:
                pool.finish(handle, terminal)
            else:
                handle.cancel()
                if _await_worker_terminal(handle, deadline=time.monotonic() + 1.0) is None:
                    pool.abort(handle, "request handler exited before worker completion")
                elif not handle.terminal:
                    pool.finish(handle, {"kind": "cancelled", "error": "request closed"})
        scheduler.release()


def _handle_worker_stream(
    handler: BaseHTTPRequestHandler,
    path: str,
    handle: Any,
    *,
    max_tokens: int,
    queue_wait_ms: float,
    started: float,
    session_id: str,
) -> None:
    pool = CTX.worker_pool
    assert pool is not None
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = _model_id()
    created = int(time.time())
    full_parts: list[str] = []
    terminal: dict[str, Any] | None = None
    cancellation_sent_at: float | None = None
    _sse_headers(handler)
    try:
        while terminal is None:
            try:
                event = handle.events.get(timeout=0.25)
            except queue.Empty:
                if CTX.request_timeout_s > 0 and time.monotonic() - started >= CTX.request_timeout_s:
                    if cancellation_sent_at is None:
                        handle.cancel()
                        cancellation_sent_at = time.monotonic()
                    elif time.monotonic() - cancellation_sent_at >= 1.5:
                        pool.abort(handle, "worker did not acknowledge stream deadline cancellation")
                        terminal = {
                            "kind": "cancelled",
                            "error": "request deadline exceeded",
                        }
                continue
            kind = str(event.get("kind", ""))
            if kind == "token":
                text = str(event.get("text", ""))
                full_parts.append(text)
                if path == "/v1/chat/completions":
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                else:
                    payload = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": text, "finish_reason": None}],
                    }
                _sse_write(handler, payload)
            elif kind not in {"ready", "cancel_ack"}:
                terminal = event
        kind = str(terminal.get("kind", "")) if terminal else "error"
        if kind == "done":
            envelope = terminal.get("validated_state_envelope")
            if session_id and envelope is not None:
                pool.record_session_state(session_id, envelope)
            metrics = terminal.get("metrics")
            metrics_dict = metrics if isinstance(metrics, dict) else {}
            if session_id:
                metrics_dict["session_parking"] = {
                    "session_id": session_id,
                    "restored": handle.state_envelope is not None,
                    "ram_bytes": (
                        pool.state_store.ram_bytes if pool.state_store is not None else 0
                    ),
                    "disk_bytes": (
                        pool.state_store.disk_bytes() if pool.state_store is not None else 0
                    ),
                }
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                ttft_ms=float(metrics_dict.get("ttft_s", 0.0) or 0.0) * 1000.0,
                tokens=int(terminal.get("tokens_generated", 0) or 0),
            )
            finish_reason = "stop"
            if path == "/v1/chat/completions":
                _sse_write(
                    handler,
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "usage": _usage_from_metrics(metrics_dict),
                    },
                )
            else:
                _sse_write(
                    handler,
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
                        "usage": _usage_from_metrics(metrics_dict),
                    },
                )
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        elif kind == "cancelled":
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                cancelled=True,
            )
        elif kind == "worker_crashed":
            _server_stats().inc("worker_failures")
            _server_stats().record_generation(
                queue_wait_ms=queue_wait_ms,
                latency_ms=(time.monotonic() - started) * 1000.0,
                failed=True,
            )
    except (BrokenPipeError, ConnectionResetError, TimeoutError):
        handle.cancel()
        terminal = _await_worker_terminal(handle, deadline=time.monotonic() + 1.0)
        if terminal is None:
            pool.abort(handle, "worker did not acknowledge client disconnect cancellation")
        _server_stats().record_generation(
            queue_wait_ms=queue_wait_ms,
            latency_ms=(time.monotonic() - started) * 1000.0,
            cancelled=True,
        )
    finally:
        if terminal is None and not handle.terminal:
            handle.cancel()
            terminal = _await_worker_terminal(handle, deadline=time.monotonic() + 1.0)
            if terminal is None:
                pool.abort(handle, "stream handler exited before worker completion")
        if terminal is not None:
            pool.finish(handle, terminal)


def _handle_stream(
    handler: BaseHTTPRequestHandler,
    path: str,
    prompt: str,
    payload: dict[str, Any],
    *,
    deadline: float | None = None,
) -> None:
    assert CTX.engine is not None
    model = _model_id()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    _sse_headers(handler)
    created = int(time.time())
    full_parts: list[str] = []
    try:
        for chunk in CTX.engine.generate_stream(prompt, deadline=deadline):
            full_parts.append(chunk)
            if path == "/v1/chat/completions":
                _sse_write(
                    handler,
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    },
                )
            else:
                _sse_write(
                    handler,
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
                    },
                )
    except GenerationCancelled as exc:
        logger.info("stream generation stopped: %s", exc)
        return
    except (BrokenPipeError, ConnectionResetError):
        logger.info("stream client disconnected; generation cancellation requested")
        return
    if path == "/v1/chat/completions":
        _sse_write(
            handler,
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": _usage_from_metrics(
                    CTX.engine.metrics.to_dict() if CTX.engine is not None else None
                ),
            },
        )
    else:
        _sse_write(
            handler,
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": _usage_from_metrics(
                    CTX.engine.metrics.to_dict() if CTX.engine is not None else None
                ),
            },
        )
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def _write_json_result(
    handler: BaseHTTPRequestHandler,
    path: str,
    text: str,
    max_tokens: int,
    cfg: Any,
    summary: str | None,
    metrics_dict: dict[str, Any] | None,
) -> None:
    model = _model_id()
    if path == "/v1/chat/completions":
        _json_response(
            handler,
            200,
            _chat_completion_response(text, model=model, metrics=metrics_dict),
        )
    elif path == "/v1/completions":
        _json_response(
            handler,
            200,
            _completion_response(text, model=model, metrics=metrics_dict),
        )
    else:
        _json_response(
            handler,
            200,
            {
                "text": text,
                "max_tokens": max_tokens,
                "mode": cfg.mode,
                "backend": cfg.backend,
                "metrics_summary": summary,
                "metrics": metrics_dict,
            },
        )


def _model_id() -> str:
    if CTX.worker_pool is not None:
        try:
            meta_path = CTX.worker_pool.config.pack_dir / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return str(meta.get("model_id") or meta.get("model_family") or "rwkv-ssd")
        except (OSError, ValueError, TypeError):
            pass
        return str(CTX.worker_pool.config.backend or "rwkv-ssd")
    if CTX.engine is None or CTX.engine.manifest is None:
        return "rwkv-ssd"
    meta = CTX.engine.manifest.meta
    return str(meta.get("model_id") or meta.get("model_family") or "rwkv-ssd")


def _model_object(model_id: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "owned_by": "local"}


def _usage_from_metrics(metrics: dict[str, Any] | None) -> dict[str, int]:
    values = metrics or {}
    completion = int(values.get("tokens_generated", 0) or 0)
    prompt = int(values.get("prompt_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _render_chat_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    system_parts: list[str] = []
    turns: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        content = _message_content_text(msg.get("content", ""))
        if role == "system":
            system_parts.append(content)
        else:
            turns.append(f"{role}: {content}")
    system_prefix = "\n".join(p for p in system_parts if p).strip()
    prompt = "\n".join(turns).strip()
    if not prompt:
        prompt = system_prefix
        system_prefix = ""
    return prompt, system_prefix


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _chat_completion_response(
    text: str, *, model: str, metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-rwkv-ssd",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    if metrics is not None:
        body["rwkv_ssd_metrics"] = metrics
    body["usage"] = _usage_from_metrics(metrics)
    return body


def _completion_response(
    text: str, *, model: str, metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl-rwkv-ssd",
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }
    if metrics is not None:
        body["rwkv_ssd_metrics"] = metrics
    body["usage"] = _usage_from_metrics(metrics)
    return body


def build_engine_from_args(args: argparse.Namespace) -> InferenceEngine:
    CTX.worker_pool = None
    CTX.state_store = None
    cfg = build_engine_config(args)
    ensure_v0_backend(cfg.backend)
    if cfg.mode != "resident" and not supports_streaming_mode(cfg.backend):
        raise SystemExit(
            "partial|streaming requires rwkvcpp, synthetic, chatrwkv, or albatross backend"
        )
    engine = InferenceEngine(cfg)
    engine.load()
    CTX.state_store = _build_state_store_from_env()
    return engine


def _build_state_store_from_env() -> Any:
    parking_dir = os.environ.get("RWKV_STATE_PARKING_DIR", "").strip()
    if not parking_dir:
        return None
    from rwkv_ssd.runtime.state_parking import HierarchicalStateStore

    ram_bytes = max(0, int(os.environ.get("RWKV_STATE_PARKING_RAM_BYTES", "0")))
    return HierarchicalStateStore(parking_dir, max_ram_bytes=ram_bytes)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="RWKV SSD HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local chat page in your default browser",
    )
    p.add_argument("--cors", action="store_true")
    p.add_argument(
        "--cors-origin",
        default=os.environ.get("RWKV_SSD_CORS_ORIGIN"),
        help="allowed CORS origin(s), comma-separated; required with --cors",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("RWKV_SSD_API_KEY"),
        help="optional API key; X-API-Key or Authorization: Bearer is accepted",
    )
    p.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(os.environ.get("RWKV_SSD_MAX_BODY_BYTES", str(4 * 1024 * 1024))),
    )
    p.add_argument(
        "--max-queue",
        type=int,
        default=int(os.environ.get("RWKV_SSD_MAX_QUEUE", "32")),
        help="bounded number of requests waiting behind the shared engine",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("RWKV_SSD_REQUEST_TIMEOUT", "30")),
        help="total seconds for queue wait plus generation; 0 disables the timeout",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RWKV_SSD_WORKERS", "1")),
        help="number of spawned inference workers (default: 1)",
    )
    p.add_argument(
        "--lightning-url",
        help="Forward OpenAI requests to an existing rwkv_lightning server",
    )
    add_engine_args(p, for_serve=True)
    args = p.parse_args()
    if args.max_body_bytes <= 0:
        raise SystemExit("--max-body-bytes must be positive")
    if args.max_queue < 0:
        raise SystemExit("--max-queue must be >= 0")
    try:
        if args.workers < 1:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("--workers must be >= 1") from exc
    if args.cors and not args.cors_origin:
        raise SystemExit("--cors requires --cors-origin; wildcard CORS is disabled")
    if args.system_prefix or args.stateless_prefix_cache:
        args.state_cache = True

    CTX.cors = bool(args.cors)
    CTX.cors_origin = args.cors_origin
    CTX.api_key = args.api_key
    CTX.max_body_bytes = args.max_body_bytes
    CTX.request_timeout_s = args.request_timeout
    CTX.scheduler = RequestScheduler(
        max_concurrent=int(args.workers), max_queue=args.max_queue
    )
    CTX.stats = ServerStats()
    CTX.lightning_url = args.lightning_url
    CTX.worker_pool = None
    CTX.state_store = None
    if CTX.lightning_url:
        CTX.engine = None
    else:
        try:
            cfg = build_engine_config(args)
            ensure_v0_backend(cfg.backend)
            if cfg.mode != "resident" and not supports_streaming_mode(cfg.backend):
                raise ValueError(
                    "partial|streaming requires rwkvcpp, synthetic, chatrwkv, or albatross backend"
                )
        except Exception as exc:
            raise SystemExit(
                f"Invalid service configuration: {exc}\n"
                "Next step: run `rwkv-ssd doctor --pack <pack> --checkpoint <checkpoint>`."
            ) from exc
        validate_worker_count(cfg, args.workers)
        CTX.state_store = _build_state_store_from_env()
        try:
            CTX.worker_pool = InferenceWorkerPool(
                cfg,
                workers=int(args.workers),
                max_restarts=int(os.environ.get("RWKV_SSD_WORKER_MAX_RESTARTS", "3")),
                start_timeout_s=float(os.environ.get("RWKV_SSD_WORKER_START_TIMEOUT", "120")),
                state_store=CTX.state_store,
            ).start()
            CTX.engine = None
        except Exception as exc:
            if CTX.worker_pool is not None:
                CTX.worker_pool.close()
                CTX.worker_pool = None
            message = str(exc)
            hint = (
                "Check the pack with `rwkv-ssd doctor --pack <pack>` and verify "
                "the selected backend's required files."
            )
            if isinstance(exc, MemoryError) or "out of memory" in message.lower():
                hint = (
                    "Try mode: streaming, a smaller model, or close other "
                    "memory-heavy applications."
                )
            raise SystemExit(
                f"Could not start the CPU inference service: {message}\n"
                f"Next step: {hint}"
            ) from exc

    InferenceHandler.server_version = f"rwkv-ssd/{RWKV_SSD_VERSION}"
    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    mode = "lightning-proxy" if CTX.lightning_url else "local-engine"
    print(
        f"rwkv-ssd serve http://{args.host}:{args.port} mode={mode} "
        "(POST /v1/chat/completions, stream supported)",
        file=sys.stderr,
    )
    if args.open_browser:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        url = f"http://{browser_host}:{server.server_address[1]}/"
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            logger.warning(
                "The browser UI is being served on a non-local interface at %s",
                url,
            )
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    finally:
        if CTX.scheduler is not None:
            CTX.scheduler.close()
        if CTX.engine is not None:
            CTX.engine.close()
        if CTX.worker_pool is not None:
            CTX.worker_pool.close()
        CTX.state_store = None
        server.server_close()


if __name__ == "__main__":
    main()
