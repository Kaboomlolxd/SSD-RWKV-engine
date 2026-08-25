"""Bounded spawned-process workers for the local HTTP service."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.errors import CapabilityNotSupportedError
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from rwkv_ssd.runtime.state_envelope import (
    STATE_SERIALIZATION_VERSION,
    StateEnvelope,
    make_state_envelope,
    model_fingerprint,
    tokenizer_fingerprint,
)
from rwkv_ssd.runtime.state_parking import (
    HierarchicalStateStore,
    StateParkingCompatibilityError,
)


class WorkerPoolError(RuntimeError):
    """A worker request cannot be completed by the process pool."""


class WorkerCrashedError(WorkerPoolError):
    """The worker assigned to a request exited unexpectedly."""


_LAYER_NAME_PATTERNS = (
    re.compile(r"^blocks\.\d+\."),
)


def _dtype_bytes(dtype: object, *, fallback: int = 2) -> int:
    value = str(dtype or "").strip().lower()
    if value in {"float64", "double", "f64"}:
        return 8
    if value in {"float32", "float", "f32"}:
        return 4
    if value in {"float16", "half", "f16", "bfloat16", "bf16"}:
        return 2
    if value in {"uint8", "int8", "u8", "i8", "bool"}:
        return 1
    return max(1, int(fallback))


def _tensor_numel(shape: object) -> int:
    total = 1
    if not isinstance(shape, list):
        return 0
    for dim in shape:
        try:
            total *= max(0, int(dim))
        except (TypeError, ValueError):
            return 0
    return total


def estimate_worker_memory(config: EngineConfig) -> tuple[int, str]:
    """Estimate one spawned engine's high-water RSS from pack metadata.

    The estimate deliberately distinguishes resident weights from streamed
    layers.  It is an admission guard, not a replacement for measured RSS:
    fixed process/runtime overhead and a bounded activation/state allowance
    are included, while mmap file size is not treated as resident RSS by
    itself.  Explicit RAM/cache budgets remain authoritative when configured.
    """
    configured_gb = max(
        float(config.ram_budget_gb or 0.0),
        float(config.cache_budget_gb or 0.0),
    )
    if configured_gb > 0.0:
        return int(configured_gb * 1e9), "configured_per_worker_budget"

    root = config.pack_dir
    manifest_path = root / "manifest.json"
    meta_path = root / "meta.json"
    raw: dict[str, Any] | None = None
    meta: dict[str, Any] = {}
    try:
        raw_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw_value, dict):
            raw = raw_value
            if isinstance(raw.get("meta"), dict):
                meta.update(raw["meta"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raw = None
    try:
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(sidecar, dict):
            for key, value in sidecar.items():
                meta.setdefault(key, value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    fixed_overhead = 256 * 1024 * 1024
    state_allowance = 64 * 1024 * 1024
    if raw is None:
        try:
            file_bytes = sum(
                path.stat().st_size
                for path in root.glob("weights*.bin")
                if path.is_file()
            )
            # Raw safetensors checkpoints have no manifest/weights.bin pair.
            # Count them so worker admission cannot under-estimate resident RSS.
            if file_bytes <= 0:
                file_bytes = sum(
                    path.stat().st_size
                    for path in root.glob("*.safetensors")
                    if path.is_file()
                )
        except OSError:
            file_bytes = 0
        if file_bytes <= 0:
            return fixed_overhead + state_allowance, "fallback_process_overhead"
        source = (
            "safetensors_file_fallback"
            if any(root.glob("*.safetensors"))
            else "weights_file_fallback"
        )
        return int(file_bytes + fixed_overhead + state_allowance), source

    tensors = raw.get("tensors", [])
    if not isinstance(tensors, list) or not tensors:
        return fixed_overhead + state_allowance, "manifest_without_tensor_entries"
    primary_dtype = meta.get("primary_dtype", "bfloat16")
    entries: list[tuple[int, bool, bool, int]] = []
    for item in tensors:
        if not isinstance(item, dict):
            continue
        logical = _tensor_numel(item.get("shape")) * _dtype_bytes(
            item.get("dtype") or primary_dtype
        )
        try:
            packed = max(0, int(item.get("length", 0)))
        except (TypeError, ValueError):
            packed = 0
        # Lossy entries are decoded to a model dtype before a dense sequence
        # backend uses them.  Native layer streaming may borrow packed bytes,
        # so retain the larger of packed and logical only for the conservative
        # active-layer estimate below.
        size = max(logical, packed)
        try:
            layer_id = int(item.get("layer_id", -1))
        except (TypeError, ValueError):
            layer_id = -1
        name = str(item.get("name", ""))
        is_layer = any(pattern.match(name) for pattern in _LAYER_NAME_PATTERNS)
        is_resident = str(item.get("residency", "resident")).lower() == "resident"
        entries.append((layer_id, is_layer, is_resident, size))

    mode = str(config.mode).strip().lower()
    if mode == "resident":
        model_bytes = sum(size for _, _, _, size in entries)
        source = "manifest_resident_dense_estimate"
    else:
        global_bytes = sum(
            size for _, is_layer, is_resident, size in entries if not is_layer or is_resident
        )
        streamed_by_layer: dict[int, int] = {}
        for layer_id, is_layer, is_resident, size in entries:
            if is_layer and not is_resident:
                streamed_by_layer[layer_id] = streamed_by_layer.get(layer_id, 0) + size
        active_layer = max(streamed_by_layer.values(), default=0)
        cache_layers = max(1, int(config.max_provider_cache_layers or 0))
        if not config.stream_layer_cache:
            cache_layers = 1
        model_bytes = global_bytes + active_layer * cache_layers
        source = "manifest_streamed_layer_estimate"

    configured_cache_bytes = max(
        int(config.max_provider_cache_bytes or 0),
        int(float(config.cache_budget_gb or 0.0) * 1e9),
    )
    if configured_cache_bytes:
        model_bytes = max(model_bytes, configured_cache_bytes)
    return int(model_bytes + fixed_overhead + state_allowance), source


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return max(0, int(psutil.virtual_memory().available))
    except Exception:
        return 0


def validate_worker_count(config: EngineConfig, workers: int) -> None:
    """Reject worker counts that clearly exceed configured/physical memory."""
    count = int(workers)
    if count < 1:
        raise ValueError("workers must be >= 1")
    maximum = os.environ.get("RWKV_SSD_MAX_WORKERS", "").strip()
    if maximum:
        try:
            if count > int(maximum):
                raise ValueError(
                    f"workers={count} exceeds RWKV_SSD_MAX_WORKERS={maximum}"
                )
        except ValueError as exc:
            if "workers=" in str(exc):
                raise
            raise ValueError("RWKV_SSD_MAX_WORKERS must be an integer") from exc

    per_worker_bytes, _ = estimate_worker_memory(config)
    available_bytes = _available_memory_bytes()
    if available_bytes > 0 and per_worker_bytes * count > available_bytes * 0.90:
        raise ValueError(
            f"workers={count} requires an estimated {per_worker_bytes * count / 1e9:.2f} GB "
            f"for {per_worker_bytes / 1e9:.2f} GB per worker, but only about "
            f"{available_bytes / 1e9:.2f} GB is available"
        )


def _worker_identity(config: EngineConfig) -> tuple[str, str, str]:
    try:
        pack_fp = model_fingerprint(config.pack_dir)
    except (OSError, ValueError):
        pack_fp = "unknown"
    try:
        tokenizer_fp = tokenizer_fingerprint(config.pack_dir)
    except OSError:
        tokenizer_fp = "missing"
    return str(config.backend), pack_fp, tokenizer_fp


def _worker_process_main(
    worker_id: int,
    config: EngineConfig,
    request_queue: Any,
    control_queue: Any,
    event_queue: Any,
) -> None:
    """Worker entry point; kept module-level so Windows spawn can import it."""
    engine: InferenceEngine | None = None
    backend_kind, pack_fp, tokenizer_fp = _worker_identity(config)
    cancel_events: dict[str, threading.Event] = {}
    cancel_lock = threading.Lock()
    stop_control = threading.Event()

    def control_loop() -> None:
        while not stop_control.is_set():
            try:
                message = control_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if not isinstance(message, dict):
                continue
            kind = str(message.get("kind", ""))
            if kind == "shutdown":
                stop_control.set()
                with cancel_lock:
                    for event in cancel_events.values():
                        event.set()
                return
            if kind == "cancel":
                request_id = str(message.get("request_id", ""))
                with cancel_lock:
                    event = cancel_events.get(request_id)
                if event is not None:
                    event.set()
                # A cancellation request is an IPC operation in its own
                # right.  The terminal ``cancelled`` event can arrive much
                # later (or a native call can fail before it reaches a token
                # boundary), so acknowledge receipt separately.  The parent
                # deliberately does not treat this event as terminal.
                event_queue.put(
                    {
                        "kind": "cancel_ack",
                        "request_id": request_id,
                        "accepted": event is not None,
                    }
                )

    control_thread = threading.Thread(
        target=control_loop,
        name=f"rwkv-worker-{worker_id}-control",
        daemon=True,
    )
    control_thread.start()
    try:
        engine = InferenceEngine(config)
        engine.load()
        event_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "backend_kind": backend_kind,
                "model_fingerprint": pack_fp,
                "tokenizer_fingerprint": tokenizer_fp,
                "pid": os.getpid(),
            }
        )
    except BaseException as exc:
        event_queue.put(
            {
                "kind": "startup_error",
                "worker_id": worker_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )
        stop_control.set()
        return

    try:
        while not stop_control.is_set():
            try:
                message = request_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if not isinstance(message, dict):
                continue
            if str(message.get("kind", "")) == "shutdown":
                break
            if str(message.get("kind", "")) != "generate":
                continue
            request_id = str(message.get("request_id", ""))
            cancel_event = threading.Event()
            with cancel_lock:
                cancel_events[request_id] = cancel_event
            old_max = engine.config.max_tokens
            old_temp = engine.config.temperature
            old_greedy = engine.config.greedy
            old_top_p = engine.config.top_p
            old_seed = engine.config.seed
            old_prefix = engine.config.system_prefix
            try:
                envelope = message.get("state_envelope")
                if envelope is not None:
                    if not isinstance(envelope, StateEnvelope):
                        raise ValueError("invalid state envelope payload")
                    envelope.validate_against(
                        backend_kind=backend_kind,
                        model_fingerprint=pack_fp,
                        tokenizer_fingerprint=tokenizer_fp,
                    )
                    engine.backend.set_recurrent_state(envelope.state_payload)

                engine.config.max_tokens = int(message["max_tokens"])
                engine.config.temperature = float(message.get("temperature", 1.0))
                engine.config.greedy = bool(message.get("greedy", True))
                engine.config.top_p = float(message.get("top_p", 1.0))
                raw_seed = message.get("seed")
                engine.config.seed = int(raw_seed) if raw_seed is not None else None
                system_prefix = str(message.get("system_prefix", "") or "")
                if system_prefix:
                    engine.config.system_prefix = system_prefix

                def on_token(token_id: int) -> None:
                    event_queue.put(
                        {
                            "kind": "token",
                            "request_id": request_id,
                            "session_id": str(message.get("session_id", "")),
                            "token_id": int(token_id),
                            "text": engine.backend.decode_text([int(token_id)]),
                        }
                    )

                if envelope is not None:
                    text = engine.generate_followup(
                        str(message.get("prompt", "")),
                        max_tokens=int(message["max_tokens"]),
                        token_callback=on_token,
                        cancel_event=cancel_event,
                        deadline=message.get("deadline"),
                    )
                    token_ids: list[int] = []
                else:
                    token_ids = engine.generate_tokens(
                        str(message.get("prompt", "")),
                        token_callback=on_token,
                        cancel_event=cancel_event,
                        deadline=message.get("deadline"),
                    )
                    text = engine.backend.decode_text(token_ids)
                state = engine.backend.get_recurrent_state()
                state_envelope = (
                    make_state_envelope(
                        state,
                        backend_kind=backend_kind,
                        model_fingerprint=pack_fp,
                        tokenizer_fingerprint=tokenizer_fp,
                    )
                    if state is not None and message.get("session_id")
                    else None
                )
                metrics = engine.metrics.to_dict()
                event_queue.put(
                    {
                        "kind": "done",
                        "request_id": request_id,
                        "text": text,
                        "token_ids": token_ids,
                        "tokens_generated": int(engine.metrics.tokens_generated),
                        "metrics": metrics,
                        "summary": engine.metrics.summary()
                        if engine.metrics.layers
                        else None,
                        "state_envelope": state_envelope,
                    }
                )
            except GenerationCancelled as exc:
                event_queue.put(
                    {
                        "kind": "cancelled",
                        "request_id": request_id,
                        "error": str(exc),
                    }
                )
            except CapabilityNotSupportedError as exc:
                event_queue.put(
                    {
                        "kind": "capability_error",
                        "request_id": request_id,
                        "error": str(exc),
                        "capability": getattr(exc, "capability", "unsupported"),
                    }
                )
            except BaseException as exc:
                event_queue.put(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
            finally:
                engine.config.max_tokens = old_max
                engine.config.temperature = old_temp
                engine.config.greedy = old_greedy
                engine.config.top_p = old_top_p
                engine.config.seed = old_seed
                engine.config.system_prefix = old_prefix
                with cancel_lock:
                    cancel_events.pop(request_id, None)
    finally:
        stop_control.set()
        if engine is not None:
            engine.close()


@dataclass
class _WorkerSlot:
    worker_id: int
    process: Any = None
    request_queue: Any = None
    control_queue: Any = None
    event_queue: Any = None
    ready: bool = False
    healthy: bool = False
    startup_error: str | None = None
    busy: bool = False
    current_request_id: str | None = None
    rss_bytes: int = 0
    restarts: int = 0
    pid: int | None = None
    reader: threading.Thread | None = None


@dataclass
class WorkerRequestHandle:
    request_id: str
    worker_id: int
    events: queue.Queue[dict[str, Any]]
    pool: "InferenceWorkerPool"
    state_envelope: StateEnvelope | None = None
    terminal: bool = False
    cancel_requested: bool = False
    cancel_acknowledged: bool = False

    def cancel(self) -> None:
        self.cancel_requested = True
        self.pool.cancel(self.request_id)


class InferenceWorkerPool:
    """Parent-side lifecycle, routing, and IPC facade for spawned engines."""

    def __init__(
        self,
        config: EngineConfig,
        workers: int = 1,
        *,
        max_restarts: int = 3,
        start_timeout_s: float = 120.0,
        max_sessions: int = 1024,
        state_store: HierarchicalStateStore | None = None,
    ) -> None:
        validate_worker_count(config, workers)
        self.config = config
        self.worker_count = int(workers)
        self.max_restarts = max(0, int(max_restarts))
        self.start_timeout_s = max(1.0, float(start_timeout_s))
        self.max_sessions = max(1, int(max_sessions))
        self.state_store = state_store
        self._ctx = mp.get_context("spawn")
        self._slots = [_WorkerSlot(index) for index in range(self.worker_count)]
        self._condition = threading.Condition()
        self._responses: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._request_worker: dict[str, int] = {}
        self._request_session: dict[str, str] = {}
        self._session_states: dict[str, StateEnvelope] = {}
        self._session_workers: dict[str, int] = {}
        self._round_robin = 0
        self._closed = False
        self._monitor: threading.Thread | None = None
        self._backend_kind, self.model_fingerprint, self.tokenizer_fingerprint = _worker_identity(config)
        (
            self.memory_estimate_bytes_per_worker,
            self.memory_estimate_source,
        ) = estimate_worker_memory(config)

    def _spawn_slot(self, slot: _WorkerSlot) -> None:
        slot.request_queue = self._ctx.Queue(maxsize=1)
        slot.control_queue = self._ctx.Queue()
        slot.event_queue = self._ctx.Queue()
        slot.ready = False
        slot.healthy = False
        slot.startup_error = None
        slot.busy = False
        slot.current_request_id = None
        process = self._ctx.Process(
            target=_worker_process_main,
            args=(
                slot.worker_id,
                self.config,
                slot.request_queue,
                slot.control_queue,
                slot.event_queue,
            ),
            name=f"rwkv-ssd-worker-{slot.worker_id}",
        )
        process.daemon = True
        slot.process = process
        process.start()
        reader = threading.Thread(
            target=self._reader_loop,
            args=(slot.worker_id, slot.event_queue),
            name=f"rwkv-worker-{slot.worker_id}-reader",
            daemon=True,
        )
        slot.reader = reader
        reader.start()

    def start(self) -> "InferenceWorkerPool":
        for slot in self._slots:
            self._spawn_slot(slot)
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="rwkv-worker-pool-monitor",
            daemon=True,
        )
        self._monitor.start()
        deadline = time.monotonic() + self.start_timeout_s
        with self._condition:
            while not self._closed:
                if any(slot.healthy for slot in self._slots):
                    # A partially started pool is usable and reports degraded
                    # capacity; a failed all-worker startup remains fatal.
                    if all(slot.ready or slot.startup_error for slot in self._slots):
                        break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(min(0.25, remaining))
        if not any(slot.healthy for slot in self._slots):
            errors = "; ".join(
                slot.startup_error or "worker did not become ready" for slot in self._slots
            )
            self.close()
            raise WorkerPoolError(f"no inference worker became ready: {errors}")
        return self

    def _reader_loop(self, worker_id: int, event_queue: Any) -> None:
        while not self._closed:
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            if not isinstance(event, dict):
                continue
            kind = str(event.get("kind", ""))
            with self._condition:
                slot = self._slots[worker_id]
                if kind == "ready":
                    slot.ready = True
                    slot.healthy = True
                    slot.pid = int(event.get("pid", 0) or 0) or None
                    if str(event.get("backend_kind", "")) != self._backend_kind:
                        slot.healthy = False
                        slot.startup_error = "worker backend identity mismatch"
                    if str(event.get("model_fingerprint", "")) != self.model_fingerprint:
                        slot.healthy = False
                        slot.startup_error = "worker model fingerprint mismatch"
                    if str(event.get("tokenizer_fingerprint", "")) != self.tokenizer_fingerprint:
                        slot.healthy = False
                        slot.startup_error = "worker tokenizer fingerprint mismatch"
                    self._condition.notify_all()
                    continue
                if kind == "startup_error":
                    slot.ready = False
                    slot.healthy = False
                    slot.startup_error = str(event.get("error", "worker startup failed"))
                    self._condition.notify_all()
                    continue
                request_id = str(event.get("request_id", ""))
                response = self._responses.get(request_id)
                if response is not None:
                    if kind == "cancel_ack":
                        # Preserve the acknowledgement in the request event
                        # stream so HTTP/tests can observe it, while leaving
                        # the worker slot busy until the terminal event.
                        response.put(event)
                        self._condition.notify_all()
                        continue
                    if kind == "done":
                        envelope = event.get("state_envelope")
                        if isinstance(envelope, StateEnvelope):
                            try:
                                envelope.validate_against(
                                    backend_kind=self._backend_kind,
                                    model_fingerprint=self.model_fingerprint,
                                    tokenizer_fingerprint=self.tokenizer_fingerprint,
                                )
                            except ValueError as exc:
                                event = {
                                    "kind": "state_error",
                                    "request_id": request_id,
                                    "error": str(exc),
                                }
                            else:
                                event["validated_state_envelope"] = envelope
                        event["session_id"] = self._request_session.get(request_id, "")
                        metrics = event.get("metrics")
                        if isinstance(metrics, dict):
                            slot.rss_bytes = int(
                                metrics.get(
                                    "process_rss_peak_bytes",
                                    metrics.get("process_rss_bytes", slot.rss_bytes),
                                )
                                or slot.rss_bytes
                            )
                    response.put(event)
                # The handler owns terminal acknowledgement.  Keeping the
                # slot busy until ``finish`` prevents a second request from
                # racing session-state recording or reusing mutable worker
                # configuration between the IPC event and HTTP response.
                if kind in {"done", "error", "cancelled", "capability_error", "state_error"}:
                    self._condition.notify_all()

    def _monitor_loop(self) -> None:
        while not self._closed:
            time.sleep(0.25)
            for slot in self._slots:
                process = slot.process
                if process is None or process.is_alive():
                    continue
                with self._condition:
                    request_id = slot.current_request_id
                    slot.healthy = False
                    slot.ready = False
                    slot.startup_error = f"worker exited with code {process.exitcode}"
                    if request_id:
                        response = self._responses.get(request_id)
                        if response is not None:
                            response.put(
                                {
                                    "kind": "worker_crashed",
                                    "request_id": request_id,
                                    "error": slot.startup_error,
                                }
                            )
                        slot.busy = False
                        slot.current_request_id = None
                    self._condition.notify_all()
                if slot.restarts < self.max_restarts and not self._closed:
                    slot.restarts += 1
                    try:
                        self._spawn_slot(slot)
                    except BaseException as exc:
                        slot.startup_error = f"worker restart failed: {exc}"

    def _choose_slot(self, session_id: str, deadline: float | None) -> _WorkerSlot:
        with self._condition:
            assigned = None
            if session_id:
                assigned = self._session_workers.get(session_id)
                if assigned is None:
                    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
                    assigned = int.from_bytes(digest[:4], "little") % self.worker_count
                    self._session_workers[session_id] = assigned
            while not self._closed:
                candidates = self._slots
                if assigned is not None:
                    candidates = [self._slots[assigned]]
                else:
                    start = self._round_robin % self.worker_count
                    candidates = self._slots[start:] + self._slots[:start]
                for slot in candidates:
                    if slot.healthy and not slot.busy:
                        self._round_robin = slot.worker_id + 1
                        slot.busy = True
                        return slot
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise WorkerPoolError("no healthy worker became available before deadline")
                self._condition.wait(0.25 if remaining is None else min(0.25, remaining))
        raise WorkerPoolError("worker pool is shutting down")

    def submit(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        greedy: bool,
        top_p: float = 1.0,
        seed: int | None = None,
        session_id: str = "",
        system_prefix: str = "",
        deadline: float | None = None,
    ) -> WorkerRequestHandle:
        if self._closed:
            raise WorkerPoolError("worker pool is closed")
        request_id = uuid.uuid4().hex
        envelope = self._get_session_state(session_id) if session_id else None
        if envelope is not None:
            envelope.validate_against(
                backend_kind=self._backend_kind,
                model_fingerprint=self.model_fingerprint,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
            )
        slot = self._choose_slot(session_id, deadline)
        response: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._condition:
            self._responses[request_id] = response
            self._request_worker[request_id] = slot.worker_id
            self._request_session[request_id] = str(session_id)
            slot.current_request_id = request_id
        try:
            slot.request_queue.put(
                {
                    "kind": "generate",
                    "request_id": request_id,
                    "prompt": str(prompt),
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "greedy": bool(greedy),
                    "top_p": float(top_p),
                    "seed": (int(seed) if seed is not None else None),
                    "session_id": str(session_id),
                    "system_prefix": str(system_prefix or ""),
                    "deadline": deadline,
                    "state_envelope": envelope,
                },
                timeout=1.0,
            )
        except BaseException:
            with self._condition:
                self._responses.pop(request_id, None)
                self._request_worker.pop(request_id, None)
                self._request_session.pop(request_id, None)
                slot.busy = False
                slot.current_request_id = None
                self._condition.notify_all()
            raise
        return WorkerRequestHandle(
            request_id,
            slot.worker_id,
            response,
            self,
            state_envelope=envelope,
        )

    def _get_session_state(self, session_id: str) -> StateEnvelope | None:
        """Load a session envelope from the parent cache or parked state."""
        if not session_id:
            return None
        with self._condition:
            envelope = self._session_states.get(session_id)
        if envelope is not None:
            return envelope
        if self.state_store is None:
            return None
        state = self.state_store.get(
            session_id,
            backend=self._backend_kind,
            model_fingerprint=self.model_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            serialization_version=STATE_SERIALIZATION_VERSION,
        )
        if state is None:
            return None
        envelope = make_state_envelope(
            state,
            backend_kind=self._backend_kind,
            model_fingerprint=self.model_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        with self._condition:
            self._session_states[session_id] = envelope
        return envelope

    def cancel(self, request_id: str) -> None:
        with self._condition:
            worker_id = self._request_worker.get(str(request_id))
            if worker_id is None:
                return
            slot = self._slots[worker_id]
        try:
            slot.control_queue.put({"kind": "cancel", "request_id": str(request_id)})
        except (OSError, ValueError):
            pass

    def finish(self, handle: WorkerRequestHandle, event: dict[str, Any]) -> None:
        if event.get("kind") == "done":
            envelope = event.get("validated_state_envelope")
            session_id = str(event.get("session_id", ""))
            # The event itself does not need to carry the session id for the
            # model; retain it in a request map supplied by the handler.
            if isinstance(envelope, StateEnvelope):
                setattr(handle, "state_envelope", envelope)
        with self._condition:
            if handle.terminal:
                return
            handle.terminal = True
            slot = self._slots[handle.worker_id]
            self._responses.pop(handle.request_id, None)
            self._request_worker.pop(handle.request_id, None)
            self._request_session.pop(handle.request_id, None)
            if slot.current_request_id == handle.request_id:
                slot.current_request_id = None
                slot.busy = False
            self._condition.notify_all()

    def abort(self, handle: WorkerRequestHandle, reason: str = "worker cancellation timed out") -> None:
        """Terminate an unresponsive worker and let the monitor restart it.

        Generation cancellation is cooperative.  If a native backend fails to
        acknowledge it within the HTTP deadline grace period, retaining the
        slot indefinitely would make the bounded pool permanently lose
        capacity.  Terminating only the assigned worker preserves isolation
        for all other requests.
        """
        with self._condition:
            if handle.terminal:
                return
            slot = self._slots[handle.worker_id]
            slot.healthy = False
            slot.ready = False
            slot.startup_error = str(reason)
            process = slot.process
        if process is not None and process.is_alive():
            try:
                process.terminate()
                process.join(timeout=1.0)
            except (OSError, ValueError):
                pass
        self.finish(
            handle,
            {"kind": "worker_crashed", "request_id": handle.request_id, "error": reason},
        )

    def record_session_state(self, session_id: str, envelope: StateEnvelope | None) -> None:
        if not session_id or envelope is None:
            return
        envelope.validate_against(
            backend_kind=self._backend_kind,
            model_fingerprint=self.model_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        if self.state_store is not None:
            self.state_store.put(
                session_id,
                envelope.state_payload,
                backend=self._backend_kind,
                model_family=self._backend_kind,
                model_fingerprint=self.model_fingerprint,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                serialization_version=STATE_SERIALIZATION_VERSION,
            )
        with self._condition:
            self._session_states[session_id] = envelope
            while len(self._session_states) > self.max_sessions:
                self._session_states.pop(next(iter(self._session_states)))

    def clear_session_state(self, session_id: str) -> None:
        with self._condition:
            self._session_states.pop(session_id, None)

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._condition:
            rows = [
                {
                    "worker_id": slot.worker_id,
                    "pid": slot.pid,
                    "healthy": bool(slot.healthy and slot.process is not None and slot.process.is_alive()),
                    "ready": bool(slot.ready),
                    "busy": bool(slot.busy),
                    "rss_bytes": int(slot.rss_bytes),
                    "restarts": int(slot.restarts),
                    "error": slot.startup_error,
                }
                for slot in self._slots
            ]
            healthy = sum(1 for row in rows if row["healthy"])
            return {
                "worker_count": self.worker_count,
                "healthy_workers": healthy,
                "degraded": healthy < self.worker_count,
                "active_generations": sum(1 for row in rows if row["busy"]),
                "memory_estimate_bytes_per_worker": int(
                    self.memory_estimate_bytes_per_worker
                ),
                "memory_estimate_bytes_total": int(
                    self.memory_estimate_bytes_per_worker * self.worker_count
                ),
                "memory_estimate_source": self.memory_estimate_source,
                "workers": rows,
            }

    def close(self) -> None:
        self._closed = True
        for slot in self._slots:
            try:
                slot.control_queue.put({"kind": "shutdown"})
                slot.request_queue.put({"kind": "shutdown"}, timeout=0.2)
            except (OSError, ValueError, queue.Full, AttributeError):
                pass
        for slot in self._slots:
            process = slot.process
            if process is None:
                continue
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            for q in (slot.request_queue, slot.control_queue, slot.event_queue):
                try:
                    q.close()
                except (AttributeError, OSError):
                    pass
        if self._monitor is not None:
            self._monitor.join(timeout=1.0)
        with self._condition:
            self._condition.notify_all()


__all__ = [
    "InferenceWorkerPool",
    "WorkerCrashedError",
    "WorkerPoolError",
    "WorkerRequestHandle",
    "estimate_worker_memory",
    "validate_worker_count",
]
