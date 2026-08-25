"""Backend conformance diagnostics.

The normal engine contract intentionally returns text or token IDs.  This
module provides the opt-in diagnostic layer used when qualifying resident,
streaming, and native implementations against a correctness oracle.  It does
not require internal states to have the same representation: token equality is
the hard contract and logits/state measurements are guardrails.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class StateObservation:
    """A backend state plus the layout needed for semantic comparison.

    Backends are allowed to own different state buffers.  In particular,
    ChatRWKV exposes RWKV-7 as three tensors per layer in
    ``[attention_prev, attention_matrix, ffn_prev]`` order, while rwkv.cpp
    exposes one flat FP32 buffer in ``[ffn_prev, attention_prev,
    attention_matrix]`` order.  Keeping this metadata on the opt-in probe
    value avoids changing the public snapshot ABI just to make diagnostics
    meaningful.
    """

    value: Any
    layout: str = "generic"
    n_layer: int | None = None
    n_embed: int | None = None
    head_count: int | None = None
    head_size: int | None = None


@dataclass(frozen=True)
class ParityThresholds:
    """Configurable quality guardrails for a parity trace."""

    min_top10_overlap: float = 0.80
    max_kl: float = 0.05
    max_state_relative_error: float = 0.10

    def __post_init__(self) -> None:
        """Reject invalid gates before a certification run starts.

        A malformed quality certificate should fail closed.  In particular,
        accepting ``NaN`` or a negative overlap threshold would turn a
        diagnostic into an accidental pass and is much harder to spot in a
        batch certification report than an input error here.
        """
        values = {
            "min_top10_overlap": float(self.min_top10_overlap),
            "max_kl": float(self.max_kl),
            "max_state_relative_error": float(self.max_state_relative_error),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("parity thresholds must be finite numbers")
        if not 0.0 <= values["min_top10_overlap"] <= 1.0:
            raise ValueError("min_top10_overlap must be between 0 and 1")
        if values["max_kl"] < 0.0:
            raise ValueError("max_kl must be non-negative")
        if values["max_state_relative_error"] < 0.0:
            raise ValueError("max_state_relative_error must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ParityThresholds":
        raw = raw or {}

        def get(*names: str, default: float) -> float:
            for name in names:
                if name in raw and raw[name] is not None:
                    return float(raw[name])
            return default

        return cls(
            min_top10_overlap=get(
                "min_top10_overlap", "top10_overlap", "top_k_overlap", default=0.80
            ),
            max_kl=get("max_kl", "kl", "max_kl_candidate_to_reference", default=0.05),
            max_state_relative_error=get(
                "max_state_relative_error",
                "max_state_relative_l2",
                "state_relative_error",
                default=0.10,
            ),
        )

    @classmethod
    def from_quality_certificate(cls, certificate: Mapping[str, Any]) -> "ParityThresholds":
        """Read thresholds from a v2 certificate's evaluated gate records."""
        gates = certificate.get("gates", {})
        values: dict[str, Any] = {}
        if isinstance(gates, Mapping):
            for name, gate in gates.items():
                if not isinstance(gate, Mapping):
                    continue
                metric = str(gate.get("metric", name))
                if "min" in gate:
                    values[metric] = gate["min"]
                elif "max" in gate:
                    values[metric] = gate["max"]
        return cls.from_mapping(values)


def _unwrap_recurrent_state(value: Any, field: str) -> Any:
    """Extract one backend-owned payload from a RecurrentState wrapper."""
    if hasattr(value, field):
        return getattr(value, field)
    return value


def _canonical_rwkv7_state(observation: StateObservation) -> np.ndarray:
    """Return a canonical FP64 RWKV-7 state vector for guardrail metrics.

    The canonical order is the order used by ChatRWKV's model state:
    ``attention_prev, attention_matrix, ffn_prev`` for every layer.  The
    conversion is only performed at the diagnostic observation boundary; the
    runtime state remains in the native backend representation.
    """
    value = observation.value
    if observation.layout == "chatrwkv_rwkv7":
        state = _unwrap_recurrent_state(value, "rwkv7_state")
        if state is None:
            return np.empty(0, dtype=np.float64)
        if not isinstance(state, (list, tuple)):
            raise ValueError("ChatRWKV RWKV-7 state must be a list or tuple")
        if len(state) % 3 != 0:
            raise ValueError(
                "ChatRWKV RWKV-7 state must contain three tensors per layer"
            )
        parts: list[np.ndarray] = []
        for layer in range(0, len(state), 3):
            # ChatRWKV's state list is [att_xx, att_heads, ffn_xx].
            parts.extend(
                _array(state[layer + offset]).reshape(-1)
                for offset in (0, 1, 2)
            )
        return np.concatenate(parts).astype(np.float64, copy=False)

    if observation.layout == "rwkv_cpp_rwkv7":
        raw = _unwrap_recurrent_state(value, "external_state")
        flat = np.asarray(_array(raw), dtype=np.float64).reshape(-1)
        n_layer = observation.n_layer
        n_embed = observation.n_embed
        head_count = observation.head_count
        head_size = observation.head_size
        if not n_layer or not n_embed or not head_count or not head_size:
            raise ValueError(
                "rwkv.cpp RWKV-7 state comparison requires model dimensions"
            )
        state_len = int(n_embed) * (2 + int(head_size))
        expected = int(n_layer) * state_len
        if flat.size != expected:
            raise ValueError(
                f"rwkv.cpp RWKV-7 state has {flat.size} values; expected {expected}"
            )
        matrix_size = int(head_count) * int(head_size) * int(head_size)
        if state_len != 2 * int(n_embed) + matrix_size:
            raise ValueError(
                "rwkv.cpp RWKV-7 state dimensions are internally inconsistent"
            )
        parts = []
        for layer in range(int(n_layer)):
            offset = layer * state_len
            ffn_prev = flat[offset : offset + int(n_embed)]
            att_prev = flat[offset + int(n_embed) : offset + 2 * int(n_embed)]
            att_matrix = flat[offset + 2 * int(n_embed) : offset + state_len]
            # Flattening the matrix is intentional after the semantic field
            # reorder.  Both backends use the same H*N*N row-major values;
            # internal tensor rank is not part of this guardrail.
            parts.extend((att_prev, att_matrix, ffn_prev))
        return np.concatenate(parts).astype(np.float64, copy=False)

    return np.asarray(_array(value), dtype=np.float64).reshape(-1)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, StateObservation):
        return _canonical_rwkv7_state(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # PyTorch intentionally does not expose ``Tensor.numpy()`` for BF16
        # on several supported versions/platforms.  Parity diagnostics are
        # numeric guardrails, so promote only at this observation boundary;
        # the backend-owned state remains in its native dtype.
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    if hasattr(value, "h") or hasattr(value, "rwkv7_state") or hasattr(value, "external_state"):
        parts: list[np.ndarray] = []
        h = getattr(value, "h", None)
        if h is not None:
            parts.append(_array(h).reshape(-1))
        rwkv7 = getattr(value, "rwkv7_state", None)
        if rwkv7 is not None:
            parts.extend(_array(item).reshape(-1) for item in rwkv7)
        external = getattr(value, "external_state", None)
        if external is not None:
            parts.append(_array(external).reshape(-1))
        sequence = getattr(value, "sequence_state", None)
        if sequence is not None:
            next_logits = getattr(sequence, "next_logits", None)
            if next_logits is not None:
                parts.append(_array(next_logits).reshape(-1))
            for layer in getattr(sequence, "layers", ()):  # Mamba/Transformer state
                for name in ("conv", "ssm", "key", "value"):
                    tensor = getattr(layer, name, None)
                    if tensor is not None:
                        parts.append(_array(tensor).reshape(-1))
        if parts:
            return np.concatenate(parts).astype(np.float64, copy=False)
    if isinstance(value, (list, tuple)):
        parts = [_array(item).reshape(-1) for item in value]
        if parts:
            return np.concatenate(parts).astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def top_k_overlap(reference_logits: Any, candidate_logits: Any, *, k: int = 10) -> float:
    reference = _array(reference_logits).reshape(-1)
    candidate = _array(candidate_logits).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate logits have different shapes")
    if reference.size == 0:
        return 1.0
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return 0.0
    width = min(max(1, int(k)), reference.size)
    ref = set(np.argpartition(reference, -width)[-width:].tolist())
    cand = set(np.argpartition(candidate, -width)[-width:].tolist())
    # The argpartition values above are positions, not token IDs.  Using the
    # selected indices is exactly what we need: overlap is about vocabulary
    # positions, and ties do not affect generation or the conformance gate.
    return float(len(ref & cand) / width)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(np.clip(shifted, -745.0, 80.0))
    total = np.sum(values)
    return values / total if total > 0 else np.full_like(values, 1.0 / values.size)


def kl_candidate_to_reference(reference_logits: Any, candidate_logits: Any) -> float:
    reference = _array(reference_logits).reshape(-1)
    candidate = _array(candidate_logits).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate logits have different shapes")
    if reference.size == 0:
        return 0.0
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return math.inf
    ref = _softmax(reference)
    cand = _softmax(candidate)
    return float(np.sum(cand * (np.log(np.maximum(cand, 1e-12)) - np.log(np.maximum(ref, 1e-12)))))


def relative_state_error(reference_state: Any, candidate_state: Any) -> float:
    reference = _array(reference_state).reshape(-1)
    candidate = _array(candidate_state).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate states have different shapes")
    if reference.size == 0:
        return 0.0
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return math.inf
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    return float(np.linalg.norm(candidate - reference) / denom)


@dataclass
class ParityStep:
    token_index: int
    reference_token_id: int | None = None
    candidate_token_id: int | None = None
    greedy_match: bool | None = None
    top10_overlap: float | None = None
    kl: float | None = None
    state_relative_error: float | None = None
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    def guardrails_pass(self, thresholds: ParityThresholds) -> bool:
        return (
            (self.top10_overlap is None or self.top10_overlap >= thresholds.min_top10_overlap)
            and (self.kl is None or self.kl <= thresholds.max_kl)
            and (
                self.state_relative_error is None
                or self.state_relative_error <= thresholds.max_state_relative_error
            )
        )


@dataclass
class ParityTrace:
    """Serializable per-prompt parity evidence."""

    backend: str
    reference_backend: str | None = None
    prompt: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    decoded_text: str = ""
    reference_decoded_text: str | None = None
    decoded_text_match: bool | None = None
    thresholds: ParityThresholds = field(default_factory=ParityThresholds)
    require_guardrails: bool = False
    diagnostics_complete: bool = True
    steps: list[ParityStep] = field(default_factory=list)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    rss_bytes: int = 0
    rss_peak_bytes: int = 0
    streamed_bytes: int = 0
    cache_hits: int = 0
    cancelled: bool = False
    deadline_exceeded: bool = False
    notes: list[str] = field(default_factory=list)
    created_unix_s: float = field(default_factory=time.time)

    def add_step(self, step: ParityStep) -> None:
        self.steps.append(step)

    @property
    def exact_greedy_match(self) -> bool:
        return all(step.greedy_match is not False for step in self.steps)

    @property
    def guardrails_passed(self) -> bool:
        return all(step.guardrails_pass(self.thresholds) for step in self.steps)

    @property
    def passed(self) -> bool:
        return (
            self.exact_greedy_match
            and self.guardrails_passed
            and (not self.require_guardrails or self.diagnostics_complete)
            and self.decoded_text_match is not False
            and not self.cancelled
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["thresholds"] = asdict(self.thresholds)
        out["exact_greedy_match"] = self.exact_greedy_match
        out["guardrails_passed"] = self.guardrails_passed
        out["passed"] = self.passed
        return out

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target


@dataclass
class BackendProbeResult:
    """Runner result accepted by :func:`run_backend_conformance`."""

    token_ids: list[int]
    text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    logits: list[Any] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def _coerce_probe_result(value: Any) -> BackendProbeResult:
    if isinstance(value, BackendProbeResult):
        return value
    if isinstance(value, Mapping):
        return BackendProbeResult(
            token_ids=[int(x) for x in value.get("token_ids", value.get("tokens", []))],
            text=str(value.get("text", "")),
            prompt_token_ids=[
                int(x) for x in value.get("prompt_token_ids", value.get("prompt_ids", []))
            ],
            logits=list(value.get("logits", [])),
            states=list(value.get("states", [])),
            prefill_ms=float(value.get("prefill_ms", 0.0)),
            decode_ms=float(value.get("decode_ms", 0.0)),
            metrics=dict(value.get("metrics", {}) or {}),
        )
    if isinstance(value, (list, tuple)):
        return BackendProbeResult(token_ids=[int(x) for x in value])
    raise TypeError("backend probe runner must return BackendProbeResult, mapping, or token IDs")


def run_backend_conformance(
    reference_runner: Callable[[str, int], Any],
    candidate_runner: Callable[[str, int], Any],
    prompts: Iterable[str],
    *,
    max_tokens: int,
    reference_backend: str = "reference",
    candidate_backend: str = "candidate",
    thresholds: ParityThresholds | None = None,
    require_guardrails: bool = False,
) -> list[ParityTrace]:
    """Run the same prompt corpus through two backend adapters.

    Runners may return only token IDs for a lightweight smoke test, or a
    :class:`BackendProbeResult` with per-step logits/states for full guardrail
    diagnostics.  The harness deliberately keeps sampling outside this helper:
    callers should pass fixed greedy runners for the hard acceptance gate.
    """
    limits = thresholds or ParityThresholds()
    traces: list[ParityTrace] = []
    for prompt in prompts:
        ref = _coerce_probe_result(reference_runner(prompt, int(max_tokens)))
        cand = _coerce_probe_result(candidate_runner(prompt, int(max_tokens)))
        trace = ParityTrace(
            backend=candidate_backend,
            reference_backend=reference_backend,
            prompt=str(prompt),
            prompt_token_ids=list(cand.prompt_token_ids or ref.prompt_token_ids),
            generated_token_ids=list(cand.token_ids),
            decoded_text=cand.text,
            reference_decoded_text=ref.text or None,
            decoded_text_match=(cand.text == ref.text) if ref.text or cand.text else None,
            thresholds=limits,
            require_guardrails=bool(require_guardrails),
            prefill_ms=cand.prefill_ms,
            decode_ms=cand.decode_ms,
            rss_bytes=int(cand.metrics.get("process_rss_bytes", 0) or 0),
            rss_peak_bytes=int(cand.metrics.get("process_rss_peak_bytes", 0) or 0),
            streamed_bytes=int(cand.metrics.get("streamed_bytes", 0) or 0),
            cache_hits=int(cand.metrics.get("cache_hits", 0) or 0),
        )
        for index in range(max(len(ref.token_ids), len(cand.token_ids))):
            ref_id = ref.token_ids[index] if index < len(ref.token_ids) else None
            cand_id = cand.token_ids[index] if index < len(cand.token_ids) else None
            step = ParityStep(
                token_index=index,
                reference_token_id=ref_id,
                candidate_token_id=cand_id,
                greedy_match=ref_id == cand_id,
            )
            if index < len(ref.logits) and index < len(cand.logits):
                step.top10_overlap = top_k_overlap(ref.logits[index], cand.logits[index])
                step.kl = kl_candidate_to_reference(ref.logits[index], cand.logits[index])
            if index < len(ref.states) and index < len(cand.states):
                step.state_relative_error = relative_state_error(ref.states[index], cand.states[index])
            trace.add_step(step)
        if require_guardrails:
            if len(ref.logits) < len(ref.token_ids) or len(cand.logits) < len(cand.token_ids):
                trace.diagnostics_complete = False
                trace.notes.append(
                    "logit probes are incomplete for one or more generated tokens"
                )
            if len(ref.states) < len(ref.token_ids) or len(cand.states) < len(cand.token_ids):
                trace.diagnostics_complete = False
                trace.notes.append(
                    "recurrent-state probes are incomplete for one or more generated tokens"
                )
        traces.append(trace)
    return traces


def probe_loaded_engine(
    engine: Any,
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> BackendProbeResult:
    """Probe an already-loaded engine without taking ownership of it.

    The callback observes the post-token recurrent state and next-token logits
    exposed by the backend.  Those observations are intentionally backend
    agnostic: RWKV-7 lists, rwkv.cpp external arrays, and sequence-model
    states are flattened only for relative-error diagnostics.
    """
    count = int(engine.config.max_tokens if max_tokens is None else max_tokens)
    old_max = engine.config.max_tokens
    engine.config.max_tokens = count
    token_ids: list[int] = []
    logits: list[Any] = []
    states: list[Any] = []

    def observe_state(state: Any) -> Any:
        backend = getattr(engine, "backend", None)
        model = getattr(backend, "_model", None)
        if getattr(state, "rwkv7_state", None) is not None:
            return StateObservation(state, layout="chatrwkv_rwkv7")
        if (
            getattr(state, "external_state", None) is not None
            and int(getattr(model, "arch_version_major", 0) or 0) == 7
        ):
            return StateObservation(
                state,
                layout="rwkv_cpp_rwkv7",
                n_layer=int(getattr(model, "n_layer", 0) or getattr(backend, "_n_layer", 0) or 0),
                n_embed=int(
                    getattr(model, "n_embed", 0)
                    or getattr(getattr(model, "header", None), "n_embed", 0)
                    or 0
                ),
                head_count=int(getattr(model, "head_count", 0) or 0),
                head_size=int(getattr(model, "head_size", 0) or 0),
            )
        return state

    def on_token(token_id: int) -> None:
        token_ids.append(int(token_id))
        state = engine.backend.get_recurrent_state()
        if state is not None:
            states.append(observe_state(state))
        current_logits = engine.backend.probe_logits(state)
        if current_logits is not None:
            logits.append(_array(current_logits).copy())

    try:
        generated = engine.generate_tokens(prompt, token_callback=on_token)
        if not token_ids:
            token_ids = [int(value) for value in generated]
        prompt_token_ids: list[int] = []
        for name in ("_encode", "_encode_fn"):
            encoder = getattr(engine.backend, name, None)
            if callable(encoder):
                try:
                    prompt_token_ids = [int(value) for value in encoder(prompt)]
                    break
                except (TypeError, ValueError, RuntimeError):
                    continue
        return BackendProbeResult(
            token_ids=token_ids,
            text=engine.backend.decode_text(token_ids),
            prompt_token_ids=prompt_token_ids,
            logits=logits,
            states=states,
            prefill_ms=float(getattr(engine.metrics, "prefill_wall_s", 0.0) or 0.0) * 1000.0,
            decode_ms=float(getattr(engine.metrics, "decode_wall_s", 0.0) or 0.0) * 1000.0,
            metrics=engine.metrics.to_dict(),
        )
    finally:
        engine.config.max_tokens = old_max


def probe_engine_backend(
    config: Any,
    prompt: str,
    *,
    max_tokens: int | None = None,
    backend_label: str | None = None,
) -> BackendProbeResult:
    """Load one engine configuration and capture per-token diagnostics."""
    del backend_label
    from rwkv_ssd.runtime.engine import InferenceEngine

    engine = InferenceEngine(config)
    try:
        engine.load()
        return probe_loaded_engine(engine, prompt, max_tokens=max_tokens)
    finally:
        engine.close()


__all__ = [
    "BackendProbeResult",
    "StateObservation",
    "ParityStep",
    "ParityThresholds",
    "ParityTrace",
    "kl_candidate_to_reference",
    "relative_state_error",
    "run_backend_conformance",
    "probe_engine_backend",
    "probe_loaded_engine",
    "top_k_overlap",
]
