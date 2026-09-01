"""rwkv.cpp CPU backend for resident and provider-backed RWKV inference."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.runtime.errors import BackendNotAvailableError
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.sampling import sample_numpy
from rwkv_ssd.runtime.state_cache import RecurrentState

logger = logging.getLogger(__name__)


def _bridge_slot_config() -> tuple[int, int]:
    """Resolve opt-in native weight-slot sizing without changing legacy RAM use."""
    values: list[int] = []
    for name in ("RWKV_GGML_SLOT_COUNT", "RWKV_GGML_SLOT_BYTES"):
        raw = os.environ.get(name, "0").strip()
        try:
            value = max(0, int(raw))
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, raw)
            value = 0
        values.append(value)
    count, size = values
    if bool(count) != bool(size):
        logger.warning(
            "GGML slots require both RWKV_GGML_SLOT_COUNT and "
            "RWKV_GGML_SLOT_BYTES; disabling slot uploads"
        )
        return 0, 0
    return count, size


def _sync_provider_every_token() -> bool:
    """Return whether to force the diagnostic per-token upload path.

    rwkv.cpp owns a complete GGML graph, including its weight tensors.  Once
    the provider pack has been uploaded before prefill, the native graph is
    already weight-stationary for the whole generation.  Re-uploading every
    layer before every token only measures conversion/copy overhead and does
    not reduce native memory.  Keep the old behavior available for explicit
    bridge/SSD diagnostics, but make the resident native graph the fast path.
    """
    raw = os.environ.get("RWKVCPP_SYNC_EVERY_TOKEN", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _native_layer_packed_head_enabled() -> bool:
    """Return whether native layer streaming should trade head speed for RAM.

    The packed vocabulary head avoids materializing a potentially very large
    FP32 host matrix, but the measured CPU GEMV is slower than the dense BLAS
    projection on the current host.  Keep the default on the speed path and
    expose the memory-saving choice explicitly for bounded-RAM deployments.
    ``auto`` is intentionally equivalent to disabled until a host-specific
    profile proves otherwise.
    """
    raw = os.environ.get("RWKVCPP_NATIVE_LAYER_PACKED_HEAD", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"", "0", "false", "no", "off", "auto"}:
        return False
    logger.warning(
        "Ignoring invalid RWKVCPP_NATIVE_LAYER_PACKED_HEAD=%r; using dense head",
        raw,
    )
    return False


def _resolve_layer_cache_bytes(default: int = 0) -> int:
    """Resolve the bounded native decoded-layer cache cap."""
    raw = os.environ.get("RWKVCPP_LAYER_CACHE_BYTES")
    if raw is None or raw.strip().lower() in {"", "auto"}:
        return max(0, int(default))
    try:
        return max(0, int(raw, 0))
    except ValueError:
        logger.warning(
            "Ignoring invalid RWKVCPP_LAYER_CACHE_BYTES=%r; using %d",
            raw,
            max(0, int(default)),
        )
        return max(0, int(default))


def _record_token_metrics(
    metrics: MetricsCollector | None,
    started: float,
    *,
    prefill_s: float = 0.0,
) -> None:
    """Record decode latency/TTFT at the backend token boundary."""
    if metrics is None:
        return
    token_ms = (time.perf_counter() - started) * 1000.0
    metrics.token_latencies_ms.append(token_ms)
    if len(metrics.token_latencies_ms) == 1 and metrics.ttft_s <= 0.0:
        metrics.ttft_s = max(0.0, float(prefill_s)) + token_ms / 1000.0


def _create_weight_bridge(model: Any):
    from rwkv_ssd.runtime.ggml_weight_bridge import GgmlWeightBridge

    slot_count, slot_bytes = _bridge_slot_config()
    return GgmlWeightBridge(
        model,
        slot_count=slot_count,
        slot_bytes=slot_bytes,
    )


def find_rwkvcpp_root() -> Path | None:
    env = os.environ.get("RWKVCPP_ROOT")
    if env:
        p = Path(env)
        if (p / "CMakeLists.txt").is_file():
            return p
    here = Path(__file__).resolve().parents[2]
    for candidate in (here / "backends" / "rwkvcpp_ref", here.parent / "rwkv.cpp"):
        if (candidate / "CMakeLists.txt").is_file():
            return candidate
    return None


def find_rwkvcpp_dll(root: Path | None = None) -> Path | None:
    root = root or find_rwkvcpp_root()
    if root is None:
        return None
    env = os.environ.get("RWKVCPP_DLL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    name = "rwkv.dll" if sys.platform == "win32" else "librwkv.so"
    names = [name]
    if sys.platform == "win32":
        names.extend(("librwkv.dll", "rwkv.dll"))
    for dll_name in names:
        for rel in (
            Path("bin") / "Release" / dll_name,
            Path("bin") / dll_name,
            Path("build") / "bin" / "Release" / dll_name,
            Path("build") / "bin" / dll_name,
            Path("build") / dll_name,
            Path(dll_name),
        ):
            candidate = root / rel
            if candidate.is_file():
                return candidate
    return None


def is_rwkvcpp_available() -> bool:
    return find_rwkvcpp_dll() is not None


def _ensure_rwkv_cpp_import(root: Path) -> None:
    py_dir = root / "python"
    rwkv_pkg = py_dir / "rwkv_cpp"
    for path in (str(py_dir), str(rwkv_pkg)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_thread_count(n_embd: int | None = None) -> int:
    """Resolve the native ggml thread count.

    ``RWKV_CPU_THREADS=N`` remains an explicit override.  When the engine has
    already read model metadata, an unset/``auto`` value can use the model
    width to avoid the one-thread default being accidentally applied to a
    multi-billion-parameter model.  Keeping the no-argument behavior at one
    thread preserves the conservative standalone/backend API behavior for
    callers that do not know the model shape yet.
    """
    raw = os.environ.get("RWKV_CPU_THREADS", "").strip()
    raw_lower = raw.lower()
    if raw_lower == "off":
        # Keep the documented off behavior conservative.  The native API
        # accepts a positive thread count, so one is the safe representation
        # of "do not autotune" here.
        return 1
    if raw_lower not in {"", "auto"}:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            # Preserve the old safe fallback for invalid settings instead of
            # silently making an unexpected large native thread pool.
            return 1
        return 1

    if n_embd is None:
        # Standalone callers do not have enough information to choose a tier.
        return 1
    try:
        width = int(n_embd)
    except (TypeError, ValueError):
        return 1
    if width <= 0:
        return 1

    cpu = os.cpu_count() or 1
    if width >= 2048:
        # rwkv.cpp's single-token quantized GEMVs are memory-bandwidth and
        # cache-pressure limited on typical AVX2 laptops.  More threads can
        # make the model slower; on the 8-thread host used for the release
        # gate, Q4_K and FP16 both peaked at two-to-four threads.  Scale only
        # up to four on wider hosts and keep a minimum of two for large
        # models.
        return max(1, min(4, max(2, cpu // 4)))
    if width >= 1280:
        return max(1, min(4, max(2, cpu // 2)))
    return 1


def _resolve_ggml_path(model_path: str) -> Path:
    explicit = os.environ.get("RWKVCPP_GGML_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"RWKVCPP_GGML_PATH not found: {explicit}")

    path = Path(model_path)
    if path.suffix.lower() == ".bin" and path.is_file():
        return path

    if path.suffix.lower() == ".pth":
        stem = path.with_suffix("")
        for candidate in (
            stem.with_name(stem.name + "-FP16.bin"),
            stem.with_suffix(".bin"),
            path.parent / f"{path.stem}-FP16.bin",
        ):
            if candidate.is_file():
                return candidate
        if os.environ.get("RWKVCPP_AUTO_CONVERT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            out = stem.with_name(stem.name + "-FP16.bin")
            _convert_pytorch_to_ggml(path, out)
            return out
        raise FileNotFoundError(
            f"No ggml .bin for {model_path}. Convert with "
            f"backends/rwkvcpp_ref/python/convert_pytorch_to_ggml.py "
            f"or set RWKVCPP_GGML_PATH."
        )

    if path.is_file():
        return path
    raise FileNotFoundError(f"Model path not found: {model_path}")


def _convert_pytorch_to_ggml(src: Path, dest: Path) -> None:
    root = find_rwkvcpp_root()
    if root is None:
        raise RuntimeError("rwkv.cpp root not found for conversion")
    script = root / "python" / "convert_pytorch_to_ggml.py"
    if not script.is_file():
        raise FileNotFoundError(f"convert script missing: {script}")
    import subprocess

    logger.info("Converting %s -> %s (FP16, may take several minutes) ...", src, dest)
    subprocess.run(
        [sys.executable, str(script), str(src), str(dest), "FP16"],
        check=True,
    )


class RWKVCppBackend(RecurrentBackend):
    """
    Thin wrapper around rwkv.cpp for FP16 / Q4 / Q5 comparison runs.

    Use when:
      - validating quantized artifacts,
      - CPU-only deployment,
      - comparing perplexity/latency vs the PyTorch path.

    Do NOT use as the primary SSD streaming GPU engine — there is no overlap
    with mmap → pinned H2D → fused CUDA in this stack.
    """

    # Pack streaming is provider-backed; ggml remains the native compute
    # engine while the current model graph is refreshed layer-by-layer.
    supports_streaming: bool = True

    def __init__(self) -> None:
        self._root: Path | None = None
        self._dll: Path | None = None
        self._model_path: str | None = None
        self._ggml_path: Path | None = None
        self._model: Any = None
        self._library: Any = None
        self._decode_fn: Any = None
        self._encode_fn: Any = None
        self._n_layer = 0
        self._state: np.ndarray | None = None
        self._last_logits: np.ndarray | None = None
        self._last_token_id = 0
        self._bridge: Any = None
        self._n_embd_hint: int | None = None
        self._native_u8_hint: bool = False
        self._native_u8_packed_only_hint: bool = False
        self._layer_streaming_hint: bool = False
        self._layer_cache_bytes_hint: int = 0
        self._native_layer_cache_bytes: int = 0
        self._native_layer_cache_seen: dict[str, int] = {
            "limit_bytes": 0,
            "used_bytes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        self._layer_all_ids: list[int] = []
        self._layer_states: list[np.ndarray] | None = None
        # Reusable contiguous buffers for the native cached layer-step ABI.
        # Avoid concatenating and re-slicing the complete recurrent state on
        # every autoregressive token.
        self._layer_state_flat: np.ndarray | None = None
        self._layer_state_scratch: np.ndarray | None = None
        self._layer_state_scratch_by_layer: list[np.ndarray] | None = None
        self._layer_v_first: np.ndarray | None = None
        self._layer_activation_scratch: tuple[np.ndarray, np.ndarray] | None = None
        self._layer_v_first_scratch: tuple[np.ndarray, np.ndarray] | None = None
        # Reusable buffers for the native sequence-prefill ABI.  A prefill
        # sweep alternates activation buffers between layers and reuses one
        # state/v_first output buffer after each layer has copied its result
        # into the persistent recurrent state.  This avoids allocating large
        # arrays once per layer on every prompt.
        self._layer_sequence_activation_scratch: tuple[np.ndarray, np.ndarray] | None = None
        self._layer_sequence_state_scratch: np.ndarray | None = None
        self._layer_sequence_v_first_scratch: np.ndarray | None = None
        self._layer_global_weights: dict[str, Any] = {}
        self._layer_packed_head: tuple[Any, int, int] | None = None
        self._layer_dense_head: np.ndarray | None = None

    @staticmethod
    def is_available() -> bool:
        return is_rwkvcpp_available()

    def set_model_width_hint(self, n_embd: int | None) -> None:
        """Supply manifest width before constructing the native GGML model.

        The GGML constructor needs its thread count up front.  The engine has
        already validated the pack manifest at that point, so passing this
        hint lets the native backend choose a useful automatic tier without
        changing the standalone ``load`` API.
        """
        try:
            width = int(n_embd) if n_embd is not None else 0
        except (TypeError, ValueError):
            width = 0
        self._n_embd_hint = width if width > 0 else None

    def set_native_u8_hint(self, enabled: bool) -> None:
        """Enable the CPU-native grouped-U8 graph for a packed load.

        The hint is supplied after manifest inspection and before the GGML
        context is created.  An explicit ``RWKVCPP_NATIVE_U8`` environment
        value still overrides it for diagnostics and rollback.
        """
        self._native_u8_hint = bool(enabled)

    def set_native_u8_packed_only_hint(self, enabled: bool) -> None:
        """Allow packed-only residency when the manifest covers every matrix.

        A mixed pack still uses the native grouped-U8 kernels, but keeps the
        dense fallback for any non-SG8 matrix.  This separate hint prevents a
        partial/mixed artifact from being loaded into shape-only GGML tensors.
        """
        self._native_u8_packed_only_hint = bool(enabled)

    def set_layer_streaming_hint(self, enabled: bool) -> None:
        """Select the bounded native one-block ABI for non-resident tiers."""
        self._layer_streaming_hint = bool(enabled)

    def set_layer_cache_bytes_hint(self, bytes_limit: int) -> None:
        """Set the profile default for the optional bounded native cache."""
        try:
            self._layer_cache_bytes_hint = max(0, int(bytes_limit))
        except (TypeError, ValueError):
            self._layer_cache_bytes_hint = 0

    def _native_layer_streaming_active(self) -> bool:
        """Return a strict capability result, including for test doubles.

        ``MagicMock`` makes every missing attribute truthy.  Treat only the
        concrete boolean capability exposed by ``RWKVModel`` as enabled so a
        compatibility/mock model continues through the legacy upload path.
        """
        if self._model is None:
            return False
        value = getattr(self._model, "supports_layer_streaming", False)
        return type(value) is bool and value

    def load(self, model_path: str, strategy: str, device: str) -> None:
        del strategy, device
        self._root = find_rwkvcpp_root()
        if self._root is None:
            raise RuntimeError(
                "rwkv.cpp not found. Clone https://github.com/RWKV/rwkv.cpp and set "
                "RWKVCPP_ROOT, or place it at backends/rwkvcpp_ref/."
            )
        self._dll = find_rwkvcpp_dll(self._root)
        if self._dll is None:
            raise BackendNotAvailableError(
                "rwkv.cpp shared library not built. Run: "
                "cmake -S backends/rwkvcpp_ref -B backends/rwkvcpp_ref/build && "
                "cmake --build backends/rwkvcpp_ref/build --config Release"
            )
        _ensure_rwkv_cpp_import(self._root)
        from rwkv_cpp import rwkv_cpp_model, rwkv_cpp_shared_library
        from rwkv_cpp import rwkv_world_tokenizer

        self._library = rwkv_cpp_shared_library.RWKVSharedLibrary(str(self._dll))
        self._model_path = model_path
        self._ggml_path = _resolve_ggml_path(model_path)
        gpu_layers = int(os.environ.get("RWKVCPP_GPU_LAYERS", "0"))
        thread_count = _resolve_thread_count(self._n_embd_hint)
        native_u8_raw = os.environ.get("RWKVCPP_NATIVE_U8", "auto").strip().lower()
        if native_u8_raw in {"1", "true", "yes", "on"}:
            native_u8 = True
        elif native_u8_raw in {"0", "false", "no", "off"}:
            native_u8 = False
        else:
            native_u8 = self._native_u8_hint
        packed_only_raw = os.environ.get(
            "RWKVCPP_NATIVE_U8_PACKED_ONLY", "auto"
        ).strip().lower()
        if packed_only_raw in {"1", "true", "yes", "on"}:
            native_u8_packed_only = True
        elif packed_only_raw in {"0", "false", "no", "off"}:
            native_u8_packed_only = False
        else:
            native_u8_packed_only = self._native_u8_packed_only_hint
        self._model = rwkv_cpp_model.RWKVModel(
            self._library,
            str(self._ggml_path),
            thread_count=thread_count,
            gpu_layer_count=gpu_layers,
            native_u8=native_u8 and gpu_layers == 0,
            native_u8_packed_only=(
                native_u8_packed_only and native_u8 and gpu_layers == 0
            ),
            layer_streaming=self._layer_streaming_hint and gpu_layers == 0,
        )
        if self._layer_streaming_hint and not bool(
            getattr(self._model, "supports_layer_streaming", False)
        ):
            self._model.free()
            self._model = None
            raise BackendNotAvailableError(
                "rwkv.cpp was built without the required native layer-streaming "
                "ABI for the selected F1-F4 tier"
            )
        self._n_layer = int(self._model.n_layer)
        self._layer_all_ids = list(range(self._n_layer))
        self._decode_fn, self._encode_fn = rwkv_world_tokenizer.get_world_tokenizer_v20230424()
        self._state = None
        self._last_logits = None
        self._last_token_id = 0
        self._layer_states = None
        self._layer_state_flat = None
        self._layer_state_scratch = None
        self._layer_state_scratch_by_layer = None
        self._layer_v_first = None
        self._layer_activation_scratch = None
        self._layer_v_first_scratch = None
        self._layer_sequence_activation_scratch = None
        self._layer_sequence_state_scratch = None
        self._layer_sequence_v_first_scratch = None
        self._layer_global_weights = {}
        self._layer_packed_head = None
        self._layer_dense_head = None
        self._native_layer_cache_bytes = 0
        if self._native_layer_streaming_active():
            set_cache = getattr(self._model, "set_layer_cache_bytes", None)
            supports_cache = bool(
                getattr(self._model, "supports_layer_cache", False)
            )
            cache_bytes = _resolve_layer_cache_bytes(self._layer_cache_bytes_hint)
            if supports_cache and callable(set_cache):
                set_cache(cache_bytes)
                self._native_layer_cache_bytes = cache_bytes
            elif cache_bytes:
                logger.warning(
                    "rwkv.cpp lacks the optional bounded native layer-cache ABI; "
                    "continuing without RWKVCPP_LAYER_CACHE_BYTES=%d",
                    cache_bytes,
                )
        self._native_layer_cache_seen = self.native_layer_cache_stats()
        # Keep resident-only compatibility with older pre-bridge DLLs.  The
        # upload ABI is required only when a streaming tier is actually used.
        self._bridge = None
        logger.info(
            "rwkv.cpp loaded %s (%d layers, %d threads, native_u8=%s, packed_only=%s, layer_streaming=%s)",
            self._ggml_path.name,
            self._n_layer,
            thread_count,
            bool(getattr(self._model, "supports_native_u8", False)),
            bool(getattr(self._model, "native_u8_packed_only", False)),
            bool(getattr(self._model, "supports_layer_streaming", False)),
        )

    def native_layer_cache_stats(self) -> dict[str, int]:
        """Return native decoded-layer cache counters when available."""
        empty = {
            "limit_bytes": 0,
            "used_bytes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        if self._model is None:
            return empty
        getter = getattr(self._model, "layer_cache_stats", None)
        if not callable(getter):
            return empty
        try:
            raw = dict(getter())
            return {
                key: max(0, int(raw.get(key, 0) or 0))
                for key in empty
            }
        except (TypeError, ValueError, RuntimeError):
            return empty

    @property
    def num_layers(self) -> int:
        return self._n_layer

    def prefill(self, prompt: str) -> tuple[list[int], Any]:
        if self._model is None:
            raise RuntimeError("backend not loaded")
        ids = self._encode_fn(prompt)
        if not ids:
            ids = [0]
        self._last_logits, self._state = self._model.eval_sequence_in_chunks(
            ids, None, None, None, use_numpy=True
        )
        self._last_token_id = ids[-1]
        return ids, self._state

    def step(self, token_id: int, state: Any) -> tuple[int, Any]:
        if self._model is None:
            raise RuntimeError("backend not loaded")
        logits, state = self._model.eval(int(token_id), state, None, None, use_numpy=True)
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        next_id = int(np.argmax(logits))
        self._state = state
        self._last_token_id = next_id
        return next_id, state

    def decode_text(self, token_ids: list[int]) -> str:
        if not token_ids or self._decode_fn is None:
            return ""
        # A corrupted/quantized native graph can produce an out-of-vocabulary
        # argmax.  The tokenizer raises KeyError for that case, which used to
        # obscure the backend/step and made streaming failures hard to debug.
        # Never silently clamp or drop a token: that would turn a numerical
        # failure into plausible-looking but incorrect text.
        try:
            return self._decode_fn([int(token_id) for token_id in token_ids])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            bad = next(
                (
                    int(token_id)
                    for token_id in token_ids
                    if int(token_id) < 0
                ),
                None,
            )
            if bad is None and isinstance(exc, KeyError):
                key = exc.args[0] if exc.args else "unknown"
                bad = int(key) if isinstance(key, (int, np.integer)) else key
            raise RuntimeError(
                "rwkv.cpp produced a token that the RWKV world tokenizer "
                f"cannot decode (token_id={bad!r}, generated={len(token_ids)}); "
                "check GGML/pack/checkpoint compatibility and codec quality"
            ) from exc

    def generate_simple(
        self,
        prompt: str,
        max_tokens: int,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> str:
        if self._model is None:
            raise RuntimeError("backend not loaded")
        ids = self._encode_fn(prompt)
        if not ids:
            ids = [0]
        logits, state = self._model.eval_sequence_in_chunks(
            ids, None, None, None, use_numpy=True
        )
        next_state = np.empty_like(state)
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        out_ids: list[int] = []
        for _ in range(max(0, int(max_tokens))):
            token = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out_ids.append(token)
            logits, next_state = self._model.eval(
                token, state, next_state, logits, use_numpy=True
            )
            state, next_state = next_state, state
        self._state = state
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        self._last_token_id = out_ids[-1] if out_ids else ids[-1]
        return self.decode_text(out_ids)

    def generate_greedy_native(
        self,
        prompt: str,
        max_tokens: int,
        *,
        metrics: MetricsCollector | None = None,
        power_percent: int = 100,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        del power_percent
        if self._model is None:
            raise RuntimeError("backend not loaded")
        import time

        ids = self._encode_fn(prompt)
        if not ids:
            ids = [0]
        t0 = time.perf_counter()
        logits, state = self._model.eval_sequence_in_chunks(
            ids, None, None, None, use_numpy=True
        )
        next_state = np.empty_like(state)
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        prefill_s = time.perf_counter() - t0
        out_ids: list[int] = []
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        decode_t0 = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            token_started = time.perf_counter()
            token = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out_ids.append(token)
            # Publish the state/logits that produced this token before the
            # streaming callback observes it.  The state is the context after
            # the prompt/previous token and the logits are the exact greedy
            # probe used for this choice; the next eval advances both.
            if control is not None:
                # Cancellation/deadline-aware callers may resume at the
                # pre-token boundary.  Uncontrolled decode has no observer
                # for this snapshot, so avoid copying the full recurrent
                # state on every token.
                self._state = np.asarray(state, dtype=np.float32).copy()
                self._last_logits = np.asarray(logits, dtype=np.float32).copy()
                control.emit(token)
            logits, next_state = self._model.eval(
                token, state, next_state, logits, use_numpy=True
            )
            state, next_state = next_state, state
            if control is not None:
                self._last_logits = np.asarray(logits, dtype=np.float32).copy()
            _record_token_metrics(metrics, token_started, prefill_s=prefill_s)
        decode_s = time.perf_counter() - decode_t0
        self._state = np.asarray(state, dtype=np.float32).copy()
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        self._last_token_id = out_ids[-1] if out_ids else ids[-1]
        if metrics is not None:
            metrics.prefill_wall_s = prefill_s
            metrics.decode_wall_s = decode_s
            metrics.tokens_generated = len(out_ids)
        return out_ids

    _LAYER_GLOBAL_NAMES = frozenset(
        {
            "emb.weight",
            "blocks.0.ln0.weight",
            "blocks.0.ln0.bias",
            "ln_out.weight",
            "ln_out.bias",
            "head.weight",
        }
    )

    def _load_layer_global_weights(
        self,
        provider: Any,
        by_layer: dict[int, list[Any]],
    ) -> None:
        """Materialize only the global tensors around the native block ABI."""
        if self._layer_global_weights:
            return
        required_names = self._LAYER_GLOBAL_NAMES
        packed_head = None
        if _native_layer_packed_head_enabled():
            prime = getattr(provider, "_prime_fused_global_blobs", None)
            if callable(prime):
                prime()
            get_blob = getattr(provider, "get_fused_lut_blob", None)
            candidate = get_blob("head.weight") if callable(get_blob) else None
            if candidate is not None:
                try:
                    blob, out_features, in_features = candidate
                    if len(blob) >= 8 and int(out_features) > 0 and int(in_features) > 0:
                        packed_head = (blob, int(out_features), int(in_features))
                except (TypeError, ValueError):
                    packed_head = None
            if packed_head is not None:
                required_names = frozenset(
                    name for name in self._LAYER_GLOBAL_NAMES if name != "head.weight"
                )
                logger.info(
                    "rwkv.cpp native layer streaming: retaining packed vocabulary head "
                    "(%dx%d); dense head materialization disabled",
                    packed_head[1],
                    packed_head[2],
                )
        groups: dict[int, list[Any]] = {}
        seen: set[str] = set()
        for entries in by_layer.values():
            for entry in entries:
                if entry.name in required_names and entry.name not in seen:
                    groups.setdefault(int(entry.layer_id), []).append(entry)
                    seen.add(entry.name)
        loaded: dict[str, Any] = {}
        for entries in groups.values():
            loaded.update(provider.load_layer_tensors_materialized(entries))
        missing = sorted(required_names.difference(loaded))
        if missing:
            raise RuntimeError(
                "rwkv.cpp native layer streaming requires provider-visible global "
                f"tensors; missing {', '.join(missing)}"
            )
        import torch

        for name in required_names:
            tensor = loaded[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"provider returned non-tensor global {name}")
            self._layer_global_weights[name] = tensor.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        if packed_head is None:
            head = self._layer_global_weights.get("head.weight")
            if isinstance(head, torch.Tensor):
                self._layer_dense_head = head.numpy()
            elif isinstance(head, np.ndarray):
                self._layer_dense_head = np.asarray(head)
        self._layer_packed_head = packed_head
        release_globals = getattr(provider, "release_native_global_cache", None)
        if callable(release_globals):
            # The native layer backend now owns the global host views.  Do not
            # leave the provider's preload aliases in its bounded cache.
            release_globals(required_names)

    def _layer_reset_state(self, state: np.ndarray | None) -> None:
        if self._model is None or not self._native_layer_streaming_active():
            raise RuntimeError("native layer-streaming model is not active")
        state_len = int(self._model.layer_state_len)
        expected = int(self._n_layer) * state_len
        if state is None or int(np.asarray(state).size) != expected:
            flat = np.zeros(expected, dtype=np.float32)
        else:
            flat = np.asarray(state, dtype=np.float32).reshape(expected).copy()
        self._layer_state_flat = flat
        self._layer_state_scratch = np.empty_like(flat)
        self._layer_states = [
            flat[i * state_len : (i + 1) * state_len]
            for i in range(self._n_layer)
        ]
        self._layer_state_scratch_by_layer = [
            np.empty_like(item) for item in self._layer_states
        ]
        self._layer_v_first = None
        self._layer_activation_scratch = None
        self._layer_v_first_scratch = None

    def _layer_flat_state(self) -> np.ndarray:
        if self._layer_state_flat is not None:
            return self._layer_state_flat
        if self._layer_states is None:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._layer_states).astype(np.float32, copy=False)

    def _store_layer_state(self, index: int, value: np.ndarray) -> None:
        """Update one recurrent layer without breaking the flat-state view."""
        if self._layer_states is None:
            raise RuntimeError("native layer state is not initialized")
        target = self._layer_states[index]
        if target.shape == value.shape and target.dtype == value.dtype:
            np.copyto(target, value)
            return
        # Defensive fallback for older/mock adapters with a different state
        # representation.  It is not used by the native ABI path.
        self._layer_states[index] = np.asarray(value, dtype=np.float32).copy()
        self._layer_state_flat = None
        self._layer_state_scratch = None
        self._layer_state_scratch_by_layer = None

    def _layer_norm_global(self, value: np.ndarray, weight_name: str, bias_name: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.asarray(value, dtype=np.float32))
        weight = self._layer_global_weights[weight_name]
        bias = self._layer_global_weights[bias_name]
        return F.layer_norm(x, (int(x.numel()),), weight=weight, bias=bias).numpy().copy()

    def _layer_logits_global(self, value: np.ndarray) -> np.ndarray:
        # The layer-local ABI already returns host FP32 activations.  Keeping
        # the final projection in Torch adds a per-token dispatcher and GEMM
        # wrapper cost (and was materially slower than the NumPy/BLAS path on
        # the CPU-only backend).  Convert the provider-owned head once at the
        # boundary and leave the hot projection in the same host array domain
        # as the native ABI.
        hidden = np.asarray(value, dtype=np.float32)
        if self._layer_packed_head is not None:
            import torch

            from rwkv_ssd.runtime.lut_gemm_fused import lut2_gemv

            blob, out_features, in_features = self._layer_packed_head
            if int(hidden.size) != int(in_features):
                raise RuntimeError(
                    f"packed head input width {in_features} is incompatible with "
                    f"activation width {int(hidden.size)}"
                )
            logits = lut2_gemv(
                blob,
                torch.from_numpy(np.ascontiguousarray(hidden)),
                out_features=out_features,
                in_features=in_features,
                # Greedy selection is particularly sensitive to head error;
                # use the normal FP32 activation path unless the operator has
                # separately opted into head INT8 activation quantization.
                activation_int8=False,
            )
            return logits.detach().cpu().numpy().astype(np.float32, copy=False)
        head = self._layer_dense_head
        if head is None:
            head = self._layer_global_weights["head.weight"]
            if not isinstance(head, np.ndarray):
                head = head.detach().cpu().numpy()
            self._layer_dense_head = head
        if head.ndim != 2:
            raise RuntimeError("head.weight must be a rank-2 tensor")
        hidden_width = int(hidden.shape[-1]) if hidden.ndim == 2 else int(hidden.size)
        if int(head.shape[1]) == hidden_width:
            logits = np.matmul(hidden, head.T)
        elif int(head.shape[0]) == hidden_width:
            logits = np.matmul(hidden, head)
        else:
            raise RuntimeError(
                f"head.weight shape {tuple(head.shape)} is incompatible with "
                f"activation width {int(hidden.size)}"
            )
        return np.asarray(logits, dtype=np.float32)

    def _record_native_layer_cache_delta(
        self,
        metrics: MetricsCollector | None,
        before: dict[str, int],
        after: dict[str, int],
    ) -> None:
        if metrics is None:
            return
        metrics.native_layer_cache_bytes = int(after.get("used_bytes", 0) or 0)
        for key, field in (
            ("hits", "native_layer_cache_hits"),
            ("misses", "native_layer_cache_misses"),
            ("evictions", "native_layer_cache_evictions"),
        ):
            setattr(
                metrics,
                field,
                int(getattr(metrics, field, 0))
                + max(
                    0,
                    int(after.get(key, 0) or 0)
                    - int(before.get(key, 0) or 0),
                ),
            )

    def _layer_advance_token(
        self,
        token_id: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
        *,
        control=None,
    ) -> np.ndarray:
        if self._model is None or self._layer_states is None:
            raise RuntimeError("native layer state is not initialized")
        if control is not None:
            control.check()
        import torch

        emb = self._layer_global_weights["emb.weight"]
        if token_id < 0 or token_id >= int(emb.shape[0]):
            raise ValueError(f"token id {token_id} is outside the provider vocabulary")
        activation = emb[int(token_id)].detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        activation = self._layer_norm_global(
            activation,
            "blocks.0.ln0.weight",
            "blocks.0.ln0.bias",
        )
        cached_step = bool(
            len(layer_ids) == self._n_layer
            and layer_ids == self._layer_all_ids
            and getattr(self._model, "supports_layer_cached_step", False)
            and callable(getattr(self._model, "layer_cache_ready", None))
            and self._model.layer_cache_ready()
        )
        if cached_step:
            state_in = self._layer_state_flat
            if state_in is None:
                state_in = self._layer_flat_state()
            state_out = self._layer_state_scratch
            if state_out is None or state_out.shape != state_in.shape:
                state_out = np.empty_like(state_in)
            activation_scratch = self._layer_activation_scratch
            if (
                activation_scratch is None
                or activation_scratch[0].shape != activation.shape
            ):
                activation_scratch = (np.empty_like(activation), np.empty_like(activation))
                self._layer_activation_scratch = activation_scratch
            activation_out = activation_scratch[0]
            cache_before = self.native_layer_cache_stats() if metrics is not None else None
            self._model.layer_step_cached(
                activation,
                activation_out,
                state_in,
                state_out,
                None,
                None,
            )
            cache_after = (
                self.native_layer_cache_stats() if metrics is not None else None
            )
            state_len = int(self._model.layer_state_len)
            self._layer_state_flat = state_out
            self._layer_state_scratch = state_in
            self._layer_states = [
                state_out[index * state_len : (index + 1) * state_len]
                for index in range(self._n_layer)
            ]
            if metrics is not None:
                metrics.native_layer_streaming = True
                assert cache_before is not None and cache_after is not None
                self._record_native_layer_cache_delta(metrics, cache_before, cache_after)
            hidden = self._layer_norm_global(
                activation_out,
                "ln_out.weight",
                "ln_out.bias",
            )
            return self._layer_logits_global(hidden)
        v_first = None
        activation_scratch = self._layer_activation_scratch
        if (
            activation_scratch is None
            or activation_scratch[0].shape != activation.shape
            or activation_scratch[1].shape != activation.shape
        ):
            activation_scratch = (np.empty_like(activation), np.empty_like(activation))
            self._layer_activation_scratch = activation_scratch
        activation_scratch_index = 0
        v_first_scratch = self._layer_v_first_scratch
        if (
            v_first_scratch is None
            or v_first_scratch[0].shape != activation.shape
            or v_first_scratch[1].shape != activation.shape
        ):
            v_first_scratch = (np.empty_like(activation), np.empty_like(activation))
            self._layer_v_first_scratch = v_first_scratch
        v_first_scratch_index = 0
        for index, layer_id in enumerate(layer_ids):
            if control is not None:
                control.check()
            if index + 1 < len(layer_ids) and getattr(provider, "_prefetch_enabled", True):
                provider.prefetch_entries(by_layer.get(layer_ids[index + 1], []))
            entries = by_layer.get(layer_id, [])
            if not entries:
                raise RuntimeError(f"provider has no entries for RWKV layer {layer_id}")
            layer_start = len(metrics.layers) if metrics is not None else 0
            self._upload_provider_layer(provider, layer_id, entries, metrics)
            activation_out = activation_scratch[activation_scratch_index]
            activation_scratch_index = 1 - activation_scratch_index
            state_in = self._layer_states[index]
            state_out = (
                self._layer_state_scratch_by_layer[index]
                if self._layer_state_scratch_by_layer is not None
                else np.empty_like(state_in)
            )
            v_first_out = v_first_scratch[v_first_scratch_index]
            v_first_scratch_index = 1 - v_first_scratch_index
            started = time.perf_counter()
            self._model.layer_step(
                int(layer_id),
                activation,
                activation_out,
                state_in,
                state_out,
                None if index == 0 else v_first,
                v_first_out,
            )
            self._store_layer_state(index, state_out)
            activation = activation_out
            v_first = v_first_out
            if metrics is not None:
                metrics.native_layer_streaming = True
                if len(metrics.layers) > layer_start:
                    metrics.layers[-1].compute_ms += (
                        time.perf_counter() - started
                    ) * 1000.0
        hidden = self._layer_norm_global(
            activation,
            "ln_out.weight",
            "ln_out.bias",
        )
        return self._layer_logits_global(hidden)

    def _layer_advance_batch_sequences(
        self,
        token_sequences: list[list[int]],
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Prefill independent prompts with one native upload per layer.

        Prompt lengths may differ, so the native sequence ABI cannot receive
        all sessions in one call.  Keeping the layer active while visiting
        each session still removes the repeated provider read and native
        upload from the old session-outer loop.  Older DLLs use the same
        schedule with their token ABI.
        """
        if self._model is None or not self._native_layer_streaming_active():
            raise RuntimeError("native layer streaming is not active")
        if not token_sequences or not layer_ids:
            raise ValueError("native batch prefill requires prompts and layers")
        import torch

        emb = self._layer_global_weights["emb.weight"]
        activations: list[np.ndarray] = []
        for token_ids in token_sequences:
            if not token_ids:
                raise ValueError("native batch prefill prompts must be non-empty")
            ids = np.asarray(token_ids, dtype=np.int64)
            if int(ids.min()) < 0 or int(ids.max()) >= int(emb.shape[0]):
                raise ValueError("token id is outside the provider vocabulary")
            activation = emb.index_select(0, torch.from_numpy(ids)).detach().cpu().numpy()
            activation = np.asarray(activation, dtype=np.float32).copy()
            activations.append(
                self._layer_norm_sequence_global(
                    activation,
                    "blocks.0.ln0.weight",
                    "blocks.0.ln0.bias",
                )
            )

        batch_size = len(activations)
        state_len = int(self._model.layer_state_len)
        state_size = int(self._n_layer) * state_len
        states = [np.zeros(state_size, dtype=np.float32) for _ in activations]
        v_first: list[np.ndarray | None] = [None] * batch_size
        use_sequence = bool(
            getattr(self._model, "supports_layer_streaming_sequence", False)
            and callable(getattr(self._model, "layer_step_sequence", None))
        )

        if metrics is not None:
            metrics.batch_size = max(int(getattr(metrics, "batch_size", 0)), batch_size)
            metrics.weight_sweeps += 1

        for index, layer_id in enumerate(layer_ids):
            entries = by_layer.get(layer_id, [])
            if not entries:
                raise RuntimeError(f"provider has no entries for RWKV layer {layer_id}")
            if index + 1 < len(layer_ids) and getattr(
                provider, "_prefetch_enabled", True
            ):
                provider.prefetch_entries(by_layer.get(layer_ids[index + 1], []))
            layer_start = len(metrics.layers) if metrics is not None else 0
            self._upload_provider_layer(provider, layer_id, entries, metrics)
            row = None
            if metrics is not None:
                row = (
                    metrics.layers[-1]
                    if len(metrics.layers) > layer_start
                    else metrics.start_layer(layer_id)
                )
            started = time.perf_counter()
            state_offset = index * state_len
            for batch_index, activation in enumerate(activations):
                state_in = states[batch_index][state_offset : state_offset + state_len]
                state_out = np.empty_like(state_in)
                activation_out = np.empty_like(activation)
                if use_sequence:
                    if index == 0:
                        v_first_out = np.empty_like(activation)
                        self._model.layer_step_sequence(
                            int(layer_id),
                            activation,
                            activation_out,
                            int(activation.shape[0]),
                            state_in,
                            state_out,
                            None,
                            v_first_out,
                        )
                        v_first[batch_index] = v_first_out
                    else:
                        v_first_in = v_first[batch_index]
                        if v_first_in is None:
                            raise RuntimeError("native batch prefill lost v_first state")
                        self._model.layer_step_sequence(
                            int(layer_id),
                            activation,
                            activation_out,
                            int(activation.shape[0]),
                            state_in,
                            state_out,
                            v_first_in,
                            None,
                        )
                else:
                    state_a = np.empty_like(state_in)
                    state_b = np.empty_like(state_in)
                    current_state = state_in
                    next_state = state_a
                    if index == 0:
                        v_first_out = np.empty_like(activation)
                        v_first[batch_index] = v_first_out
                    else:
                        v_first_in = v_first[batch_index]
                        if v_first_in is None:
                            raise RuntimeError("native batch prefill lost v_first state")
                    for token_index in range(int(activation.shape[0])):
                        token_v_out = (
                            np.empty(activation.shape[1], dtype=np.float32)
                            if index == 0
                            else None
                        )
                        self._model.layer_step(
                            int(layer_id),
                            activation[token_index],
                            activation_out[token_index],
                            current_state,
                            next_state,
                            None
                            if index == 0
                            else v_first_in[token_index],
                            token_v_out,
                        )
                        if index == 0:
                            v_first[batch_index][token_index] = token_v_out
                        current_state, next_state = next_state, current_state
                    np.copyto(state_out, current_state)
                np.copyto(state_in, state_out)
                activations[batch_index] = activation_out
            if row is not None:
                row.compute_ms += (time.perf_counter() - started) * 1000.0
            if metrics is not None:
                metrics.native_layer_streaming = True
                metrics.weight_layer_loads += 1

        logits: list[np.ndarray] = []
        for activation in activations:
            hidden = self._layer_norm_sequence_global(
                activation[-1:],
                "ln_out.weight",
                "ln_out.bias",
            )
            logits.append(self._layer_logits_global(hidden[0]))
        return logits, states

    def _layer_advance_batch(
        self,
        token_ids: list[int],
        states: list[np.ndarray],
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Advance independent native states with one upload per layer.

        The rwkv.cpp layer ABI is token-shaped, but a layer plan remains
        active between calls.  Calling that ABI for every session while the
        same layer is active removes repeated provider reads and uploads from
        multi-request greedy decode without requiring a new native batch ABI.
        """
        if self._model is None or not self._native_layer_streaming_active():
            raise RuntimeError("native layer streaming is not active")
        if not token_ids or len(token_ids) != len(states):
            raise ValueError("token_ids and states must be equally sized and non-empty")
        if not layer_ids:
            raise ValueError("native batch decode requires at least one layer")
        import torch

        emb = self._layer_global_weights["emb.weight"]
        ids = np.asarray(token_ids, dtype=np.int64)
        if int(ids.min()) < 0 or int(ids.max()) >= int(emb.shape[0]):
            raise ValueError("token id is outside the provider vocabulary")
        activation = emb.index_select(0, torch.from_numpy(ids)).detach().cpu().numpy()
        activation = np.asarray(activation, dtype=np.float32).copy()
        activation = self._layer_norm_sequence_global(
            activation,
            "blocks.0.ln0.weight",
            "blocks.0.ln0.bias",
        )
        batch_size = len(token_ids)
        state_len = int(self._model.layer_state_len)
        expected_state = int(self._n_layer) * state_len
        state_in: list[np.ndarray] = []
        state_out: list[np.ndarray] = []
        for state in states:
            current = np.asarray(state, dtype=np.float32)
            if int(current.size) != expected_state:
                raise ValueError(
                    f"native batch state has {int(current.size)} elements; "
                    f"expected {expected_state}"
                )
            current = np.ascontiguousarray(current.reshape(expected_state)).copy()
            state_in.append(current)
            state_out.append(np.empty_like(current))
        next_activation = np.empty_like(activation)
        v_first = np.empty_like(activation)

        if metrics is not None:
            metrics.batch_size = batch_size
            metrics.weight_sweeps += 1

        for index, layer_id in enumerate(layer_ids):
            entries = by_layer.get(layer_id, [])
            if not entries:
                raise RuntimeError(f"provider has no entries for RWKV layer {layer_id}")
            if index + 1 < len(layer_ids) and getattr(
                provider, "_prefetch_enabled", True
            ):
                provider.prefetch_entries(by_layer.get(layer_ids[index + 1], []))
            layer_start = len(metrics.layers) if metrics is not None else 0
            self._upload_provider_layer(provider, layer_id, entries, metrics)
            row = None
            if metrics is not None:
                row = (
                    metrics.layers[-1]
                    if len(metrics.layers) > layer_start
                    else metrics.start_layer(layer_id)
                )
            started = time.perf_counter()
            state_offset = index * state_len
            for batch_index in range(batch_size):
                v_first_in = None if index == 0 else v_first[batch_index]
                v_first_out = v_first[batch_index] if index == 0 else None
                self._model.layer_step(
                    int(layer_id),
                    activation[batch_index],
                    next_activation[batch_index],
                    state_in[batch_index][state_offset : state_offset + state_len],
                    state_out[batch_index][state_offset : state_offset + state_len],
                    v_first_in,
                    v_first_out,
                )
                np.copyto(
                    state_in[batch_index][state_offset : state_offset + state_len],
                    state_out[batch_index][state_offset : state_offset + state_len],
                )
            activation, next_activation = next_activation, activation
            if row is not None:
                row.compute_ms += (time.perf_counter() - started) * 1000.0
            if metrics is not None:
                metrics.native_layer_streaming = True

        hidden = self._layer_norm_sequence_global(
            activation,
            "ln_out.weight",
            "ln_out.bias",
        )
        if self._layer_packed_head is not None:
            # The packed head GEMV ABI is intentionally scalar.  Keep the
            # memory-saving mode correct for batches, while dense BLAS uses a
            # single matrix multiplication below.
            logits = np.stack(
                [self._layer_logits_global(row) for row in hidden], axis=0
            )
        else:
            logits = self._layer_logits_global(hidden)
        return np.asarray(logits, dtype=np.float32), state_in

    def generate_greedy_batch_streaming(
        self,
        prompts: list[str],
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None = None,
    ) -> list[list[int]]:
        """Greedy batch generation using a native layer-outer decode sweep."""
        if self._model is None or self._encode_fn is None:
            raise RuntimeError("backend not loaded")
        if not self._native_layer_streaming_active():
            raise RuntimeError(
                "rwkv.cpp batch streaming requires the native layer-local ABI"
            )
        if not prompts:
            return []
        self._load_layer_global_weights(provider, by_layer)
        prompt_sequences = [self._encode_fn(prompt) or [0] for prompt in prompts]
        logits_list: list[np.ndarray] = []
        states: list[np.ndarray] = []
        prefill_started = time.perf_counter()
        logits_list, states = self._layer_advance_batch_sequences(
            [[int(token_id) for token_id in token_ids] for token_ids in prompt_sequences],
            provider,
            by_layer,
            layer_ids,
            metrics,
        )
        logits_list = [np.asarray(logits, dtype=np.float32).copy() for logits in logits_list]
        prefill_s = time.perf_counter() - prefill_started
        if metrics is not None:
            metrics.batch_size = len(prompts)
            metrics.batch_prefill_wall_s = prefill_s
            metrics.prefill_wall_s = prefill_s

        outputs = [[] for _ in prompts]
        count = max(0, int(max_tokens))
        decode_started = time.perf_counter()
        batch_logits = np.stack(logits_list, axis=0)
        for step in range(count):
            generated = [int(np.argmax(row)) for row in batch_logits]
            for output, token in zip(outputs, generated, strict=True):
                output.append(token)
            if step + 1 >= count:
                break
            batch_logits, states = self._layer_advance_batch(
                generated,
                states,
                provider,
                by_layer,
                layer_ids,
                metrics,
            )
        decode_s = time.perf_counter() - decode_started
        if states:
            self._state = states[0].copy()
            self._last_logits = batch_logits[0].copy()
            self._last_token_id = int(
                outputs[0][-1] if outputs[0] else prompt_sequences[0][-1]
            )
        if metrics is not None:
            metrics.batch_decode_wall_s = decode_s
            metrics.decode_wall_s = decode_s
            metrics.tokens_generated = len(prompts) * count
            metrics.native_layer_streaming = True
        return outputs

    def _layer_advance_sequence(
        self,
        token_ids: list[int],
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
        *,
        control=None,
    ) -> np.ndarray:
        """Prefill a token chunk while each active layer is uploaded once.

        The native sequence ABI consumes row-major FP32 activations and
        returns the final recurrent state for one layer.  The fallback keeps
        the same semantics for older qualified DLLs by invoking the existing
        single-token ABI without changing the bounded residency contract.
        """
        if self._model is None or self._layer_states is None:
            raise RuntimeError("native layer state is not initialized")
        if not token_ids:
            raise ValueError("layer sequence prefill requires at least one token")
        if control is not None:
            control.check()
        import torch

        emb = self._layer_global_weights["emb.weight"]
        ids = np.asarray(token_ids, dtype=np.int64)
        if int(ids.min()) < 0 or int(ids.max()) >= int(emb.shape[0]):
            raise ValueError("token id is outside the provider vocabulary")
        activation = emb.index_select(0, torch.from_numpy(ids)).detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        activation = self._layer_norm_sequence_global(
            activation,
            "blocks.0.ln0.weight",
            "blocks.0.ln0.bias",
        )
        v_first: np.ndarray | None = None
        use_sequence = bool(
            self._native_layer_streaming_active()
            and getattr(self._model, "supports_layer_streaming_sequence", False)
            and callable(getattr(self._model, "layer_step_sequence", None))
        )
        sequence_activation_scratch = self._layer_sequence_activation_scratch
        sequence_state_scratch = self._layer_sequence_state_scratch
        sequence_v_first_scratch = self._layer_sequence_v_first_scratch
        if use_sequence:
            if (
                sequence_activation_scratch is None
                or any(item.shape != activation.shape for item in sequence_activation_scratch)
                or any(item.dtype != activation.dtype for item in sequence_activation_scratch)
            ):
                sequence_activation_scratch = (
                    np.empty_like(activation),
                    np.empty_like(activation),
                )
                self._layer_sequence_activation_scratch = sequence_activation_scratch
            if (
                sequence_state_scratch is None
                or sequence_state_scratch.shape != self._layer_states[0].shape
                or sequence_state_scratch.dtype != self._layer_states[0].dtype
            ):
                sequence_state_scratch = np.empty_like(self._layer_states[0])
                self._layer_sequence_state_scratch = sequence_state_scratch
            if (
                sequence_v_first_scratch is None
                or sequence_v_first_scratch.shape != activation.shape
                or sequence_v_first_scratch.dtype != activation.dtype
            ):
                sequence_v_first_scratch = np.empty_like(activation)
                self._layer_sequence_v_first_scratch = sequence_v_first_scratch
        for index, layer_id in enumerate(layer_ids):
            if control is not None:
                control.check()
            if index + 1 < len(layer_ids) and getattr(provider, "_prefetch_enabled", True):
                provider.prefetch_entries(by_layer.get(layer_ids[index + 1], []))
            entries = by_layer.get(layer_id, [])
            if not entries:
                raise RuntimeError(f"provider has no entries for RWKV layer {layer_id}")
            self._upload_provider_layer(provider, layer_id, entries, metrics)
            state_in = self._layer_states[index]
            if use_sequence:
                assert sequence_activation_scratch is not None
                assert sequence_state_scratch is not None
                activation_out = sequence_activation_scratch[index & 1]
                state_out = sequence_state_scratch
            else:
                activation_out = np.empty_like(activation)
                state_out = np.empty_like(state_in)
            started = time.perf_counter()
            if use_sequence:
                if index == 0:
                    assert sequence_v_first_scratch is not None
                    v_first_out = sequence_v_first_scratch
                    self._model.layer_step_sequence(
                        int(layer_id),
                        activation,
                        activation_out,
                        len(token_ids),
                        state_in,
                        state_out,
                        None,
                        v_first_out,
                    )
                    v_first = v_first_out
                else:
                    self._model.layer_step_sequence(
                        int(layer_id),
                        activation,
                        activation_out,
                        len(token_ids),
                        state_in,
                        state_out,
                        v_first,
                        None,
                    )
            else:
                # Older native DLLs are still correct; keep one uploaded layer
                # active while stepping all prompt tokens through it.
                for token_index in range(len(token_ids)):
                    if control is not None:
                        control.check()
                    token_state_in = state_in if token_index == 0 else state_out
                    next_state = np.empty_like(state_out)
                    token_v_in = None if index == 0 else v_first[token_index]
                    token_v_out = np.empty(activation.shape[1], dtype=np.float32)
                    self._model.layer_step(
                        int(layer_id),
                        activation[token_index],
                        activation_out[token_index],
                        token_state_in,
                        next_state,
                        token_v_in,
                        token_v_out,
                    )
                    state_out = next_state
                    if index == 0:
                        if v_first is None:
                            v_first = np.empty_like(activation)
                        v_first[token_index] = token_v_out
            self._store_layer_state(index, state_out)
            activation = activation_out
            if metrics is not None:
                metrics.native_layer_streaming = True
                if metrics.layers:
                    metrics.layers[-1].compute_ms += (time.perf_counter() - started) * 1000.0
        hidden = self._layer_norm_sequence_global(
            activation[-1:],
            "ln_out.weight",
            "ln_out.bias",
        )
        return self._layer_logits_global(hidden[0])

    def _layer_norm_sequence_global(
        self, values: np.ndarray, weight_name: str, bias_name: str
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.asarray(values, dtype=np.float32))
        weight = self._layer_global_weights[weight_name]
        bias = self._layer_global_weights[bias_name]
        return F.layer_norm(x, (int(x.shape[-1]),), weight=weight, bias=bias).numpy().copy()

    def _generate_greedy_layer_streaming(
        self,
        prompt: str,
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
        *,
        system_prefix: str | None = None,
        prefix_cache: Any | None = None,
        prefix_cache_mode: str = "system",
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        if self._model is None or self._encode_fn is None:
            raise RuntimeError("backend not loaded")
        from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState

        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if control is not None:
            control.check()

        if system_prefix and prefix_cache_mode == "system" and isinstance(
            prefix_cache, PrefixStateCache
        ):
            full_prompt = system_prefix + prompt if not prompt.startswith(system_prefix) else prompt
            user_text = full_prompt[len(system_prefix) :]
            cached = prefix_cache.get(system_prefix)
            if cached is not None and cached.external_state is not None:
                state_in = np.asarray(cached.external_state, dtype=np.float32).copy()
                prompt_ids = self._encode_fn(user_text) or [0]
                if metrics is not None:
                    metrics.state_cache_hit = True
            else:
                system_ids = self._encode_fn(system_prefix) or [0]
                self._load_layer_global_weights(provider, by_layer)
                self._layer_reset_state(None)
                self._layer_advance_sequence(
                    [int(token_id) for token_id in system_ids],
                    provider,
                    by_layer,
                    layer_ids,
                    metrics,
                    control=control,
                )
                state_in = self._layer_flat_state()
                prefix_cache.put(
                    system_prefix,
                    RecurrentState(last_token_id=int(system_ids[-1]), external_state=state_in.copy()),
                )
                prompt_ids = self._encode_fn(user_text) or [0]
                if metrics is not None:
                    metrics.state_cache_hit = False
        else:
            if system_prefix and not prompt.startswith(system_prefix):
                prompt = system_prefix + prompt
            prompt_ids = self._encode_fn(prompt) or [0]
            state_in = None

        self._load_layer_global_weights(provider, by_layer)
        self._layer_reset_state(state_in)
        prefill_t0 = time.perf_counter()
        logits = self._layer_advance_sequence(
            [int(token_id) for token_id in prompt_ids],
            provider,
            by_layer,
            layer_ids,
            metrics,
            control=control,
        )
        prefill_s = time.perf_counter() - prefill_t0
        self._last_logits = logits.copy()
        out_ids: list[int] = []
        decode_t0 = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            token_started = time.perf_counter()
            next_id = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out_ids.append(next_id)
            if control is not None:
                # Preserve the state before applying the emitted token only
                # for controlled generation, where cancellation can interrupt
                # at this exact boundary.  The normal path publishes once at
                # the end of the request.
                self._state = self._layer_flat_state().copy()
                self._last_logits = logits.copy()
                control.emit(next_id)
            logits = self._layer_advance_token(
                int(next_id),
                provider,
                by_layer,
                layer_ids,
                metrics,
                control=control,
            )
            if control is not None:
                self._last_logits = logits.copy()
            _record_token_metrics(metrics, token_started, prefill_s=prefill_s)
        decode_s = time.perf_counter() - decode_t0
        self._state = self._layer_flat_state().copy()
        self._last_logits = logits.copy()
        self._last_token_id = int(out_ids[-1] if out_ids else prompt_ids[-1])
        if metrics is not None:
            metrics.prefill_wall_s = prefill_s
            metrics.decode_wall_s = decode_s
            metrics.tokens_generated = len(out_ids)
            metrics.native_layer_streaming = True
        return out_ids

    def _generate_layer_from_state(
        self,
        state: RecurrentState,
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
        *,
        layers_already_synced: bool = False,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        self._load_layer_global_weights(provider, by_layer)
        self._layer_reset_state(np.asarray(state.external_state, dtype=np.float32))
        del layers_already_synced
        out: list[int] = []
        current = int(state.last_token_id)
        last_logits: np.ndarray | None = None
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        t0 = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            token_started = time.perf_counter()
            logits = self._layer_advance_token(
                current,
                provider,
                by_layer,
                layer_ids,
                metrics,
                control=control,
            )
            last_logits = logits
            current = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out.append(current)
            if control is not None:
                # A controlled request must leave a resumable post-token
                # state if its callback cancels the next iteration.  Skip the
                # large copy for uninterrupted decode and publish at return.
                self._state = self._layer_flat_state().copy()
                self._last_logits = logits.copy()
                control.emit(current)
            _record_token_metrics(metrics, token_started)
        self._state = self._layer_flat_state().copy()
        if last_logits is not None:
            self._last_logits = last_logits.copy()
        self._last_token_id = current
        if metrics is not None:
            metrics.decode_wall_s = time.perf_counter() - t0
            metrics.tokens_generated = len(out)
            metrics.native_layer_streaming = True
        return out

    def _upload_provider_layer(
        self,
        provider: Any,
        layer_id: int,
        entries: list[Any],
        metrics: MetricsCollector | None,
    ) -> None:
        """Decode one provider layer and upload it into the ggml graph."""
        # The native one-block ABI receives only block payloads.  The
        # embedding and final/output norms are loaded once by
        # ``_load_layer_global_weights``; layer 0's manifest bucket also
        # contains those global entries, so passing the unfiltered bucket here
        # would re-read and re-materialize them on every prefill/decode sweep.
        # Only the bounded one-block ABI has a separate Python-side global
        # path.  The promoted/F5 native graph still receives the complete
        # provider bucket, including embedding/output tensors; removing
        # those entries there leaves packed-only GGML readiness incomplete.
        if self._native_layer_streaming_active():
            entries = [
                entry
                for entry in entries
                if getattr(entry, "name", None) not in self._LAYER_GLOBAL_NAMES
            ]
        if not entries or self._bridge is None:
            if not entries:
                return
            self._bridge = _create_weight_bridge(self._model)
            set_invalidator = getattr(
                provider, "set_native_layer_invalidator", None
            )
            invalidate = getattr(
                self._bridge, "invalidate_borrowed_layer", None
            )
            if callable(set_invalidator) and callable(invalidate):
                set_invalidator(invalidate)
        t0 = time.perf_counter()
        layer_started = False
        # Cache counters are diagnostic-only.  Avoid five ctypes calls before
        # and after every layer upload when the caller did not request
        # metrics, which is the hot path for direct backend users.
        cache_before = self.native_layer_cache_stats() if metrics is not None else None
        if self._native_layer_streaming_active():
            begin_layer = getattr(self._bridge, "begin_layer", None)
            if callable(begin_layer):
                # This begins the native plan before touching the provider.
                # When every payload is in the native bounded cache, the
                # provider decode/read path is skipped for this token.
                ready = bool(begin_layer(layer_id))
                layer_started = True
                cache_after = (
                    self.native_layer_cache_stats() if metrics is not None else None
                )
                if ready:
                    if metrics is not None:
                        if metrics.layers:
                            metrics.layers[-1].staging_ms += (
                                time.perf_counter() - t0
                            ) * 1000.0
                        # This is a complete decoded-layer restore.  It is
                        # distinct from the native tensor-level hit counter:
                        # provider read/decode/materialization was skipped.
                        metrics.native_decoded_cache_hits += 1
                        assert cache_before is not None and cache_after is not None
                        self._record_native_layer_cache_delta(metrics, cache_before, cache_after)
                        metrics.native_layer_streaming = True
                    return
        set_borrowed_persistent = getattr(
            self._bridge, "set_borrowed_persistent", None
        )
        set_packed_borrowed_persistent = getattr(
            self._bridge, "set_packed_borrowed_persistent", None
        )
        if callable(set_borrowed_persistent):
            persistent_payloads = bool(getattr(provider, "_stream_layer_cache", False))
            persistent_resolver = getattr(
                provider, "native_dense_layer_payloads_persistent", None
            )
            if not callable(persistent_resolver):
                persistent_resolver = getattr(
                    provider, "native_layer_payloads_persistent", None
                )
            if callable(persistent_resolver):
                persistent_payloads = bool(persistent_resolver(entries))
            set_borrowed_persistent(
                persistent_payloads
            )
            if callable(set_packed_borrowed_persistent):
                packed_persistent = persistent_payloads
                packed_resolver = getattr(
                    provider, "native_packed_layer_payloads_persistent", None
                )
                if callable(packed_resolver):
                    packed_persistent = bool(packed_resolver(entries))
                set_packed_borrowed_persistent(packed_persistent)
        if bool(getattr(self._model, "supports_native_u8", False)):
            load_native = getattr(provider, "load_layer_tensors_native", None)
        else:
            load_native = None
        if callable(load_native):
            tensors = load_native(entries)
        else:
            tensors = provider.load_layer_tensors_materialized(entries)
        cache_before = self.native_layer_cache_stats() if metrics is not None else None
        stats = self._bridge.upload_layer(
            layer_id,
            tensors,
            layer_started=layer_started,
        )
        cache_after = (
            self.native_layer_cache_stats() if metrics is not None else None
        )
        cache_delta = {
            key: max(0, int(cache_after.get(key, 0)) - int(cache_before.get(key, 0)))
            for key in ("hits", "misses", "evictions")
        } if cache_before is not None and cache_after is not None else {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
        if cache_after is not None:
            self._native_layer_cache_seen = cache_after
        if metrics is not None and metrics.layers:
            # ``load_layer_tensors_materialized`` owns the provider read row;
            # charge bridge conversion/upload to its staging bucket.
            metrics.layers[-1].staging_ms += (time.perf_counter() - t0) * 1000.0
            metrics.layers[-1].layer_cache_hits += stats.skipped_tensors
            metrics.native_upload_bytes += int(stats.uploaded_bytes)
            metrics.native_packed_bytes += int(stats.packed_bytes)
            metrics.native_active_bytes += int(stats.active_bytes)
            if cache_after is not None:
                metrics.native_layer_cache_bytes = int(
                    cache_after.get("used_bytes", 0) or 0
                )
            metrics.native_layer_cache_hits += cache_delta["hits"]
            metrics.native_layer_cache_misses += cache_delta["misses"]
            metrics.native_layer_cache_evictions += cache_delta["evictions"]
            # ``LayerTiming.bytes_read`` is the source of truth for streamed
            # bytes.  Upload bytes are reported separately above; counting
            # the logical layer payload here would charge cached native views
            # as fresh SSD traffic on every autoregressive token.
            slot_stats = self._bridge.slot_stats()
            metrics.native_evictions = max(
                metrics.native_evictions,
                int(slot_stats.get("evictions", 0) or 0),
            )
            if self._native_layer_streaming_active():
                metrics.native_layer_streaming = True

    def _sync_provider_layers(
        self,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None,
    ) -> None:
        """Make the native graph reflect the current provider view of layers."""
        # Submit one best-effort aggregate prefetch.  The provider's
        # non-blocking drain semantics are preserved; if it is not ready the
        # normal layer read remains the fallback.
        effective_layer_ids = list(layer_ids)
        # The normal engine scheduler exposes only block IDs.  A native U8
        # graph also owns the vocabulary head, whose manifest sentinel is
        # 9999; upload it once so the final logits projection does not remain
        # a dense resident checkpoint tensor.
        if 9999 in by_layer and 9999 not in effective_layer_ids:
            effective_layer_ids.append(9999)
        if getattr(provider, "_prefetch_enabled", True):
            all_entries = [
                e for lid in effective_layer_ids for e in by_layer.get(lid, [])
            ]
            if all_entries:
                provider.prefetch_entries(all_entries)
        for layer_id in effective_layer_ids:
            entries = by_layer.get(layer_id, [])
            if entries:
                self._upload_provider_layer(provider, layer_id, entries, metrics)

    def prepare_streaming_layers(
        self,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None = None,
        *,
        upload_layers: bool = False,
    ) -> None:
        """Warm provider/global state without making a full graph resident.

        ``upload_layers`` is used by the startup prewarm path.  It fills the
        bounded native layer plans and their borrowed-payload cache, but does
        not evaluate a token or allocate a resident all-layer GGML graph.
        """
        if self._model is not None and self._native_layer_streaming_active():
            self._load_layer_global_weights(provider, by_layer)
            if upload_layers:
                for layer_id in layer_ids:
                    entries = by_layer.get(layer_id, [])
                    if entries:
                        self._upload_provider_layer(
                            provider, layer_id, entries, metrics
                        )
            return
        self._sync_provider_layers(provider, by_layer, layer_ids, metrics)

    def generate_greedy_pack_streaming(
        self,
        prompt: str,
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None = None,
        *,
        system_prefix: str | None = None,
        prefix_cache: Any | None = None,
        prefix_cache_mode: str = "system",
        power_percent: int = 100,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        """Greedy rwkv.cpp decode backed by the shared F1-F5 provider stack.

        ggml evaluates the whole recurrent graph natively.  The provider pack
        is synchronized before prefill.  By default the native graph then
        keeps those weights stationary for the rest of the generation; a
        complete GGML graph is already resident, so per-token re-uploads add
        cost without reducing native memory.  Set
        ``RWKVCPP_SYNC_EVERY_TOKEN=1`` to restore the strict diagnostic path.
        """
        del power_percent
        if self._model is None or self._encode_fn is None:
            raise RuntimeError("backend not loaded")
        if self._native_layer_streaming_active():
            return self._generate_greedy_layer_streaming(
                prompt,
                max_tokens,
                provider,
                by_layer,
                layer_ids,
                metrics,
                system_prefix=system_prefix,
                prefix_cache=prefix_cache,
                prefix_cache_mode=prefix_cache_mode,
                temperature=temperature,
                greedy=greedy,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        from rwkv_ssd.runtime.state_cache import PrefixStateCache, RecurrentState

        if system_prefix and prefix_cache is not None and not prompt.startswith(system_prefix):
            prompt = system_prefix + prompt

        prompt_ids = self._encode_fn(prompt) or [0]
        sync_every_token = _sync_provider_every_token()
        layers_synced = False
        state_in: np.ndarray | None = None
        if prefix_cache_mode == "transcript" and isinstance(prefix_cache, PrefixStateCache):
            from rwkv_ssd.runtime.transcript_cache import longest_cached_prefix

            cache_key, stable, suffix = longest_cached_prefix(
                prompt,
                has_entry=prefix_cache.contains,
            )
            cached = prefix_cache.get(cache_key) if cache_key else None
            if cached is not None and cached.external_state is not None:
                if metrics is not None:
                    metrics.state_cache_hit = True
                state_in = np.asarray(cached.external_state, dtype=np.float32).copy()
                prompt_ids = self._encode_fn(suffix) or [0]
            else:
                if metrics is not None:
                    metrics.state_cache_hit = False
                if stable:
                    self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
                    layers_synced = True
                    stable_ids = self._encode_fn(stable) or [0]
                    _, state_in = self._model.eval_sequence_in_chunks(
                        stable_ids, None, None, None, use_numpy=True
                    )
                    if cache_key:
                        prefix_cache.put(
                            cache_key,
                            RecurrentState(
                                last_token_id=int(stable_ids[-1]),
                                external_state=np.asarray(state_in, dtype=np.float32).copy(),
                            ),
                        )
                    prompt_ids = self._encode_fn(suffix) or [0]
        elif system_prefix and isinstance(prefix_cache, PrefixStateCache):
            user_text = prompt[len(system_prefix) :]
            system_ids = self._encode_fn(system_prefix)
            cached = prefix_cache.get(system_prefix)
            if cached is not None and cached.external_state is not None:
                if metrics is not None:
                    metrics.state_cache_hit = True
                state_in = np.asarray(cached.external_state, dtype=np.float32).copy()
                prompt_ids = self._encode_fn(user_text) or [0]
            else:
                if metrics is not None:
                    metrics.state_cache_hit = False
                self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
                layers_synced = True
                t_prefill = time.perf_counter()
                _, state_in = self._model.eval_sequence_in_chunks(
                    system_ids or [0], None, None, None, use_numpy=True
                )
                if metrics is not None:
                    metrics.prefill_wall_s = time.perf_counter() - t_prefill
                prefix_cache.put(
                    system_prefix,
                    RecurrentState(
                        last_token_id=int(system_ids[-1]) if system_ids else 0,
                        external_state=np.asarray(state_in, dtype=np.float32).copy(),
                    ),
                )
                prompt_ids = self._encode_fn(user_text) or [0]

        t_prefill = time.perf_counter()
        if not layers_synced:
            self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
        logits, state = self._model.eval_sequence_in_chunks(
            prompt_ids, state_in, None, None, use_numpy=True
        )
        next_state = np.empty_like(state)
        self._last_logits = np.asarray(logits, dtype=np.float32).copy()
        if metrics is not None:
            metrics.prefill_wall_s = max(metrics.prefill_wall_s, time.perf_counter() - t_prefill)
            metrics.layers.clear()

        out_ids: list[int] = []
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        last_id = int(prompt_ids[-1]) if prompt_ids else 0
        decode_t0 = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            # Synchronization happens after choosing the next token so native
            # decode timing excludes provider upload for the previous token.
            next_id = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out_ids.append(next_id)
            if control is not None:
                control.emit(next_id)
            if sync_every_token:
                self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
            logits, next_state = self._model.eval(
                next_id, state, next_state, logits, use_numpy=True
            )
            state, next_state = next_state, state
            self._last_logits = np.asarray(logits, dtype=np.float32).copy()
            last_id = next_id
        decode_s = time.perf_counter() - decode_t0
        self._state = np.asarray(state, dtype=np.float32).copy()
        self._last_token_id = last_id
        if metrics is not None:
            metrics.decode_wall_s = decode_s
            metrics.tokens_generated = len(out_ids)
            if metrics.layers:
                per_layer = decode_s * 1000.0 / max(1, len(metrics.layers))
                for row in metrics.layers:
                    row.compute_ms += per_layer
        return out_ids

    def generate_greedy_from_state(
        self,
        state: RecurrentState,
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list[Any]],
        layer_ids: list[int],
        metrics: MetricsCollector | None = None,
        *,
        layers_already_synced: bool = False,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        """Continue from an external rwkv.cpp state using provider layers."""
        if self._model is None or state.external_state is None:
            raise RuntimeError("rwkv.cpp state is not available")
        if self._native_layer_streaming_active():
            return self._generate_layer_from_state(
                state,
                max_tokens,
                provider,
                by_layer,
                layer_ids,
                metrics,
                layers_already_synced=layers_already_synced,
                temperature=temperature,
                greedy=greedy,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        current = np.asarray(state.external_state, dtype=np.float32).copy()
        next_state = np.empty_like(current)
        last = int(state.last_token_id)
        out: list[int] = []
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        t0 = time.perf_counter()
        sync_every_token = _sync_provider_every_token()
        last_logits: np.ndarray | None = None
        if not layers_already_synced:
            self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            token_started = time.perf_counter()
            if sync_every_token:
                self._sync_provider_layers(provider, by_layer, layer_ids, metrics)
            logits, next_state = self._model.eval(
                last, current, next_state, None, use_numpy=True
            )
            current, next_state = next_state, current
            last_logits = np.asarray(logits, dtype=np.float32)
            last = sample_numpy(logits, temperature=temperature, greedy=greedy)
            out.append(last)
            if control is not None:
                self._state = np.asarray(current, dtype=np.float32).copy()
                self._last_logits = np.asarray(logits, dtype=np.float32).copy()
                control.emit(last)
            _record_token_metrics(metrics, token_started)
        self._state = np.asarray(current, dtype=np.float32).copy()
        if last_logits is not None:
            self._last_logits = last_logits.copy()
        self._last_token_id = last
        if metrics is not None:
            metrics.decode_wall_s = time.perf_counter() - t0
            metrics.tokens_generated = len(out)
        return out

    def get_recurrent_state(self) -> RecurrentState | None:
        if self._state is None:
            return None
        return RecurrentState(
            last_token_id=self._last_token_id,
            external_state=self._state.copy(),
        )

    def probe_logits(self, state: RecurrentState | None = None) -> np.ndarray | None:
        del state
        return self._last_logits.copy() if self._last_logits is not None else None

    def set_recurrent_state(self, state: RecurrentState) -> None:
        if state.external_state is not None:
            self._state = np.asarray(state.external_state, dtype=np.float32).copy()
        self._last_logits = None
        self._last_token_id = int(state.last_token_id)

    def close(self) -> None:
        self._bridge = None
        # Release any provider-backed packed head view before the engine closes
        # the mmap/store that owns its bytes.
        self._layer_packed_head = None
        if self._model is not None and hasattr(self._model, "free"):
            try:
                self._model.free()
            except ValueError:
                pass
            self._model = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
