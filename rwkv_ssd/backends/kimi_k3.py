"""Resident CPU backend for the Hugging Face Kimi-K3 text model.

Kimi-K3 is a hybrid KDA/MLA sparse-MoE model and is not a packed RWKV or
generic Transformer checkpoint.  This backend deliberately loads its local
Hugging Face directory with the model's own remote code and uses the narrow
CPU FLA compatibility layer in :mod:`kimi_k3_compat` when Triton is absent.

The first useful integration is resident CPU execution.  The backend exposes
the same token/state/cancellation contract as the other engines, while
returning an explicit ``layer_streaming`` capability error for partial or
streaming modes.  That boundary prevents a full 720 MB resident model from
being mislabeled as a low-RAM F1-F4 execution.
"""

from __future__ import annotations

import gc
import importlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.backends.kimi_k3_compat import cpu_compat_modules
from rwkv_ssd.runtime.errors import BackendNotAvailableError, CapabilityNotSupportedError
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.sampling import sample_torch
from rwkv_ssd.runtime.state_cache import RecurrentState


_STATE_MAGIC = 4_736_291.0
_STATE_VERSION = 1.0

# External-state record roles.  Keeping the schema numeric makes it compatible
# with the existing float32 state parking/snapshot container.
_ROLE_CONV_Q = 1
_ROLE_CONV_K = 2
_ROLE_CONV_V = 3
_ROLE_RECURRENT = 4
_ROLE_KEY = 5
_ROLE_VALUE = 6
_ROLE_NEXT_LOGITS = 7


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_execution_dtype(strategy: str) -> tuple[torch.dtype, bool]:
    """Resolve the resident Kimi dtype and optional dynamic-int8 mode.

    The checkpoint is FP32 on disk, but the model's own text configuration is
    BF16-oriented.  Keep the engine's existing ``cpu fp32`` default explicit
    for reproducibility, while allowing the normal strategy spelling to select
    a half-width resident model.  Dynamic int8 is deliberately opt-in: it is a
    useful memory experiment, not a silent change to the correctness path.
    ``RWKV_KIMI_DTYPE`` and ``RWKV_KIMI_DYNAMIC_INT8`` are process-level
    overrides for HTTP workers and shell-based benchmarks.
    """

    requested = os.environ.get("RWKV_KIMI_DTYPE", "").strip().lower()
    if not requested:
        requested = str(strategy).strip().lower()
    if requested in {"fp16", "float16", "half"} or " fp16" in requested:
        dtype = torch.float16
    elif requested in {"bf16", "bfloat16"} or " bf16" in requested:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    dynamic_int8 = _env_flag("RWKV_KIMI_DYNAMIC_INT8") or any(
        marker in requested for marker in ("int8", "q8", "quant")
    )
    if dynamic_int8:
        # Dynamic quantized Linear modules consume floating-point activations;
        # FP32 is the portable reference activation dtype on CPU.
        dtype = torch.float32
    return dtype, dynamic_int8


def _configure_cpu_threads() -> int:
    """Tune Torch's small-matrix thread policy for Kimi's large vocab heads."""

    explicit = os.environ.get("RWKV_KIMI_CPU_THREADS", "").strip()

    def configure_interop() -> None:
        raw_interop = os.environ.get(
            "RWKV_KIMI_INTEROP_THREADS",
            os.environ.get("RWKV_CPU_INTEROP_THREADS", ""),
        ).strip()
        if not raw_interop:
            raw_interop = os.environ.get("TORCH_NUM_INTEROP_THREADS", "").strip()
        if raw_interop:
            if raw_interop.lower() in {"0", "false", "off", "no"}:
                return
            try:
                interop = max(1, int(raw_interop))
            except ValueError as exc:
                raise ValueError(
                    "RWKV_KIMI_INTEROP_THREADS must be a positive integer"
                ) from exc
        else:
            interop = max(1, min(4, os.cpu_count() or 4))
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            # A process may already have initialized Torch parallel work (for
            # example when a second engine is created).  In that case keep
            # the established pool rather than failing model load.
            pass

    if explicit:
        try:
            threads = max(1, int(explicit))
        except ValueError as exc:
            raise ValueError(
                f"RWKV_KIMI_CPU_THREADS must be a positive integer, got {explicit!r}"
            ) from exc
        torch.set_num_threads(threads)
        configure_interop()
        return threads

    # Honor the shared CPU knob and the standard BLAS/OpenMP knobs.  The
    # engine's generic small-model default is one thread, which is good for
    # tiny RWKV GEMVs but unnecessarily slow for Kimi's 163k-token lm_head.
    shared = os.environ.get("RWKV_CPU_THREADS", "").strip().lower()
    if shared and shared not in {"auto", "0", "false", "off", "no"}:
        try:
            threads = max(1, int(shared))
        except ValueError:
            threads = torch.get_num_threads()
        torch.set_num_threads(threads)
        configure_interop()
        return threads
    if any(os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")):
        configure_interop()
        return int(torch.get_num_threads())

    threads = max(1, min(4, os.cpu_count() or 4))
    torch.set_num_threads(threads)
    configure_interop()
    return threads


def _model_weight_bytes(model: torch.nn.Module) -> int:
    """Estimate loaded parameter bytes, including dynamic quantized weights."""

    total = sum(_tensor_nbytes(parameter) for parameter in model.parameters())
    for module in model.modules():
        module_name = type(module).__module__
        if "torch.ao.nn.quantized.dynamic" not in module_name:
            continue
        weight = getattr(module, "weight", None)
        if not callable(weight):
            continue
        try:
            quantized = weight()
        except (RuntimeError, TypeError):
            continue
        total += _tensor_nbytes(quantized)
        # Per-channel quantizers carry scale and zero-point arrays outside the
        # int8 storage.  Include them so metrics do not under-report RSS.
        try:
            total += _tensor_nbytes(quantized.q_per_channel_scales())
            total += _tensor_nbytes(quantized.q_per_channel_zero_points())
        except (RuntimeError, AttributeError):
            try:
                total += 8  # scalar scale + zero point, conservatively
            except Exception:
                pass
    return int(total)


def _load_safetensors_in_target_dtype(
    root: Path,
    *,
    auto_config: Any,
    auto_model: Any,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Build the remote model in-place and copy one safetensor at a time.

    ``from_pretrained(dtype=...)`` is convenient, but on Windows it can leave
    the original FP32 checkpoint pages resident while the converted model is
    alive.  For FP16/BF16 resident Kimi this defeats much of the memory goal.
    Constructing from config with the target dtype and copying each tensor
    directly keeps the peak bounded by the largest checkpoint tensor plus the
    target model.  This path is intentionally limited to the single-file raw
    Kimi layout validated by :meth:`KimiK3CPUBackend.load`.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Kimi-K3 reduced-dtype loading requires the safetensors package"
        ) from exc

    with cpu_compat_modules():
        config = auto_config.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=True
        )
        # Transformers 5 calls this field ``dtype``; older remote-code stacks
        # may consult ``torch_dtype``.  Set both without passing the deprecated
        # loader keyword to the modern API.
        try:
            config.dtype = dtype
        except (AttributeError, TypeError):
            pass
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            try:
                text_config.dtype = dtype
            except (AttributeError, TypeError):
                pass
        model = auto_model.from_config(config, trust_remote_code=True)
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        targets: dict[str, torch.Tensor] = dict(parameters)
        targets.update(buffers)
        loaded: set[str] = set()
        with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                target = targets.get(name)
                if target is None:
                    raise ValueError(
                        f"Kimi-K3 checkpoint tensor {name!r} is not present in the model"
                    )
                source = handle.get_tensor(name)
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Kimi-K3 tensor shape mismatch for {name!r}: "
                        f"checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
                    )
                with torch.no_grad():
                    target.copy_(source.to(dtype=target.dtype, device=target.device))
                loaded.add(name)
        # The language-only Kimi checkpoint intentionally omits one vision
        # positional buffer that the multimodal wrapper can initialize lazily.
        # Missing trainable parameters are never acceptable; missing buffers
        # may retain the remote model's deterministic initialization because
        # the CPU text path never consumes them.
        missing = sorted(set(parameters) - loaded)
        if missing:
            raise ValueError(
                "Kimi-K3 checkpoint is missing model tensors: "
                + ", ".join(missing[:4])
            )
        return model


class KimiK3CPUBackend(RecurrentBackend):
    """Reference-quality, resident CPU executor for Kimi-K3."""

    model_family = "kimi_k3"
    sequence_kind = "kimi_k3"
    supports_batch = False

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._root: Path | None = None
        self._device = torch.device("cpu")
        self._cache: Any | None = None
        self._cache_type: type | None = None
        self._text_config: Any | None = None
        self._next_logits: torch.Tensor | None = None
        self._last_token_id = 0
        self._position = 0
        self._num_layers = 0
        self._vocab_size = 0
        self._eos_ids: frozenset[int] = frozenset()
        self._state: RecurrentState | None = None
        self._weight_dtype = torch.float32
        self._dynamic_int8 = False
        self._thread_count = 1
        self._weight_bytes = 0

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def weight_dtype(self) -> torch.dtype:
        return self._weight_dtype

    @property
    def dynamic_int8(self) -> bool:
        return self._dynamic_int8

    @property
    def thread_count(self) -> int:
        return self._thread_count

    def _state_dtype(self) -> torch.dtype:
        """Return the floating dtype expected by the loaded model cache."""

        if self._model is not None:
            for parameter in self._model.parameters():
                if parameter.is_floating_point():
                    return parameter.dtype
        return self._weight_dtype

    def _cache_tensor_dtype(self, role: int) -> torch.dtype:
        # The compatibility KDA recurrence intentionally keeps its matrix
        # state in FP32 for numerical stability.  Convolution and MLA KV
        # caches must match the model activation dtype for torch.cat/linear
        # kernels, especially after FP16/BF16 state parking.
        if role == _ROLE_RECURRENT:
            return torch.float32
        return self._state_dtype()

    def load(self, model_path: str, strategy: str, device: str) -> None:
        root = Path(model_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Kimi-K3 model directory does not exist: {root}")
        if not (root / "config.json").is_file():
            raise ValueError(f"Kimi-K3 model directory is missing config.json: {root}")
        if not (root / "model.safetensors").is_file():
            raise ValueError(
                "Kimi-K3 CPU backend currently requires a local model.safetensors "
                f"checkpoint in {root}"
            )
        requested = torch.device(device)
        if requested.type != "cpu":
            raise CapabilityNotSupportedError(
                "device",
                "kimi_k3",
                "the current Kimi-K3 reference path is CPU-only",
            )
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Kimi-K3 requires the optional transformers package; "
                "install the local CPU/HF dependencies first"
            ) from exc

        dtype, dynamic_int8 = _resolve_execution_dtype(strategy)
        self._thread_count = _configure_cpu_threads()
        self._root = root
        self._device = requested
        self._weight_dtype = dtype
        self._dynamic_int8 = dynamic_int8
        # Tokenizer remote code does not depend on FLA.  Loading it outside the
        # shim keeps the replacement module tree limited to model import.
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=True
        )
        try:
            if dtype is not torch.float32 and not dynamic_int8:
                self._model = _load_safetensors_in_target_dtype(
                    root,
                    auto_config=AutoConfig,
                    auto_model=AutoModelForCausalLM,
                    dtype=dtype,
                )
            else:
                with cpu_compat_modules():
                    try:
                        self._model = AutoModelForCausalLM.from_pretrained(
                            str(root),
                            local_files_only=True,
                            trust_remote_code=True,
                            dtype=dtype,
                        )
                    except TypeError:
                        # Transformers 4.x calls this argument torch_dtype.
                        self._model = AutoModelForCausalLM.from_pretrained(
                            str(root),
                            local_files_only=True,
                            trust_remote_code=True,
                            torch_dtype=dtype,
                        )
                    # The compatibility backend is correctness-first, but an
                    # explicit ``cpu int8`` strategy is useful on machines where
                    # the resident FP32 vocabulary heads are the limiting factor.
                    # Keep quantization after HF loading so remote-code model
                    # construction remains unchanged and auditable.
                    if dynamic_int8:
                        try:
                            from torch.ao.quantization import quantize_dynamic

                            self._model = quantize_dynamic(
                                self._model,
                                {torch.nn.Linear},
                                dtype=torch.qint8,
                                inplace=True,
                            )
                        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                            raise BackendNotAvailableError(
                                "Kimi-K3 dynamic int8 requires the CPU Torch quantization "
                                "operators available in this environment"
                            ) from exc
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Kimi-K3 remote code could not be loaded. Ensure the local model "
                "directory includes its custom modeling/tokenization Python files."
            ) from exc
        assert self._model is not None
        self._model.to(self._device)
        self._model.eval()
        self._weight_bytes = _model_weight_bytes(self._model)
        self._text_config = getattr(self._model.config, "text_config", self._model.config)
        # The custom remote module keeps KimiDynamicCache as a module-global
        # rather than exporting it from the top-level model object. Discover
        # it once at load time so a fresh worker can restore a parked state
        # before it has performed its first prefill.
        for owner in (
            getattr(self._model, "language_model", None),
            getattr(getattr(self._model, "language_model", None), "model", None),
        ):
            function = getattr(owner, "forward", None)
            globals_dict = getattr(function, "__globals__", {})
            candidate = globals_dict.get("KimiDynamicCache")
            if isinstance(candidate, type):
                self._cache_type = candidate
                break
        if self._cache_type is None:
            for owner in (
                getattr(self._model, "language_model", None),
                getattr(getattr(self._model, "language_model", None), "model", None),
            ):
                module_name = getattr(type(owner), "__module__", "")
                if not module_name:
                    continue
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                candidate = getattr(module, "KimiDynamicCache", None)
                if isinstance(candidate, type):
                    self._cache_type = candidate
                    break
        self._num_layers = int(getattr(self._text_config, "num_hidden_layers", 0))
        self._vocab_size = int(
            getattr(self._text_config, "vocab_size", getattr(self._model.config, "vocab_size", 0))
        )
        if self._num_layers <= 0 or self._vocab_size <= 0:
            raise ValueError("Kimi-K3 config did not expose a valid layer/vocabulary shape")
        generation_config = getattr(self._model, "generation_config", None)
        eos = getattr(generation_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self._text_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
        eos_values = eos if isinstance(eos, (list, tuple, set)) else [eos]
        self._eos_ids = frozenset(int(value) for value in eos_values if value is not None)
        self._cache = None
        self._next_logits = None
        self._state = None
        self._position = 0
        self._last_token_id = 0

    def close(self) -> None:
        self._state = None
        self._next_logits = None
        self._cache = None
        self._cache_type = None
        self._model = None
        self._tokenizer = None
        self._weight_bytes = 0
        self._dynamic_int8 = False
        gc.collect()

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Kimi-K3 tokenizer is not loaded")
        token_ids = [int(value) for value in self._tokenizer.encode(str(text))]
        if token_ids:
            return token_ids
        bos = getattr(self._tokenizer, "bos_token_id", None)
        if bos is None:
            raise ValueError("Kimi-K3 tokenizer produced no tokens and has no BOS token")
        return [int(bos)]

    def decode_text(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Kimi-K3 tokenizer is not loaded")
        # The custom tokenizer uses the no-keyword path for its fast tiktoken
        # decoder.  Filter special markers only when the tokenizer's generic
        # path is explicitly available; ordinary model output is unchanged.
        try:
            return str(self._tokenizer.decode([int(value) for value in token_ids]))
        except (TypeError, ValueError):
            return str(
                self._tokenizer.decode(
                    [int(value) for value in token_ids], skip_special_tokens=True
                )
            )

    def _new_cache(self) -> Any:
        if self._cache_type is None or self._text_config is None:
            raise RuntimeError("Kimi-K3 cache type is unavailable; run prefill first")
        try:
            return self._cache_type(config=self._text_config)
        except TypeError:
            return self._cache_type(self._text_config)

    def _forward(
        self,
        token_ids: list[int],
        *,
        cache: Any | None,
    ) -> tuple[torch.Tensor, Any]:
        if self._model is None:
            raise RuntimeError("Kimi-K3 model is not loaded")
        ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        kwargs: dict[str, Any] = {
            "input_ids": ids,
            "use_cache": True,
            "return_dict": True,
        }
        if cache is None:
            kwargs["attention_mask"] = torch.ones_like(ids)
        else:
            kwargs["past_key_values"] = cache
        with torch.inference_mode():
            output = self._model(**kwargs)
        logits = output.logits[:, -1, :].detach()
        next_cache = output.past_key_values
        if next_cache is None:
            raise RuntimeError("Kimi-K3 did not return a cache despite use_cache=True")
        if self._cache_type is None:
            self._cache_type = type(next_cache)
        return logits, next_cache

    def _set_live_state(
        self,
        *,
        cache: Any,
        logits: torch.Tensor,
        last_token_id: int,
        position: int,
    ) -> None:
        self._cache = cache
        self._next_logits = logits.detach()
        self._last_token_id = int(last_token_id)
        self._position = int(position)
        self._state = None

    @staticmethod
    def _cache_tensors(cache: Any) -> list[tuple[int, int, torch.Tensor]]:
        records: list[tuple[int, int, torch.Tensor]] = []
        conv_states = getattr(cache, "conv_states", [])
        recurrent_states = getattr(cache, "recurrent_states", [])
        key_cache = getattr(cache, "key_cache", [])
        value_cache = getattr(cache, "value_cache", [])
        for layer_id, value in enumerate(conv_states):
            if value is None:
                continue
            if len(value) != 3:
                raise ValueError("Kimi-K3 convolution cache must contain q/k/v states")
            for role, tensor in zip(
                (_ROLE_CONV_Q, _ROLE_CONV_K, _ROLE_CONV_V), value
            ):
                if tensor is not None:
                    records.append((role, layer_id, tensor))
        for layer_id, tensor in enumerate(recurrent_states):
            if tensor is not None:
                records.append((_ROLE_RECURRENT, layer_id, tensor))
        for layer_id, (key, value) in enumerate(zip(key_cache, value_cache)):
            if key is not None:
                records.append((_ROLE_KEY, layer_id, key))
            if value is not None:
                records.append((_ROLE_VALUE, layer_id, value))
        return records

    def _pack_live_state(self) -> np.ndarray:
        if self._cache is None or self._next_logits is None:
            raise RuntimeError("Kimi-K3 has no live state to serialize")
        records = self._cache_tensors(self._cache)
        records.append((_ROLE_NEXT_LOGITS, -1, self._next_logits))
        parts: list[np.ndarray] = [
            np.asarray(
                [
                    _STATE_MAGIC,
                    _STATE_VERSION,
                    float(len(records)),
                    float(self._position),
                    float(self._num_layers),
                ],
                dtype=np.float32,
            )
        ]
        for role, layer_id, tensor in records:
            # NumPy has no native bfloat16 conversion on all supported
            # versions.  State envelopes use the established float32 wire
            # representation, so convert through Torch explicitly instead of
            # relying on ``np.asarray(torch_bfloat16)``.
            array = (
                tensor.detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
            )
            flat = array.reshape(-1)
            parts.append(
                np.asarray(
                    [float(role), float(layer_id), float(array.ndim)]
                    + [float(dim) for dim in array.shape]
                    + [float(flat.size)],
                    dtype=np.float32,
                )
            )
            parts.append(flat)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _unpack_state(
        self, payload: object
    ) -> tuple[Any, torch.Tensor, int]:
        raw = np.asarray(payload, dtype=np.float32).reshape(-1)
        if raw.size < 5 or float(raw[0]) != _STATE_MAGIC or float(raw[1]) != _STATE_VERSION:
            raise ValueError("invalid Kimi-K3 external state magic or version")
        record_count = int(round(float(raw[2])))
        position = int(round(float(raw[3])))
        layers = int(round(float(raw[4])))
        if layers != self._num_layers or record_count <= 0:
            raise ValueError("Kimi-K3 external state model shape mismatch")
        cache = self._new_cache()
        next_logits: torch.Tensor | None = None
        conv_by_layer: dict[int, list[torch.Tensor | None]] = {}
        cursor = 5
        for _ in range(record_count):
            if cursor + 3 > raw.size:
                raise ValueError("truncated Kimi-K3 external state record")
            role = int(round(float(raw[cursor])))
            layer_id = int(round(float(raw[cursor + 1])))
            ndim = int(round(float(raw[cursor + 2])))
            cursor += 3
            if ndim < 0 or cursor + ndim + 1 > raw.size:
                raise ValueError("invalid Kimi-K3 external state tensor rank")
            shape = tuple(int(round(float(value))) for value in raw[cursor : cursor + ndim])
            cursor += ndim
            count = int(round(float(raw[cursor])))
            cursor += 1
            if count < 0 or cursor + count > raw.size:
                raise ValueError("truncated Kimi-K3 external state tensor")
            expected = int(np.prod(shape, dtype=np.int64))
            if expected != count:
                raise ValueError("Kimi-K3 external state tensor shape/count mismatch")
            tensor = torch.from_numpy(raw[cursor : cursor + count].copy().reshape(shape))
            cursor += count
            if role == _ROLE_NEXT_LOGITS:
                if layer_id != -1:
                    raise ValueError("Kimi-K3 next-logits record has an invalid layer")
                next_logits = tensor.to(dtype=self._state_dtype())
            elif role in (_ROLE_CONV_Q, _ROLE_CONV_K, _ROLE_CONV_V):
                if not 0 <= layer_id < self._num_layers:
                    raise ValueError("Kimi-K3 convolution state layer is out of range")
                row = conv_by_layer.setdefault(layer_id, [None, None, None])
                row[role - _ROLE_CONV_Q] = tensor.to(dtype=self._cache_tensor_dtype(role))
            elif role == _ROLE_RECURRENT:
                if not 0 <= layer_id < self._num_layers:
                    raise ValueError("Kimi-K3 recurrent state layer is out of range")
                cache.recurrent_states[layer_id] = tensor.to(dtype=self._cache_tensor_dtype(role))
            elif role == _ROLE_KEY:
                if not 0 <= layer_id < self._num_layers:
                    raise ValueError("Kimi-K3 key-cache layer is out of range")
                cache.key_cache[layer_id] = tensor.to(dtype=self._cache_tensor_dtype(role))
            elif role == _ROLE_VALUE:
                if not 0 <= layer_id < self._num_layers:
                    raise ValueError("Kimi-K3 value-cache layer is out of range")
                cache.value_cache[layer_id] = tensor.to(dtype=self._cache_tensor_dtype(role))
            else:
                raise ValueError(f"unknown Kimi-K3 external state record role {role}")
        if cursor != raw.size or next_logits is None:
            raise ValueError("Kimi-K3 external state has trailing or missing data")
        for layer_id, values in conv_by_layer.items():
            if any(value is None for value in values):
                raise ValueError("Kimi-K3 external state is missing a convolution cache")
            cache.conv_states[layer_id] = tuple(values)  # type: ignore[assignment]
        return cache, next_logits, position

    def get_recurrent_state(self) -> RecurrentState | None:
        if self._cache is None or self._next_logits is None:
            return None
        payload = self._pack_live_state()
        state = RecurrentState(
            last_token_id=int(self._last_token_id), external_state=payload
        )
        self._state = state
        return state.clone()

    def set_recurrent_state(self, state: RecurrentState) -> None:
        if state.h is not None or state.rwkv7_state is not None or state.sequence_state is not None:
            raise ValueError("Kimi-K3 accepts only its external state representation")
        if state.external_state is None:
            raise ValueError("Kimi-K3 state payload is empty")
        if not 0 <= int(state.last_token_id) < self._vocab_size:
            raise ValueError(
                "Kimi-K3 state last_token_id outside vocabulary: "
                f"{state.last_token_id} (vocab_size={self._vocab_size})"
            )
        cache, logits, position = self._unpack_state(state.external_state)
        self._set_live_state(
            cache=cache,
            logits=logits,
            last_token_id=int(state.last_token_id),
            position=position,
        )
        self._state = state.clone()

    def probe_logits(self, state: RecurrentState | None = None) -> torch.Tensor | None:
        if state is None:
            return self._next_logits.detach().clone() if self._next_logits is not None else None
        if state.external_state is None:
            return None
        _cache, logits, _position = self._unpack_state(state.external_state)
        return logits.detach().clone()

    def _refresh_metrics(self, metrics: MetricsCollector) -> None:
        cache_bytes = 0 if self._cache is None else sum(
            _tensor_nbytes(tensor)
            for _role, _layer, tensor in self._cache_tensors(self._cache)
        )
        cache_bytes += _tensor_nbytes(self._next_logits)
        metrics.kv_cache_bytes = int(cache_bytes)
        metrics.weight_cache_bytes = int(self._weight_bytes)
        metrics.sequence_context_tokens = int(self._position)
        metrics.sequence_kernel = "kimi_cpu_reference"

    def _prefill_ids_live(
        self,
        token_ids: list[int],
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        continue_live: bool = False,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> None:
        control = make_generation_control(cancel_event=cancel_event, deadline=deadline)
        if control is not None:
            control.check()
        if not token_ids:
            token_ids = self._encode("")
        normalized_ids = [int(value) for value in token_ids]
        invalid = [value for value in normalized_ids if not 0 <= value < self._vocab_size]
        if invalid:
            raise ValueError(
                "Kimi-K3 token id outside vocabulary: "
                f"{invalid[0]} (vocab_size={self._vocab_size})"
            )
        base_position = 0
        cache: Any | None = None
        if initial_state is not None:
            self.set_recurrent_state(initial_state)
            base_position = self._position
            cache = self._cache
        elif continue_live and self._cache is not None:
            base_position = self._position
            cache = self._cache
        started = time.perf_counter()
        logits, cache = self._forward(
            normalized_ids,
            cache=cache,
        )
        self._set_live_state(
            cache=cache,
            logits=logits,
            last_token_id=normalized_ids[-1],
            position=base_position + len(normalized_ids),
        )
        metrics.prefill_wall_s += time.perf_counter() - started
        metrics.prompt_tokens = max(int(metrics.prompt_tokens), len(normalized_ids))
        self._refresh_metrics(metrics)
        if control is not None:
            control.check()

    def prefill_ids(
        self,
        token_ids: list[int],
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        self._prefill_ids_live(
            token_ids,
            metrics,
            initial_state=initial_state,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        state = self.get_recurrent_state()
        assert state is not None
        return state

    def prefill_text(
        self,
        text: str,
        provider: Any | None = None,
        metrics: MetricsCollector | None = None,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        del provider
        collector = metrics if metrics is not None else MetricsCollector()
        return self.prefill_ids(
            self._encode(text),
            collector,
            initial_state=initial_state,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def _prefill_text_live(
        self,
        text: str,
        *,
        metrics: MetricsCollector,
        continue_live: bool = False,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> None:
        self._prefill_ids_live(
            self._encode(text),
            metrics,
            continue_live=continue_live,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def _decode_live(
        self,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float,
        greedy: bool,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        if self._cache is None or self._next_logits is None:
            raise RuntimeError("Kimi-K3 has no recurrent state to decode")
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        generated: list[int] = []
        decode_started = time.perf_counter()
        try:
            for _ in range(max(0, int(max_tokens))):
                if control is not None:
                    control.check()
                token_started = time.perf_counter()
                token_id = sample_torch(
                    self._next_logits,
                    temperature=float(temperature) if not greedy else 0.0,
                    greedy=bool(greedy),
                )
                if not 0 <= int(token_id) < self._vocab_size:
                    raise RuntimeError(
                        f"Kimi-K3 sampler produced invalid vocabulary id {token_id}"
                    )
                generated.append(int(token_id))
                if control is not None:
                    control.emit(int(token_id))
                logits, cache = self._forward([int(token_id)], cache=self._cache)
                self._set_live_state(
                    cache=cache,
                    logits=logits,
                    last_token_id=int(token_id),
                    position=self._position + 1,
                )
                metrics.token_latencies_ms.append((time.perf_counter() - token_started) * 1000.0)
                if len(generated) == 1:
                    metrics.ttft_s = metrics.prefill_wall_s + (
                        time.perf_counter() - token_started
                    )
                if int(token_id) in self._eos_ids:
                    break
        finally:
            metrics.decode_wall_s += time.perf_counter() - decode_started
            metrics.tokens_generated = len(generated)
            self._refresh_metrics(metrics)
        return generated

    def decode_greedy(
        self,
        state: RecurrentState,
        provider: Any | None,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
        greedy: bool | None = None,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        del provider
        if self._cache is None or self._next_logits is None or state.external_state is not None:
            self.set_recurrent_state(state)
        return self._decode_live(
            max_tokens,
            metrics,
            temperature=float(temperature),
            greedy=bool(greedy) if greedy is not None else temperature <= 0.0,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_greedy_native(
        self,
        prompt: str,
        max_tokens: int,
        *,
        metrics: MetricsCollector,
        power_percent: int = 100,
        temperature: float = 0.0,
        greedy: bool = True,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        del power_percent
        # Generation does not need a serialized state until the caller asks
        # for a snapshot/prefix entry.  Avoid packing the entire MLA cache to
        # float32 immediately before every decode request.
        self._prefill_text_live(
            prompt,
            metrics=metrics,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        return self._decode_live(
            max_tokens,
            metrics,
            temperature=float(temperature),
            greedy=bool(greedy),
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_followup_native(
        self,
        suffix: str,
        max_tokens: int,
        *,
        metrics: MetricsCollector,
        temperature: float = 0.0,
        greedy: bool = True,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        if self._cache is None or self._next_logits is None:
            return self.generate_greedy_native(
                suffix,
                max_tokens,
                metrics=metrics,
                temperature=temperature,
                greedy=greedy,
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
        # Continue the live cache directly.  The public ``prefill_text`` path
        # still performs a full state transfer when a caller supplies an
        # explicit parked state; follow-up on the same engine need not do that
        # serialize/unpack round trip.
        self._prefill_text_live(
            suffix,
            metrics=metrics,
            continue_live=True,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        return self._decode_live(
            max_tokens,
            metrics,
            temperature=float(temperature),
            greedy=bool(greedy),
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    # Compatibility aliases used by generic backend callers and diagnostics.
    def generate(self, prompt: str, provider: Any, max_tokens: int, metrics: MetricsCollector, *, temperature: float = 0.0, greedy: bool = True, token_callback: Any | None = None, cancel_event: Any | None = None, deadline: float | None = None) -> list[int]:
        del provider
        return self.generate_greedy_native(
            prompt,
            max_tokens,
            metrics=metrics,
            temperature=temperature,
            greedy=greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_simple(
        self,
        prompt: str,
        max_tokens: int,
        *,
        greedy: bool = True,
        temperature: float = 0.0,
    ) -> str:
        metrics = MetricsCollector()
        return self.decode_text(
            self.generate_greedy_native(
                prompt,
                max_tokens,
                metrics=metrics,
                temperature=temperature,
                greedy=greedy,
            )
        )


__all__ = ["KimiK3CPUBackend"]
