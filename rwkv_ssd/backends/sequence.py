"""Pack-backed Mamba-2 and Hugging Face decoder-only backends.

The implementations deliberately use dense PyTorch operators.  The runtime
provider still owns residency, SSD reads, prefetching, and materialized codec
decoding, so the same backend can run resident, partial, or true streaming
without loading the original checkpoint into RAM.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from rwkv_ssd.backends.pack_backend import PackBackend
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.sampling import sample_torch
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.runtime.sequence_state import (
    MambaLayerState,
    SequenceState,
    TransformerKVState,
    validate_sequence_state,
)
from rwkv_ssd.runtime.state_cache import RecurrentState


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight.to(
        dtype=x.dtype
    )


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(
        x,
        (x.shape[-1],),
        weight.to(dtype=x.dtype),
        bias.to(dtype=x.dtype) if bias is not None else None,
        eps,
    )


def _linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # F.linear selects the backend's optimized addmm path and fuses the bias
    # add for the common one-token decode shape.
    return F.linear(
        x,
        weight.to(dtype=x.dtype),
        bias.to(dtype=x.dtype) if bias is not None else None,
    )


def _gpt2_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # GPT-2 Conv1D stores [in_features, out_features], unlike nn.Linear.
    if weight.ndim == 2 and weight.shape[0] == x.shape[-1]:
        return F.linear(
            x,
            weight.to(dtype=x.dtype).transpose(-1, -2),
            bias.to(dtype=x.dtype) if bias is not None else None,
        )
    else:
        return F.linear(
            x,
            weight.to(dtype=x.dtype),
            bias.to(dtype=x.dtype) if bias is not None else None,
        )


def _first(
    tensors: Mapping[str, torch.Tensor],
    *names: str,
    required: bool = True,
) -> torch.Tensor | None:
    for name in names:
        value = tensors.get(name)
        if value is not None:
            return value
    if required:
        raise ValueError(f"packed model is missing required tensor; tried {names!r}")
    return None


class _HFTokenizer:
    """Lazy, local-only tokenizer adapter.

    Model math does not depend on ``transformers``.  Text generation does,
    unless callers use the fixed-token-ID methods exposed by the backends.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._tokenizer: Any | None = None

    def _load(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "text generation for an HF pack requires the optional "
                "'transformers' package; install it in the active environment "
                "or use the backend's fixed-token-ID API"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.root),
                local_files_only=True,
                use_fast=True,
                trust_remote_code=False,
            )
        except Exception as exc:  # transformers raises several model-specific types
            raise RuntimeError(
                f"could not load a local tokenizer from {self.root}; ensure the "
                "pack contains tokenizer files and install compatible optional "
                "tokenizer dependencies"
            ) from exc
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        tokenizer = self._load()
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=True)]

    def decode(self, token_ids: list[int]) -> str:
        tokenizer = self._load()
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))

    def bos_id(self) -> int | None:
        value = getattr(self._load(), "bos_token_id", None)
        return int(value) if value is not None else None


class SequencePackBackend(PackBackend):
    """Shared pack/provider/state plumbing for dense sequence models."""

    sequence_kind = "sequence"
    supports_batch = False

    def __init__(self) -> None:
        self._manifest: Manifest | None = None
        self._config: dict[str, Any] = {}
        self._device = torch.device("cpu")
        self._layer_entries: dict[int, list[TensorEntry]] = {}
        self._global_entries: dict[int, list[TensorEntry]] = {}
        self._num_layers = 0
        self._state: RecurrentState | None = None
        self._globals: dict[str, torch.Tensor] | None = None
        # Resident layers are immutable after pack load.  Keeping their
        # already-resolved name -> tensor mapping avoids rebuilding the same
        # dictionary on every decode token while preserving streaming memory
        # bounds for middle layers.
        self._resident_layer_weights: dict[int, dict[str, torch.Tensor]] = {}
        self._tokenizer: _HFTokenizer | None = None

    def load(self, model_path: str, strategy: str, device: str) -> None:
        del model_path, strategy, device
        raise RuntimeError(
            f"{type(self).__name__} is pack-backed; load it through "
            "InferenceEngine with EngineConfig.pack_dir"
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def model_family(self) -> str:
        return str(self._manifest.model_family if self._manifest else self.sequence_kind)

    def _layer_id_for_name(self, name: str) -> int | None:
        raise NotImplementedError

    def _validate_pack_config(self) -> None:
        raise NotImplementedError

    def load_pack(self, manifest: Manifest, device: str) -> None:
        self._manifest = manifest
        self._device = torch.device(device)
        config_path = (manifest.pack_dir or manifest.weights_path.parent) / "config.json"
        if not config_path.is_file():
            raise ValueError(
                f"{type(self).__name__} requires a self-contained HF config.json in {config_path.parent}"
            )
        try:
            self._config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid HF config.json in {config_path}") from exc
        if not isinstance(self._config, dict):
            raise ValueError("HF config.json must contain an object")

        unsupported = {
            "trinity",
            "trinity_lut2",
            "trinity_layer",
        }
        bad = {
            str(entry.dequant).strip().lower()
            for entry in manifest.tensors
            if str(entry.dequant).strip().lower() in unsupported
        }
        if bad and not manifest.has_bf16_shadow():
            raise ValueError(
                f"{type(self).__name__} needs dense tensors; codecs {sorted(bad)!r} "
                "require a BF16 shadow or a repack with --pack-codec none/scale_u8/scale_u4"
            )

        self._layer_entries.clear()
        self._global_entries.clear()
        for entry in manifest.tensors:
            layer_id = self._layer_id_for_name(entry.name)
            if layer_id is None:
                self._global_entries.setdefault(entry.layer_id, []).append(entry)
            else:
                self._layer_entries.setdefault(layer_id, []).append(entry)
        configured = self._config.get("num_hidden_layers", self._config.get("n_layer"))
        if configured is not None:
            self._num_layers = int(configured)
        elif self._layer_entries:
            self._num_layers = max(self._layer_entries) + 1
        else:
            raise ValueError("packed sequence model has no numbered layer tensors")
        missing = [layer for layer in range(self._num_layers) if layer not in self._layer_entries]
        if missing:
            raise ValueError(f"packed sequence model is missing layer tensors for {missing[:8]!r}")
        self._validate_pack_config()
        root = manifest.pack_dir or manifest.weights_path.parent
        self._tokenizer = _HFTokenizer(root)
        self._globals = None
        self._resident_layer_weights.clear()
        self._state = None

    def refresh_manifest(self, manifest: Manifest) -> None:
        """Refresh entry residency after the engine applies its policy.

        ``InferenceEngine.load`` discovers the backend layer count before it
        applies resident/partial/streaming overrides.  Sequence backends keep
        an entry index for layer-outer execution, so rebuild that index after
        the policy rather than retaining pre-policy ``TensorEntry`` objects.
        """
        self._manifest = manifest
        self._layer_entries.clear()
        self._global_entries.clear()
        for entry in manifest.tensors:
            layer_id = self._layer_id_for_name(entry.name)
            if layer_id is None:
                self._global_entries.setdefault(entry.layer_id, []).append(entry)
            else:
                self._layer_entries.setdefault(layer_id, []).append(entry)
        if len(self._layer_entries) != self._num_layers:
            raise ValueError("residency refresh changed the sequence layer index")
        self._globals = None
        self._resident_layer_weights.clear()

    def _load_global_tensors(self, provider: Any, metrics: MetricsCollector) -> dict[str, torch.Tensor]:
        if self._globals is not None:
            return self._globals
        out: dict[str, torch.Tensor] = {}
        for layer_id in sorted(self._global_entries):
            entries = self._global_entries[layer_id]
            out.update(provider.load_layer_tensors_dense(entries))
        self._globals = out
        return out

    def _load_layer_tensors(
        self,
        layer_id: int,
        provider: Any,
        metrics: MetricsCollector,
    ) -> dict[str, torch.Tensor]:
        entries = self._layer_entries[layer_id]
        if layer_id in self._resident_layer_weights:
            return self._resident_layer_weights[layer_id]
        if layer_id + 1 in self._layer_entries:
            try:
                provider.prefetch_layer(self._layer_entries[layer_id + 1])
            except (OSError, RuntimeError, ValueError):
                # Prefetch is an optimization and must not change correctness.
                pass
        metrics.weight_layer_loads += 1
        weights = provider.load_layer_tensors_dense(entries)
        # ``resident`` is a manifest property, not merely a provider mode:
        # partial/streaming runs intentionally keep edge/global layers hot.
        # Cache only an all-resident layer so a streamed layer can still be
        # reclaimed by the provider's bounded cache.
        if entries and all(entry.residency == "resident" for entry in entries):
            self._resident_layer_weights[layer_id] = weights
        return weights

    def _global(
        self,
        provider: Any,
        metrics: MetricsCollector,
        *names: str,
        required: bool = True,
    ) -> torch.Tensor | None:
        return _first(self._load_global_tensors(provider, metrics), *names, required=required)

    def _encode(self, text: str) -> list[int]:
        assert self._tokenizer is not None
        ids = self._tokenizer.encode(text)
        if ids:
            return ids
        bos = self._tokenizer.bos_id()
        if bos is None:
            raise ValueError("prompt encoded to zero tokens and the tokenizer has no BOS token")
        return [bos]

    def _decode(self, token_ids: list[int]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.decode(token_ids)

    def decode_text(self, token_ids: list[int]) -> str:
        return self._decode(token_ids)

    def get_recurrent_state(self) -> RecurrentState | None:
        return self._state.clone() if self._state is not None else None

    def probe_logits(self, state: RecurrentState | None = None) -> torch.Tensor | None:
        """Return a detached next-token logit vector for parity diagnostics."""
        current = state if state is not None else self._state
        if current is None or current.sequence_state is None:
            return None
        logits = current.sequence_state.next_logits
        return logits.detach().clone() if logits is not None else None

    def set_recurrent_state(self, state: RecurrentState) -> None:
        if state.h is not None or state.rwkv7_state is not None or state.external_state is not None:
            raise ValueError(f"{type(self).__name__} accepts only sequence_state snapshots")
        if state.sequence_state is None:
            raise ValueError("sequence backend requires a sequence_state")
        validate_sequence_state(state.sequence_state, kind=self.sequence_kind)
        if len(state.sequence_state.layers) != self._num_layers:
            raise ValueError(
                f"sequence state has {len(state.sequence_state.layers)} layers; expected {self._num_layers}"
            )
        self._state = state.clone()

    def close(self) -> None:
        self._globals = None
        self._resident_layer_weights.clear()
        self._state = None

    def prefill_text(
        self,
        text: str,
        provider: Any,
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        return self.prefill_ids(
            self._encode(text),
            provider,
            metrics,
            initial_state=initial_state,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def prefill_ids(
        self,
        token_ids: list[int],
        provider: Any,
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        raise NotImplementedError

    def _step_state(
        self,
        state: RecurrentState,
        token_id: int,
        provider: Any,
        metrics: MetricsCollector,
        *,
        clone_state: bool = True,
    ) -> RecurrentState:
        raise NotImplementedError

    def decode_ids(
        self,
        state: RecurrentState,
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
        greedy: bool = True,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        if state.sequence_state is None or state.sequence_state.next_logits is None:
            raise ValueError("cannot decode a sequence state without next-token logits")
        validate_sequence_state(state.sequence_state, kind=self.sequence_kind)
        current = state.clone()
        assert current.sequence_state is not None
        if current.sequence_state.batch_size != 1:
            raise ValueError("single-session decode received a batched sequence state")
        generated: list[int] = []
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        metrics.sequence_kernel = str(getattr(self, "_sequence_kernel", "torch"))
        decode_started = time.perf_counter()
        try:
            for _ in range(max(0, int(max_tokens))):
                if control is not None:
                    control.check()
                token_started = time.perf_counter()
                logits = current.sequence_state.next_logits
                assert logits is not None
                token_id = sample_torch(
                    logits,
                    temperature=temperature,
                    greedy=greedy,
                )
                generated.append(token_id)
                if control is not None:
                    control.emit(token_id)
                # ``current`` is already an exclusive clone of the caller's
                # state.  Avoid cloning every layer/KV buffer again on every
                # generated token; this changes Transformer continuation from
                # quadratic cache copying to append-only KV growth.
                current = self._step_state(
                    current, token_id, provider, metrics, clone_state=False
                )
                token_ms = (time.perf_counter() - token_started) * 1000.0
                metrics.token_latencies_ms.append(token_ms)
                if len(generated) == 1 and metrics.ttft_s <= 0.0:
                    metrics.ttft_s = token_ms / 1000.0
        finally:
            # Keep timing and partial-token accounting visible when a deadline
            # or cancellation interrupts decode.  Follow-up/prefix paths may
            # invoke this method more than once, so timings are additive.
            metrics.decode_wall_s += time.perf_counter() - decode_started
            self._state = current
            metrics.tokens_generated = len(generated)
            self._record_sequence_metrics(current, metrics)
        return generated

    def decode_greedy(
        self,
        state: RecurrentState,
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
        temperature: float = 0.0,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        return self.decode_ids(
            state,
            provider,
            max_tokens,
            metrics,
            temperature=temperature,
            greedy=temperature <= 0,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_greedy(
        self,
        prompt: str,
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[int]:
        state = self.prefill_text(prompt, provider, metrics)
        return self.decode_ids(state, provider, max_tokens, metrics)

    def generate_greedy_ids(
        self,
        token_ids: list[int],
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[int]:
        state = self.prefill_ids(token_ids, provider, metrics)
        return self.decode_ids(state, provider, max_tokens, metrics)

    def _record_sequence_metrics(self, state: RecurrentState, metrics: MetricsCollector) -> None:
        if state.sequence_state is None:
            return
        metrics.weight_cache_bytes = max(metrics.weight_cache_bytes, 0)
        metrics.sequence_context_tokens = state.sequence_state.context_length
        if self.sequence_kind == "mamba2":
            metrics.mamba_state_bytes = state.sequence_state.nbytes()
        else:
            metrics.kv_cache_bytes = state.sequence_state.nbytes()


class Mamba2PackBackend(SequencePackBackend):
    sequence_kind = "mamba2"
    supports_batch = True

    def _layer_id_for_name(self, name: str) -> int | None:
        match = re.match(r"^backbone\.layers\.(\d+)\.", name)
        return int(match.group(1)) if match else None

    def _validate_pack_config(self) -> None:
        model_type = str(self._config.get("model_type", "")).lower()
        family = str(self._manifest.model_family if self._manifest else "").lower()
        if model_type not in {"mamba2", "mamba_2"} and family not in {"mamba2", "mamba"}:
            raise ValueError(f"Mamba2PackBackend cannot load model_type={model_type!r}")
        self.hidden_size = int(self._config["hidden_size"])
        self.d_inner = int(
            self._config.get("intermediate_size", self.hidden_size * int(self._config.get("expand", 2)))
        )
        self.state_size = int(self._config["state_size"])
        self.n_heads = int(self._config["num_heads"])
        self.head_dim = int(self._config["head_dim"])
        self.n_groups = int(self._config["n_groups"])
        self.conv_kernel = int(self._config["conv_kernel"])
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.state_size
        self.eps = float(self._config.get("layer_norm_epsilon", 1e-5))
        self._residual_in_fp32 = bool(self._config.get("residual_in_fp32", False))
        self._time_step_floor = float(
            self._config.get(
                "time_step_floor",
                self._config.get("time_step_limit", [0.0, float("inf")])[0]
                if isinstance(self._config.get("time_step_limit"), (list, tuple))
                else 0.0,
            )
        )
        configured_time_max = self._config.get("time_step_max")
        if configured_time_max is None:
            limit = self._config.get("time_step_limit")
            configured_time_max = (
                limit[1] if isinstance(limit, (list, tuple)) and len(limit) > 1 else float("inf")
            )
        self._time_step_max = float(configured_time_max)
        # The grouped form changes floating-point reduction order.  Keep the
        # reference recurrence as the default so packed logits retain the
        # existing exact/tolerance contract; callers can opt into the
        # experimental grouped kernel explicitly after validating its dtype.
        grouped_raw = os.environ.get("RWKV_MAMBA_GROUPED_SSM", "0").strip().lower()
        self._grouped_ssm = grouped_raw not in {"0", "false", "off", "no"}
        # The direct recurrence avoids a general conv1d launch per token.  It
        # accumulates BF16/FP16 taps in float32 before restoring activation
        # dtype, matching the reference kernel's numerical contract.
        direct_conv_raw = os.environ.get("RWKV_MAMBA_DIRECT_CONV", "1").strip().lower()
        self._direct_conv = direct_conv_raw not in {"0", "false", "off", "no"}
        self._sequence_kernel = (
            "torch_grouped_ssm_direct_conv"
            if self._grouped_ssm and self._direct_conv
            else "torch_grouped_ssm"
            if self._grouped_ssm
            else "torch_direct_conv"
            if self._direct_conv
            else "torch"
        )
        if self.n_heads * self.head_dim != self.d_inner:
            raise ValueError("Mamba-2 config has inconsistent num_heads * head_dim")
        if self.n_heads % self.n_groups:
            raise ValueError("Mamba-2 requires num_heads divisible by n_groups")

    def _initial_state(self, provider: Any, metrics: MetricsCollector, batch: int = 1) -> SequenceState:
        emb = self._global(provider, metrics, "backbone.embeddings.weight", "model.embed_tokens.weight")
        assert emb is not None
        dtype = emb.dtype
        layers = [
            MambaLayerState(
                torch.zeros(batch, self.conv_dim, self.conv_kernel - 1, dtype=dtype, device=self._device),
                torch.zeros(
                    batch,
                    self.n_heads,
                    self.state_size,
                    self.head_dim,
                    dtype=torch.float32,
                    device=self._device,
                ),
            )
            for _ in range(self._num_layers)
        ]
        return SequenceState(
            kind="mamba2",
            position=0,
            layers=layers,
            batch_size=batch,
            next_logits=None,
        )

    def _mamba_layer_step(
        self,
        layer_id: int,
        x: torch.Tensor,
        previous: MambaLayerState,
        weights: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, MambaLayerState]:
        prefix = f"backbone.layers.{layer_id}."
        hidden = _linear(x, _first(weights, prefix + "mixer.in_proj.weight"))
        conv_input, gate, dt_input = torch.split(
            hidden, [self.conv_dim, self.d_inner, self.n_heads], dim=-1
        )
        conv_weight = _first(weights, prefix + "mixer.conv1d.weight")
        conv_bias = _first(weights, prefix + "mixer.conv1d.bias", required=False)
        assert conv_weight is not None
        conv_input = conv_input.to(dtype=conv_weight.dtype)
        if self._direct_conv:
            # Only the newest depthwise-convolution sample is needed for a
            # recurrent step.  Avoid constructing ``window`` and launching a
            # general conv1d kernel for every token/layer.
            kernel = conv_weight.reshape(self.conv_dim, -1)
            conv_dtype = (
                torch.float32
                if conv_input.dtype in {torch.float16, torch.bfloat16}
                else conv_input.dtype
            )
            conv = (
                previous.conv.to(dtype=conv_dtype)
                * kernel[:, :-1].to(dtype=conv_dtype).unsqueeze(0)
            ).sum(dim=-1) + conv_input.to(dtype=conv_dtype) * kernel[:, -1].to(dtype=conv_dtype).unsqueeze(0)
            if conv_bias is not None:
                conv = conv + conv_bias.to(dtype=conv_dtype)
            conv = conv.to(dtype=conv_input.dtype)
            next_conv = torch.cat((previous.conv[..., 1:], conv_input.unsqueeze(-1)), dim=-1)
        else:
            window = torch.cat((previous.conv, conv_input.unsqueeze(-1)), dim=-1)
            conv = F.conv1d(window, conv_weight, conv_bias, groups=self.conv_dim)[:, :, -1]
            next_conv = window[:, :, 1:]
        conv = F.silu(conv)
        u = conv[:, : self.d_inner]
        b = conv[:, self.d_inner : self.d_inner + self.n_groups * self.state_size]
        c = conv[:, self.d_inner + self.n_groups * self.state_size :]
        b = b.view(x.shape[0], self.n_groups, self.state_size)
        c = c.view(x.shape[0], self.n_groups, self.state_size)
        dt_bias = _first(weights, prefix + "mixer.dt_bias")
        a_log = _first(weights, prefix + "mixer.A_log")
        d = _first(weights, prefix + "mixer.D")
        assert dt_bias is not None and a_log is not None and d is not None
        dt = F.softplus(dt_input + dt_bias.to(dtype=dt_input.dtype))
        dt = dt.clamp_min(self._time_step_floor)
        if self._time_step_max > 0:
            dt = dt.clamp_max(self._time_step_max)
        a = -torch.exp(a_log.float()).view(1, self.n_heads)
        discrete_a = torch.exp(dt.float() * a)
        u_head = u.view(x.shape[0], self.n_heads, self.head_dim)
        ssm = discrete_a[:, :, None, None] * previous.ssm.float()
        per_group = self.n_heads // self.n_groups
        if self._grouped_ssm and self.n_groups < self.n_heads:
            # Avoid materializing repeated B/C tensors for GQA-like Mamba
            # groups.  The grouped view has the same head ordering as
            # repeat_interleave but leaves the broadcasted group dimension
            # virtual until the elementwise update/reduction.
            grouped = ssm.view(
                x.shape[0], self.n_groups, per_group, self.state_size, self.head_dim
            )
            grouped = grouped + (
                dt.float().view(x.shape[0], self.n_groups, per_group, 1, 1)
                * b[:, :, None, :, None]
                * u.view(x.shape[0], self.n_groups, per_group, 1, self.head_dim)
            )
            # Keep the updated grouped state, not just the temporary output
            # view.  Returning the decayed pre-input ``ssm`` here makes the
            # experimental grouped path silently forget every token after
            # the first one.
            ssm = grouped.reshape(x.shape[0], self.n_heads, self.state_size, self.head_dim)
            y = (
                grouped * c[:, :, None, :, None]
            ).sum(dim=3).reshape(x.shape[0], self.n_heads, self.head_dim)
        else:
            b_head = b.repeat_interleave(per_group, dim=1)
            c_head = c.repeat_interleave(per_group, dim=1)
            ssm = ssm + dt.float()[:, :, None, None] * b_head[:, :, :, None] * u_head[:, :, None, :]
            y = (ssm * c_head[:, :, :, None]).sum(dim=2)
        y = y + d.float().view(1, self.n_heads, 1) * u_head.float()
        y = y.reshape(x.shape[0], self.d_inner).to(dtype=x.dtype)
        norm_weight = _first(weights, prefix + "mixer.norm.weight")
        out_weight = _first(weights, prefix + "mixer.out_proj.weight")
        assert norm_weight is not None and out_weight is not None
        y = _rms_norm(y, norm_weight, self.eps)
        gated = y * F.silu(gate.to(dtype=y.dtype))
        if bool(self._config.get("norm_before_gate", True)):
            y = gated
        else:
            y = _rms_norm(gated, norm_weight, self.eps)
        y = _linear(y, out_weight)
        return y, MambaLayerState(next_conv.detach(), ssm.detach())

    def _mamba_block_step(
        self,
        layer_id: int,
        x: torch.Tensor,
        previous: MambaLayerState,
        weights: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, MambaLayerState]:
        prefix = f"backbone.layers.{layer_id}."
        norm = _first(weights, prefix + "norm.weight")
        assert norm is not None
        # Hugging Face Mamba-2 keeps the residual stream in FP32 when the
        # checkpoint requests it, while the mixer itself still consumes the
        # norm-weight dtype.  This is the recurrent analogue of keeping RWKV
        # state/update accumulation in a wider dtype.
        residual = x.float() if self._residual_in_fp32 else x
        y, next_state = self._mamba_layer_step(
            layer_id, _rms_norm(x.to(dtype=norm.dtype), norm, self.eps), previous, weights
        )
        return residual + y, next_state

    def _mamba_logits(
        self,
        hidden: torch.Tensor,
        provider: Any,
        metrics: MetricsCollector,
    ) -> torch.Tensor:
        norm = self._global(provider, metrics, "backbone.norm_f.weight", "model.norm.weight")
        head = self._global(provider, metrics, "lm_head.weight", "head.weight", "output.weight", required=False)
        emb = self._global(provider, metrics, "backbone.embeddings.weight", "model.embed_tokens.weight")
        assert norm is not None and emb is not None
        if head is None:
            head = emb
        return _linear(_rms_norm(hidden.to(dtype=norm.dtype), norm, self.eps), head)

    def _coerce_initial_state(
        self,
        initial_state: RecurrentState | None,
        provider: Any,
        metrics: MetricsCollector,
    ) -> SequenceState:
        if initial_state is None:
            return self._initial_state(provider, metrics)
        if initial_state.sequence_state is None:
            raise ValueError("Mamba prefill received a non-sequence initial state")
        validate_sequence_state(initial_state.sequence_state, kind="mamba2")
        current = initial_state.sequence_state.clone()
        if len(current.layers) != self._num_layers or current.batch_size != 1:
            raise ValueError("Mamba initial state has incompatible layer or batch shape")
        return current

    def prefill_ids(
        self,
        token_ids: list[int],
        provider: Any,
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        control = make_generation_control(
            cancel_event=cancel_event,
            deadline=deadline,
        )
        started = time.perf_counter()
        current: SequenceState | None = None
        try:
            if control is not None:
                control.check()
            current = self._coerce_initial_state(initial_state, provider, metrics)
            if not token_ids:
                if current.next_logits is None:
                    raise ValueError("cannot prefill an empty Mamba prompt without an existing state")
                return RecurrentState(last_token_id=initial_state.last_token_id if initial_state else 0, sequence_state=current)
            emb = self._global(provider, metrics, "backbone.embeddings.weight", "model.embed_tokens.weight")
            assert emb is not None
            ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
            hidden_seq = emb.index_select(0, ids)
            for layer_id in range(self._num_layers):
                if control is not None:
                    control.check()
                metrics.weight_sweeps += 1
                weights = self._load_layer_tensors(layer_id, provider, metrics)
                # Reuse one output buffer per layer.  Building a Python list and
                # concatenating every prompt layer creates an extra copy and makes
                # prefill cost grow with prompt length.
                next_hidden = torch.empty(
                    hidden_seq.shape,
                    dtype=torch.float32 if self._residual_in_fp32 else hidden_seq.dtype,
                    device=hidden_seq.device,
                )
                layer_state = current.layers[layer_id]
                assert isinstance(layer_state, MambaLayerState)
                for t in range(hidden_seq.shape[0]):
                    if control is not None:
                        control.check()
                    value, layer_state = self._mamba_block_step(
                        layer_id, hidden_seq[t : t + 1], layer_state, weights
                    )
                    next_hidden[t : t + 1] = value
                current.layers[layer_id] = layer_state
                hidden_seq = next_hidden
            logits = self._mamba_logits(hidden_seq[-1:], provider, metrics)
            current.position += len(token_ids)
            current.next_logits = logits.detach()
            state = RecurrentState(last_token_id=int(token_ids[-1]), sequence_state=current)
            self._record_sequence_metrics(state, metrics)
            return state
        finally:
            metrics.prefill_wall_s += time.perf_counter() - started

    def _step_state(
        self,
        state: RecurrentState,
        token_id: int,
        provider: Any,
        metrics: MetricsCollector,
        *,
        clone_state: bool = True,
    ) -> RecurrentState:
        assert state.sequence_state is not None
        current = state.sequence_state.clone() if clone_state else state.sequence_state
        emb = self._global(provider, metrics, "backbone.embeddings.weight", "model.embed_tokens.weight")
        assert emb is not None
        hidden = emb[int(token_id)].reshape(1, -1)
        for layer_id in range(self._num_layers):
            metrics.weight_sweeps += 1
            weights = self._load_layer_tensors(layer_id, provider, metrics)
            layer_state = current.layers[layer_id]
            assert isinstance(layer_state, MambaLayerState)
            hidden, layer_state = self._mamba_block_step(layer_id, hidden, layer_state, weights)
            current.layers[layer_id] = layer_state
        current.position += 1
        current.next_logits = self._mamba_logits(hidden, provider, metrics).detach()
        state.last_token_id = int(token_id)
        state.sequence_state = current
        return state

    def generate_greedy_batch(
        self,
        prompts: list[str],
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        return self.generate_greedy_ids_batch(
            [self._encode(prompt) for prompt in prompts], provider, max_tokens, metrics
        )

    def generate_greedy_ids_batch(
        self,
        token_batches: list[list[int]],
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        if not token_batches:
            return []
        if any(not ids for ids in token_batches):
            raise ValueError("Mamba batch prompts must contain at least one token")
        batch = len(token_batches)
        batch_prefill_started = time.perf_counter()
        emb = self._global(provider, metrics, "backbone.embeddings.weight", "model.embed_tokens.weight")
        assert emb is not None
        lengths = [len(ids) for ids in token_batches]
        max_length = max(lengths)
        hidden_sequences = torch.zeros(
            batch,
            max_length,
            self.hidden_size,
            dtype=torch.float32 if self._residual_in_fp32 else emb.dtype,
            device=self._device,
        )
        for session_id, ids in enumerate(token_batches):
            token_tensor = torch.tensor(ids, dtype=torch.long, device=self._device)
            hidden_sequences[session_id, : len(ids)] = emb.index_select(0, token_tensor)

        # Capture each session's state at its own final prompt token while the
        # layer is still resident.  The padded tail is allowed to run through
        # the vectorized kernel, but is never exposed as a session state.
        finish_at: dict[int, list[int]] = {}
        for session_id, length in enumerate(lengths):
            finish_at.setdefault(length - 1, []).append(session_id)
        session_layers: list[list[MambaLayerState]] = [[] for _ in token_batches]
        metrics.batch_size = batch
        for layer_id in range(self._num_layers):
            metrics.weight_sweeps += 1
            weights = self._load_layer_tensors(layer_id, provider, metrics)
            layer_state = MambaLayerState(
                torch.zeros(
                    batch,
                    self.conv_dim,
                    self.conv_kernel - 1,
                    dtype=emb.dtype,
                    device=self._device,
                ),
                torch.zeros(
                    batch,
                    self.n_heads,
                    self.state_size,
                    self.head_dim,
                    dtype=torch.float32,
                    device=self._device,
                ),
            )
            outputs = torch.empty_like(hidden_sequences)
            for t in range(max_length):
                value, layer_state = self._mamba_block_step(
                    layer_id, hidden_sequences[:, t, :], layer_state, weights
                )
                outputs[:, t, :] = value
                for session_id in finish_at.get(t, ()):
                    session_layers[session_id].append(
                        MambaLayerState(
                            layer_state.conv[session_id : session_id + 1].clone(),
                            layer_state.ssm[session_id : session_id + 1].clone(),
                        )
                    )
            hidden_sequences = outputs

        batch_layers = [
            MambaLayerState(
                torch.cat([session_layers[i][layer_id].conv for i in range(batch)], dim=0),
                torch.cat([session_layers[i][layer_id].ssm for i in range(batch)], dim=0),
            )
            for layer_id in range(self._num_layers)
        ]
        last_indices = torch.tensor(
            [length - 1 for length in lengths], dtype=torch.long, device=self._device
        )
        last_hidden = hidden_sequences[torch.arange(batch, device=self._device), last_indices]
        logits = self._mamba_logits(last_hidden, provider, metrics).detach()
        metrics.batch_prefill_wall_s = time.perf_counter() - batch_prefill_started
        metrics.prefill_wall_s = metrics.batch_prefill_wall_s
        generated = [[] for _ in token_batches]
        batch_decode_started = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            chosen = torch.argmax(logits, dim=-1)
            for index, token in enumerate(chosen.tolist()):
                generated[index].append(int(token))
            hidden = emb.index_select(0, chosen).reshape(batch, self.hidden_size)
            for layer_id in range(self._num_layers):
                metrics.weight_sweeps += 1
                weights = self._load_layer_tensors(layer_id, provider, metrics)
                hidden, batch_layers[layer_id] = self._mamba_block_step(
                    layer_id, hidden, batch_layers[layer_id], weights
                )
            logits = self._mamba_logits(hidden, provider, metrics).detach()
        metrics.batch_decode_wall_s = time.perf_counter() - batch_decode_started
        metrics.decode_wall_s = metrics.batch_decode_wall_s
        metrics.tokens_generated = len(generated) * batch
        metrics.sequence_kernel = str(getattr(self, "_sequence_kernel", "torch"))
        if batch_layers:
            summary = SequenceState(
                kind="mamba2",
                position=max(lengths) + max(0, int(max_tokens)),
                layers=batch_layers,
                batch_size=batch,
                next_logits=logits,
            )
            self._record_sequence_metrics(
                RecurrentState(
                    last_token_id=(int(chosen[-1].item()) if max_tokens > 0 else 0),
                    sequence_state=summary,
                ),
                metrics,
            )
        return generated


class HFTransformerPackBackend(SequencePackBackend):
    """Dense PyTorch backend for common decoder-only HF checkpoints."""

    sequence_kind = "transformer"

    def __init__(self) -> None:
        super().__init__()
        self._rope_tables: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._sdpa_enabled = False
        self._sequence_kernel = "torch"

    def _layer_id_for_name(self, name: str) -> int | None:
        for pattern in (
            r"^model\.layers\.(\d+)\.",
            r"^model\.model\.layers\.(\d+)\.",
            r"^language_model\.model\.layers\.(\d+)\.",
            r"^model\.language_model\.model\.layers\.(\d+)\.",
            r"^transformer\.h\.(\d+)\.",
        ):
            match = re.match(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def _validate_pack_config(self) -> None:
        model_type = str(self._config.get("model_type", "")).lower()
        family = str(self._manifest.model_family if self._manifest else "").lower()
        if model_type in {"kimi_k3", "kimi_linear"} or family in {
            "kimi_k3",
            "kimi_linear",
        }:
            self.capability_error(
                "model_architecture",
                "Kimi-K3 uses custom KDA/MLA attention and sparse MoE blocks; "
                "the packed dense Transformer backend does not implement this "
                "architecture. Use the local Transformers trust_remote_code "
                "reference path or add a Kimi-specific adapter.",
            )
        self.style = "gpt2" if model_type in {"gpt2", "gpt_neo", "gptj"} or family == "gpt2" else "llama"
        if self.style == "llama":
            supported = {"llama", "mistral", "qwen2", "qwen3", "transformer", "hf_transformer"}
            if model_type not in supported and family not in supported:
                raise ValueError(
                    f"HFTransformerPackBackend does not recognize model_type={model_type!r}; "
                    "supported families are Llama/Mistral/Qwen-style or GPT-2-style"
                )
            self.hidden_size = int(self._config["hidden_size"])
            self.n_head = int(self._config["num_attention_heads"])
            self.n_kv_head = int(self._config.get("num_key_value_heads", self.n_head))
            self.head_dim = int(self._config.get("head_dim", self.hidden_size // self.n_head))
            self.intermediate_size = int(self._config["intermediate_size"])
            self.eps = float(self._config.get("rms_norm_eps", 1e-6))
            rope = self._config.get("rope_parameters", {}) or {}
            self.rope_theta = float(rope.get("rope_theta", self._config.get("rope_theta", 10000.0)))
            rope_scaling = self._config.get("rope_scaling", rope.get("rope_scaling"))
            # Newer Transformers configs put the scaling type/factor directly
            # in ``rope_parameters`` instead of a nested ``rope_scaling``
            # object.  Treat both layouts identically.
            if rope_scaling is None and str(rope.get("rope_type", "default")).lower() != "default":
                rope_scaling = rope
            self.rope_scale = 1.0
            if isinstance(rope_scaling, dict):
                rope_type = str(
                    rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
                ).lower()
                if rope_type == "linear":
                    self.rope_scale = float(rope_scaling.get("factor", 1.0))
                elif rope_type not in {"default", "none"}:
                    raise ValueError(
                        f"unsupported RoPE scaling type {rope_type!r}; "
                        "repack with default/linear RoPE or add an adapter"
                    )
            self.sliding_window = int(self._config.get("sliding_window", 0) or 0)
            self.max_position = int(self._config.get("max_position_embeddings", 0) or 0)
            sample_name = next(iter(self._layer_entries[0]))
            if sample_name.name.startswith("model.model.layers."):
                self.layer_prefix = "model.model.layers"
            else:
                self.layer_prefix = "model.layers"
        else:
            self.hidden_size = int(self._config.get("n_embd", self._config.get("hidden_size")))
            self.n_head = int(self._config.get("n_head", self._config.get("num_attention_heads")))
            self.n_kv_head = self.n_head
            self.head_dim = self.hidden_size // self.n_head
            self.intermediate_size = int(self._config.get("n_inner", 4 * self.hidden_size))
            self.eps = float(self._config.get("layer_norm_epsilon", 1e-5))
            self.rope_theta = 10000.0
            self.sliding_window = 0
            self.max_position = int(self._config.get("n_positions", self._config.get("max_position_embeddings", 0)) or 0)
        if self.n_head <= 0 or self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            raise ValueError("invalid Transformer attention head configuration")
        if self.head_dim % 2:
            raise ValueError("Transformer RoPE requires an even head_dim")
        if self.n_head * self.head_dim != self.hidden_size:
            raise ValueError("Transformer config has inconsistent hidden_size/head_dim")
        # SDPA is an opt-in accelerator.  Its backend-specific accumulation
        # order can differ from the float32 reference attention enough to
        # change a greedy token on small-margin logits.
        sdpa_raw = os.environ.get("RWKV_SEQUENCE_SDPA", "auto").strip().lower()
        self._sdpa_enabled = (
            hasattr(F, "scaled_dot_product_attention")
            and sdpa_raw in {"1", "true", "on", "yes"}
        )
        if self._sdpa_enabled and self.n_kv_head == self.n_head:
            # SDPA is used for multi-token prefill; one-token decode remains
            # on the validated manual path in _attention.
            self._sequence_kernel = "sdpa_prefill_manual_decode"
        elif self.n_kv_head != self.n_head:
            # Grouped GQA/MQA attention intentionally uses the no-repeat
            # manual kernel; the generic SDPA API would expand or require an
            # optional enable_gqa implementation.
            self._sequence_kernel = "torch_gqa"
        else:
            self._sequence_kernel = "torch"
        self._rope_tables.clear()
        batch_raw = os.environ.get(
            "RWKV_TRANSFORMER_BATCH",
            self._config.get("rwkv_ssd_batch", self._config.get("supports_batch", False)),
        )
        if isinstance(batch_raw, str):
            self.supports_batch = batch_raw.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self.supports_batch = bool(batch_raw)

    def _initial_state(
        self,
        provider: Any,
        metrics: MetricsCollector,
        batch: int = 1,
    ) -> SequenceState:
        emb = self._global(
            provider,
            metrics,
            "model.embed_tokens.weight",
            "model.model.embed_tokens.weight",
            "backbone.embeddings.weight",
            "transformer.wte.weight",
        )
        dtype = emb.dtype if emb is not None else torch.float32
        batch = max(1, int(batch))
        layers = [
            TransformerKVState(
                torch.empty(batch, self.n_kv_head, 0, self.head_dim, dtype=dtype, device=self._device),
                torch.empty(batch, self.n_kv_head, 0, self.head_dim, dtype=dtype, device=self._device),
            )
            for _ in range(self._num_layers)
        ]
        return SequenceState(kind="transformer", position=0, layers=layers, batch_size=batch, context_limit=self.max_position or None)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        half = self.head_dim // 2
        if positions.numel() == 0:
            return x
        needed = int(positions.max().item()) + 1
        key = (str(x.device), x.dtype)
        cached = self._rope_tables.get(key)
        if cached is None or cached[0].shape[0] < needed:
            current = int(cached[0].shape[0]) if cached is not None else 0
            capacity = max(needed, self.max_position or 0, max(1, current * 2))
            inv = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half)
            )
            table_positions = torch.arange(
                capacity, dtype=torch.float32, device=x.device
            )
            angles = (table_positions / float(self.rope_scale))[:, None] * inv[None, :]
            cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype=x.dtype)
            sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype=x.dtype)
            cached = (cos, sin)
            self._rope_tables[key] = cached
        cos = cached[0].index_select(0, positions.to(device=x.device, dtype=torch.long))
        sin = cached[1].index_select(0, positions.to(device=x.device, dtype=torch.long))
        first, second = x[..., :half], x[..., half : 2 * half]
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]

    def _attention(
        self,
        q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        grouped_gqa = self.n_kv_head != self.n_head
        repeat = self.n_head // self.n_kv_head if grouped_gqa else 1
        if grouped_gqa:
            # Keep K/V grouped for GQA/MQA.  Repeating the cache to
            # ``n_head`` materializes a large tensor on every decode step and
            # is especially expensive for long contexts.  The reference math
            # below broadcasts the KV head over its query-head group instead.
            q = q.reshape(q.shape[0], self.n_kv_head, repeat, q.shape[-2], q.shape[-1])
        scale = self.head_dim**-0.5
        query_length = int(q.shape[-2])
        # A sliding-window cache only retains the tail, so its first key is
        # not necessarily position zero.  Derive the absolute key offset
        # from the known end position instead of masking a compacted cache as
        # though it still contained the original prefix.
        key_start = max(0, int(position) + query_length - int(full_k.shape[-2]))
        # Decode appends one token at a time.  At that point every cached key
        # is causal by construction; avoid rebuilding an O(context) boolean
        # mask for the common no-window path.  A configured sliding window can
        # use a bounded tail with the same result.
        if q.shape[-2] == 1:
            if self.sliding_window > 0 and full_k.shape[-2] > self.sliding_window:
                full_k = full_k[..., -self.sliding_window :, :]
                full_v = full_v[..., -self.sliding_window :, :]
            scores = q.float() @ full_k.float().unsqueeze(2).transpose(-1, -2) if grouped_gqa else q.float() @ full_k.float().transpose(-1, -2)
            scores = scores * scale
            attended = torch.softmax(scores, dim=-1).to(dtype=q.dtype) @ (
                full_v.to(dtype=q.dtype).unsqueeze(2) if grouped_gqa else full_v.to(dtype=q.dtype)
            )
            return attended.reshape(q.shape[0], self.n_head, q.shape[-2], self.head_dim) if grouped_gqa else attended
        if self._sdpa_enabled and not grouped_gqa and q.dtype != torch.float64:
            if self.sliding_window > 0:
                query_positions = torch.arange(
                    position,
                    position + q.shape[-2],
                    dtype=torch.long,
                    device=q.device,
                )
                key_positions = key_start + torch.arange(
                    full_k.shape[-2], dtype=torch.long, device=q.device
                )
                allowed = key_positions[None, :] <= query_positions[:, None]
                allowed = allowed & (
                    key_positions[None, :]
                    >= query_positions[:, None] - self.sliding_window + 1
                )
                return F.scaled_dot_product_attention(
                    q,
                    full_k,
                    full_v,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )
            if position == 0 and key_start == 0 and full_k.shape[-2] == query_length:
                return F.scaled_dot_product_attention(
                    q,
                    full_k,
                    full_v,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=scale,
                )
            # ``is_causal=True`` assumes the query and key sequences start at
            # the same position.  Prefix continuation violates that
            # assumption: every new query may attend to the already-cached
            # prefix.  Supply the offset-aware mask explicitly in that case.
            query_positions = torch.arange(
                position, position + query_length, dtype=torch.long, device=q.device
            )
            key_positions = key_start + torch.arange(
                full_k.shape[-2], dtype=torch.long, device=q.device
            )
            allowed = key_positions[None, :] <= query_positions[:, None]
            return F.scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        scores = q.float() @ full_k.float().unsqueeze(2).transpose(-1, -2) if grouped_gqa else q.float() @ full_k.float().transpose(-1, -2)
        scores = scores * scale
        query_positions = torch.arange(
            position, position + q.shape[-2], dtype=torch.long, device=q.device
        )
        key_positions = key_start + torch.arange(
            full_k.shape[-2], dtype=torch.long, device=q.device
        )
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.sliding_window > 0:
            allowed = allowed & (
                key_positions[None, :] >= query_positions[:, None] - self.sliding_window + 1
            )
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        attended = torch.softmax(scores, dim=-1).to(dtype=q.dtype) @ (
            full_v.to(dtype=q.dtype).unsqueeze(2) if grouped_gqa else full_v.to(dtype=q.dtype)
        )
        return attended.reshape(q.shape[0], self.n_head, q.shape[-2], self.head_dim) if grouped_gqa else attended

    def _split_qkv(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_prefix = prefix + "self_attn."
        fused = _first(
            weights,
            attn_prefix + "qkv_proj.weight",
            attn_prefix + "W_pack.weight",
            required=False,
        )
        if fused is not None:
            bias = _first(weights, attn_prefix + "qkv_proj.bias", attn_prefix + "W_pack.bias", required=False)
            packed = _linear(x, fused, bias)
            q_width = self.n_head * self.head_dim
            kv_width = self.n_kv_head * self.head_dim
            if packed.shape[-1] != q_width + 2 * kv_width:
                raise ValueError(
                    f"unsupported fused QKV width {packed.shape[-1]} for layer {prefix.rstrip('.')!r}"
                )
            return torch.split(packed, [q_width, kv_width, kv_width], dim=-1)
        q = _linear(
            x,
            _first(weights, attn_prefix + "q_proj.weight", attn_prefix + "q_attn.weight"),
            _first(weights, attn_prefix + "q_proj.bias", required=False),
        )
        k = _linear(
            x,
            _first(weights, attn_prefix + "k_proj.weight"),
            _first(weights, attn_prefix + "k_proj.bias", required=False),
        )
        v = _linear(
            x,
            _first(weights, attn_prefix + "v_proj.weight"),
            _first(weights, attn_prefix + "v_proj.bias", required=False),
        )
        return q, k, v

    def _transformer_layer(
        self,
        layer_id: int,
        x: torch.Tensor,
        previous: TransformerKVState,
        weights: Mapping[str, torch.Tensor],
        position: int,
    ) -> tuple[torch.Tensor, TransformerKVState]:
        prefix = f"{self.layer_prefix}.{layer_id}."
        input_norm = _first(weights, prefix + "input_layernorm.weight")
        post_norm = _first(weights, prefix + "post_attention_layernorm.weight")
        assert input_norm is not None and post_norm is not None
        normed = _rms_norm(x, input_norm, self.eps)
        q, k, v = self._split_qkv(normed, weights, prefix)
        q = q.view(x.shape[0], x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.n_kv_head, self.head_dim).transpose(1, 2)
        q_norm = _first(weights, prefix + "self_attn.q_norm.weight", required=False)
        k_norm = _first(weights, prefix + "self_attn.k_norm.weight", required=False)
        if q_norm is not None:
            q = _rms_norm(q, q_norm, self.eps)
        if k_norm is not None:
            k = _rms_norm(k, k_norm, self.eps)
        positions = torch.arange(position, position + x.shape[1], dtype=torch.long, device=x.device)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        previous.append(k, v)
        if self.sliding_window > 0 and previous.length > self.sliding_window:
            # Compact after computing this chunk so multi-token prefill keeps
            # only the state needed by the next recurrent continuation.
            previous.compact_left(self.sliding_window)
        attended = self._attention(q, previous.key, previous.value, position)
        attended = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.hidden_size)
        o_weight = _first(weights, prefix + "self_attn.o_proj.weight")
        assert o_weight is not None
        hidden = x + _linear(attended, o_weight, _first(weights, prefix + "self_attn.o_proj.bias", required=False))
        normed = _rms_norm(hidden, post_norm, self.eps)
        gate_up = _first(weights, prefix + "mlp.gate_up_proj.weight", required=False)
        if gate_up is not None:
            gate_up_value = _linear(normed, gate_up, _first(weights, prefix + "mlp.gate_up_proj.bias", required=False))
            gate, up = torch.chunk(gate_up_value, 2, dim=-1)
        else:
            gate = _linear(normed, _first(weights, prefix + "mlp.gate_proj.weight"), _first(weights, prefix + "mlp.gate_proj.bias", required=False))
            up = _linear(normed, _first(weights, prefix + "mlp.up_proj.weight"), _first(weights, prefix + "mlp.up_proj.bias", required=False))
        down = _first(weights, prefix + "mlp.down_proj.weight")
        assert down is not None
        hidden = hidden + _linear(F.silu(gate) * up, down, _first(weights, prefix + "mlp.down_proj.bias", required=False))
        return hidden, previous

    def _gpt2_layer(
        self,
        layer_id: int,
        x: torch.Tensor,
        previous: TransformerKVState,
        weights: Mapping[str, torch.Tensor],
        position: int,
    ) -> tuple[torch.Tensor, TransformerKVState]:
        prefix = f"transformer.h.{layer_id}."
        ln1_w = _first(weights, prefix + "ln_1.weight")
        ln1_b = _first(weights, prefix + "ln_1.bias", required=False)
        ln2_w = _first(weights, prefix + "ln_2.weight")
        ln2_b = _first(weights, prefix + "ln_2.bias", required=False)
        assert ln1_w is not None and ln2_w is not None
        normed = _layer_norm(x, ln1_w, ln1_b, self.eps)
        attn_prefix = prefix + "attn."
        qkv_weight = _first(weights, attn_prefix + "c_attn.weight", attn_prefix + "qkv_proj.weight")
        qkv_bias = _first(weights, attn_prefix + "c_attn.bias", attn_prefix + "qkv_proj.bias", required=False)
        assert qkv_weight is not None
        qkv = _gpt2_linear(normed, qkv_weight, qkv_bias)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        previous.append(k, v)
        if self.sliding_window > 0 and previous.length > self.sliding_window:
            previous.compact_left(self.sliding_window)
        attended = self._attention(q, previous.key, previous.value, position)
        attended = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.hidden_size)
        proj_weight = _first(weights, attn_prefix + "c_proj.weight", attn_prefix + "o_proj.weight")
        proj_bias = _first(weights, attn_prefix + "c_proj.bias", attn_prefix + "o_proj.bias", required=False)
        assert proj_weight is not None
        hidden = x + _gpt2_linear(attended, proj_weight, proj_bias)
        normed = _layer_norm(hidden, ln2_w, ln2_b, self.eps)
        fc_weight = _first(weights, prefix + "mlp.c_fc.weight", prefix + "mlp.fc_in.weight")
        fc_bias = _first(weights, prefix + "mlp.c_fc.bias", prefix + "mlp.fc_in.bias", required=False)
        proj_weight = _first(weights, prefix + "mlp.c_proj.weight", prefix + "mlp.fc_out.weight")
        proj_bias = _first(weights, prefix + "mlp.c_proj.bias", prefix + "mlp.fc_out.bias", required=False)
        assert fc_weight is not None and proj_weight is not None
        hidden = hidden + _gpt2_linear(F.gelu(_gpt2_linear(normed, fc_weight, fc_bias)), proj_weight, proj_bias)
        return hidden, previous

    def _embedding(self, token_ids: torch.Tensor, position: int, provider: Any, metrics: MetricsCollector) -> torch.Tensor:
        if self.max_position and position + int(token_ids.shape[-1]) > self.max_position:
            raise ValueError(
                f"sequence length {position + int(token_ids.shape[-1])} exceeds the "
                f"checkpoint context limit {self.max_position}"
            )
        if self.style == "gpt2":
            word = self._global(provider, metrics, "transformer.wte.weight", "model.embed_tokens.weight")
            pos = self._global(provider, metrics, "transformer.wpe.weight")
            assert word is not None and pos is not None
            positions = torch.arange(position, position + token_ids.shape[-1], dtype=torch.long, device=self._device)
            return word.index_select(0, token_ids.reshape(-1)).view(token_ids.shape[0], token_ids.shape[1], -1) + pos.index_select(0, positions)[None, :, :]
        word = self._global(provider, metrics, "model.embed_tokens.weight", "model.model.embed_tokens.weight", "backbone.embeddings.weight")
        assert word is not None
        return word.index_select(0, token_ids.reshape(-1)).view(token_ids.shape[0], token_ids.shape[1], -1)

    def _final_logits(self, hidden: torch.Tensor, provider: Any, metrics: MetricsCollector) -> torch.Tensor:
        if self.style == "gpt2":
            norm_w = self._global(provider, metrics, "transformer.ln_f.weight")
            norm_b = self._global(provider, metrics, "transformer.ln_f.bias", required=False)
            head = self._global(provider, metrics, "lm_head.weight", "transformer.wte.weight")
            assert norm_w is not None and head is not None
            return _gpt2_linear(_layer_norm(hidden, norm_w, norm_b, self.eps), head)
        norm = self._global(provider, metrics, "model.norm.weight", "model.model.norm.weight", "backbone.norm_f.weight")
        head = self._global(provider, metrics, "lm_head.weight", "model.lm_head.weight", required=False)
        emb = self._global(provider, metrics, "model.embed_tokens.weight", "model.model.embed_tokens.weight", "backbone.embeddings.weight")
        assert norm is not None and emb is not None
        if head is None:
            head = emb
        return _linear(_rms_norm(hidden, norm, self.eps), head)

    def _run_layer(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        layer_state: TransformerKVState,
        provider: Any,
        metrics: MetricsCollector,
        position: int,
    ) -> tuple[torch.Tensor, TransformerKVState]:
        metrics.weight_sweeps += 1
        weights = self._load_layer_tensors(layer_id, provider, metrics)
        if self.style == "gpt2":
            return self._gpt2_layer(layer_id, hidden, layer_state, weights, position)
        return self._transformer_layer(layer_id, hidden, layer_state, weights, position)

    def prefill_ids(
        self,
        token_ids: list[int],
        provider: Any,
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        control = make_generation_control(
            cancel_event=cancel_event,
            deadline=deadline,
        )
        started = time.perf_counter()
        current: SequenceState | None = None
        try:
            if control is not None:
                control.check()
            if initial_state is None:
                current = self._initial_state(provider, metrics)
            else:
                if initial_state.sequence_state is None:
                    raise ValueError("Transformer prefill received a non-sequence initial state")
                validate_sequence_state(initial_state.sequence_state, kind="transformer")
                current = initial_state.sequence_state.clone()
            if len(current.layers) != self._num_layers or current.batch_size != 1:
                raise ValueError("Transformer initial state has incompatible layer or batch shape")
            if not token_ids:
                if current.next_logits is None:
                    raise ValueError("cannot prefill an empty Transformer prompt without an existing state")
                return RecurrentState(last_token_id=initial_state.last_token_id if initial_state else 0, sequence_state=current)
            ids = torch.tensor(token_ids, dtype=torch.long, device=self._device).reshape(1, -1)
            position = int(current.position)
            hidden = self._embedding(ids, position, provider, metrics)
            for layer_id in range(self._num_layers):
                if control is not None:
                    control.check()
                hidden, layer_state = self._run_layer(
                    layer_id, hidden, current.layers[layer_id], provider, metrics, position
                )
                current.layers[layer_id] = layer_state
            current.position += len(token_ids)
            current.next_logits = self._final_logits(hidden[:, -1:, :], provider, metrics).detach()
            state = RecurrentState(last_token_id=int(token_ids[-1]), sequence_state=current)
            self._record_sequence_metrics(state, metrics)
            return state
        finally:
            metrics.prefill_wall_s += time.perf_counter() - started

    def _step_state(
        self,
        state: RecurrentState,
        token_id: int,
        provider: Any,
        metrics: MetricsCollector,
        *,
        clone_state: bool = True,
    ) -> RecurrentState:
        assert state.sequence_state is not None
        current = state.sequence_state.clone() if clone_state else state.sequence_state
        ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=self._device)
        hidden = self._embedding(ids, current.position, provider, metrics)
        position = int(current.position)
        for layer_id in range(self._num_layers):
            hidden, layer_state = self._run_layer(
                layer_id, hidden, current.layers[layer_id], provider, metrics, position
            )
            current.layers[layer_id] = layer_state
        current.position += 1
        current.next_logits = self._final_logits(hidden, provider, metrics).detach()
        state.last_token_id = int(token_id)
        state.sequence_state = current
        return state

    def generate_greedy_ids_batch(
        self,
        token_batches: list[list[int]],
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        """Decode a qualified equal-length Transformer batch in one layer sweep.

        All sessions share the prompt length in this path.  That lets every
        layer consume one ``[batch, sequence, hidden]`` activation and one
        batched KV cache, while each row keeps an independent recurrent/KV
        state.  Unequal lengths are rejected with a capability-specific error
        instead of silently falling back to sequential generation.
        """
        if not self.supports_batch:
            self.capability_error(
                "batch",
                "Transformer pack did not opt into padded/packed batch decoding",
            )
        if not token_batches:
            return []
        if any(not ids for ids in token_batches):
            raise ValueError("Transformer batch prompts must contain at least one token")
        lengths = {len(ids) for ids in token_batches}
        if len(lengths) != 1:
            self.capability_error(
                "batch_shape",
                "qualified Transformer batching currently requires equal-length prompts; "
                "group prompts by length or use individual generate() calls",
            )
        batch = len(token_batches)
        prompt_length = next(iter(lengths))
        prefill_started = time.perf_counter()
        current = self._initial_state(provider, metrics, batch=batch)
        ids = torch.tensor(
            [[int(value) for value in row] for row in token_batches],
            dtype=torch.long,
            device=self._device,
        )
        hidden = self._embedding(ids, int(current.position), provider, metrics)
        for layer_id in range(self._num_layers):
            hidden, layer_state = self._run_layer(
                layer_id,
                hidden,
                current.layers[layer_id],
                provider,
                metrics,
                int(current.position),
            )
            current.layers[layer_id] = layer_state
        current.position += prompt_length
        current.next_logits = self._final_logits(
            hidden[:, -1:, :], provider, metrics
        ).detach()
        metrics.batch_prefill_wall_s = time.perf_counter() - prefill_started
        metrics.prefill_wall_s = metrics.batch_prefill_wall_s

        outputs = [[] for _ in token_batches]
        decode_started = time.perf_counter()
        for _ in range(max(0, int(max_tokens))):
            token_started = time.perf_counter()
            logits = current.next_logits
            assert logits is not None
            chosen = torch.argmax(logits[:, -1, :], dim=-1)
            for index, token in enumerate(chosen.tolist()):
                outputs[index].append(int(token))
            hidden = self._embedding(
                chosen.reshape(batch, 1),
                int(current.position),
                provider,
                metrics,
            )
            for layer_id in range(self._num_layers):
                hidden, layer_state = self._run_layer(
                    layer_id,
                    hidden,
                    current.layers[layer_id],
                    provider,
                    metrics,
                    int(current.position),
                )
                current.layers[layer_id] = layer_state
            current.position += 1
            current.next_logits = self._final_logits(hidden, provider, metrics).detach()
            token_ms = (time.perf_counter() - token_started) * 1000.0
            metrics.token_latencies_ms.append(token_ms)
            if len(outputs[0]) == 1 and metrics.ttft_s <= 0.0:
                metrics.ttft_s = token_ms / 1000.0
        metrics.batch_decode_wall_s = time.perf_counter() - decode_started
        metrics.decode_wall_s = metrics.batch_decode_wall_s
        metrics.batch_size = len(token_batches)
        metrics.tokens_generated = sum(len(row) for row in outputs)
        self._last_batch_states = []
        assert current.next_logits is not None
        for index, row in enumerate(token_batches):
            layers: list[TransformerKVState] = []
            for layer in current.layers:
                assert isinstance(layer, TransformerKVState)
                layers.append(
                    TransformerKVState(
                        layer.key[index : index + 1].clone(),
                        layer.value[index : index + 1].clone(),
                    )
                )
            summary = SequenceState(
                kind="transformer",
                position=int(current.position),
                layers=layers,
                batch_size=1,
                context_limit=current.context_limit,
                next_logits=current.next_logits[index : index + 1].clone(),
            )
            last_token = outputs[index][-1] if outputs[index] else int(row[-1])
            self._last_batch_states.append(
                RecurrentState(last_token_id=int(last_token), sequence_state=summary)
            )
        self._sequence_kernel = str(getattr(self, "_sequence_kernel", "torch"))
        return outputs

    def generate_greedy_batch(
        self,
        prompts: list[str],
        provider: Any,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        return self.generate_greedy_ids_batch(
            [self._encode(prompt) for prompt in prompts],
            provider,
            max_tokens,
            metrics,
        )

    def close(self) -> None:
        self._rope_tables.clear()
        super().close()


__all__ = [
    "HFTransformerPackBackend",
    "Mamba2PackBackend",
    "SequencePackBackend",
]
