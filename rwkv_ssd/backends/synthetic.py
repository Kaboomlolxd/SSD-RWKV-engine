"""
Minimal layer-wise recurrent model for V0 correctness tests (CPU-only).

Architecture (meta in manifest):
  - embed: [vocab, n_embd]
  - blocks.{i}.weight: [n_embd, n_embd]
  - head.weight: [n_embd, vocab]

State h is n_embd. One layer: h = tanh(h @ W + x).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from rwkv_ssd.backends.pack_backend import PackBackend
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.sampling import sample_torch
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.metrics import MetricsCollector, Timer
from rwkv_ssd.runtime.power import throttle_after_work
from rwkv_ssd.runtime.state_cache import RecurrentState
from rwkv_ssd.runtime.provider_factory import prefetch_ahead
from rwkv_ssd.runtime.weight_provider import ManifestWeightProvider, WeightProvider

logger = logging.getLogger(__name__)


class SyntheticBackend(PackBackend):
    supports_batch = True

    def __init__(self) -> None:
        self._manifest: Manifest | None = None
        self._device = torch.device("cpu")
        self._n_layer = 0
        self._n_embd = 0
        self._vocab = 256
        self._layer_ids: list[int] = []
        self._w_embed: torch.Tensor | None = None
        self._w_head: torch.Tensor | None = None

    @property
    def num_layers(self) -> int:
        return self._n_layer

    def load(self, model_path: str, strategy: str, device: str) -> None:
        raise BackendLoadError(
            "SyntheticBackend uses load_pack() with a runtime pack directory, not load()."
        )

    def load_pack(self, manifest: Manifest, device: str) -> None:
        self._manifest = manifest
        self._device = torch.device(device)
        meta = manifest.meta
        if meta.get("model_type") != "synthetic_rwkv":
            logger.warning(
                "manifest model_type=%r — synthetic backend expects 'synthetic_rwkv'",
                meta.get("model_type"),
            )
        self._n_layer = int(meta.get("n_layer", 4))
        self._n_embd = int(meta.get("n_embd", 32))
        self._vocab = int(meta.get("vocab_size", 256))
        by_layer = manifest.by_layer()
        self._layer_ids = sorted(
            lid
            for lid in by_layer
            if lid >= 0 and any("blocks" in t.name for t in by_layer[lid])
        )
        if not self._layer_ids:
            self._layer_ids = list(range(self._n_layer))

    def _encode(self, prompt: str) -> list[int]:
        return [b % self._vocab for b in prompt.encode("utf-8")] or [0]

    def _get_weight(
        self,
        tensors: dict[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        for key, val in tensors.items():
            if key == name or key.endswith(name):
                return val
        raise KeyError(f"missing weight {name} in {list(tensors)}")

    def _load_embed_head(
        self, provider: WeightProvider
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._manifest is not None
        embed_entries = [t for t in self._manifest.tensors if "embed" in t.name]
        head_entries = [t for t in self._manifest.tensors if "head" in t.name]
        if not embed_entries or not head_entries:
            raise ValueError("synthetic pack must contain embed.* and head.* tensors")
        embed = provider.load_layer_tensors(embed_entries)
        head = provider.load_layer_tensors(head_entries)
        w_embed = self._get_weight(embed, "embed.weight")
        try:
            w_head = self._get_weight(head, "head.weight")
        except KeyError:
            w_head = self._materialize_head(provider, head_entries[0])
        self._w_embed = w_embed
        self._w_head = w_head
        return w_embed, w_head

    def _materialize_head(
        self, provider: WeightProvider, entry: TensorEntry
    ) -> torch.Tensor:
        """Materialize a fused LUT ``head.weight`` blob to bf16.

        The synthetic backend has no fused head GEMV, so when the provider leaves
        ``head.weight`` as a registered LUT blob (``load_layer_tensors`` returns
        no tensor for it) we force a bf16 materialize via ``_load_one``.
        """
        from rwkv_ssd.runtime.dequant import decode_weight_to_tensor
        from rwkv_ssd.runtime.metrics import Timer

        timing = provider.begin_layer(entry.layer_id)
        if entry.name in provider._cache:
            return provider._cache[entry.name]
        raw = provider._read_packed_bytes(entry, timing)
        with Timer() as t_stage:
            t = decode_weight_to_tensor(
                raw,
                entry,
                provider._device,
                trinity_layer_cache=provider._trinity_layer_cache,
                decode_device=provider._decode_device,
            )
        timing.staging_ms += t_stage.elapsed_ms
        if provider._stream_layer_cache and provider._should_stream(entry):
            provider._cache[entry.name] = t
        return t

    def _forward_layers(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        provider: WeightProvider,
        by_layer: dict[int, list],
        prefetch: bool,
        metrics: MetricsCollector,
    ) -> torch.Tensor:
        for i, layer_id in enumerate(self._layer_ids):
            entries = by_layer.get(layer_id, [])
            if not entries:
                continue
            # Prefetch is only useful when the decoded tensors will be
            # reused on the next call (stream_layer_cache). For the
            # no-cache / strict-fused path, the prefetch thread blocks
            # the main thread on ``begin_layer`` for the I/O time with
            # no compute to overlap against — it just adds latency and
            # a ``ThreadPoolExecutor.submit`` round-trip per layer per
            # token. Skip it.
            if (
                prefetch
                and isinstance(provider, ManifestWeightProvider)
                and provider._stream_layer_cache
            ):
                prefetch_ahead(provider, provider.planner, self._layer_ids, i, by_layer)

            layer_tensors = provider.load_layer_tensors(entries)
            with Timer() as t_compute:
                w = self._get_weight(layer_tensors, f"blocks.{layer_id}.weight")
                h = torch.tanh(h @ w + x)
            if metrics.layers:
                metrics.layers[-1].compute_ms += t_compute.elapsed_ms
        return h

    def _forward_layers_batch(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        provider: WeightProvider,
        by_layer: dict[int, list],
        metrics: MetricsCollector,
    ) -> torch.Tensor:
        """Advance a batch through one weight-stationary layer sweep."""
        for layer_id in self._layer_ids:
            entries = by_layer.get(layer_id, [])
            if not entries:
                continue
            layer_tensors = provider.load_layer_tensors(entries)
            metrics.weight_layer_loads += 1
            with Timer() as t_compute:
                w = self._get_weight(layer_tensors, f"blocks.{layer_id}.weight")
                h = torch.tanh(h @ w + x)
            if metrics.layers:
                metrics.layers[-1].compute_ms += t_compute.elapsed_ms
        metrics.weight_sweeps += 1
        return h

    def prefill_batch_texts(
        self,
        texts: list[str],
        provider: WeightProvider,
        metrics: MetricsCollector,
    ) -> list[RecurrentState]:
        """Prefill variable-length prompts while sharing each layer load."""
        if not texts:
            return []
        assert self._manifest is not None
        w_embed, _ = self._load_embed_head(provider)
        by_layer = self._manifest.by_layer()
        encoded = [self._encode(text) for text in texts]
        h = torch.zeros((len(texts), self._n_embd), device=self._device)
        last = torch.zeros(len(texts), dtype=torch.long, device=self._device)
        for position in range(max(len(ids) for ids in encoded)):
            active = [index for index, ids in enumerate(encoded) if position < len(ids)]
            token_ids = torch.tensor(
                [encoded[index][position] for index in active],
                dtype=torch.long,
                device=self._device,
            )
            active_index = torch.tensor(active, dtype=torch.long, device=self._device)
            advanced = self._forward_layers_batch(
                h.index_select(0, active_index),
                w_embed.index_select(0, token_ids),
                provider,
                by_layer,
                metrics,
            )
            h.index_copy_(0, active_index, advanced)
            last.index_copy_(0, active_index, token_ids)
        return [
            RecurrentState(h=h[index].clone(), last_token_id=int(last[index].item()))
            for index in range(len(texts))
        ]

    def decode_greedy_batch(
        self,
        states: list[RecurrentState],
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
    ) -> list[list[int]]:
        """Decode multiple recurrent states with one layer read per sweep."""
        if not states:
            return []
        assert self._manifest is not None
        w_embed = self._w_embed
        w_head = self._w_head
        if w_embed is None or w_head is None:
            w_embed, w_head = self._load_embed_head(provider)
        by_layer = self._manifest.by_layer()
        h = torch.stack([state.h.clone() for state in states])
        last = torch.tensor(
            [state.last_token_id for state in states],
            dtype=torch.long,
            device=self._device,
        )
        outputs = [[] for _ in states]
        for _ in range(max(0, int(max_tokens))):
            h = self._forward_layers_batch(
                h,
                w_embed.index_select(0, last),
                provider,
                by_layer,
                metrics,
            )
            logits = h @ w_head
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                last = torch.multinomial(probs, 1).reshape(-1)
            else:
                last = logits.argmax(dim=-1)
            for index, token_id in enumerate(last.tolist()):
                outputs[index].append(int(token_id))
        metrics.batch_size = len(states)
        metrics.tokens_generated = sum(len(output) for output in outputs)
        self._last_batch_states = [
            RecurrentState(h=h[index].clone(), last_token_id=int(last[index].item()))
            for index in range(len(states))
        ]
        return outputs

    def generate_greedy_batch(
        self,
        prompts: list[str],
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        states = self.prefill_batch_texts(prompts, provider, metrics)
        return self.decode_greedy_batch(states, provider, max_tokens, metrics)

    def prefill_text(
        self,
        text: str,
        provider: WeightProvider,
        metrics: MetricsCollector,
        initial_state: RecurrentState | None = None,
        *,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> RecurrentState:
        assert self._manifest is not None
        w_embed, _ = self._load_embed_head(provider)
        by_layer = self._manifest.by_layer()
        token_ids = self._encode(text)
        if initial_state is not None:
            h = initial_state.h.clone()
            last = initial_state.last_token_id
        else:
            h = torch.zeros(self._n_embd, device=self._device)
            last = 0

        control = make_generation_control(
            cancel_event=cancel_event,
            deadline=deadline,
        )
        for tid in token_ids:
            if control is not None:
                control.check()
            x = w_embed[tid]
            h = self._forward_layers(
                h, x, provider, by_layer, prefetch=True, metrics=metrics
            )
            last = tid

        return RecurrentState(h=h, last_token_id=last)

    def decode_greedy(
        self,
        state: RecurrentState,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        assert self._manifest is not None
        w_embed = self._w_embed
        w_head = self._w_head
        if w_embed is None or w_head is None:
            w_embed, w_head = self._load_embed_head(provider)
        by_layer = self._manifest.by_layer()

        h = state.h.clone()
        last = state.last_token_id
        out: list[int] = []
        power_percent = getattr(metrics, "power_percent", 100)
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        for _ in range(max_tokens):
            if control is not None:
                control.check()
            t_token = time.perf_counter()
            x = w_embed[last]
            h = self._forward_layers(
                h, x, provider, by_layer, prefetch=True, metrics=metrics
            )
            logits = h @ w_head
            last = sample_torch(logits, temperature=temperature, greedy=temperature <= 0)
            out.append(last)
            # Keep the diagnostic state/logit probe live while incremental
            # callbacks are being delivered.  This is also what makes a
            # cancellation/streaming trace useful instead of exposing only
            # the state after the entire decode loop.
            self._last_recurrent_state = RecurrentState(
                h=h.clone(), last_token_id=int(last)
            )
            self._last_logits = logits.detach().clone()
            if control is not None:
                control.emit(last)
            throttle_after_work(t_token, power_percent)

        metrics.tokens_generated = len(out)
        self._last_recurrent_state = RecurrentState(h=h.clone(), last_token_id=last)
        return out

    def generate_greedy(
        self,
        prompt: str,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[int]:
        state = self.prefill_text(prompt, provider, metrics)
        token_ids = self.decode_greedy(state, provider, max_tokens, metrics)
        if not hasattr(self, "_last_recurrent_state") or self._last_recurrent_state is None:
            self._last_recurrent_state = RecurrentState(
                h=state.h.clone() if state.h is not None else None,
                last_token_id=token_ids[-1] if token_ids else state.last_token_id,
            )
        return token_ids

    def decode_text(self, token_ids: list[int]) -> str:
        return bytes(t % 256 for t in token_ids).decode("utf-8", errors="replace")

    def _build_recurrent_state(
        self, provider: WeightProvider, *, last_token_id: int
    ) -> RecurrentState:
        """Build a synthetic recurrent state for snapshotting.

        For synthetic, the only "recurrent" quantity is the current h tensor
        plus the last emitted token id. Re-running prefill with this state as
        ``initial_state`` produces an h that is identical to the current one
        modulo the h-clone in prefill.
        """
        return RecurrentState(
            h=torch.zeros(self._n_embd, device=self._device),
            rwkv7_state=None,
            last_token_id=int(last_token_id),
        )

    def get_recurrent_state(self) -> RecurrentState | None:
        return getattr(self, "_last_recurrent_state", None)

    def probe_logits(self, state: RecurrentState | None = None) -> torch.Tensor | None:
        current = state if state is not None else self.get_recurrent_state()
        if current is None or current.h is None or self._w_head is None:
            return None
        return (current.h @ self._w_head).detach().clone()

    def set_recurrent_state(self, state: RecurrentState) -> None:
        self._last_recurrent_state = state


class BackendLoadError(RuntimeError):
    pass
