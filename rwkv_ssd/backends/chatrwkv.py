"""ChatRWKV adapter — uses BlinkDL ChatRWKV rwkv pip package (RWKV-7 + legacy)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.runtime.checkpoint_meta import (
    default_rwkv_strategy,
    is_rwkv7_checkpoint,
    load_checkpoint_tensors,
    model_path_base,
)
from rwkv_ssd.runtime.deepembed import (
    DEEP_EMBED_RWKV7A_V1,
    detect_deepembed_variant,
)
from rwkv_ssd.runtime.state_cache import RecurrentState, clone_rwkv7_state
from rwkv_ssd.runtime.sampling import sample_torch


def find_chatrwkv_root() -> Path | None:
    env = os.environ.get("CHATRWKV_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[2]
    if os.environ.get("RWKV_SSD_SKIP_BUNDLED_CHATRWKV") != "1":
        candidates.append(here / "test_model" / "ChatRWKV")
    candidates.extend(
        [
            here.parent / "ChatRWKV",
            here / "backends" / "chatrwkv_ref",
        ]
    )
    for candidate in candidates:
        pip = candidate / "rwkv_pip_package" / "src" / "rwkv" / "model.py"
        legacy = candidate / "src" / "model_run.py"
        if pip.is_file() or legacy.is_file():
            return candidate
    return None


def _rwkv_package_src(root: Path) -> Path:
    pip = root / "rwkv_pip_package" / "src"
    if pip.is_dir():
        return pip
    return root / "src"


def _configure_rwkv_env(
    rwkv7: bool,
    strategy: str = "",
    *,
    deepembed_variant: str | None = None,
) -> None:
    # Must be set before ``import rwkv.model``. ChatRWKV enables JIT unless this
    # is explicitly "0"; default off for compatibility, but allow user override.
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("RWKV_CUDA_ON", "0")
    if rwkv7:
        os.environ["RWKV_V7_ON"] = "1"
    if deepembed_variant == DEEP_EMBED_RWKV7A_V1:
        os.environ["RWKV_DE_VERSION"] = "1"
    else:
        # Do not let a previous v1 load change an ordinary RWKV-7 import in
        # the same test/server process.  ChatRWKV reads this at import time.
        os.environ.pop("RWKV_DE_VERSION", None)


class ChatRWKVBackend(RecurrentBackend):
    def __init__(self) -> None:
        self._model = None
        self._pipeline = None
        self._tokenizer = None
        self._pipe_args = None
        self._rwkv7 = False
        self._deepembed = False
        self._deepembed_streaming = False
        self._deepembed_streaming_reference = False
        self._deepembed_variant: str | None = None
        self._n_layer = 0

    def load(
        self,
        model_path: str,
        strategy: str,
        device: str,
        *,
        pack_meta: dict | None = None,
        pack_dir: Path | None = None,
        skeleton_load: bool = False,
        resident_layer_ids: set[int] | None = None,
    ) -> None:
        import sys

        from rwkv_ssd.runtime.device import resolve_device, resolve_strategy

        root = find_chatrwkv_root()
        if root is None:
            raise RuntimeError(
                "ChatRWKV not found. Clone https://github.com/BlinkDL/ChatRWKV "
                "into test_model/ChatRWKV or set CHATRWKV_ROOT."
            )

        ckpt = Path(model_path)
        if not ckpt.is_file() and Path(f"{model_path}.pth").is_file():
            ckpt = Path(f"{model_path}.pth")

        # DeepEmbed is a RWKV-7-derived model contract with an extra DEA/qkv
        # path and context-indexed tables.  The ordinary ChatRWKV RWKV class
        # does not consume those tensors, so select the small reference model
        # explicitly instead of silently dropping them.
        deepembed = bool(pack_meta and pack_meta.get("deepembed"))
        deepembed_variant = (
            str(pack_meta.get("deepembed_variant"))
            if pack_meta and pack_meta.get("deepembed_variant")
            else None
        )
        checkpoint_tensors: dict[str, Any] | None = None
        if not deepembed and ckpt.is_file() and not pack_meta:
            checkpoint_tensors = load_checkpoint_tensors(ckpt)
            deepembed_variant = detect_deepembed_variant(checkpoint_tensors)
            deepembed = deepembed_variant is not None
        elif deepembed and deepembed_variant is None and ckpt.is_file():
            # Older packs predate the variant field.  Inspecting the local
            # checkpoint makes the safe choice possible without guessing.
            checkpoint_tensors = load_checkpoint_tensors(ckpt)
            deepembed_variant = detect_deepembed_variant(checkpoint_tensors)
        if deepembed and deepembed_variant is None:
            # A legacy metadata-only qkv/DEA pack has no variant field.
            deepembed_variant = "qkv_dea"
        if deepembed:
            self._rwkv7 = True
            self._deepembed = True
            self._deepembed_variant = deepembed_variant
            self._deepembed_streaming = True
        elif pack_meta and pack_meta.get("rwkv_version") == 7:
            self._rwkv7 = True
        elif pack_meta and "rwkv_version" in pack_meta:
            self._rwkv7 = False
        else:
            print("Inspecting checkpoint format...", file=sys.stderr, flush=True)
            tensors = checkpoint_tensors or load_checkpoint_tensors(ckpt)
            self._rwkv7 = is_rwkv7_checkpoint(tensors)
            del tensors

        dev = resolve_device(device)
        dev_label = dev.type
        if strategy.strip().lower() in ("cpu fp32", "cuda fp16", "auto", ""):
            if pack_meta and pack_meta.get("primary_dtype"):
                dtype = str(pack_meta["primary_dtype"])
                kind = "bf16" if dtype == "bfloat16" else "fp32"
                strategy = f"{dev_label} {kind}"
            else:
                print("Reading checkpoint for strategy...", file=sys.stderr, flush=True)
                tensors = checkpoint_tensors or load_checkpoint_tensors(ckpt)
                strategy = default_rwkv_strategy(tensors, dev_label)
                del tensors
        else:
            strategy = resolve_strategy(strategy, dev, rwkv7=self._rwkv7)

        _configure_rwkv_env(
            self._rwkv7,
            strategy,
            deepembed_variant=deepembed_variant,
        )

        src = str(_rwkv_package_src(root))
        if src not in sys.path:
            sys.path.insert(0, src)

        # ``rwkv.model`` specializes its CMix signatures at import time.  A
        # long-lived process may load an ordinary RWKV model before v1 (or
        # vice versa), so reload only when the desired specialization differs.
        import importlib

        existing_rwkv_model = sys.modules.get("rwkv.model")
        desired_de_version = (
            "1" if deepembed_variant == DEEP_EMBED_RWKV7A_V1 else None
        )
        if existing_rwkv_model is not None and getattr(
            existing_rwkv_model, "_rwkv_ssd_de_version", None
        ) != desired_de_version:
            importlib.reload(existing_rwkv_model)

        from rwkv.model import RWKV  # type: ignore[import-untyped]
        from rwkv.utils import PIPELINE, PIPELINE_ARGS  # type: ignore[import-untyped]
        rwkv_module = sys.modules.get("rwkv.model")
        if rwkv_module is not None:
            setattr(rwkv_module, "_rwkv_ssd_de_version", desired_de_version)

        use_skeleton = skeleton_load and self._rwkv7 and pack_dir is not None
        if dev.type == "xpu" and deepembed and deepembed_variant != DEEP_EMBED_RWKV7A_V1:
            raise RuntimeError(
                "ChatRWKV qkv/DEA DeepEmbed is not implemented on Intel XPU; "
                "use the CPU reference backend or an ordinary RWKV-7/RWKV7a-v1 pack."
            )
        if dev.type == "xpu" and self._rwkv7 and not use_skeleton:
            raise RuntimeError(
                "ChatRWKV XPU requires the pack-only RWKV-7 skeleton path. "
                "Keep skeleton_load enabled and provide --model/pack_dir."
            )
        if use_skeleton and deepembed and deepembed_variant != DEEP_EMBED_RWKV7A_V1:
            from rwkv_ssd.runtime.deepembed import DeepEmbedReferenceModel, DeepEmbedSidecar

            candidates = [pack_dir / "DeepEmbed.bin"] if pack_dir is not None else []
            candidates.append(ckpt.with_name("DeepEmbed.bin"))
            sidecar_path = next((path for path in candidates if path.is_file()), None)
            if sidecar_path is None:
                raise RuntimeError(
                    "qkv/DEA DeepEmbed CPU streaming requires DeepEmbed.bin; "
                    "repack without --no-deepembed-sidecar"
                )
            if checkpoint_tensors is None:
                checkpoint_tensors = load_checkpoint_tensors(ckpt)
            print(
                "Loading qkv/DEA DeepEmbed CPU streaming reference with sidecar "
                f"{sidecar_path} ...",
                file=sys.stderr,
                flush=True,
            )
            self._model = DeepEmbedReferenceModel(
                checkpoint_tensors,
                DeepEmbedSidecar(sidecar_path),
                resident_layers=set(),
            )
            self._deepembed_streaming_reference = True
        elif use_skeleton:
            from rwkv_ssd.runtime.rwkv7_skeleton import (
                build_rwkv7_skeleton_from_pack,
                estimate_z_bytes,
            )

            print(
                f"Loading RWKV-7 skeleton from pack {pack_dir} [{strategy}] ...",
                file=sys.stderr,
                flush=True,
            )
            self._model = build_rwkv7_skeleton_from_pack(
                pack_dir,
                strategy,
                resident_layer_ids=resident_layer_ids,
            )
            if deepembed_variant == DEEP_EMBED_RWKV7A_V1:
                setattr(self._model, "_rwkv7a_deepembed_v1", True)
                setattr(self._model, "_rwkv_ssd_s_emb_merged", False)
            print(
                f"Skeleton ready ({estimate_z_bytes(self._model.z) / 1e6:.1f} MB in z).",
                file=sys.stderr,
                flush=True,
            )
        elif deepembed and deepembed_variant != DEEP_EMBED_RWKV7A_V1:
            from rwkv_ssd.runtime.deepembed import DeepEmbedReferenceModel

            candidates = []
            if pack_dir is not None:
                candidates.append(pack_dir / "DeepEmbed.bin")
            candidates.append(ckpt.with_name("DeepEmbed.bin"))
            sidecar = next((path for path in candidates if path.is_file()), None)
            print(
                "Loading DeepEmbed reference model (resident CPU path); "
                f"sidecar={sidecar or 'derived in RAM'} ...",
                file=sys.stderr,
                flush=True,
            )
            if checkpoint_tensors is None:
                checkpoint_tensors = load_checkpoint_tensors(ckpt)
            self._model = DeepEmbedReferenceModel(checkpoint_tensors, sidecar)
            print("DeepEmbed reference model ready.", file=sys.stderr, flush=True)
        else:
            base = model_path_base(ckpt)
            print(f"Loading RWKV weights: {base}.pth [{strategy}] ...", file=sys.stderr, flush=True)
            if self._rwkv7:
                self._model = RWKV(base, strategy=strategy)
            else:
                self._model = RWKV(base, strategy=strategy, verbose=False)
            if deepembed_variant == DEEP_EMBED_RWKV7A_V1:
                setattr(self._model, "_rwkv7a_deepembed_v1", True)
                setattr(self._model, "_rwkv_ssd_s_emb_merged", True)
            print("Checkpoint loaded.", file=sys.stderr, flush=True)
            if skeleton_load and self._rwkv7:
                from rwkv_ssd.runtime.rwkv7_skeleton import (
                    apply_skeleton_to_loaded_model,
                    estimate_z_bytes,
                )

                before = estimate_z_bytes(self._model.z)
                freed = apply_skeleton_to_loaded_model(
                    self._model,
                    resident_layer_ids=resident_layer_ids,
                )
                after = estimate_z_bytes(self._model.z)
                print(
                    f"Skeleton evict: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB "
                    f"(freed {freed / 1e6:.1f} MB)",
                    file=sys.stderr,
                    flush=True,
                )
        self._n_layer = int(getattr(self._model, "n_layer", 0) or getattr(self._model.args, "n_layer", 0))
        self._pipeline = PIPELINE(self._model, "rwkv_vocab_v20230424")
        self._tokenizer = self._pipeline.tokenizer
        self._pipe_args = PIPELINE_ARGS(temperature=0, top_p=0)

    @property
    def num_layers(self) -> int:
        return self._n_layer

    def prefill(self, prompt: str) -> tuple[list[int], Any]:
        if self._pipeline is None:
            raise RuntimeError("backend not loaded")
        ids = self._pipeline.encode(prompt)
        state = None
        if self._rwkv7 and hasattr(self._model, "generate_zero_state"):
            state = self._model.generate_zero_state()
        return ids, state

    def step(self, token_id: int, state: Any) -> tuple[int, Any]:
        if self._model is None:
            raise RuntimeError("backend not loaded")
        out, state = self._model.forward([token_id], state)
        next_id = int(out.argmax().item())
        return next_id, state

    def decode_text(self, token_ids: list[int]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(token_ids)
        return bytes(token_ids).decode("utf-8", errors="replace")

    def get_recurrent_state(self) -> RecurrentState | None:
        if self._model is None or not self._rwkv7:
            return None
        state = getattr(self._model, "_rwkv_ssd_last_state", None)
        if state is None:
            return None
        return RecurrentState(
            last_token_id=int(getattr(self._model, "_rwkv_ssd_last_token_id", 0)),
            rwkv7_state=clone_rwkv7_state(state),
        )

    def probe_logits(self, state: RecurrentState | None = None) -> torch.Tensor | None:
        del state
        if self._model is None:
            return None
        logits = getattr(self._model, "_rwkv_ssd_last_logits", None)
        if logits is None:
            return None
        if not hasattr(logits, "detach"):
            return None
        observed = logits.detach().clone()
        # Keep the additive diagnostic ABI semantic: one vector for the
        # distribution that selected the observed token.  Older streaming
        # paths may still leave a [steps, vocab] tensor behind, so normalize
        # at this boundary as well as at the producer.
        if observed.ndim > 1:
            observed = observed.reshape(-1, observed.shape[-1])[-1]
        return observed

    def set_recurrent_state(self, state: RecurrentState) -> None:
        if self._model is None or state.rwkv7_state is None:
            return
        setattr(self._model, "_rwkv_ssd_last_state", clone_rwkv7_state(state.rwkv7_state))
        setattr(self._model, "_rwkv_ssd_last_token_id", int(state.last_token_id))
        setattr(self._model, "_rwkv_ssd_last_logits", None)

    def generate_simple(
        self,
        prompt: str,
        max_tokens: int,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> str:
        if self._pipeline is None or self._pipe_args is None:
            raise RuntimeError("backend not loaded")
        args = self._pipe_args
        if not greedy:
            from rwkv.utils import PIPELINE_ARGS  # type: ignore[import-untyped]

            args = PIPELINE_ARGS(temperature=float(temperature), top_p=0.85)
        return self._pipeline.generate(prompt, token_count=max_tokens, args=args)

    def generate_greedy_native(
        self,
        prompt: str,
        max_tokens: int,
        metrics: Any | None = None,
        *,
        power_percent: int = 100,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        """Greedy token ids via ``model.forward`` (matches streaming+cache decode path)."""
        if self._model is None or self._pipeline is None:
            raise RuntimeError("backend not loaded")
        from rwkv_ssd.backends.rwkv7_forward import greedy_token_ids_native

        return greedy_token_ids_native(
            self._model,
            self._pipeline,
            prompt,
            max_tokens,
            metrics=metrics,
            power_percent=power_percent,
            temperature=temperature,
            greedy=greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_greedy_pack_streaming(
        self,
        prompt: str,
        max_tokens: int,
        provider: Any,
        by_layer: dict[int, list],
        layer_ids: list[int],
        metrics: Any = None,
        *,
        system_prefix: str | None = None,
        prefix_cache: Any = None,
        prefix_cache_mode: str = "system",
        power_percent: int = 100,
        temperature: float = 1.0,
        greedy: bool = True,
        token_callback=None,
        cancel_event=None,
        deadline: float | None = None,
    ) -> list[int]:
        if not self._rwkv7:
            raise RuntimeError("pack streaming forward requires RWKV-7 (rwkv_version=7)")
        if self._model is None or self._pipeline is None:
            raise RuntimeError("backend not loaded")
        if self._deepembed_streaming_reference:
            from rwkv_ssd.runtime.deepembed import is_deepembed_tensor_name
            from rwkv_ssd.runtime.generation_control import make_generation_control
            from rwkv_ssd.backends.rwkv7_forward import _remember_rwkv7_state

            stream_layers = {
                layer_id: [
                    entry
                    for entry in entries
                    if not is_deepembed_tensor_name(entry.name)
                ]
                for layer_id, entries in by_layer.items()
            }
            model = self._model
            ids = self._pipeline.encode(prompt)
            state = model.generate_zero_state()
            control = make_generation_control(
                token_callback=token_callback,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            if control is not None:
                control.check()
            prefill_started = time.perf_counter()
            # Prefill already produces the distribution for the first
            # generated token.  Keep it instead of replaying the final prompt
            # token in the decode loop; replaying advances qkv/DEA state twice
            # and makes the reference path disagree with resident inference.
            prefill_ids = ids or [0]
            logits, state = model.forward_streaming(
                prefill_ids,
                state,
                provider,
                stream_layers,
            )
            if metrics is not None:
                metrics.prefill_wall_s = time.perf_counter() - prefill_started
            output: list[int] = []
            decode_started = time.perf_counter()
            context_last = int(prefill_ids[-1])
            for _ in range(max(0, int(max_tokens))):
                if control is not None:
                    control.check()
                next_id = sample_torch(
                    logits,
                    temperature=temperature,
                    greedy=greedy,
                )
                output.append(next_id)
                if control is not None:
                    _remember_rwkv7_state(model, state, context_last, logits)
                    control.emit(next_id)
                context_last = int(next_id)
                logits, state = model.forward_streaming(
                    [context_last],
                    state,
                    provider,
                    stream_layers,
                )
                if metrics is not None:
                    metrics.tokens_generated = len(output)
            if metrics is not None:
                metrics.decode_wall_s = time.perf_counter() - decode_started
                metrics.token_latencies_ms.extend(
                    [metrics.decode_wall_s * 1000.0 / len(output)] * len(output)
                    if output
                    else []
                )
            _remember_rwkv7_state(model, state, context_last, logits)
            return output
        from rwkv_ssd.backends.rwkv7_forward import greedy_token_ids_streaming

        return greedy_token_ids_streaming(
            self._model,
            self._pipeline,
            prompt,
            max_tokens,
            provider,
            layer_ids,
            by_layer,
            metrics=metrics,
            system_prefix=system_prefix,
            prefix_cache=prefix_cache,
            prefix_cache_mode=prefix_cache_mode,
            power_percent=power_percent,
            temperature=temperature,
            greedy=greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_greedy_batch_streaming(self, prompts, max_tokens, provider, by_layer, layer_ids, metrics):
        """Shared-layer qkv/DEA prefill and dense decode, or ordinary RWKV batch decode.

        The final prefill forward already produces the distribution for the
        first generated token.  Replaying the prompt's final token here would
        advance every recurrent state twice.  Keep those logits per session,
        emit the first token from them, and use the shared layer sweep only to
        advance the state after a generated token.
        """
        if self._model is None or self._pipeline is None or not self._rwkv7:
            raise RuntimeError("RWKV-7 ChatRWKV backend not loaded")
        from rwkv_ssd.backends.rwkv7_batch import forward_batch_one_dense
        from rwkv_ssd.backends.rwkv7_forward import forward_one, prefill_text_streaming
        if self._deepembed_streaming_reference:
            from rwkv_ssd.runtime.deepembed import is_deepembed_tensor_name

            # qkv/DEA uses a context-indexed sidecar and therefore cannot use
            # the ordinary ChatRWKV sequence prefill helper.  Its reference
            # model nevertheless supports a useful shared-layer path: load
            # each ordinary layer once, then advance every independent
            # context through that layer.
            stream_layers = {
                layer_id: [
                    entry
                    for entry in entries
                    if not is_deepembed_tensor_name(str(getattr(entry, "name", "")))
                ]
                for layer_id, entries in by_layer.items()
            }
            model = self._model
            states = []
            prompt_sequences = []
            for prompt in prompts:
                ids = self._pipeline.encode(prompt)
                state = model.generate_zero_state()
                prompt_sequences.append(ids or [0])
                states.append(state)

            # qkv/DEA lookup rows are context-dependent, so the reference
            # implementation keeps each session's exact sequence/mask while
            # sharing the ordinary layer transaction across all prompts.
            prefill_logits, states = model.forward_batch_prefill_streaming(
                prompt_sequences,
                states,
                provider,
                stream_layers,
                metrics=metrics,
            )
            outputs = [[] for _ in prompts]
            count = max(0, int(max_tokens))
            for _ in range(count):
                # The prefill sweep (and each preceding decode sweep) already
                # produced the next-token distribution.  Sampling it directly
                # avoids consuming every prompt's final token twice.
                generated = [int(value.argmax().item()) for value in prefill_logits]
                for output, token in zip(outputs, generated, strict=True):
                    output.append(token)
                if len(outputs[0]) < count:
                    prefill_logits, states = model.forward_batch_streaming(
                        generated,
                        states,
                        provider,
                        stream_layers,
                        metrics=metrics,
                    )
            metrics.tokens_generated = len(prompts) * count
            return outputs
        states = []
        prefill_logits = []
        for prompt in prompts:
            state = self._model.generate_zero_state()
            state, last = prefill_text_streaming(self._model, self._pipeline, prompt, state, provider, by_layer, layer_ids, metrics)
            logits = getattr(self._model, "_rwkv_ssd_last_logits", None)
            if logits is not None and hasattr(logits, "detach"):
                logits = logits.detach().clone()
                if logits.ndim > 1:
                    logits = logits.reshape(-1, logits.shape[-1])[-1]
            else:
                # Empty prompts (or older reference kernels that do not keep
                # the prefill distribution) have no cached next-token logits.
                # Resolve only those rows with the normal exact token path;
                # non-empty rows still retain the shared batch fast path.
                logits, state = forward_one(
                    self._model,
                    last,
                    state,
                    provider,
                    by_layer,
                    layer_ids=layer_ids,
                    metrics=metrics,
                )
            states.append(state)
            prefill_logits.append(logits)
        outputs = [[] for _ in prompts]
        count = max(0, int(max_tokens))
        for step in range(count):
            # These are the logits after the prompt (first iteration) or
            # after the previously generated token (later iterations).
            generated = [int(value.argmax().item()) for value in prefill_logits]
            for output, token in zip(outputs, generated):
                output.append(token)
            if step + 1 >= count:
                break
            prefill_logits, states = forward_batch_one_dense(
                self._model,
                generated,
                states,
                provider,
                by_layer,
                layer_ids,
                metrics,
            )
        metrics.tokens_generated = len(prompts) * max(0, int(max_tokens))
        return outputs

    def close(self) -> None:
        """Release a qkv/DEA sidecar opened for the CPU reference stream."""
        sidecar = getattr(self._model, "sidecar", None)
        if sidecar is not None and hasattr(sidecar, "close"):
            sidecar.close()
