"""Backend capability flags for engine dispatch (avoid isinstance chains)."""

from __future__ import annotations

from dataclasses import dataclass

from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.backends.pack_backend import PackBackend


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_pack_streaming: bool
    supports_skeleton: bool
    supports_resident: bool
    supports_state_snapshot: bool
    supports_batch: bool = False
    supports_prefix_cache: bool = False
    supports_tokenizer: bool = False
    state_kind: str = "legacy"
    supported_codecs: tuple[str, ...] = ()
    requires_ggml_bin: bool = False
    supports_deepembed: bool = False
    supports_deepembed_streaming: bool = False
    # Common engine contract.  These fields are additive so older callers
    # that only inspect residency/state flags continue to work unchanged.
    supports_token_generation: bool = True
    supports_logits_probe: bool = False
    supports_state_transfer: bool = False
    supports_followup_generation: bool = False
    supports_sampling: bool = False
    supports_incremental_streaming: bool = False
    supports_cancellation: bool = False
    supports_deadlines: bool = False
    supports_process_workers: bool = True
    supports_native_layer_streaming: bool = False
    supports_native_cached_layer_step: bool = False


def probe_backend_capabilities(backend: RecurrentBackend) -> dict[str, object]:
    """Combine declared capabilities with callable/runtime-native probes."""
    declared = backend_capabilities(backend)
    model = getattr(backend, "_model", None)
    return {
        "name": declared.name,
        "declared": {
            "pack_streaming": declared.supports_pack_streaming,
            "skeleton": declared.supports_skeleton,
            "resident": declared.supports_resident,
            "state_snapshot": declared.supports_state_snapshot,
            "batch": declared.supports_batch,
            "prefix_cache": declared.supports_prefix_cache,
            "tokenizer": declared.supports_tokenizer,
            "state_kind": declared.state_kind,
            "supported_codecs": list(declared.supported_codecs),
            "requires_ggml_bin": declared.requires_ggml_bin,
            "deepembed": declared.supports_deepembed,
            "deepembed_streaming": declared.supports_deepembed_streaming,
            "token_generation": declared.supports_token_generation,
            "logits_probe": declared.supports_logits_probe,
            "state_transfer": declared.supports_state_transfer,
            "followup_generation": declared.supports_followup_generation,
            "sampling": declared.supports_sampling,
            "incremental_streaming": declared.supports_incremental_streaming,
            "cancellation": declared.supports_cancellation,
            "deadlines": declared.supports_deadlines,
            "process_workers": declared.supports_process_workers,
            "native_layer_streaming": declared.supports_native_layer_streaming,
            "native_cached_layer_step": declared.supports_native_cached_layer_step,
        },
        "runtime": {
            "loaded": model is not None,
            "token_generation": callable(getattr(backend, "generate", None))
            or callable(getattr(backend, "generate_greedy", None))
            or callable(getattr(backend, "generate_greedy_native", None)),
            "logits_probe": callable(getattr(backend, "probe_logits", None))
            or bool(getattr(model, "supports_logits_probe", False)),
            "state_get": callable(getattr(backend, "get_recurrent_state", None)),
            "state_set": callable(getattr(backend, "set_recurrent_state", None)),
            "followup_generation": callable(getattr(backend, "prefill_text", None))
            and callable(getattr(backend, "decode_greedy", None)),
            "incremental_streaming": callable(
                getattr(backend, "decode_greedy", None)
            ),
            "sampling": callable(getattr(backend, "generate", None))
            or callable(getattr(backend, "generate_greedy_native", None)),
            "cancellation": True,
            "deadlines": True,
            "batch_generate": (
                declared.supports_batch
                or callable(getattr(backend, "generate_greedy_batch", None))
                or callable(getattr(backend, "generate_greedy_batch_streaming", None))
            ),
            "tensor_upload": bool(getattr(model, "supports_tensor_upload", False)),
            "tensor_slot_upload": bool(
                getattr(model, "supports_tensor_slot_upload", False)
                and callable(getattr(model, "register_tensor_slot", None))
                and callable(getattr(model, "set_tensor_from_slot", None))
            ),
            "native_layer_streaming": bool(
                getattr(model, "supports_layer_streaming", False)
                and callable(getattr(model, "layer_step", None))
            ),
            "native_cached_layer_step": bool(
                getattr(model, "supports_layer_cached_step", False)
                and callable(getattr(model, "layer_step_cached", None))
                and callable(getattr(model, "layer_cache_ready", None))
            ),
            "deepembed": {
                "sidecar_reader": bool(declared.supports_deepembed),
                "model_forward": bool(getattr(backend, "_deepembed", False)),
                "streaming": bool(getattr(backend, "_deepembed_streaming", False)),
            },
        },
    }


def backend_capabilities(backend: RecurrentBackend) -> BackendCapabilities:
    from rwkv_ssd.backends.albatross import AlbatrossBackend
    from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
    from rwkv_ssd.backends.kimi_k3 import KimiK3CPUBackend
    from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend

    if isinstance(backend, AlbatrossBackend):
        return BackendCapabilities(
            name="albatross",
            supports_pack_streaming=True,
            supports_skeleton=False,
            supports_resident=True,
            # Arbitrary CUDA extension state is not portable through the
            # generic snapshot serializer; in-memory state transfer remains
            # supported through RecurrentState.external_state.
            supports_state_snapshot=False,
            supports_batch=False,
            supports_prefix_cache=True,
            supports_tokenizer=True,
            state_kind="external",
            supported_codecs=(
                "none",
                "scale_u8",
                "scale_u8_grouped",
                "scale_u4",
                "trinity_lut2",
            ),
            supports_logits_probe=True,
            supports_state_transfer=True,
            supports_followup_generation=True,
            supports_sampling=True,
            supports_incremental_streaming=True,
            supports_cancellation=True,
            supports_deadlines=True,
            supports_process_workers=True,
            supports_native_layer_streaming=True,
            supports_native_cached_layer_step=False,
        )
    if isinstance(backend, PackBackend):
        sequence_kind = str(getattr(backend, "sequence_kind", "legacy"))
        return BackendCapabilities(
            name=type(backend).__name__,
            supports_pack_streaming=True,
            supports_skeleton=False,
            supports_resident=True,
            supports_state_snapshot=True,
            supports_batch=bool(getattr(backend, "supports_batch", False)),
            supports_prefix_cache=False,
            supports_tokenizer=False,
            state_kind=sequence_kind,
            supported_codecs=("none", "scale_u8", "scale_u8_grouped", "scale_u4"),
            supports_deepembed=False,
            supports_deepembed_streaming=False,
            supports_logits_probe=False,
            supports_state_transfer=True,
            supports_followup_generation=True,
            supports_sampling=True,
            supports_incremental_streaming=True,
            supports_cancellation=True,
            supports_deadlines=True,
            supports_process_workers=True,
        )
    if isinstance(backend, ChatRWKVBackend):
        return BackendCapabilities(
            name="chatrwkv",
            supports_pack_streaming=True,
            supports_skeleton=True,
            supports_resident=True,
            supports_state_snapshot=True,
            supports_prefix_cache=True,
            state_kind="rwkv7",
            supports_deepembed=True,
            # Both DeepEmbed contracts have CPU reference streaming; qkv/DEA
            # additionally requires its DeepEmbed.bin sidecar.
            supports_deepembed_streaming=True,
            supports_logits_probe=True,
            supports_state_transfer=True,
            supports_followup_generation=True,
            supports_sampling=True,
            supports_incremental_streaming=True,
            supports_cancellation=True,
            supports_deadlines=True,
            supports_process_workers=True,
        )
    if isinstance(backend, KimiK3CPUBackend):
        return BackendCapabilities(
            name="kimi_k3",
            supports_pack_streaming=False,
            supports_skeleton=False,
            supports_resident=True,
            supports_state_snapshot=True,
            supports_batch=False,
            supports_prefix_cache=True,
            supports_tokenizer=True,
            state_kind="kimi_k3",
            supports_logits_probe=True,
            supports_state_transfer=True,
            supports_followup_generation=True,
            supports_sampling=True,
            supports_incremental_streaming=True,
            supports_cancellation=True,
            supports_deadlines=True,
            supports_process_workers=True,
            supports_native_layer_streaming=False,
            supports_native_cached_layer_step=False,
        )
    if isinstance(backend, RWKVCppBackend):
        return BackendCapabilities(
            name="rwkvcpp",
            supports_pack_streaming=True,
            supports_skeleton=True,
            supports_resident=True,
            supports_state_snapshot=True,
            state_kind="external",
            requires_ggml_bin=True,
            supports_deepembed=False,
            supports_deepembed_streaming=False,
            supports_logits_probe=True,
            supports_state_transfer=True,
            supports_followup_generation=True,
            supports_sampling=True,
            supports_incremental_streaming=True,
            supports_cancellation=True,
            supports_deadlines=True,
            supports_process_workers=True,
            supports_native_layer_streaming=False,
            supports_native_cached_layer_step=False,
        )
    return BackendCapabilities(
        name=type(backend).__name__,
        supports_pack_streaming=False,
        supports_skeleton=False,
        supports_resident=True,
        supports_state_snapshot=False,
        supports_token_generation=False,
        supports_process_workers=False,
    )


def supports_true_streaming(backend: RecurrentBackend) -> bool:
    return backend_capabilities(backend).supports_pack_streaming


def capability_supported(backend: RecurrentBackend, capability: str) -> bool:
    """Query one common engine capability by its public snake-case name."""
    capabilities = backend_capabilities(backend)
    key = str(capability).strip().lower()
    aliases = {
        "generation": "supports_token_generation",
        "token_generation": "supports_token_generation",
        "logits": "supports_logits_probe",
        "logits_probe": "supports_logits_probe",
        "state": "supports_state_transfer",
        "state_transfer": "supports_state_transfer",
        "followup": "supports_followup_generation",
        "followup_generation": "supports_followup_generation",
        "streaming": "supports_incremental_streaming",
        "incremental_streaming": "supports_incremental_streaming",
        "sampling": "supports_sampling",
        "batch": "supports_batch",
        "batching": "supports_batch",
        "prefix_cache": "supports_prefix_cache",
        "state_snapshot": "supports_state_snapshot",
        "stream": "supports_incremental_streaming",
        "deadline": "supports_deadlines",
        "process_worker": "supports_process_workers",
        "process_workers": "supports_process_workers",
        "native_layer_streaming": "supports_native_layer_streaming",
        "native_cached_layer_step": "supports_native_cached_layer_step",
        "cached_layer_step": "supports_native_cached_layer_step",
        "cached_token_step": "supports_native_cached_layer_step",
        "cached_token": "supports_native_cached_layer_step",
    }
    attr = aliases.get(key, key if key.startswith("supports_") else f"supports_{key}")
    declared_value = bool(getattr(capabilities, attr, False))
    if declared_value:
        return True
    # Native rwkv.cpp layer streaming is a loaded-library capability rather
    # than a property of the Python wrapper alone.  Keep the static report
    # conservative before load, but let callers query the actual ABI after a
    # model has been constructed.
    if attr == "supports_native_layer_streaming":
        model = getattr(backend, "_model", None)
        return bool(
            type(getattr(model, "supports_layer_streaming", False)) is bool
            and getattr(model, "supports_layer_streaming", False)
        )
    if attr == "supports_native_cached_layer_step":
        model = getattr(backend, "_model", None)
        return bool(
            getattr(model, "supports_layer_cached_step", False)
            and callable(getattr(model, "layer_step_cached", None))
            and callable(getattr(model, "layer_cache_ready", None))
        )
    return False
