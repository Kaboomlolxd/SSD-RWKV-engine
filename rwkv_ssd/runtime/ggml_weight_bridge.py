"""Translate provider tensors into the byte layout expected by rwkv.cpp.

The pack runtime deliberately exposes PyTorch-shaped tensors.  rwkv.cpp's
ggml files use the same logical matrices but store a few RWKV-7 projections
transposed and concatenate the six time-mix vectors into ``x_rwkvag``.  This
module keeps that compatibility logic in one place and uploads only changed
layers through the small native tensor-upload ABI.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass
from typing import Any

import torch

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.ggml_slot_arena import LayerSlotArena
from rwkv_ssd.runtime.prepared_artifact_cache import PreparedArtifactCache


class GgmlBridgeError(RuntimeError):
    """Raised when a pack tensor cannot be represented by the loaded ggml model."""


_RWKV7_TRANSPOSED = frozenset(
    {"att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2"}
)
_RWKV7_X_ORDER = ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
_RWKV7_X_SOURCE_SUFFIXES = tuple(f"att.{name}" for name in _RWKV7_X_ORDER)

# Keep these values local to the bridge so the Python runtime does not need to
# import private ggml headers.  They are stable public enum values in the
# vendored ggml ABI.
_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_BF16 = 30


@dataclass(frozen=True)
class UploadStats:
    layer_id: int
    uploaded_tensors: int
    uploaded_bytes: int
    skipped_tensors: int
    # ``uploaded_bytes`` is the source payload sent through the ABI.  For a
    # grouped-U8 record that is packed bytes; ``active_bytes`` is the decoded
    # GGML tensor footprint that becomes resident in the one active layer plan.
    packed_bytes: int = 0
    active_bytes: int = 0


def _layer_prefix(layer_id: int) -> str:
    return f"blocks.{layer_id}."


def _native_name(layer_id: int, suffix: str) -> str:
    return f"blocks.{layer_id}.{suffix}"


def _source_for_target(
    target: str,
    tensors: dict[str, Any],
    layer_id: int,
) -> Any | None:
    """Resolve one ggml target from pack/PyTorch tensor names."""
    prefix = _layer_prefix(layer_id)
    suffix = target.removeprefix(prefix)
    direct = tensors.get(target)
    if direct is not None:
        return direct
    if suffix == "att.x_rwkvag":
        parts = [tensors.get(prefix + "att." + name) for name in _RWKV7_X_ORDER]
        if all(t is not None for t in parts):
            return torch.cat([t.reshape(-1) for t in parts if t is not None], dim=0)
        return None
    return None


def _convert_for_ggml(target: str, tensor: torch.Tensor) -> torch.Tensor:
    """Mirror ``convert_pytorch_to_ggml.py`` for one RWKV-7 tensor."""
    t = tensor.detach().to(device="cpu").contiguous()
    suffix = target.split(".", 2)[-1]
    # suffix is ``att.w1`` / ``att.key.weight`` etc.
    if suffix in _RWKV7_TRANSPOSED and t.ndim >= 2:
        t = t.transpose(0, 1).contiguous()
    if ".time_" in target:
        t = t.squeeze().contiguous()
    return t


def _native_tensor(
    tensor: torch.Tensor,
    expected_nbytes: int,
    target: str,
    target_type: int | None = None,
) -> torch.Tensor:
    """Convert values to the exact dense dtype stored by the GGML target.

    Byte size alone cannot distinguish FP16 from BF16.  New rwkv.cpp builds
    provide ``tensor_type``; an older library is accepted for unambiguous
    FP32 uploads but rejects dense two-byte uploads rather than reinterpreting
    BF16 bits as FP16 (or vice versa).
    """
    dtype_by_type = {
        _GGML_TYPE_F32: torch.float32,
        _GGML_TYPE_F16: torch.float16,
        _GGML_TYPE_BF16: torch.bfloat16,
    }
    if target_type is not None:
        try:
            dtype = dtype_by_type[int(target_type)]
        except (KeyError, TypeError, ValueError) as exc:
            raise GgmlBridgeError(
                f"{target}: loaded rwkv.cpp tensor type {target_type!r} is not "
                "a supported dense FP32/FP16/BF16 target"
            ) from exc
        expected_element_size = 4 if dtype == torch.float32 else 2
        if tensor.numel() * expected_element_size != expected_nbytes:
            raise GgmlBridgeError(
                f"{target}: ggml type {target_type} expects {expected_nbytes} "
                f"bytes, but the pack tensor has {tensor.numel()} elements"
            )
    elif tensor.numel() * 4 == expected_nbytes:
        dtype = torch.float32
    elif tensor.numel() * 2 == expected_nbytes:
        if tensor.dtype in (torch.float16, torch.bfloat16):
            raise GgmlBridgeError(
                f"{target}: loaded rwkv.cpp library does not expose tensor "
                "dtype introspection; refusing an ambiguous two-byte upload "
                "(rebuild the vendored backend with rwkv_get_tensor_type)"
            )
        dtype = torch.float16
    else:
        raise GgmlBridgeError(
            f"{target}: pack tensor has {tensor.numel()} elements but rwkv.cpp "
            f"expects {expected_nbytes} bytes; native quantized upload is not "
            "safe without a ggml quantizer"
        )
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _native_bytes(
    tensor: torch.Tensor,
    expected_nbytes: int,
    target: str,
    target_type: int | None = None,
) -> bytes:
    return _native_payload(
        _native_tensor(tensor, expected_nbytes, target, target_type)
    )


def _native_payload(tensor: torch.Tensor) -> bytes:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


class GgmlWeightBridge:
    """Upload provider-decoded layers into a resident rwkv.cpp graph.

    The graph itself stays resident in ggml, so uploads do not invalidate its
    allocations or scheduler.  ``set_tensor`` is skipped when provider cache
    returns the same tensor objects, which is the important F2/F3/F5 fast path.
    """

    def __init__(
        self,
        model: Any,
        *,
        slot_count: int = 0,
        slot_bytes: int = 0,
        artifact_cache: PreparedArtifactCache | None = None,
    ) -> None:
        self.model = model
        self._layer_upload = bool(getattr(model, "supports_layer_streaming", False))
        self._persistent_borrowed = False
        # Packed grouped-U8 records may safely borrow a read-only mmap for the
        # lifetime of the engine even when the accompanying dense controls are
        # transient. Keep those lifetime policies separate so a temporary
        # decoded tensor is never retained through a native pointer.
        self._persistent_packed_borrowed = False
        if not self._layer_upload and not bool(
            getattr(model, "supports_tensor_upload", False)
        ):
            raise GgmlBridgeError(
                "rwkv.cpp library lacks both the resident tensor-upload ABI "
                "and the native layer-streaming ABI; rebuild "
                "backends/rwkvcpp_ref before using pack streaming"
            )
        self._signatures: dict[int, tuple[tuple[str, int, int], ...]] = {}
        cache_dir = os.environ.get("RWKV_PREPARED_ARTIFACT_CACHE", "").strip()
        self._artifact_cache = artifact_cache or (
            PreparedArtifactCache(cache_dir) if cache_dir else None
        )
        self._slot_arena = (
            LayerSlotArena(slot_count, slot_bytes)
            if slot_count > 0 and slot_bytes > 0
            else None
        )
        self._slot_upload = bool(
            not self._layer_upload
            and self._slot_arena is not None
            and getattr(model, "supports_tensor_slot_upload", False)
            and callable(getattr(model, "set_tensor_from_slot", None))
            and callable(getattr(model, "register_tensor_slot", None))
        )
        if self._slot_upload:
            for lease in self._slot_arena.leases:
                model.register_tensor_slot(lease.slot_id, lease.buffer)

    def set_borrowed_persistent(self, enabled: bool) -> None:
        """Allow packed layer views to be retained across layer switches.

        The provider uses this only when its own bounded layer cache owns the
        memoryview objects.  Strict F1 leaves it disabled, so the native ABI
        borrows only until the next active layer.
        """
        self._persistent_borrowed = bool(enabled)
        self._persistent_packed_borrowed = bool(enabled)

    def set_packed_borrowed_persistent(self, enabled: bool) -> None:
        """Set the lifetime policy for packed grouped-U8 payloads only."""
        self._persistent_packed_borrowed = bool(enabled)

    def _signature(self, tensors: dict[str, Any], layer_id: int) -> tuple[tuple[str, int, int], ...]:
        prefix = _layer_prefix(layer_id)
        global_names = (
            set()
            if self._layer_upload
            else ({"emb.weight"} if layer_id == 0 else set())
        )
        if layer_id == 9999 and not self._layer_upload:
            global_names.add("head.weight")
        items: list[tuple[str, int, int]] = []
        for name, tensor in sorted(tensors.items()):
            if name.startswith(prefix) or name in global_names:
                if isinstance(tensor, torch.Tensor):
                    size = int(tensor.numel())
                elif isinstance(tensor, (bytes, bytearray, memoryview)):
                    size = len(tensor)
                else:
                    size = 0
                items.append((name, id(tensor), size))
        # The concatenated x_rwkvag target depends on six source objects.
        for name in _RWKV7_X_ORDER:
            tensor = tensors.get(prefix + "att." + name)
            if tensor is not None:
                items.append((prefix + "att.x_rwkvag:" + name, id(tensor), int(tensor.numel())))
        return tuple(items)

    def upload_layer(
        self,
        layer_id: int,
        tensors: dict[str, Any],
        *,
        force: bool = False,
        layer_started: bool = False,
    ) -> UploadStats:
        prefix = _layer_prefix(layer_id)
        signature = self._signature(tensors, layer_id)
        if (
            not self._layer_upload
            and not force
            and signature
            and self._signatures.get(layer_id) == signature
        ):
            return UploadStats(layer_id, 0, 0, 0)

        if self._layer_upload and not layer_started:
            if layer_id == 9999:
                raise GgmlBridgeError(
                    "native layer-streaming accepts block tensors only; "
                    "global tensors must be handled by the provider/global path"
                )
            self.model.layer_begin(layer_id)

        uploaded = 0
        skipped = 0
        total_bytes = 0
        packed_bytes = 0
        active_bytes = 0
        slot_lease = None
        grouped_batch: list[tuple[str, bytes | memoryview | bytearray]] = []
        # Borrowing is the cheapest dense path when the provider owns the
        # payload for the lifetime of the layer.  For streamed dense layers it
        # also prevents rwkv.cpp's bounded decoded-layer cache from retaining
        # the already-converted target bytes: the pointer is invalidated at
        # the next layer begin and Python has to decode/convert it again.  An
        # explicit native cache budget opts those layers into the owned upload
        # ABI.  The native cache remains bounded; payloads that do not fit fall
        # back to the normal active-layer path.
        native_cache_limit = 0
        if self._layer_upload:
            cache_stats = getattr(self.model, "layer_cache_stats", None)
            if callable(cache_stats):
                try:
                    native_cache_limit = max(
                        0, int(dict(cache_stats()).get("limit_bytes", 0) or 0)
                    )
                except (TypeError, ValueError, RuntimeError):
                    native_cache_limit = 0
        native_owned_dense = native_cache_limit > 0
        if self._slot_upload and self._slot_arena is not None:
            slot_lease, evicted_layer = self._slot_arena.acquire(layer_id)
            if evicted_layer is not None:
                self._signatures.pop(evicted_layer, None)
        # Read the loaded ggml model's target names from the canonical RWKV-7
        # layout. Unknown optional names (notably block-0 v0/v1/v2) are fine.
        candidates: set[str] = {
            name
            for name in tensors
            if name.startswith(prefix)
            and not (
                self._layer_upload
                and layer_id == 0
                and name.startswith("blocks.0.ln0.")
            )
        }
        # The pack groups the embedding with block 0 and the output head in
        # the global 9999 bucket, while neither target follows the
        # ``blocks.N.`` naming prefix.  Include them explicitly so a
        # packed-only native graph receives every matrix required by its
        # readiness check.
        if layer_id == 0 and not self._layer_upload:
            candidates.add("emb.weight")
        if layer_id == 9999 and not self._layer_upload:
            candidates.add("head.weight")
        candidates.add(prefix + "att.x_rwkvag")

        # Direct source names plus the synthetic concatenated target.
        targets = sorted(
            {
                name
                for name in candidates
                if not name.endswith(_RWKV7_X_SOURCE_SUFFIXES)
            }
        )
        for target in targets:
            source = _source_for_target(target, tensors, layer_id)
            if source is None:
                skipped += 1
                continue
            expected = int(self.model.tensor_nbytes(target))
            if expected <= 0:
                skipped += 1
                continue

            # Native grouped-U8 records stay packed through the provider and
            # are copied into rwkv.cpp's model-owned blob store.  The native
            # graph interprets the RWKV-7 transposed adapter names itself, so
            # no Python transpose or float staging allocation is needed here.
            if isinstance(source, (bytes, bytearray, memoryview)):
                if not self._layer_upload and not bool(
                    getattr(self.model, "supports_native_u8", False)
                ):
                    raise GgmlBridgeError(
                        f"{target}: received a packed grouped-U8 blob but the "
                        "loaded rwkv.cpp model has no native-U8 ABI"
                    )
                if self._layer_upload:
                    batch = getattr(
                        self.model,
                        "layer_set_tensors_scale_u8_grouped",
                        None,
                    )
                    borrowed = (
                        getattr(
                            self.model,
                            "layer_set_tensor_scale_u8_grouped_borrowed_persistent",
                            None,
                        )
                        if self._persistent_packed_borrowed
                        else getattr(
                            self.model,
                            "layer_set_tensor_scale_u8_grouped_borrowed",
                            None,
                        )
                    )
                    if callable(batch):
                        grouped_batch.append((target, source))
                    elif callable(borrowed):
                        borrowed(target, source)
                    else:
                        self.model.layer_set_tensor_scale_u8_grouped(target, source)
                else:
                    self.model.set_tensor_scale_u8_grouped(target, source)
                uploaded += 1
                total_bytes += len(source)
                packed_bytes += len(source)
                active_bytes += expected
                continue

            if not isinstance(source, torch.Tensor):
                skipped += 1
                continue
            tensor_type_getter = getattr(self.model, "tensor_type", None)
            target_type: int | None = None
            if callable(tensor_type_getter):
                queried_type = int(tensor_type_getter(target))
                if queried_type >= 0:
                    target_type = queried_type

            native_tensor = _native_tensor(
                _convert_for_ggml(target, source),
                expected,
                target,
                target_type,
            )
            if self._layer_upload:
                borrowed_dense = None
                if self._persistent_borrowed and bool(
                    getattr(self.model, "supports_layer_persistent_dense", False)
                ):
                    borrowed_dense = getattr(
                        self.model, "layer_set_tensor_borrowed_persistent", None
                    )
                elif (
                    not native_owned_dense
                    and bool(getattr(self.model, "supports_layer_borrowed_dense", False))
                ):
                    # Native grouped-U8 and dense controls can coexist in the
                    # same layer-local plan.  The C ABI rejects this route for
                    # a tensor that has a grouped-U8 descriptor, while dense
                    # vectors are safe to borrow until the next layer_begin.
                    # Keeping the guard at the ABI boundary lets mixed packs
                    # avoid copying every normalization/time-mix control on
                    # every autoregressive revisit without retaining a second
                    # model-sized dense payload.
                    borrowed_dense = getattr(
                        self.model, "layer_set_tensor_borrowed", None
                    )
                if callable(borrowed_dense):
                    # Keep the contiguous native-dtype tensor alive in the
                    # model wrapper.  The native plan binds tensor->data to
                    # this provider-owned buffer and never copies it into its
                    # one-block arena.  Persistent binding is selected only
                    # by the provider lifetime policy above; strict F1 uses
                    # the per-layer form and releases it at begin_layer.
                    borrowed_dense(target, native_tensor)
                    uploaded += 1
                    total_bytes += expected
                    active_bytes += expected
                    continue
            if self._artifact_cache is not None:
                source_cpu = source.detach().to(device="cpu").contiguous()
                if source_cpu.dtype == torch.bfloat16:
                    source_bytes = source_cpu.view(torch.uint16).numpy().tobytes()
                else:
                    source_bytes = source_cpu.numpy().tobytes()
                artifact_meta = {
                    "abi": "ggml-weight-bridge-v2",
                    "target": target,
                    "expected_nbytes": expected,
                    "target_type": target_type,
                    "target_dtype": str(native_tensor.dtype),
                    "dtype": str(source_cpu.dtype),
                    "shape": list(source_cpu.shape),
                }
                artifact_key = self._artifact_cache.make_key(
                    "ggml-weight", source_bytes, artifact_meta
                )
                payload = self._artifact_cache.get_or_build(
                    artifact_key,
                    lambda: _native_payload(native_tensor),
                    metadata=artifact_meta,
                )
            else:
                payload = None
            if slot_lease is not None:
                if payload is None:
                    payload = _native_payload(native_tensor)
                slot_offset, slot_length = slot_lease.write(payload)
                self.model.set_tensor_from_slot(
                    target,
                    slot_lease.slot_id,
                    slot_offset,
                    slot_length,
                    slot_lease.generation,
                )
            elif self._layer_upload:
                self.model.layer_set_tensor(target, _native_payload(native_tensor))
            elif payload is not None:
                self.model.set_tensor(target, payload)
            elif callable(getattr(self.model, "set_tensor_from_pointer", None)):
                self.model.set_tensor_from_pointer(
                    target,
                    int(native_tensor.data_ptr()),
                    int(native_tensor.numel() * native_tensor.element_size()),
                )
            else:
                self.model.set_tensor(target, _native_payload(native_tensor))
            uploaded += 1
            total_bytes += expected
            active_bytes += expected

        if grouped_batch:
            batch = getattr(self.model, "layer_set_tensors_scale_u8_grouped")
            batch(grouped_batch, persistent=self._persistent_packed_borrowed)

        if signature:
            self._signatures[layer_id] = signature
        return UploadStats(
            layer_id,
            uploaded,
            total_bytes,
            skipped,
            packed_bytes=packed_bytes,
            active_bytes=active_bytes,
        )

    def begin_layer(self, layer_id: int) -> bool:
        """Begin a native layer and report whether its bounded cache is ready.

        A ready layer has already been restored into the single active native
        plan.  The provider can therefore skip decoding and converting its
        tensors entirely for this token while retaining the same bounded
        residency semantics.
        """
        if not self._layer_upload:
            return False
        if layer_id == 9999:
            raise GgmlBridgeError(
                "native layer-streaming accepts block tensors only; global "
                "tensors must be handled by the provider/global path"
            )
        self.model.layer_begin(layer_id)
        ready = getattr(self.model, "layer_ready", None)
        return bool(callable(ready) and ready(layer_id))

    def invalidate_borrowed_layer(self, layer_id: int) -> None:
        """Invalidate provider-owned pointers before a layer view is evicted."""
        invalidate = getattr(self.model, "layer_invalidate_borrowed_cache", None)
        if callable(invalidate):
            invalidate(int(layer_id))

    def invalidate(self, layer_id: int | None = None) -> None:
        if layer_id is None:
            self._signatures.clear()
        else:
            self._signatures.pop(int(layer_id), None)
        if self._slot_arena is not None:
            self._slot_arena.invalidate(layer_id)

    def slot_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self._slot_upload,
            "slot_count": self._slot_arena.slot_count if self._slot_arena else 0,
            "slot_bytes": self._slot_arena.slot_bytes if self._slot_arena else 0,
            "allocated_bytes": self._slot_arena.allocated_bytes if self._slot_arena else 0,
            "evictions": self._slot_arena.evictions if self._slot_arena else 0,
        }

    def artifact_stats(self) -> dict[str, int]:
        if self._artifact_cache is None:
            return {"hits": 0, "misses": 0, "writes": 0, "corruptions": 0, "evictions": 0, "disk_bytes": 0}
        stats = self._artifact_cache.stats
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "writes": stats.writes,
            "corruptions": stats.corruptions,
            "evictions": stats.evictions,
            "disk_bytes": self._artifact_cache.disk_bytes(),
        }


def bridge_conversion_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
