from __future__ import annotations

import pytest
import torch

from rwkv_ssd.runtime.ggml_slot_arena import LayerSlotArena
from rwkv_ssd.runtime.ggml_weight_bridge import GgmlBridgeError, GgmlWeightBridge


def test_layer_slot_arena_reuses_fixed_memory_and_evicts_lru() -> None:
    arena = LayerSlotArena(slot_count=2, slot_bytes=32)
    first, evicted = arena.acquire(0)
    assert evicted is None
    assert first.write(b"abcd") == (0, 4)
    second, _ = arena.acquire(1)
    assert second.slot_id != first.slot_id
    arena.acquire(0)  # layer 0 becomes most recently used
    third, evicted = arena.acquire(2)
    assert evicted == 1
    assert third.slot_id == second.slot_id
    assert arena.allocated_bytes == 64
    assert arena.evictions == 1


def test_layer_slot_arena_rejects_payload_over_capacity() -> None:
    arena = LayerSlotArena(slot_count=1, slot_bytes=4)
    lease, _ = arena.acquire(0)
    try:
        lease.write(b"12345")
    except MemoryError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("oversized slot payload should fail")


class _SlotModel:
    supports_tensor_upload = True
    supports_tensor_slot_upload = True

    def __init__(self) -> None:
        self.slots = {}
        self.uploads = []

    def register_tensor_slot(self, slot_id, buffer) -> None:
        self.slots[slot_id] = buffer

    def tensor_nbytes(self, name: str) -> int:
        return 8 if name.endswith(".weight") else 0

    def set_tensor(self, name: str, payload: bytes) -> None:
        raise AssertionError("slot-capable model should not use copying upload")

    def set_tensor_from_slot(
        self, name, slot_id, offset, length, generation
    ) -> None:
        payload = bytes(self.slots[slot_id][offset : offset + length])
        self.uploads.append((name, slot_id, generation, payload))


def test_ggml_bridge_uses_slot_abi_and_invalidates_evicted_signature() -> None:
    model = _SlotModel()
    bridge = GgmlWeightBridge(model, slot_count=1, slot_bytes=32)
    layer0 = {"blocks.0.weight": torch.tensor([1.0, 2.0])}
    layer1 = {"blocks.1.weight": torch.tensor([3.0, 4.0])}
    assert bridge.upload_layer(0, layer0).uploaded_tensors == 1
    assert bridge.upload_layer(1, layer1).uploaded_tensors == 1
    assert bridge.upload_layer(0, layer0).uploaded_tensors == 1
    assert len(model.uploads) == 3
    assert bridge.slot_stats() == {
        "enabled": True,
        "slot_count": 1,
        "slot_bytes": 32,
        "allocated_bytes": 32,
        "evictions": 2,
    }


class _PointerModel:
    supports_tensor_upload = True
    supports_tensor_slot_upload = False

    def __init__(self) -> None:
        self.uploads = []

    def tensor_nbytes(self, name: str) -> int:
        return 8 if name.endswith(".weight") else 0

    def set_tensor(self, name: str, payload: bytes) -> None:
        raise AssertionError("pointer-capable model should not allocate a bytes payload")

    def set_tensor_from_pointer(self, name: str, address: int, nbytes: int) -> None:
        self.uploads.append((name, address, nbytes))


def test_ggml_bridge_uses_pointer_upload_without_python_bytes_copy() -> None:
    model = _PointerModel()
    bridge = GgmlWeightBridge(model)
    stats = bridge.upload_layer(0, {"blocks.0.weight": torch.tensor([1.0, 2.0])})
    assert stats.uploaded_tensors == 1
    assert stats.uploaded_bytes == 8
    assert len(model.uploads) == 1
    assert model.uploads[0][0] == "blocks.0.weight"
    assert model.uploads[0][1] > 0
    assert model.uploads[0][2] == 8


class _NativeU8Model:
    supports_tensor_upload = True
    supports_tensor_slot_upload = False
    supports_native_u8 = True

    def __init__(self) -> None:
        self.uploads = []

    def tensor_nbytes(self, name: str) -> int:
        return 8 if name in {"emb.weight", "head.weight"} else 0

    def set_tensor_scale_u8_grouped(self, name: str, payload) -> None:
        self.uploads.append((name, bytes(payload)))


def test_ggml_bridge_uploads_global_native_u8_matrices() -> None:
    model = _NativeU8Model()
    bridge = GgmlWeightBridge(model)
    blob = b"SG8\x01" + (4).to_bytes(4, "little") + b"payload"

    emb = bridge.upload_layer(0, {"emb.weight": blob})
    head = bridge.upload_layer(9999, {"head.weight": blob})

    assert emb.uploaded_tensors == 1
    assert head.uploaded_tensors == 1
    assert [name for name, _ in model.uploads] == ["emb.weight", "head.weight"]


class _MixedNativeLayerModel:
    supports_tensor_upload = True
    supports_tensor_slot_upload = False
    supports_native_u8 = True
    supports_layer_streaming = True
    supports_layer_borrowed_dense = True

    def __init__(self) -> None:
        self.grouped: list[tuple[str, bytes]] = []
        self.dense: list[tuple[str, bytes]] = []

    def layer_begin(self, layer_id: int) -> None:
        self.active_layer = layer_id

    def tensor_nbytes(self, name: str) -> int:
        if name.endswith("key.weight"):
            return 12
        if name.endswith("ln1.weight"):
            return 4
        return 0

    def tensor_type(self, name: str) -> int:
        return 1 if name.endswith("ln1.weight") else -1

    def layer_set_tensor_scale_u8_grouped_borrowed(
        self, name: str, payload
    ) -> None:
        self.grouped.append((name, bytes(payload)))

    def layer_set_tensor_borrowed(self, name: str, tensor: torch.Tensor) -> None:
        self.dense.append((name, tensor.numpy().tobytes()))


def test_ggml_bridge_borrows_mixed_grouped_u8_and_dense_layer_controls() -> None:
    model = _MixedNativeLayerModel()
    bridge = GgmlWeightBridge(model)
    grouped = b"SG8\x01" + (4).to_bytes(4, "little") + b"payload"
    dense = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)

    stats = bridge.upload_layer(
        0,
        {
            "blocks.0.att.key.weight": grouped,
            "blocks.0.ln1.weight": dense,
        },
    )

    assert stats.uploaded_tensors == 2
    assert model.grouped == [("blocks.0.att.key.weight", grouped)]
    expected = dense.to(dtype=torch.float16).numpy().tobytes()
    assert model.dense == [("blocks.0.ln1.weight", expected)]


class _DenseDtypeModel:
    supports_tensor_upload = True
    supports_tensor_slot_upload = False

    def __init__(self, target_type: int | None) -> None:
        self.target_type = target_type
        self.uploads: list[tuple[str, bytes]] = []

    def tensor_nbytes(self, name: str) -> int:
        if name != "blocks.0.weight":
            return 0
        return 12 if self.target_type == 0 else 6

    def tensor_type(self, name: str) -> int:
        if self.target_type is None:
            return -1
        return self.target_type

    def set_tensor(self, name: str, payload: bytes) -> None:
        self.uploads.append((name, bytes(payload)))


@pytest.mark.parametrize(
    ("target_type", "target_dtype"),
    [(0, torch.float32), (1, torch.float16), (30, torch.bfloat16)],
)
def test_ggml_bridge_converts_dense_values_to_loaded_target_dtype(
    target_type: int, target_dtype: torch.dtype
) -> None:
    model = _DenseDtypeModel(target_type)
    source = torch.tensor([1.25, -2.5, 3.75], dtype=torch.bfloat16)

    stats = GgmlWeightBridge(model).upload_layer(0, {"blocks.0.weight": source})

    assert stats.uploaded_tensors == 1
    expected = source.to(dtype=target_dtype).contiguous()
    expected_bytes = (
        expected.view(torch.uint16).numpy().tobytes()
        if target_dtype == torch.bfloat16
        else expected.numpy().tobytes()
    )
    assert model.uploads == [("blocks.0.weight", expected_bytes)]


def test_ggml_bridge_rejects_ambiguous_two_byte_upload_without_type_abi() -> None:
    model = _DenseDtypeModel(None)
    source = torch.tensor([1.25, -2.5, 3.75], dtype=torch.bfloat16)

    with pytest.raises(GgmlBridgeError, match="dtype introspection"):
        GgmlWeightBridge(model).upload_layer(0, {"blocks.0.weight": source})
