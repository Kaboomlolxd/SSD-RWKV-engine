"""
RWKV-7 skeleton model: resident globals in ``model.z``, block weights from pack each layer.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.device import resolve_device
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.rwkv7_weights import prepare_rwkv7_tensor_for_z, tensors_from_entries
from rwkv_ssd.runtime.tensor_loader import tensor_from_bytes
from rwkv_ssd.runtime.weight_store import WeightStore, open_weight_store


def parse_rwkv7_strategy(strategy: str) -> tuple[str, torch.dtype]:
    parts = strategy.strip().split()
    device = parts[0] if parts else "cpu"
    kind = parts[1].lower() if len(parts) > 1 else "fp32"
    if kind == "bf16":
        return device, torch.bfloat16
    if kind == "fp16":
        return device, torch.half
    return device, torch.float32


def is_global_z_key(name: str) -> bool:
    if name in ("emb.weight", "head.weight"):
        return True
    return name.startswith("ln_out.")


def layer_id_from_z_key(name: str) -> int | None:
    if "blocks." not in name:
        return None
    try:
        return int(name.split("blocks.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def estimate_z_bytes(z: dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in z.values())


def entries_for_init(
    manifest: Manifest,
    *,
    resident_layer_ids: set[int] | None = None,
) -> list[TensorEntry]:
    """Tensors to keep in ``z`` after skeleton build (globals + optional hot layers)."""
    resident = resident_layer_ids or set()
    out: list[TensorEntry] = []
    for entry in manifest.tensors:
        if is_global_z_key(entry.name):
            out.append(entry)
            continue
        if entry.name in ("blocks.0.ln0.weight", "blocks.0.ln0.bias"):
            out.append(entry)
            continue
        if not entry.name.startswith("blocks."):
            continue
        lid = entry.layer_id if entry.layer_id >= 0 else layer_id_from_z_key(entry.name)
        if lid is not None and lid in resident:
            out.append(entry)
    return out


def merge_emb_ln0(z: dict[str, torch.Tensor], n_embd: int) -> None:
    ln0_w = z.pop("blocks.0.ln0.weight", None)
    ln0_b = z.pop("blocks.0.ln0.bias", None)
    if ln0_w is None or ln0_b is None:
        return
    z["emb.weight"] = F.layer_norm(
        z["emb.weight"], (n_embd,), weight=ln0_w, bias=ln0_b
    )


def head_dims_from_manifest(manifest: Manifest) -> tuple[int, int]:
    for entry in manifest.tensors:
        if entry.name == "blocks.0.att.r_k":
            if len(entry.shape) == 2:
                return int(entry.shape[0]), int(entry.shape[1])
            flat = int(entry.shape[0]) if entry.shape else 1
            return flat, 1
    raise ValueError("pack missing blocks.0.att.r_k (needed for head_size)")


def evict_block_weights_from_z(
    z: dict[str, torch.Tensor],
    *,
    resident_layer_ids: set[int] | None = None,
) -> int:
    """Drop block tensors from ``z``. Returns approximate bytes freed."""
    resident = resident_layer_ids or set()
    freed = 0
    for key in list(z.keys()):
        if is_global_z_key(key):
            continue
        lid = layer_id_from_z_key(key)
        if lid is None:
            continue
        if lid in resident:
            continue
        tensor = z.pop(key)
        freed += int(tensor.numel() * tensor.element_size())
    return freed


def attach_rwkv7_decode_runtime(model: Any, device: torch.device, dtype: torch.dtype) -> None:
    """Skeleton models skip ``RWKV_x070.__init__``; bind state + ChatRWKV globals."""
    n_embd = model.n_embd
    n_layer = model.n_layer
    head_size = model.head_size

    def generate_zero_state() -> list[torch.Tensor]:
        state: list[torch.Tensor | None] = [None] * (n_layer * 3)
        for i in range(n_layer):
            state[i * 3 + 0] = torch.zeros(n_embd, dtype=dtype, device=device)
            state[i * 3 + 1] = torch.zeros(
                (n_embd // head_size, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )
            state[i * 3 + 2] = torch.zeros(n_embd, dtype=dtype, device=device)
        return state  # type: ignore[return-value]

    model.generate_zero_state = generate_zero_state

    cls = type(model)
    for name in ("forward", "forward_one", "forward_seq"):
        if not hasattr(model, name):
            attr = getattr(cls, name, None)
            if attr is not None:
                setattr(model, name, attr.__get__(model, cls))

    try:
        import rwkv.model as rwkv_model  # type: ignore[import-untyped]

        rwkv_model.DEVICE = device
        rwkv_model.DTYPE = dtype
    except ImportError:
        pass


def build_rwkv7_skeleton_from_pack(
    pack_dir: Path,
    strategy: str,
    *,
    resident_layer_ids: set[int] | None = None,
) -> Any:
    """
    Construct RWKV-7 ``RWKV_x070`` with only global (and optional resident layer) weights.

    Does not read block weights from the ``.pth`` checkpoint.
    """
    device_name, dtype = parse_rwkv7_strategy(strategy)
    device = resolve_device(device_name)
    # ``resolve_device`` may demote an XPU that can allocate tensors but
    # cannot create a matrix engine. Do not leave a CPU fallback carrying an
    # accelerator BF16/FP16 strategy.
    if device.type == "cpu" and device_name.lower().startswith("xpu"):
        dtype = torch.float32
    manifest = Manifest.load(pack_dir)
    meta = manifest.meta
    n_embd = int(meta.get("n_embd", 768))
    n_layer = int(meta.get("n_layer", 12))
    vocab_size = int(meta.get("vocab_size", 65536))
    n_head, head_size = head_dims_from_manifest(manifest)

    init_entries = entries_for_init(manifest, resident_layer_ids=resident_layer_ids)
    # Sharded pack (M-class): the skeleton's ``emb.weight`` and
    # ``head.weight`` may live in any shard (round-robin by
    # ``layer_id`` puts emb in shard 0 and head in shard
    # ``(n_layer - 1) % n_shards``). Use the sharded store so the
    # per-entry ``shard_file`` is honored.
    if manifest.is_sharded():
        from rwkv_ssd.runtime.weight_store_sharded import (
            open_sharded_weight_store,
        )

        store = open_sharded_weight_store(
            manifest, parallel_workers=len(manifest.shard_files)
        )
    else:
        store = open_weight_store(manifest.weights_path)
    try:
        raw = {e.name: store.read_bytes(e) for e in init_entries}
        z = tensors_from_entries(raw, init_entries, device)
        for name, tensor in z.items():
            z[name] = tensor.to(device=device, dtype=dtype)
        if "emb.weight" in z:
            n_embd = int(z["emb.weight"].shape[1])
        merge_emb_ln0(z, n_embd)
    finally:
        store.close()

    from rwkv.model import RWKV  # type: ignore[import-untyped]  # RWKV_x070 when V7 on

    model = object.__new__(RWKV)
    model.args = types.SimpleNamespace(
        MODEL_NAME="",
        n_embd=n_embd,
        n_layer=n_layer,
        head_size=head_size,
        vocab_size=vocab_size,
    )
    model.z = z
    model.n_embd = n_embd
    model.n_layer = n_layer
    model.n_head = n_head
    model.head_size = head_size
    attach_rwkv7_decode_runtime(model, device, dtype)
    return model


def apply_skeleton_to_loaded_model(
    model: Any,
    manifest: Manifest,
    *,
    resident_layer_ids: set[int] | None = None,
) -> int:
    """Evict block weights from an already-loaded model (full ``.pth`` path)."""
    return evict_block_weights_from_z(model.z, resident_layer_ids=resident_layer_ids)
