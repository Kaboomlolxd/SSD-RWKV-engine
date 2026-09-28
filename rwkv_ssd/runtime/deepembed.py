"""DeepEmbed checkpoint metadata, sidecar I/O, and a small reference model.

DeepEmbed is a RWKV-7-derived model variant, not an ordinary RWKV-7 tensor
layout.  The public reference implementation precomputes three vocabulary
tables (``s_emb``, ``k_emb`` and ``v_emb``) and stores them in a separate
memory-mapped file.  This module keeps that file format explicit so packs can
be inspected and served without pretending that ChatRWKV/rwkv.cpp understand
the extra qkv/DEA path.

The reference model below is deliberately CPU-safe and dependency-light. It is
useful for resident inference and now exposes a correctness-first CPU
layer-streaming path for qkv/DEA when a sidecar and provider are supplied. A
fused production SSD forward path still needs a variant-aware scheduler because
a DeepEmbed step consumes ordinary layer weights and context-indexed rows.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


DE_BASE_SUFFIXES = ("s_emb", "k_emb", "v_emb")
DEEP_EMBED_QKV_DEA = "qkv_dea"
DEEP_EMBED_RWKV7A_V1 = "rwkv7a_v1"
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.uint8: 5,
}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}
_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.uint8: 1,
}


def is_deepembed_tensor_name(name: str) -> bool:
    """Return whether a checkpoint key belongs to the DeepEmbed family."""
    parts = name.replace("/", ".").split(".")
    return any(
        part in DE_BASE_SUFFIXES
        or part in {"s_emb_x", "k_emb_x", "v_emb_x"}
        or part.endswith("_emb")
        for part in parts
    )


def detect_deepembed_variant(
    tensors: Mapping[str, torch.Tensor],
) -> str | None:
    """Identify the supported DeepEmbed tensor contract, if present."""
    keys = tuple(tensors)
    has_v1_base = any(
        key.startswith("blocks.") and ".ffn.s_emb.weight" in key
        for key in keys
    )
    has_v1_projection = any(
        key.startswith("blocks.") and ".ffn.s_emb_x.weight" in key
        for key in keys
    )
    has_qkv_lookup = any(
        ".qkv.k_emb" in key or ".qkv.v_emb" in key for key in keys
    )
    has_dea = any(
        ".qkv.qq." in key or ".qkv.k1" in key or ".qkv.v1" in key
        for key in keys
    )
    if has_qkv_lookup and has_dea:
        return DEEP_EMBED_QKV_DEA
    if has_v1_base and has_v1_projection:
        return DEEP_EMBED_RWKV7A_V1
    return None


def is_full_deepembed_checkpoint(tensors: Mapping[str, torch.Tensor]) -> bool:
    """Return whether ``tensors`` use the qkv/DEA DeepEmbed contract."""
    return detect_deepembed_variant(tensors) == DEEP_EMBED_QKV_DEA


def is_rwkv7a_deepembed_checkpoint(tensors: Mapping[str, torch.Tensor]) -> bool:
    """Return whether ``tensors`` use ChatRWKV's DeepEmbed-v1 contract."""
    return detect_deepembed_variant(tensors) == DEEP_EMBED_RWKV7A_V1


def is_deepembed_checkpoint(tensors: Mapping[str, torch.Tensor]) -> bool:
    """Detect either supported DeepEmbed model contract."""
    return detect_deepembed_variant(tensors) is not None


def deepembed_layer_ids(tensors: Mapping[str, torch.Tensor]) -> list[int]:
    ids: set[int] = set()
    for key in tensors:
        if ".ffn.s_emb" not in key and ".qkv.k_emb" not in key:
            continue
        if not key.startswith("blocks."):
            continue
        try:
            ids.add(int(key.split(".", 2)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def infer_deepembed_meta(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Return manifest-friendly facts without loading any lookup rows."""
    variant = detect_deepembed_variant(tensors)
    if variant is None:
        return {"deepembed": False}
    keys = sorted(key for key in tensors if is_deepembed_tensor_name(key))
    is_qkv = variant == DEEP_EMBED_QKV_DEA
    return {
        "deepembed": True,
        "deepembed_variant": variant,
        "deepembed_format": (
            "rwkv_deepembed_mmap_v1" if is_qkv else "rwkv7a_deepembed_v1"
        ),
        "deepembed_layers": deepembed_layer_ids(tensors),
        "deepembed_tensor_keys": keys,
        "deepembed_sidecar_required": is_qkv,
        # Both contracts have a CPU reference stream.  qkv/DEA requires the
        # sidecar-aware adapter; RWKV7a-v1 uses native ChatRWKV's path.
        "deepembed_streaming_supported": True,
    }


def _lookup_key(kind: str, layer_id: int) -> str:
    if kind not in DE_BASE_SUFFIXES:
        raise ValueError(f"DeepEmbed kind must be one of {DE_BASE_SUFFIXES}, got {kind!r}")
    return f"{kind}.{int(layer_id)}"


@dataclass(frozen=True)
class DeepEmbedEntry:
    key: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def numel(self) -> int:
        out = 1
        for dim in self.shape:
            out *= int(dim)
        return out

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPE_BYTES[self.dtype]


class DeepEmbedSidecar:
    """Read the public DeepEmbed ``.bin`` format with mmap and exact checks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"DeepEmbed sidecar not found: {self.path}")
        self._file = self.path.open("rb")
        self._size = self.path.stat().st_size
        if self._size < 16:
            self.close()
            raise ValueError(f"DeepEmbed sidecar is too small: {self.path}")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        index_offset, index_size = struct.unpack_from("<QQ", self._mmap, self._size - 16)
        if index_offset > self._size - 16 or index_size > self._size - 16 - index_offset:
            self.close()
            raise ValueError("DeepEmbed sidecar index lies outside the data region")
        try:
            raw_index = json.loads(
                self._mmap[index_offset : index_offset + index_size].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.close()
            raise ValueError("DeepEmbed sidecar index is not valid JSON") from exc
        if not isinstance(raw_index, dict):
            self.close()
            raise ValueError("DeepEmbed sidecar index must be a JSON object")
        self.entries: dict[str, DeepEmbedEntry] = {}
        for key, value in raw_index.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                self.close()
                raise ValueError("DeepEmbed sidecar contains an invalid index entry")
            try:
                offset = int(value["offset"])
                shape = tuple(int(dim) for dim in value["shape"])
                dtype = _CODE_TO_DTYPE[int(value["dtype"])]
            except (KeyError, TypeError, ValueError) as exc:
                self.close()
                raise ValueError(f"invalid DeepEmbed index entry {key!r}") from exc
            if offset < 0 or any(dim < 0 for dim in shape):
                self.close()
                raise ValueError(f"invalid DeepEmbed shape/offset for {key!r}")
            entry = DeepEmbedEntry(key, offset, shape, dtype)
            if offset + entry.nbytes > index_offset:
                self.close()
                raise ValueError(f"DeepEmbed tensor {key!r} overlaps the index/footer")
            self.entries[key] = entry

    def close(self) -> None:
        mapping = getattr(self, "_mmap", None)
        if mapping is not None:
            mapping.close()
            self._mmap = None
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()
            self._file = None

    def __enter__(self) -> "DeepEmbedSidecar":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _tensor_from_raw(
        raw: bytes | bytearray | memoryview,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Decode one already-bounded sidecar byte range.

        ``torch.frombuffer`` does not accept every dtype consistently across
        the supported PyTorch versions.  Keep the BF16 representation explicit
        just as :meth:`tensor` does, while allowing ``lookup`` to copy only the
        requested rows instead of materializing a whole vocabulary table.
        """
        if dtype is torch.bfloat16:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(
                torch.bfloat16
            )
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
        return tensor.reshape(shape)

    def tensor(self, key: str, *, device: torch.device | str | None = None) -> torch.Tensor:
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(f"DeepEmbed tensor {key!r} is not present in {self.path}")
        raw = self._mmap[entry.offset : entry.offset + entry.nbytes]
        tensor = self._tensor_from_raw(raw, entry.dtype, entry.shape).clone()
        return tensor.to(device=device) if device is not None else tensor

    def lookup(
        self,
        kind: str,
        layer_id: int,
        token_ids: Iterable[int] | torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return rows for ``kind.layer_id`` in the original token order."""
        entry = self.entries.get(_lookup_key(kind, layer_id))
        if entry is None:
            raise KeyError(
                f"DeepEmbed tensor {_lookup_key(kind, layer_id)!r} is not present "
                f"in {self.path}"
            )
        ids = (
            token_ids
            if isinstance(token_ids, torch.Tensor)
            else torch.tensor(list(token_ids), dtype=torch.long)
        )
        ids = ids.to(dtype=torch.long, device="cpu")
        if ids.ndim != 1:
            raise ValueError("DeepEmbed lookup token_ids must be one-dimensional")

        rows = int(entry.shape[0]) if entry.shape else 0
        if ids.numel():
            low = int(ids.min().item())
            high = int(ids.max().item())
            if low < 0 or high >= rows:
                raise IndexError(
                    f"DeepEmbed lookup index out of range for {_lookup_key(kind, layer_id)!r}: "
                    f"valid range is [0, {rows})"
                )
        tail_shape = tuple(entry.shape[1:])
        result = torch.empty((int(ids.numel()), *tail_shape), dtype=entry.dtype)
        if ids.numel() and rows:
            row_bytes = entry.nbytes // rows
            id_values = [int(value) for value in ids.tolist()]
            position = 0
            # Coalesce consecutive token IDs.  A prompt often contains runs of
            # adjacent IDs, and one mmap slice/copy is materially cheaper than
            # opening a separate Python byte range for each row.  The output
            # remains in the caller's original token order.
            while position < len(id_values):
                first_id = id_values[position]
                end_position = position + 1
                while (
                    end_position < len(id_values)
                    and id_values[end_position] == first_id + end_position - position
                ):
                    end_position += 1
                count = end_position - position
                start = entry.offset + first_id * row_bytes
                stop = start + count * row_bytes
                chunk = self._tensor_from_raw(
                    self._mmap[start:stop],
                    entry.dtype,
                    (count, *tail_shape),
                )
                result[position:end_position].copy_(chunk)
                position = end_position
        if device is not None:
            result = result.to(device=device)
        return result


def _layer_norm_embedding(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    emb = state["emb.weight"].detach().cpu().contiguous()
    weight = state["blocks.0.ln0.weight"].detach().cpu().reshape(-1)
    bias = state["blocks.0.ln0.bias"].detach().cpu().reshape(-1)
    return F.layer_norm(emb, (emb.shape[-1],), weight=weight, bias=bias)


def _derived_table(
    state: Mapping[str, torch.Tensor],
    layer_id: int,
    kind: str,
    norm_emb: torch.Tensor,
) -> torch.Tensor:
    block = f"blocks.{layer_id}."
    if kind == "s_emb":
        base_name, proj_name = block + "ffn.s_emb.weight", block + "ffn.s_emb_x.weight"
    elif kind == "k_emb":
        base_name, proj_name = block + "qkv.k_emb.weight", block + "qkv.k_emb_x.weight"
    else:
        base_name, proj_name = block + "qkv.v_emb.weight", block + "qkv.v_emb_x.weight"
    base = state.get(base_name)
    if base is None:
        raise KeyError(f"missing DeepEmbed tensor {base_name}")
    base = base.detach().cpu().squeeze().contiguous()
    projection = state.get(proj_name)
    if projection is None:
        return base
    projection = projection.detach().cpu().squeeze().contiguous()
    candidate = norm_emb @ projection.transpose(-1, -2)
    if candidate.shape != base.shape:
        raise ValueError(
            f"DeepEmbed {kind}.{layer_id} derived shape {tuple(candidate.shape)} "
            f"does not match base shape {tuple(base.shape)}"
        )
    return base + candidate


def write_deepembed_sidecar(
    state: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, object]:
    """Materialize public-compatible ``DeepEmbed.bin`` from a checkpoint."""
    if not is_full_deepembed_checkpoint(state):
        raise ValueError("checkpoint does not contain the DeepEmbed qkv/lookup contract")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    norm_emb = _layer_norm_embedding(state)
    tables: dict[str, torch.Tensor] = {}
    for layer_id in deepembed_layer_ids(state):
        for kind in DE_BASE_SUFFIXES:
            tables[_lookup_key(kind, layer_id)] = _derived_table(state, layer_id, kind, norm_emb)

    index: dict[str, dict[str, object]] = {}
    with output.open("wb") as handle:
        for key, tensor in tables.items():
            tensor = tensor.detach().cpu().contiguous()
            if tensor.dtype not in _DTYPE_TO_CODE:
                tensor = tensor.float()
            raw = tensor.view(torch.uint16).numpy().tobytes() if tensor.dtype is torch.bfloat16 else tensor.numpy().tobytes()
            offset = handle.tell()
            handle.write(raw)
            index[key] = {
                "offset": offset,
                "shape": list(tensor.shape),
                "dtype": _DTYPE_TO_CODE[tensor.dtype],
            }
        index_offset = handle.tell()
        index_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")
        handle.write(index_bytes)
        handle.write(struct.pack("<QQ", index_offset, len(index_bytes)))
    return {
        "path": output.name,
        "format": "rwkv_deepembed_mmap_v1",
        "tensor_count": len(index),
        "layers": deepembed_layer_ids(state),
        "bytes": output.stat().st_size,
    }


class DeepEmbedReferenceModel:
    """CPU reference for the qkv/DEA DeepEmbed equations.

    It intentionally mirrors the public demo's state layout rather than
    pretending to be a ChatRWKV ``RWKV`` object.  The adapter is sufficient for
    resident parity experiments and makes the sidecar usable today.  With
    ``resident_layers`` it also supports correctness-first CPU pack streaming;
    this still does not claim CUDA/JIT or production throughput.
    """

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        sidecar: DeepEmbedSidecar | None = None,
        *,
        dtype: torch.dtype = torch.float32,
        resident_layers: set[int] | None = None,
    ) -> None:
        if not is_full_deepembed_checkpoint(state):
            raise ValueError("DeepEmbedReferenceModel requires a DeepEmbed checkpoint")
        layer_ids = deepembed_layer_ids(state)
        if resident_layers is not None and sidecar is None:
            raise ValueError("layer-streamed DeepEmbed requires a DeepEmbed.bin sidecar")
        keep_layers = set(layer_ids if resident_layers is None else resident_layers)
        self.z = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("blocks."):
                try:
                    layer_id = int(key.split(".", 2)[1])
                except (IndexError, ValueError):
                    layer_id = -1
                if layer_id not in keep_layers:
                    continue
            tensor = value.detach().cpu()
            # ChatRWKV checkpoints commonly store r_k as [n_head, head_size],
            # while many other rank-one parameters have harmless singleton
            # dimensions.  Preserve this shape long enough to infer the
            # attention geometry before flattening it for the equation below.
            if not key.endswith("att.r_k"):
                tensor = tensor.squeeze()
            self.z[key] = tensor.to(dtype=dtype).contiguous()
        r_k_source = state["blocks.0.att.r_k"]
        r_k_shape = tuple(int(dim) for dim in r_k_source.shape)
        if len(r_k_shape) != 2:
            raise ValueError(f"DeepEmbed att.r_k must be [n_head, head_size], got {r_k_shape}")
        if "blocks.0.att.r_k" in self.z:
            self.z["blocks.0.att.r_k"] = self.z["blocks.0.att.r_k"].flatten()
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.n_layer = len(layer_ids)
        self.n_embd = int(self.z["emb.weight"].shape[-1])
        self.n_head, self.head_size = r_k_shape
        self.sidecar = sidecar
        if self.sidecar is None:
            self._tables = {
                _lookup_key(kind, layer_id): _derived_table(state, layer_id, kind, _layer_norm_embedding(state)).to(dtype=dtype)
                for layer_id in layer_ids
                for kind in DE_BASE_SUFFIXES
            }
        else:
            self._tables = None
        self._streaming_layers = resident_layers is not None
        self._k_dim = self._output_dim(state["blocks.0.qkv.k1"].detach().cpu().squeeze(), self.n_embd)
        self._v_dim = self._output_dim(state["blocks.0.qkv.v1"].detach().cpu().squeeze(), self.n_embd)
        self._q_dim = self._output_dim(state["blocks.0.qkv.qq.weight"].detach().cpu().squeeze(), self.n_embd)
        self._rwkv_ssd_last_state = None
        self._rwkv_ssd_last_token_id = 0
        # Layer entries are stable for the lifetime of a streamed model, but
        # the public forward methods are called once per token. Cache the
        # already-filtered ordinary block entries so each decode step does not
        # rebuild the same list and repeatedly stringify manifest names.
        self._stream_entries_source_id: int | None = None
        self._stream_entries_cache: dict[int, list[Any]] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, sidecar_path: str | Path | None = None) -> "DeepEmbedReferenceModel":
        path = Path(checkpoint)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"unsupported DeepEmbed checkpoint: {path}")
        sidecar = DeepEmbedSidecar(sidecar_path) if sidecar_path is not None and Path(sidecar_path).is_file() else None
        return cls(state, sidecar)

    def generate_zero_state(self) -> list[torch.Tensor]:
        state: list[torch.Tensor] = []
        att_state_shape = (self.n_head, self.head_size, self.head_size)
        for _ in range(self.n_layer):
            state.extend([
                torch.zeros(self.n_embd, dtype=self.dtype),
                torch.zeros(att_state_shape, dtype=torch.float32),
                torch.zeros(self.n_embd, dtype=self.dtype),
            ])
        base = self.n_layer * 3
        state.append(torch.empty(0, dtype=torch.long))
        k_dim = self._k_dim
        v_dim = self._v_dim
        for _ in range(self.n_layer):
            state.extend([torch.empty((0, k_dim), dtype=self.dtype), torch.empty((0, v_dim), dtype=self.dtype)])
        q_dim = self._q_dim
        state.extend(torch.zeros(q_dim, dtype=self.dtype) for _ in range(self.n_layer))
        assert len(state) == base + 1 + 3 * self.n_layer
        return state

    def _table(self, kind: str, layer_id: int, token_ids: torch.Tensor) -> torch.Tensor:
        key = _lookup_key(kind, layer_id)
        if self.sidecar is not None:
            return self.sidecar.lookup(kind, layer_id, token_ids).to(dtype=self.dtype)
        assert self._tables is not None
        return self._tables[key].index_select(0, token_ids)

    @staticmethod
    def _matmul(x: torch.Tensor, weight: torch.Tensor, input_dim: int) -> torch.Tensor:
        if weight.ndim != 2:
            return x * weight
        return x @ (weight if weight.shape[0] == input_dim else weight.transpose(0, 1))

    @staticmethod
    def _output_dim(weight: torch.Tensor, input_dim: int) -> int:
        if weight.ndim != 2:
            return int(weight.numel())
        return int(weight.shape[1] if weight.shape[0] == input_dim else weight.shape[0])

    def _install_stream_layer(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Install a streamed layer without copying the resident dictionary."""
        overwritten = {key: self.z[key] for key in tensors if key in self.z}
        self.z.update(tensors)
        return overwritten

    def _restore_stream_layer(
        self,
        tensors: Mapping[str, torch.Tensor],
        overwritten: Mapping[str, torch.Tensor],
    ) -> None:
        for key in tensors:
            if key in overwritten:
                self.z[key] = overwritten[key]
            else:
                self.z.pop(key, None)

    def _stream_entries_for_layer(
        self,
        layer_entries: Mapping[int, list[Any]],
        layer_id: int,
    ) -> list[Any]:
        """Return the cached ordinary tensors for one streamed layer."""
        source_id = id(layer_entries)
        if source_id != self._stream_entries_source_id:
            self._stream_entries_source_id = source_id
            self._stream_entries_cache = {}
        cached = self._stream_entries_cache.get(layer_id)
        if cached is None:
            cached = [
                entry
                for entry in layer_entries.get(layer_id, [])
                if not is_deepembed_tensor_name(str(getattr(entry, "name", "")))
            ]
            self._stream_entries_cache[layer_id] = cached
        return cached

    def _dea(self, layer_id: int, x: torch.Tensor, state: list[torch.Tensor], ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = f"blocks.{layer_id}.qkv."
        rows = int(x.shape[0])
        q = self._matmul(x, self.z[qkv + "qq.weight"], self.n_embd)
        k_proj = self._matmul(x, self.z[qkv + "k1"], self.n_embd)
        v_proj = self._matmul(x, self.z[qkv + "v1"], self.n_embd)
        base = self.n_layer * 3 + 1 + layer_id * 2
        k_c = torch.cat((state[base], k_proj), dim=0)
        v_c = torch.cat((state[base + 1], v_proj), dim=0)
        k = self._matmul(k_c, self.z[qkv + "k2"], int(k_c.shape[-1])) * self._table("k_emb", layer_id, ctx)
        v = torch.tanh(self._matmul(v_c, self.z[qkv + "v2"], int(v_c.shape[-1]))) * self._table("v_emb", layer_id, ctx)
        q_prev = state[self.n_layer * 3 + 1 + 2 * self.n_layer + layer_id]
        x_q = self.z[qkv + "x_q"]
        x_k = self.z[qkv + "x_k"]
        x_v = self.z[qkv + "x_v"]
        q_prev_row = q_prev.unsqueeze(0)
        if rows == 1:
            q = q + (q_prev_row - q) * x_q
        else:
            q = q + (torch.cat((q_prev_row, q[:-1]), dim=0) - q) * x_q
        if k.shape[0] > 1:
            k = k + (F.pad(k, (0, 0, 1, -1)) - k) * x_k
            v = v + (F.pad(v, (0, 0, 1, -1)) - v) * x_v
        q = F.layer_norm(q, (q.shape[-1],), weight=self.z[qkv + "lnq.weight"], bias=self.z[qkv + "lnq.bias"])
        k = F.layer_norm(k, (k.shape[-1],), weight=self.z[qkv + "lnk.weight"], bias=self.z[qkv + "lnk.bias"])
        v = F.layer_norm(v, (v.shape[-1],), weight=self.z[qkv + "lnv.weight"], bias=self.z[qkv + "lnv.bias"])
        scores = 64.0 * torch.tanh((q @ k.transpose(-1, -2)) / 1024.0)
        if rows == 1:
            # A single decode query is always the final context position, so
            # every key is causal. Avoid allocating arange/mask tensors on
            # every layer/token; prompt prefill keeps the masked path below.
            qkv_out = scores.softmax(dim=-1) @ v
        else:
            query_positions = torch.arange(ctx.numel() - rows, ctx.numel())
            key_positions = torch.arange(ctx.numel())
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            qkv_out = scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ v
        state[base] = k_c.detach()
        state[base + 1] = v_c.detach()
        state[self.n_layer * 3 + 1 + 2 * self.n_layer + layer_id] = q[-1].detach()
        return qkv_out, k_c, v_c, state[self.n_layer * 3 + 1 + 2 * self.n_layer + layer_id]

    def _tmix_seq(
        self,
        layer_id: int,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        state: torch.Tensor,
        v_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = f"blocks.{layer_id}.att."
        if x.shape[0] == 1:
            xx = x_prev.unsqueeze(0) - x
        else:
            xx = torch.cat((x_prev.unsqueeze(0), x[:-1]), dim=0) - x
        xr, xw, xk, xv, xa, xg = (x + xx * self.z[b + key] for key in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        r = self._matmul(xr, self.z[b + "receptance.weight"], self.n_embd)
        w = torch.tanh(self._matmul(xw, self.z[b + "w1"], self.n_embd)) @ self.z[b + "w2"]
        k = self._matmul(xk, self.z[b + "key.weight"], self.n_embd)
        v = self._matmul(xv, self.z[b + "value.weight"], self.n_embd)
        a = torch.sigmoid(self.z[b + "a0"] + self._matmul(self._matmul(xa, self.z[b + "a1"], self.n_embd), self.z[b + "a2"], int(self.z[b + "a1"].shape[-1])))
        g = torch.sigmoid(self._matmul(xg, self.z[b + "g1"], self.n_embd) @ self.z[b + "g2"])
        H, N = self.n_head, self.head_size
        kk = F.normalize((k * self.z[b + "k_k"]).view(x.shape[0], H, N), dim=-1).view(x.shape[0], H * N)
        k = k * (1 + (a - 1) * self.z[b + "k_a"])
        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.z[b + "v0"] + (xv @ self.z[b + "v1"]) @ self.z[b + "v2"])
        w = torch.exp(-0.606531 * torch.sigmoid((self.z[b + "w0"] + w).float()))
        if x.shape[0] == 1:
            # Decode has one recurrent step. Keep the update order identical
            # to the general loop while avoiding a one-row output allocation
            # and Python loop on every streamed layer.
            vt, wt, kt, kkt, at, rt = v[0], w[0], k[0], kk[0], a[0], r[0]
            vk = vt.view(H, N, 1) @ kt.view(H, 1, N)
            ab = (-kkt).view(H, N, 1) @ (kkt * at).view(H, 1, N)
            state = state * wt.view(H, 1, N) + state @ ab.float() + vk.float()
            y = (state.to(dtype=x.dtype) @ rt.view(H, N, 1)).view(1, H * N)
            y = F.group_norm(
                y,
                num_groups=H,
                weight=self.z[b + "ln_x.weight"],
                bias=self.z[b + "ln_x.bias"],
                eps=64e-5,
            )
            y = y + (
                (
                    (r * k * self.z[b + "r_k"])
                    .view(-1, H, N)
                    .sum(dim=-1, keepdim=True)
                    * v.view(-1, H, N)
                )
                .view(-1, H * N)
            )
            return (y * g) @ self.z[b + "output.weight"], x[-1], state, v_first

        outputs = torch.empty_like(x)
        for t in range(x.shape[0]):
            vt, wt, kt, kkt, at, rt = v[t], w[t], k[t], kk[t], a[t], r[t]
            vk = vt.view(H, N, 1) @ kt.view(H, 1, N)
            ab = (-kkt).view(H, N, 1) @ (kkt * at).view(H, 1, N)
            state = state * wt.view(H, 1, N) + state @ ab.float() + vk.float()
            y = state.to(dtype=x.dtype) @ rt.view(H, N, 1)
            outputs[t] = y.view(H * N)
        y = outputs
        y = F.group_norm(y, num_groups=H, weight=self.z[b + "ln_x.weight"], bias=self.z[b + "ln_x.bias"], eps=64e-5)
        y = y + ((r * k * self.z[b + "r_k"]).view(-1, H, N).sum(dim=-1, keepdim=True) * v.view(-1, H, N)).view(-1, H * N)
        return (y * g) @ self.z[b + "output.weight"], x[-1], state, v_first

    def _cmix_seq(self, layer_id: int, x: torch.Tensor, x_prev: torch.Tensor, s_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = f"blocks.{layer_id}.ffn."
        if x.shape[0] == 1:
            xx = x_prev.unsqueeze(0) - x
        else:
            xx = torch.cat((x_prev.unsqueeze(0), x[:-1]), dim=0) - x
        k = torch.relu(x + xx * self.z[b + "x_k"])
        k = self._matmul(k, self.z[b + "key.weight"], self.n_embd) ** 2
        s1 = self._matmul(x, self.z[b + "s1"], self.n_embd)
        s2 = s_emb.view(x.shape[0], -1, 32)
        ss = (s1.view(x.shape[0], 1, -1) @ s2).reshape(x.shape[0], -1)
        k = k * (self._matmul(ss, self.z[b + "s2"], int(ss.shape[-1])) + self.z[b + "s0"])
        return self._matmul(k, self.z[b + "value.weight"], int(k.shape[-1])), x[-1]

    @torch.no_grad()
    def forward(self, idx: int | list[int], state: list[torch.Tensor] | None = None, full_output: bool = False):
        if self._streaming_layers:
            raise RuntimeError(
                "layer-streamed DeepEmbed requires forward_streaming() with a provider"
            )
        ids = torch.tensor([idx] if isinstance(idx, int) else list(idx), dtype=torch.long)
        if state is None:
            state = self.generate_zero_state()
        history = state[self.n_layer * 3]
        ctx = torch.cat((history, ids))
        x = self.z["emb.weight"].index_select(0, ids)
        v_first = torch.empty_like(x)
        for layer_id in range(self.n_layer):
            b = f"blocks.{layer_id}."
            qkv, _, _, _ = self._dea(layer_id, x, state, ctx)
            xx = F.layer_norm(x, (self.n_embd,), weight=self.z[b + "ln1.weight"], bias=self.z[b + "ln1.bias"])
            xx, next_prev, next_att, v_first = self._tmix_seq(layer_id, xx, state[layer_id * 3], state[layer_id * 3 + 1], v_first)
            state[layer_id * 3] = next_prev
            state[layer_id * 3 + 1] = next_att
            x = x + xx + qkv
            xx = F.layer_norm(x, (self.n_embd,), weight=self.z[b + "ln2.weight"], bias=self.z[b + "ln2.bias"])
            semb = self._table("s_emb", layer_id, ids)
            xx, state[layer_id * 3 + 2] = self._cmix_seq(layer_id, xx, state[layer_id * 3 + 2], semb)
            x = x + xx
        state[self.n_layer * 3] = ctx.detach()
        if not full_output:
            x = x[-1]
        x = F.layer_norm(x, (self.n_embd,), weight=self.z["ln_out.weight"], bias=self.z["ln_out.bias"])
        logits = self._matmul(x, self.z["head.weight"], self.n_embd)
        self._rwkv_ssd_last_state = [tensor.detach().clone() for tensor in state]
        self._rwkv_ssd_last_token_id = int(ids[-1])
        return logits, state

    def _normalize_stream_layer(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Match checkpoint-shape normalization used by the resident adapter."""
        normalized: dict[str, torch.Tensor] = {}
        for key, value in tensors.items():
            tensor = value.detach().cpu()
            if not key.endswith("att.r_k"):
                tensor = tensor.squeeze()
            else:
                tensor = tensor.flatten()
            normalized[key] = tensor.to(dtype=self.dtype).contiguous()
        return normalized

    @torch.no_grad()
    def forward_streaming(
        self,
        idx: int | list[int],
        state: list[torch.Tensor] | None,
        provider: Any,
        layer_entries: Mapping[int, list[Any]],
        *,
        full_output: bool = False,
    ):
        """Forward with one provider transaction per layer.

        The qkv/DEA lookup tables come from ``DeepEmbed.bin`` while ordinary
        block tensors come from the manifest provider.  DeepEmbed tensors that
        were materialized only to build the sidecar are excluded from the
        layer transaction, so streaming does not read duplicate vocabulary
        tables from ``weights.bin``.
        """
        if not self._streaming_layers:
            raise RuntimeError("forward_streaming() requires a streaming model")
        ids = torch.tensor([idx] if isinstance(idx, int) else list(idx), dtype=torch.long)
        if state is None:
            state = self.generate_zero_state()
        history = state[self.n_layer * 3]
        ctx = torch.cat((history, ids))
        x = self.z["emb.weight"].index_select(0, ids)
        v_first = torch.empty_like(x)
        for layer_id in range(self.n_layer):
            entries = self._stream_entries_for_layer(layer_entries, layer_id)
            if not entries:
                raise KeyError(f"missing streamable DeepEmbed layer entries for layer {layer_id}")
            layer_tensors = self._normalize_stream_layer(
                provider.load_layer_tensors(entries)
            )
            overwritten = self._install_stream_layer(layer_tensors)
            try:
                b = f"blocks.{layer_id}."
                qkv, _, _, _ = self._dea(layer_id, x, state, ctx)
                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=self.z[b + "ln1.weight"],
                    bias=self.z[b + "ln1.bias"],
                )
                xx, next_prev, next_att, v_first = self._tmix_seq(
                    layer_id,
                    xx,
                    state[layer_id * 3],
                    state[layer_id * 3 + 1],
                    v_first,
                )
                state[layer_id * 3] = next_prev
                state[layer_id * 3 + 1] = next_att
                x = x + xx + qkv
                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=self.z[b + "ln2.weight"],
                    bias=self.z[b + "ln2.bias"],
                )
                semb = self._table("s_emb", layer_id, ids)
                xx, state[layer_id * 3 + 2] = self._cmix_seq(
                    layer_id,
                    xx,
                    state[layer_id * 3 + 2],
                    semb,
                )
                x = x + xx
            finally:
                self._restore_stream_layer(layer_tensors, overwritten)
        state[self.n_layer * 3] = ctx.detach()
        if not full_output:
            x = x[-1]
        x = F.layer_norm(
            x,
            (self.n_embd,),
            weight=self.z["ln_out.weight"],
            bias=self.z["ln_out.bias"],
        )
        logits = self._matmul(x, self.z["head.weight"], self.n_embd)
        self._rwkv_ssd_last_state = [tensor.detach().clone() for tensor in state]
        self._rwkv_ssd_last_token_id = int(ids[-1])
        return logits, state

    @torch.no_grad()
    def forward_batch_streaming(
        self,
        token_ids: Iterable[int] | torch.Tensor,
        states: list[list[torch.Tensor]],
        provider: Any,
        layer_entries: Mapping[int, list[Any]],
        *,
        metrics: Any | None = None,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Advance independent qkv/DEA states with one layer load per sweep.

        qkv/DEA remains a CPU reference implementation, but it can still use
        the same weight-stationary batching principle as ordinary RWKV-7:
        load each ordinary layer once, then run that layer for every session.
        The DeepEmbed lookup rows remain session-specific because each state
        has its own context history.  This method handles one decode token per
        session; ``forward_batch_prefill_streaming`` provides the corresponding
        shared-layer path for equal or variable-length prompt sequences.
        """
        ids = (
            token_ids.to(dtype=torch.long, device="cpu")
            if isinstance(token_ids, torch.Tensor)
            else torch.tensor(list(token_ids), dtype=torch.long)
        )
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("DeepEmbed batch tokens must be a non-empty 1-D sequence")
        if len(states) != int(ids.numel()):
            raise ValueError("DeepEmbed batch tokens and states must be equally sized")

        batch_size = int(ids.numel())
        if metrics is not None:
            metrics.batch_size = batch_size
            metrics.weight_sweeps += 1

        history_slot = self.n_layer * 3
        contexts = [
            torch.cat((state[history_slot], ids[index : index + 1]))
            for index, state in enumerate(states)
        ]
        xs = [
            self.z["emb.weight"].index_select(0, ids[index : index + 1])
            for index in range(batch_size)
        ]
        v_first = [torch.empty_like(x) for x in xs]

        for layer_id in range(self.n_layer):
            entries = self._stream_entries_for_layer(layer_entries, layer_id)
            if not entries:
                raise KeyError(
                    f"missing streamable DeepEmbed layer entries for layer {layer_id}"
                )
            layer_tensors = self._normalize_stream_layer(
                provider.load_layer_tensors(entries)
            )
            overwritten = self._install_stream_layer(layer_tensors)
            try:
                for index, state in enumerate(states):
                    x = xs[index]
                    qkv, _, _, _ = self._dea(
                        layer_id, x, state, contexts[index]
                    )
                    block = f"blocks.{layer_id}."
                    xx = F.layer_norm(
                        x,
                        (self.n_embd,),
                        weight=self.z[block + "ln1.weight"],
                        bias=self.z[block + "ln1.bias"],
                    )
                    xx, next_prev, next_att, next_v_first = self._tmix_seq(
                        layer_id,
                        xx,
                        state[layer_id * 3],
                        state[layer_id * 3 + 1],
                        v_first[index],
                    )
                    state[layer_id * 3] = next_prev
                    state[layer_id * 3 + 1] = next_att
                    x = x + xx + qkv
                    xx = F.layer_norm(
                        x,
                        (self.n_embd,),
                        weight=self.z[block + "ln2.weight"],
                        bias=self.z[block + "ln2.bias"],
                    )
                    semb = self._table(
                        "s_emb", layer_id, ids[index : index + 1]
                    )
                    xx, state[layer_id * 3 + 2] = self._cmix_seq(
                        layer_id,
                        xx,
                        state[layer_id * 3 + 2],
                        semb,
                    )
                    xs[index] = x + xx
                    v_first[index] = next_v_first
            finally:
                self._restore_stream_layer(layer_tensors, overwritten)

            if metrics is not None:
                # ``load_layer_tensors`` owns read timing.  Count one logical
                # streamed layer load for this shared sweep; per-session
                # compute timing is intentionally left to the reference path.
                metrics.weight_layer_loads += 1

        for state, context in zip(states, contexts, strict=True):
            state[history_slot] = context.detach()

        logits = []
        for x in xs:
            x = x[-1]
            x = F.layer_norm(
                x,
                (self.n_embd,),
                weight=self.z["ln_out.weight"],
                bias=self.z["ln_out.bias"],
            )
            logits.append(self._matmul(x, self.z["head.weight"], self.n_embd))
        if states:
            self._rwkv_ssd_last_state = [
                tensor.detach().clone() for tensor in states[-1]
            ]
            self._rwkv_ssd_last_token_id = int(ids[-1])
        return logits, states

    @torch.no_grad()
    def forward_batch_prefill_streaming(
        self,
        token_sequences: Iterable[Iterable[int] | torch.Tensor],
        states: list[list[torch.Tensor]],
        provider: Any,
        layer_entries: Mapping[int, list[Any]],
        *,
        metrics: Any | None = None,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Prefill independent qkv/DEA sessions with one load per layer.

        qkv/DEA has context-indexed lookup rows, so a single padded tensor
        cannot represent sessions with different prompt lengths without
        changing its masking and state contract.  This method keeps the exact
        per-session equations, but makes the scheduler layer-outer: each
        ordinary layer is loaded once, then consumed by every prompt.  It is
        therefore a safe CPU capacity optimization for both equal and
        variable-length prompts, while remaining explicitly a reference path
        until a fused variant-specific kernel exists.
        """
        sequences = [
            (
                values.to(dtype=torch.long, device="cpu")
                if isinstance(values, torch.Tensor)
                else torch.tensor(list(values), dtype=torch.long)
            )
            for values in token_sequences
        ]
        if not sequences or len(states) != len(sequences):
            raise ValueError("DeepEmbed prefill sequences and states must be equally sized and non-empty")
        if any(values.ndim != 1 or values.numel() == 0 for values in sequences):
            raise ValueError("DeepEmbed prefill sequences must be non-empty 1-D token lists")

        batch_size = len(sequences)
        if metrics is not None:
            metrics.batch_size = max(int(getattr(metrics, "batch_size", 0)), batch_size)
            metrics.weight_sweeps += 1

        history_slot = self.n_layer * 3
        contexts = [
            torch.cat((state[history_slot], values))
            for state, values in zip(states, sequences, strict=True)
        ]
        xs = [self.z["emb.weight"].index_select(0, values) for values in sequences]
        v_first = [torch.empty_like(x) for x in xs]

        for layer_id in range(self.n_layer):
            entries = self._stream_entries_for_layer(layer_entries, layer_id)
            if not entries:
                raise KeyError(
                    f"missing streamable DeepEmbed layer entries for layer {layer_id}"
                )
            layer_tensors = self._normalize_stream_layer(
                provider.load_layer_tensors(entries)
            )
            overwritten = self._install_stream_layer(layer_tensors)
            try:
                block = f"blocks.{layer_id}."
                for index, state in enumerate(states):
                    x = xs[index]
                    qkv, _, _, _ = self._dea(
                        layer_id, x, state, contexts[index]
                    )
                    xx = F.layer_norm(
                        x,
                        (self.n_embd,),
                        weight=self.z[block + "ln1.weight"],
                        bias=self.z[block + "ln1.bias"],
                    )
                    xx, next_prev, next_att, next_v_first = self._tmix_seq(
                        layer_id,
                        xx,
                        state[layer_id * 3],
                        state[layer_id * 3 + 1],
                        v_first[index],
                    )
                    state[layer_id * 3] = next_prev
                    state[layer_id * 3 + 1] = next_att
                    x = x + xx + qkv
                    xx = F.layer_norm(
                        x,
                        (self.n_embd,),
                        weight=self.z[block + "ln2.weight"],
                        bias=self.z[block + "ln2.bias"],
                    )
                    semb = self._table("s_emb", layer_id, sequences[index])
                    xx, state[layer_id * 3 + 2] = self._cmix_seq(
                        layer_id,
                        xx,
                        state[layer_id * 3 + 2],
                        semb,
                    )
                    xs[index] = x + xx
                    v_first[index] = next_v_first
            finally:
                self._restore_stream_layer(layer_tensors, overwritten)

            if metrics is not None:
                metrics.weight_layer_loads += 1

        for state, context in zip(states, contexts, strict=True):
            state[history_slot] = context.detach()

        logits = []
        for x in xs:
            final = F.layer_norm(
                x[-1],
                (self.n_embd,),
                weight=self.z["ln_out.weight"],
                bias=self.z["ln_out.bias"],
            )
            logits.append(self._matmul(final, self.z["head.weight"], self.n_embd))
        if states:
            self._rwkv_ssd_last_state = [
                tensor.detach().clone() for tensor in states[-1]
            ]
            self._rwkv_ssd_last_token_id = int(sequences[-1][-1])
        return logits, states


__all__ = [
    "DE_BASE_SUFFIXES",
    "DEEP_EMBED_QKV_DEA",
    "DEEP_EMBED_RWKV7A_V1",
    "DeepEmbedEntry",
    "DeepEmbedReferenceModel",
    "DeepEmbedSidecar",
    "detect_deepembed_variant",
    "deepembed_layer_ids",
    "infer_deepembed_meta",
    "is_full_deepembed_checkpoint",
    "is_deepembed_checkpoint",
    "is_deepembed_tensor_name",
    "is_rwkv7a_deepembed_checkpoint",
    "write_deepembed_sidecar",
]
