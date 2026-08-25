"""Content-addressed exact recurrent-state DAG with compressed XOR deltas."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from rwkv_ssd.runtime.snapshot import (
    _decode_state,
    _encode_state,
    _external_spec,
    _state_kind,
    _tensor_specs,
)
from rwkv_ssd.runtime.state_cache import RecurrentState

_MAGIC = b"RSD\x01"


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR delta requires equal payload lengths")
    return bytes(a ^ b for a, b in zip(left, right))


class StateDeltaDAG:
    def __init__(
        self,
        root: str | Path,
        *,
        checkpoint_interval: int = 8,
        compression_level: int = 6,
    ) -> None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = int(checkpoint_interval)
        self.compression_level = max(0, min(9, int(compression_level)))

    def _path(self, node_id: str) -> Path:
        if len(node_id) != 64 or any(ch not in "0123456789abcdef" for ch in node_id):
            raise ValueError("invalid state DAG node id")
        return self.root / node_id[:2] / f"{node_id}.rsd"

    def _encode(self, state: RecurrentState) -> tuple[dict, bytes]:
        specs = _tensor_specs(state)
        payload = _encode_state(state)
        external = _external_spec(state)
        descriptor: dict = {
            "state_kind": _state_kind(state),
            "last_token_id": int(state.last_token_id),
            "specs": [
                {"shape": shape, "dtype_code": code, "nbytes": nbytes}
                for shape, code, nbytes in specs
            ],
        }
        if external is not None:
            descriptor["external_spec"] = {
                "shape": external[0],
                "dtype_code": external[1],
                "nbytes": len(external[2]),
            }
            payload += external[2]
        return descriptor, payload

    def _read_node(self, node_id: str) -> tuple[dict, bytes]:
        path = self._path(node_id)
        raw = path.read_bytes()
        if len(raw) < 8 or raw[:4] != _MAGIC:
            raise ValueError(f"bad state DAG node magic: {node_id}")
        (header_len,) = struct.unpack_from("<I", raw, 4)
        header_end = 8 + header_len
        if header_end > len(raw):
            raise ValueError(f"truncated state DAG node header: {node_id}")
        header = json.loads(raw[8:header_end].decode("utf-8"))
        compressed = raw[header_end:]
        try:
            stored_payload = zlib.decompress(compressed)
        except zlib.error as exc:
            raise ValueError(f"corrupt state DAG payload: {node_id}") from exc
        return header, stored_payload

    def _raw_payload(self, node_id: str, *, seen: set[str] | None = None) -> tuple[dict, bytes]:
        seen = set() if seen is None else seen
        if node_id in seen:
            raise ValueError("state DAG contains a cycle")
        seen.add(node_id)
        header, stored = self._read_node(node_id)
        kind = header.get("storage")
        if kind == "full":
            payload = stored
        elif kind == "xor":
            parent = str(header.get("parent", ""))
            _parent_header, parent_payload = self._raw_payload(parent, seen=seen)
            payload = _xor_bytes(stored, parent_payload)
        else:
            raise ValueError(f"unsupported state DAG storage kind: {kind!r}")
        expected = str(header.get("payload_sha256", ""))
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"state DAG payload checksum mismatch: {node_id}")
        return header, payload

    def store(self, state: RecurrentState, *, parent_id: str | None = None) -> str:
        descriptor, payload = self._encode(state)
        identity = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        node_id = hashlib.sha256(identity + payload).hexdigest()
        target = self._path(node_id)
        if target.is_file():
            return node_id
        storage = "full"
        stored = payload
        depth = 0
        parent = None
        if parent_id is not None:
            parent_header, parent_payload = self._raw_payload(parent_id)
            parent_depth = int(parent_header.get("depth", 0))
            if len(parent_payload) == len(payload) and parent_depth + 1 < self.checkpoint_interval:
                delta = _xor_bytes(payload, parent_payload)
                if len(zlib.compress(delta, self.compression_level)) < len(
                    zlib.compress(payload, self.compression_level)
                ):
                    storage = "xor"
                    stored = delta
                    depth = parent_depth + 1
                    parent = parent_id
        header = {
            "version": 1,
            "node_id": node_id,
            "storage": storage,
            "parent": parent,
            "depth": depth,
            "descriptor": descriptor,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_bytes": len(payload),
        }
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        encoded = _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes
        encoded += zlib.compress(stored, self.compression_level)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, target)
        return node_id

    def load(self, node_id: str) -> RecurrentState:
        header, payload = self._raw_payload(node_id)
        descriptor = header.get("descriptor") or {}
        specs = [
            (list(item["shape"]), int(item["dtype_code"]), int(item["nbytes"]))
            for item in descriptor.get("specs", [])
        ]
        tensor_bytes = sum(spec[2] for spec in specs)
        state = _decode_state(
            payload[:tensor_bytes],
            specs,
            last_token_id=int(descriptor.get("last_token_id", 0)),
            state_kind=str(descriptor.get("state_kind", "empty")),
        )
        external = descriptor.get("external_spec")
        if external:
            shape = tuple(int(value) for value in external["shape"])
            nbytes = int(external["nbytes"])
            blob = payload[tensor_bytes : tensor_bytes + nbytes]
            if len(blob) != nbytes:
                raise ValueError("truncated external state DAG payload")
            state.external_state = np.frombuffer(blob, dtype=np.float32).reshape(shape).copy()
        return state

    def node_info(self, node_id: str) -> dict:
        header, _stored = self._read_node(node_id)
        return dict(header)

    def storage_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*/*.rsd"))


__all__ = ["StateDeltaDAG"]
