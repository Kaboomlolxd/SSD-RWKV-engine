"""Small, repeatable A/B probes for the maintained CPU optimizations.

Examples::

    python bench/bench_cpu_optimizations.py --pack test_model/runtime_pack

The grouped-U8 probe compares the current broadcast decode against the old
per-element group-indexing shape.  The pack probe compares mmap on the raw
pack with one-shot zstd expansion plus the same logical reads.  These are
microbenchmarks, not end-to-end tok/s claims.  The state probe measures the
full recurrent-state copy that controlled generation must retain and that the
uninterrupted fast path now avoids between tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rwkv_ssd.runtime.deepembed import (
    DeepEmbedReferenceModel,
    DeepEmbedSidecar,
    write_deepembed_sidecar,
)
from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
from rwkv_ssd.runtime.pack_codec import (
    decode_scale_u8_grouped_to_tensor,
    encode_scale_u8_grouped,
)
from rwkv_ssd.runtime.trinity_codec import (
    decode_trinity_lut2_to_tensor,
    encode_trinity_lut2,
)
from rwkv_ssd.runtime.weight_store import open_weight_store
from rwkv_ssd.tools.pack_runtime import _finalize_weight_compression


def _median_ms(fn, repeats: int) -> float:
    for _ in range(2):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return sorted(samples)[len(samples) // 2]


def _legacy_grouped_decode(data: bytes, entry: TensorEntry, group_size: int) -> torch.Tensor:
    groups = (entry.numel + group_size - 1) // group_size
    scales = np.frombuffer(data, dtype=np.float32, offset=8, count=groups * 2).reshape(
        groups, 2
    )
    quant = np.frombuffer(data, dtype=np.uint8, offset=8 + groups * 8, count=entry.numel)
    values = np.empty(entry.numel, dtype=np.float32)
    batch_groups = max(1, 1_000_000 // group_size)
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, entry.numel - start)
        q = quant[start : start + available].astype(np.float32)
        ids = np.arange(available, dtype=np.intp) // group_size
        batch = scales[first : first + count]
        values[start : start + available] = batch[ids, 0] + (
            batch[ids, 1] - batch[ids, 0]
        ) * (q / 255.0)
    return torch.from_numpy(values).reshape(entry.shape).to(dtype=torch.float32)


def run_grouped_u8(repeats: int = 5) -> dict[str, float | int]:
    group_size = 64
    tensor = torch.randn(2048, 2048, dtype=torch.float32)
    blob = encode_scale_u8_grouped(tensor, group_size=group_size)
    entry = TensorEntry(
        "bench", 0, "float32", [2048, 2048], 0, len(blob), 4096, "streamed",
        dequant="scale_u8_grouped",
    )
    legacy_ms = _median_ms(
        lambda: _legacy_grouped_decode(blob, entry, group_size), repeats
    )
    optimized_ms = _median_ms(
        lambda: decode_scale_u8_grouped_to_tensor(blob, entry, torch.device("cpu")),
        repeats,
    )
    return {
        "group_size": group_size,
        "numel": entry.numel,
        "legacy_decode_ms": round(legacy_ms, 3),
        "optimized_decode_ms": round(optimized_ms, 3),
        "speedup": round(legacy_ms / optimized_ms, 3) if optimized_ms else 0.0,
    }


def _legacy_grouped_lut2_decode(
    data: bytes, entry: TensorEntry, group_size: int
) -> torch.Tensor:
    """Reference grouped-LUT2 decoder retained for an honest A/B probe."""
    groups = (entry.numel + group_size - 1) // group_size
    codebooks = np.frombuffer(
        data, dtype=np.float32, offset=8, count=groups * 4
    ).reshape(groups, 4)
    packed_offset = 8 + groups * 16
    packed = np.frombuffer(data, dtype=np.uint8, offset=packed_offset)
    positions = np.arange(entry.numel, dtype=np.intp)
    indices = (packed[positions // 4] >> ((positions % 4) * 2)) & 3
    values = np.empty(entry.numel, dtype=np.float32)
    for group in range(groups):
        start = group * group_size
        end = min(start + group_size, entry.numel)
        values[start:end] = codebooks[group, indices[start:end]]
    return torch.from_numpy(values).reshape(entry.shape)


def run_grouped_lut2(repeats: int = 3) -> dict[str, float | int]:
    group_size = 128
    tensor = torch.randn(1024, 1024, dtype=torch.float32)
    blob = encode_trinity_lut2(
        tensor, codebook="groupwise_kmeans", group_size=group_size
    )
    entry = TensorEntry(
        "bench", 0, "float32", [1024, 1024], 0, len(blob), 4096, "streamed",
        dequant="trinity_lut2",
    )
    legacy_ms = _median_ms(
        lambda: _legacy_grouped_lut2_decode(blob, entry, group_size), repeats
    )
    optimized_ms = _median_ms(
        lambda: decode_trinity_lut2_to_tensor(blob, entry, torch.device("cpu")),
        repeats,
    )
    return {
        "group_size": group_size,
        "numel": entry.numel,
        "legacy_decode_ms": round(legacy_ms, 3),
        "optimized_decode_ms": round(optimized_ms, 3),
        "speedup": round(legacy_ms / optimized_ms, 3) if optimized_ms else 0.0,
    }


def _write_lookup_sidecar(path: Path, *, vocab: int = 32768, width: int = 512) -> int:
    """Write one large synthetic lookup table for the row-read A/B probe."""
    rng = np.random.default_rng(17)
    table = rng.standard_normal((vocab, width), dtype=np.float32)
    raw = table.tobytes()
    index = {
        "k_emb.0": {
            "offset": 0,
            "shape": [vocab, width],
            "dtype": 0,
        }
    }
    index_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(raw)
        index_offset = handle.tell()
        handle.write(index_bytes)
        handle.write(struct.pack("<QQ", index_offset, len(index_bytes)))
    return len(raw)


def run_deepembed_sidecar_lookup(repeats: int = 3) -> dict[str, object]:
    """Compare full-table lookup with direct mmap row reads."""
    with tempfile.TemporaryDirectory(prefix="rwkv_deepembed_lookup_") as temp:
        path = Path(temp) / "DeepEmbed.bin"
        table_bytes = _write_lookup_sidecar(path)
        with DeepEmbedSidecar(path) as sidecar:
            cases = {
                "single": [12345],
                "contiguous_128": list(range(12345, 12473)),
                "scattered_128": [
                    (index * 9973) % 32768 for index in range(128)
                ],
            }
            rows: dict[str, object] = {}
            for label, ids in cases.items():
                ids_tensor = torch.tensor(ids, dtype=torch.long)

                def legacy(ids: torch.Tensor = ids_tensor) -> torch.Tensor:
                    return sidecar.tensor("k_emb.0").index_select(0, ids)

                def direct(ids: torch.Tensor = ids_tensor) -> torch.Tensor:
                    return sidecar.lookup("k_emb", 0, ids)

                legacy_ms = _median_ms(legacy, repeats)
                direct_ms = _median_ms(direct, repeats)
                rows[label] = {
                    "rows": len(ids),
                    "legacy_full_table_ms": round(legacy_ms, 3),
                    "direct_row_read_ms": round(direct_ms, 3),
                    "speedup": round(legacy_ms / direct_ms, 3)
                    if direct_ms
                    else 0.0,
                }
            return {
                "table_bytes": table_bytes,
                "table_mb": round(table_bytes / (1024**2), 2),
                "cases": rows,
            }


def _deepembed_bench_state(
    *, vocab: int = 256, width: int = 16, hidden: int = 8
) -> dict[str, torch.Tensor]:
    """Create a small complete qkv/DEA checkpoint for the batch probe."""
    torch.manual_seed(23)
    state: dict[str, torch.Tensor] = {
        "emb.weight": torch.randn(vocab, width),
        "blocks.0.ln0.weight": torch.ones(width),
        "blocks.0.ln0.bias": torch.zeros(width),
        "blocks.0.att.r_k": torch.randn(1, width),
        "ln_out.weight": torch.ones(width),
        "ln_out.bias": torch.zeros(width),
        "head.weight": torch.randn(width, vocab),
        "blocks.0.ffn.s_emb.weight": torch.randn(vocab, 32 * 32),
        "blocks.0.ffn.s_emb_x.weight": torch.randn(32 * 32, width),
        "blocks.0.ffn.s1": torch.randn(width, 32),
        "blocks.0.ffn.s2": torch.randn(32, width),
        "blocks.0.ffn.s0": torch.randn(width),
        "blocks.0.ffn.x_k": torch.randn(width),
        "blocks.0.ffn.key.weight": torch.randn(width, width),
        "blocks.0.ffn.value.weight": torch.randn(width, width),
        "blocks.0.qkv.k_emb.weight": torch.randn(vocab, width),
        "blocks.0.qkv.k_emb_x.weight": torch.randn(width, width),
        "blocks.0.qkv.v_emb.weight": torch.randn(vocab, width),
        "blocks.0.qkv.v_emb_x.weight": torch.randn(width, width),
        "blocks.0.qkv.qq.weight": torch.randn(width, width),
        "blocks.0.qkv.k1": torch.randn(width, width),
        "blocks.0.qkv.k2": torch.randn(width, width),
        "blocks.0.qkv.v1": torch.randn(width, width),
        "blocks.0.qkv.v2": torch.randn(width, width),
        "blocks.0.qkv.x_q": torch.randn(width),
        "blocks.0.qkv.x_k": torch.randn(width),
        "blocks.0.qkv.x_v": torch.randn(width),
        "blocks.0.qkv.lnq.weight": torch.ones(width),
        "blocks.0.qkv.lnq.bias": torch.zeros(width),
        "blocks.0.qkv.lnk.weight": torch.ones(width),
        "blocks.0.qkv.lnk.bias": torch.zeros(width),
        "blocks.0.qkv.lnv.weight": torch.ones(width),
        "blocks.0.qkv.lnv.bias": torch.zeros(width),
    }
    for prefix in ("blocks.0.ln1", "blocks.0.ln2"):
        state[prefix + ".weight"] = torch.ones(width)
        state[prefix + ".bias"] = torch.zeros(width)
    for key in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "k_k", "k_a"):
        state["blocks.0.att." + key] = torch.randn(width)
    state["blocks.0.att.w1"] = torch.randn(width, width)
    state["blocks.0.att.w2"] = torch.randn(width, width)
    state["blocks.0.att.a0"] = torch.randn(width)
    state["blocks.0.att.a1"] = torch.randn(width, hidden)
    state["blocks.0.att.a2"] = torch.randn(hidden, width)
    state["blocks.0.att.v0"] = torch.randn(width)
    state["blocks.0.att.v1"] = torch.randn(width, hidden)
    state["blocks.0.att.v2"] = torch.randn(hidden, width)
    state["blocks.0.att.g1"] = torch.randn(width, width)
    state["blocks.0.att.g2"] = torch.randn(width, width)
    for name in ("receptance", "key", "value", "output"):
        state[f"blocks.0.att.{name}.weight"] = torch.randn(width, width)
    state["blocks.0.att.ln_x.weight"] = torch.ones(width)
    state["blocks.0.att.ln_x.bias"] = torch.zeros(width)
    return state


def run_deepembed_batch(repeats: int = 3) -> dict[str, object]:
    """Measure shared-layer qkv/DEA prefill+decode versus independent sessions."""
    state = _deepembed_bench_state()
    prompts = ([1], [3])
    decode_tokens = ([4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15])
    with tempfile.TemporaryDirectory(prefix="rwkv_deepembed_batch_") as temp:
        sidecar_path = Path(temp) / "DeepEmbed.bin"
        write_deepembed_sidecar(state, sidecar_path)
        with DeepEmbedSidecar(sidecar_path) as sidecar:
            independent_model = DeepEmbedReferenceModel(
                state, sidecar, resident_layers=set()
            )
            batch_model = DeepEmbedReferenceModel(
                state, sidecar, resident_layers=set()
            )
            entries = {
                0: [
                    SimpleNamespace(name=name)
                    for name in state
                    if name.startswith("blocks.0.")
                ]
            }

            class Provider:
                def __init__(self) -> None:
                    self.loads = 0

                def load_layer_tensors(self, layer_entries):
                    self.loads += 1
                    return {entry.name: state[entry.name] for entry in layer_entries}

            def independent_run() -> int:
                provider = Provider()
                states = [independent_model.generate_zero_state() for _ in prompts]
                for prompt, state_i in zip(prompts, states, strict=True):
                    independent_model.forward_streaming(
                        prompt, state_i, provider, entries
                    )
                for position in range(len(decode_tokens[0])):
                    for index, state_i in enumerate(states):
                        independent_model.forward_streaming(
                            [decode_tokens[index][position]],
                            state_i,
                            provider,
                            entries,
                        )
                return provider.loads

            def batch_run() -> int:
                provider = Provider()
                states = [batch_model.generate_zero_state() for _ in prompts]
                batch_model.forward_batch_prefill_streaming(
                    prompts,
                    states,
                    provider,
                    entries,
                )
                for position in range(len(decode_tokens[0])):
                    batch_model.forward_batch_streaming(
                        [tokens[position] for tokens in decode_tokens],
                        states,
                        provider,
                        entries,
                    )
                return provider.loads

            independent_ms = _median_ms(independent_run, repeats)
            batch_ms = _median_ms(batch_run, repeats)
            independent_loads = independent_run()
            batch_loads = batch_run()
            return {
                "sessions": len(prompts),
                "decode_steps": len(decode_tokens[0]),
                "independent_ms": round(independent_ms, 3),
                "batched_ms": round(batch_ms, 3),
                "wall_speedup": round(independent_ms / batch_ms, 3)
                if batch_ms
                else 0.0,
                "independent_layer_loads": independent_loads,
                "batched_layer_loads": batch_loads,
                "layer_load_reduction": round(
                    1.0 - batch_loads / independent_loads, 3
                )
                if independent_loads
                else 0.0,
            }


def run_state_publication(repeats: int = 3, tokens: int = 64) -> dict[str, object]:
    """Measure the allocation avoided by eliding uncontrolled state snapshots.

    Native rwkv.cpp keeps one flat FP32 state array, while ChatRWKV keeps a
    list of tensors.  The native-shaped probe is the larger and more direct
    cost because each old publication copied the complete array once per
    generated token.  Controlled generation still performs that copy so a
    callback/cancellation boundary remains resumable.
    """
    specs = {
        "0.1b_like": 2_320 * 1024,
        "2.9b_like": 25_620 * 1024,
    }
    rows: dict[str, object] = {}
    for label, state_bytes in specs.items():
        elements = max(1, state_bytes // np.dtype(np.float32).itemsize)
        state = np.empty(elements, dtype=np.float32)
        copy_ms = _median_ms(lambda state=state: state.copy(), repeats)
        rows[label] = {
            "state_bytes": int(state.nbytes),
            "snapshot_copy_ms": round(copy_ms, 3),
            "snapshot_copy_ms_per_64_tokens": round(copy_ms * tokens, 3),
            "uninterrupted_snapshot_copies": 0,
        }
    return {"tokens": int(tokens), "cases": rows}


def run_pack(pack_dir: Path, repeats: int = 3) -> dict[str, object]:
    source = Path(pack_dir)
    manifest = Manifest.load(source)
    if manifest.is_sharded() or manifest.weights_path.name != "weights.bin":
        raise ValueError("--pack must be an uncompressed single-file pack")
    with tempfile.TemporaryDirectory(prefix="rwkv_cpu_opt_") as temp:
        compressed = Path(temp) / "compressed"
        shutil.copytree(source, compressed)
        _finalize_weight_compression(compressed, "zstd", quiet=True)
        # Compression intentionally changes the physical weights identity.  A
        # copied deployment certificate is therefore no longer valid for this
        # disposable storage microbenchmark.  Disable certificate metadata in
        # the temporary copy only; the source pack and its certificate are
        # never modified.
        compressed_manifest_path = compressed / "manifest.json"
        compressed_manifest = json.loads(
            compressed_manifest_path.read_text(encoding="utf-8")
        )
        compressed_meta = compressed_manifest.get("meta")
        if isinstance(compressed_meta, dict):
            compressed_meta.pop("quality_certificate_file", None)
            compressed_meta.pop("quality_certificate_sha256", None)
        compressed_manifest_path.write_text(
            json.dumps(compressed_manifest, indent=2), encoding="utf-8"
        )
        compressed_sidecar_path = compressed / "meta.json"
        if compressed_sidecar_path.is_file():
            compressed_sidecar = json.loads(
                compressed_sidecar_path.read_text(encoding="utf-8")
            )
            if isinstance(compressed_sidecar, dict):
                compressed_sidecar.pop("quality_certificate_file", None)
                compressed_sidecar.pop("quality_certificate_sha256", None)
                compressed_sidecar_path.write_text(
                    json.dumps(compressed_sidecar, indent=2), encoding="utf-8"
                )
        certificate_path = compressed / "quality_certificate.json"
        if certificate_path.exists():
            certificate_path.unlink()
        rows = {}
        for label, path in (("raw_mmap", source), ("zstd_load_plus_read", compressed)):
            current = Manifest.load(path)
            current_weights_path = current.weights_path
            current_entries = current.streamed_tensors() or current.tensors

            def read_all(
                weights_path: Path = current_weights_path,
                entries: list[TensorEntry] = current_entries,
            ) -> None:
                with open_weight_store(weights_path, backend="mmap") as store:
                    for entry in entries:
                        store.read_bytes(entry)

            rows[label] = {
                "median_ms": round(_median_ms(read_all, repeats), 3),
                "physical_bytes": current.weights_path.stat().st_size,
            }
        rows["logical_bytes"] = sum(
            entry.length for entry in manifest.streamed_tensors() or manifest.tensors
        )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "grouped_u8": run_grouped_u8(max(1, args.repeats)),
        "grouped_lut2": run_grouped_lut2(max(1, args.repeats)),
        "deepembed_sidecar_lookup": run_deepembed_sidecar_lookup(
            max(1, args.repeats)
        ),
        "deepembed_batch": run_deepembed_batch(max(1, args.repeats)),
        "state_publication": run_state_publication(max(1, args.repeats)),
        "whole_pack_zstd": run_pack(args.pack, max(1, args.repeats)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
