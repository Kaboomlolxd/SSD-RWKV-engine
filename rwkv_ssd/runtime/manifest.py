"""Runtime pack manifest: weights.bin + manifest.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ALIGNMENT = 4096


def _pack_path(root: Path, value: str, field: str) -> Path:
    """Resolve a manifest path and require it to stay inside the pack root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {field} must be a non-empty relative path")
    candidate = (root / value.replace("\\", "/")).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"manifest {field} escapes pack root: {value!r}") from exc
    if candidate == root_resolved:
        raise ValueError(f"manifest {field} must name a file inside the pack")
    return candidate


def _requires_real_quality_certificate(raw: dict[str, Any], meta: dict[str, Any]) -> bool:
    """Return whether a real lossy RWKV pack needs a quality certificate.

    Tiny synthetic and unit-test packs deliberately exercise codec plumbing
    without a model-level quality certificate.  Real RWKV-7 packs, however,
    must carry evidence that the codec preserves generation quality.
    """
    family = str(raw.get("model_family") or meta.get("model_family") or "").lower()
    codec = str(meta.get("pack_codec") or "").lower()
    if not codec:
        codecs = {
            str(item.get("dequant", "none")).lower()
            for item in raw.get("tensors", [])
            if isinstance(item, dict)
        }
        if codecs & {
            "trinity",
            "trinity_lut2",
            "trinity_layer",
            "scale_u8",
            "scale_u8_grouped",
            "scale_u4",
        }:
            codec = next(iter(codecs & {
                "trinity",
                "trinity_lut2",
                "trinity_layer",
                "scale_u8",
                "scale_u8_grouped",
                "scale_u4",
            }))
    try:
        n_layer = int(meta.get("n_layer", 0))
        vocab_size = int(meta.get("vocab_size", 0))
    except (TypeError, ValueError):
        return False
    return (
        family.startswith("rwkv7")
        and codec in {
            "trinity",
            "trinity_lut2",
            "trinity_layer",
            "scale_u8",
            "scale_u8_grouped",
            "scale_u4",
        }
        and n_layer >= 8
        and vocab_size >= 4096
    )


@dataclass(frozen=True)
class TensorEntry:
    name: str
    layer_id: int
    dtype: str
    shape: list[int]
    offset: int
    length: int
    alignment: int
    residency: str  # resident | streamed
    dequant: str = "none"
    inner_offset: int = 0  # byte offset inside decompressed layer blob (trinity_layer)
    inner_length: int = 0  # slice length inside decompressed layer blob
    fast_offset: int = -1  # bf16 shadow in shadow.bin (-1 = absent)
    fast_length: int = 0
    fast_shard_file: str = ""  # relative shadow shard for a striped shadow
    # Optional manifest-v2 shadow extents.  Logical offsets are relative to
    # this tensor; ``fast_offset`` remains the stable global shadow address so
    # legacy layer-span decoding can still place the gathered payload.
    fast_stripes: tuple[dict[str, Any], ...] = ()
    shard_file: str = ""  # relative path to the shard file (empty = legacy single-file pack)
    # Optional manifest-v2 striped extents.  Each item has ``shard_file``,
    # local ``offset``, ``length`` and optional logical ``offset``.  An empty
    # tuple keeps the legacy one-file-per-tensor representation unchanged.
    stripes: tuple[dict[str, Any], ...] = ()

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class Manifest:
    version: int
    model_family: str
    weights_path: Path
    tensors: list[TensorEntry]
    meta: dict[str, Any]
    pack_dir: Path | None = field(default=None, repr=False, compare=False)
    # Sharded-pack support (M-class): when the pack is split across multiple
    # files (``pack_runtime --shard N``), each tensor carries the relative
    # path of the file it lives in. ``shard_files`` is the deduplicated
    # ordered list of those paths; ``primary_weights_path`` is the first
    # one (kept for backward-compatible single-file paths that still
    # touch ``weights_path``).
    shard_files: list[Path] = field(default_factory=list)
    _by_layer_cache: dict[int, list[TensorEntry]] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def load(
        cls,
        pack_dir: str | Path,
        *,
        require_quality_certificate: bool = False,
    ) -> Manifest:
        root = Path(pack_dir).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.json not found in {root}")

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"manifest.json must be a JSON object, got {type(raw).__name__}")
        meta = raw.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("manifest.json 'meta' must be a JSON object")

        # Merge sidecar metadata needed for compatibility and certificate
        # policy.  ``pack_composition`` is intentionally left manifest-only:
        # its accessor has historically described the build-time block in
        # manifest.json, and old packs may retain a stale sidecar copy.
        sidecar_path = root / "meta.json"
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(sidecar, dict):
                raise ValueError("meta.json must be a JSON object")
            for key, value in sidecar.items():
                if key != "pack_composition":
                    meta.setdefault(key, value)

        tensors_raw = raw.get("tensors")
        if not isinstance(tensors_raw, list) or not tensors_raw:
            raise ValueError("manifest.json must have a non-empty 'tensors' list")

        weights_name = raw.get("weights_file", "weights.bin")
        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("manifest.json 'weights_file' must be a non-empty string")
        weights_path = _pack_path(root, weights_name, "weights_file")

        version = int(raw.get("version", 1))

        # Sharded packs: ``weights_files`` is a list of relative paths. If
        # absent, fall back to the legacy single-file ``weights.bin``.  Keep
        # the distinction between an absent key and an empty list: a v2
        # striped manifest must declare every file that an extent can use.
        weights_files = raw.get("weights_files")
        if weights_files is not None:
            if not isinstance(weights_files, list) or not all(
                isinstance(path, str) and path for path in weights_files
            ):
                raise ValueError(
                    "manifest.json 'weights_files' must be a list of non-empty strings"
                )
            if not weights_files:
                raise ValueError("manifest.json 'weights_files' must not be empty")
            normalized_weights_files = [
                str(path).replace("\\", "/") for path in weights_files
            ]
            if len(set(normalized_weights_files)) != len(normalized_weights_files):
                raise ValueError("manifest.json 'weights_files' contains duplicates")
            shard_files = [
                _pack_path(root, str(p), "weights_files entry")
                for p in weights_files
            ]
            for p in shard_files:
                if not p.is_file():
                    raise FileNotFoundError(f"shard file not found: {p}")
            # Legacy single-file path is the first shard (or the explicit
            # ``weights_file`` if present).
            if not weights_path.is_file() and shard_files:
                weights_path = shard_files[0]
        else:
            shard_files = []
            if not weights_path.is_file():
                raise FileNotFoundError(f"weights file not found: {weights_path}")

        shadow_files_raw = meta.get("shadow_files")
        if shadow_files_raw is not None:
            if not isinstance(shadow_files_raw, list) or not all(
                isinstance(path, str) and path for path in shadow_files_raw
            ):
                raise ValueError("manifest meta 'shadow_files' must be a list of non-empty strings")
            normalized_shadow_files = [
                str(path).replace("\\", "/") for path in shadow_files_raw
            ]
            if len(set(normalized_shadow_files)) != len(normalized_shadow_files):
                raise ValueError("manifest meta 'shadow_files' contains duplicates")
            shadow_files = [
                _pack_path(root, str(path), "shadow_files entry")
                for path in shadow_files_raw
            ]
            for path in shadow_files:
                if not path.is_file():
                    raise FileNotFoundError(f"shadow shard file not found: {path}")
        else:
            shadow_file = meta.get("shadow_file")
            shadow_files = (
                [_pack_path(root, str(shadow_file), "shadow_file")]
                if shadow_file
                else []
            )
            for path in shadow_files:
                if not path.is_file():
                    raise FileNotFoundError(f"shadow file not found: {path}")

        tensors: list[TensorEntry] = []
        names: set[str] = set()
        declared_shards = (
            set(normalized_weights_files) if weights_files is not None else set()
        )
        declared_shadow = set(normalized_shadow_files) if shadow_files_raw is not None else set()
        if shadow_files_raw is None and meta.get("shadow_file"):
            declared_shadow.add(str(meta["shadow_file"]).replace("\\", "/"))
        declared_shadow.update(path.name.replace("\\", "/") for path in shadow_files)
        shadow_sizes: dict[str, int] = {}
        for path in shadow_files:
            try:
                shadow_sizes[path.name.replace("\\", "/")] = path.stat().st_size
                shadow_sizes[str(path.relative_to(root)).replace("\\", "/")] = path.stat().st_size
            except OSError:
                pass
        for i, t in enumerate(tensors_raw):
            if not isinstance(t, dict):
                raise ValueError(f"tensors[{i}] must be a JSON object")
            required = ("name", "dtype", "shape", "offset", "length")
            for key in required:
                if key not in t:
                    raise ValueError(f"tensors[{i}] missing required key '{key}'")
            name = t["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"tensors[{i}] 'name' must be a non-empty string")
            if name in names:
                raise ValueError(f"manifest.json has duplicate tensor name {name!r}")
            names.add(name)
            dtype = t["dtype"]
            if not isinstance(dtype, str) or not dtype:
                raise ValueError(f"tensors[{i}] 'dtype' must be a non-empty string")
            shape = t["shape"]
            if not isinstance(shape, (list, tuple)) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in shape
            ):
                raise ValueError(f"tensors[{i}] 'shape' must contain non-negative integers")
            offset = int(t["offset"])
            length = int(t["length"])
            alignment = int(t.get("alignment", ALIGNMENT))
            if offset < 0 or length < 0:
                raise ValueError(f"tensors[{i}] offset/length must be non-negative")
            if alignment <= 0:
                raise ValueError(f"tensors[{i}] alignment must be positive")
            shard_file = str(t.get("shard_file", ""))
            if shard_file:
                _pack_path(root, shard_file, f"tensors[{i}].shard_file")
            stripes_raw = t.get("stripes", [])
            if stripes_raw is None:
                stripes_raw = []
            if not isinstance(stripes_raw, list):
                raise ValueError(f"tensors[{i}] 'stripes' must be a list")
            stripes: list[dict[str, Any]] = []
            for stripe in stripes_raw:
                if not isinstance(stripe, dict):
                    raise ValueError(f"tensors[{i}] stripe must be an object")
                sf = str(stripe.get("shard_file", ""))
                off = int(stripe.get("offset", -1))
                length_part = int(stripe.get("length", 0))
                logical = int(stripe.get("logical_offset", 0))
                physical_length = int(stripe.get("physical_length", length_part))
                if (
                    not sf
                    or off < 0
                    or length_part < 0
                    or logical < 0
                    or physical_length < length_part
                ):
                    raise ValueError(f"tensors[{i}] has invalid stripe extent")
                _pack_path(root, sf, f"tensors[{i}] stripe shard_file")
                stripes.append(
                    {
                        "shard_file": sf,
                        "offset": off,
                        "length": length_part,
                        "logical_offset": logical,
                        "physical_length": physical_length,
                    }
                )
            residency = t.get("residency", "streamed")
            dequant = t.get("dequant", "none")
            if not isinstance(residency, str) or not isinstance(dequant, str):
                raise ValueError(f"tensors[{i}] residency/dequant must be strings")
            inner_offset = int(t.get("inner_offset", 0))
            inner_length = int(t.get("inner_length", 0))
            fast_offset = int(t.get("fast_offset", -1))
            fast_length = int(t.get("fast_length", 0))
            fast_shard_file = str(t.get("fast_shard_file", ""))
            if fast_shard_file:
                _pack_path(root, fast_shard_file, f"tensors[{i}].fast_shard_file")
            if (
                inner_offset < 0
                or inner_length < 0
                or fast_offset < -1
                or fast_length < 0
            ):
                raise ValueError(f"tensors[{i}] inner/fast offsets must be non-negative")
            fast_stripes_raw = t.get("fast_stripes", []) or []
            if not isinstance(fast_stripes_raw, list):
                raise ValueError(f"tensors[{i}] 'fast_stripes' must be a list")
            fast_stripes: list[dict[str, Any]] = []
            for stripe in fast_stripes_raw:
                if not isinstance(stripe, dict):
                    raise ValueError(f"tensors[{i}] fast stripe must be an object")
                sf = str(stripe.get("shard_file", ""))
                off = int(stripe.get("offset", -1))
                length_part = int(stripe.get("length", 0))
                logical = int(stripe.get("logical_offset", 0))
                physical_length = int(stripe.get("physical_length", length_part))
                if (
                    not sf
                    or off < 0
                    or length_part < 0
                    or logical < 0
                    or physical_length < length_part
                ):
                    raise ValueError(f"tensors[{i}] has invalid fast stripe extent")
                _pack_path(root, sf, f"tensors[{i}] fast stripe shard_file")
                fast_stripes.append(
                    {
                        "shard_file": sf,
                        "offset": off,
                        "length": length_part,
                        "logical_offset": logical,
                        "physical_length": physical_length,
                    }
                )
            normalized_shard_file = shard_file.replace("\\", "/")
            if weights_files is not None and shard_file:
                if normalized_shard_file not in declared_shards:
                    raise ValueError(
                        f"tensors[{i}] shard_file {shard_file!r} is not listed in weights_files"
                    )
            if stripes:
                if version < 2:
                    raise ValueError(
                        f"tensors[{i}] uses stripes but manifest version is {version}; "
                        "striped extents require manifest version 2"
                    )
                if weights_files is None:
                    raise ValueError(
                        f"tensors[{i}] uses stripes but manifest.json has no "
                        "'weights_files' declaration"
                    )
                for stripe in stripes:
                    if (
                        stripe["shard_file"].replace("\\", "/")
                        not in declared_shards
                    ):
                        raise ValueError(
                            f"tensors[{i}] stripe shard_file {stripe['shard_file']!r} "
                            "is not listed in weights_files"
                        )
            if fast_stripes:
                if not shadow_files:
                    raise ValueError(
                        f"tensors[{i}] uses fast_stripes but no shadow files are declared"
                    )
                for stripe in fast_stripes:
                    if stripe["shard_file"].replace("\\", "/") not in declared_shadow:
                        raise ValueError(
                            f"tensors[{i}] fast stripe shard_file {stripe['shard_file']!r} "
                            "is not listed in shadow_files"
                        )
            if fast_shard_file and fast_shard_file.replace("\\", "/") not in declared_shadow:
                raise ValueError(
                    f"tensors[{i}] fast_shard_file {fast_shard_file!r} "
                    "is not listed in shadow_files"
                )
            if fast_stripes:
                if fast_offset < 0 or fast_length <= 0:
                    raise ValueError(
                        f"tensors[{i}] fast_stripes require fast_offset and fast_length"
                    )
                pieces = sorted(fast_stripes, key=lambda stripe: int(stripe["logical_offset"]))
                cursor = 0
                for index, stripe in enumerate(pieces):
                    logical = int(stripe["logical_offset"])
                    length_part = int(stripe["length"])
                    physical_offset = int(stripe["offset"])
                    physical_length = int(stripe["physical_length"])
                    shard_name = stripe["shard_file"].replace("\\", "/")
                    if length_part <= 0 or physical_length <= 0 or logical != cursor:
                        raise ValueError(
                            f"tensors[{i}] fast stripes have a gap, overlap, or empty extent"
                        )
                    if logical + length_part > fast_length:
                        raise ValueError(
                            f"tensors[{i}] fast stripe range exceeds fast_length"
                        )
                    if physical_offset % alignment != 0 or physical_length % alignment != 0:
                        raise ValueError(
                            f"tensors[{i}] fast stripe physical extent is not aligned"
                        )
                    file_size = shadow_sizes.get(shard_name)
                    if file_size is None:
                        raise FileNotFoundError(
                            f"fast stripe shadow file not found: {root / shard_name}"
                        )
                    if physical_offset + physical_length > file_size:
                        raise ValueError(
                            f"tensors[{i}] fast stripe [{physical_offset}, "
                            f"{physical_offset + physical_length}) exceeds shadow shard "
                            f"{shard_name} size {file_size}"
                        )
                    cursor = logical + length_part
                if cursor != fast_length:
                    raise ValueError(
                        f"tensors[{i}] fast stripes cover {cursor} logical bytes, "
                        f"expected exactly {fast_length}"
                    )
            tensors.append(
                TensorEntry(
                    name=name,
                    layer_id=int(t.get("layer_id", -1)),
                    dtype=dtype,
                    shape=list(shape),
                    offset=offset,
                    length=length,
                    alignment=alignment,
                    residency=residency,
                    dequant=dequant,
                    inner_offset=inner_offset,
                    inner_length=inner_length,
                    fast_offset=fast_offset,
                    fast_length=fast_length,
                    fast_shard_file=fast_shard_file,
                    fast_stripes=tuple(fast_stripes),
                    shard_file=shard_file,
                    stripes=tuple(stripes),
                )
            )

    # Basic consistency check: offsets should fit within the weights files.
        _validate_tensor_offsets(
            tensors,
            weights_files,
            root,
            weights_path,
            manifest_version=version,
            logical_size=(
                int(meta["weights_uncompressed_bytes"])
                if str(meta.get("weights_compression", "")).lower() == "zstd"
                and meta.get("weights_uncompressed_bytes") is not None
                else None
            ),
        )
        if meta.get("quality_certificate_file"):
            from rwkv_ssd.runtime.quality_certificate import verify_quality_certificate

            verify_quality_certificate(root, meta)
        elif require_quality_certificate and _requires_real_quality_certificate(raw, meta):
            if os.environ.get("RWKV_ALLOW_UNCERTIFIED_TRINITY", "").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                raise ValueError(
                    "real RWKV-7 lossy packs require a passing "
                    "quality_certificate.json; run rwkv_ssd.tools.quality_certificate "
                    "or set RWKV_ALLOW_UNCERTIFIED_TRINITY=1 only for diagnostics"
                )
        return cls(
            version=version,
            model_family=raw.get("model_family", "rwkv"),
            weights_path=weights_path,
            tensors=tensors,
            meta=meta,
            pack_dir=root,
            shard_files=shard_files,
        )

    def streamed_tensors(self) -> list[TensorEntry]:
        return [t for t in self.tensors if t.residency == "streamed"]

    def resident_tensors(self) -> list[TensorEntry]:
        return [t for t in self.tensors if t.residency == "resident"]

    def shadow_path(self) -> Path | None:
        name = self.meta.get("shadow_file")
        if not name:
            raw = self.meta.get("shadow_files")
            if isinstance(raw, list) and raw:
                name = raw[0]
        if not name:
            return None
        root = self.pack_dir or self.weights_path.parent
        path = root / str(name)
        return path if path.is_file() else None

    def shadow_paths(self) -> list[Path]:
        """Return contiguous or striped BF16 shadow files in manifest order."""
        raw = self.meta.get("shadow_files")
        if isinstance(raw, list) and raw:
            root = self.pack_dir or self.weights_path.parent
            return [root / str(name) for name in raw if (root / str(name)).is_file()]
        path = self.shadow_path()
        return [path] if path is not None else []

    def has_bf16_shadow(self) -> bool:
        return bool(self.shadow_paths()) and any(
            t.fast_offset >= 0 or bool(t.fast_stripes) for t in self.tensors
        )

    def by_layer(self) -> dict[int, list[TensorEntry]]:
        if self._by_layer_cache is not None:
            return self._by_layer_cache
        out: dict[int, list[TensorEntry]] = {}
        for t in self.tensors:
            out.setdefault(t.layer_id, []).append(t)
        self._by_layer_cache = out
        return out

    def is_sharded(self) -> bool:
        """True when the pack was built with ``pack_runtime --shard N``."""
        return len(self.shard_files) > 1 or any(
            t.shard_file for t in self.tensors
        )

    def shard_path_for(self, entry: TensorEntry) -> Path:
        """Return the absolute path of the shard file holding ``entry``.

        Falls back to ``weights_path`` for legacy single-file packs
        (where ``shard_file`` is empty).
        """
        if entry.shard_file:
            root = self.pack_dir or self.weights_path.parent
            return root / entry.shard_file
        return self.weights_path

    def pack_composition(self) -> dict:
        """Honest byte breakdown of this pack on disk (B3).

        Reads the build-time ``meta.json["pack_composition"]`` block written by
        :func:`rwkv_ssd.tools.pack_runtime._compute_pack_composition`. If
        absent (e.g. older packs), returns an empty dict.
        """
        return dict(self.meta.get("pack_composition") or {})


def _validate_tensor_offsets(
    tensors: list[TensorEntry],
    weights_files: list | None,
    root: Path,
    weights_path: Path,
    *,
    manifest_version: int = 1,
    logical_size: int | None = None,
) -> None:
    """Validate v2 extents and warn about legacy tensor offsets.

    Legacy manifests historically tolerated an offset beyond the current file
    size because some packs were assembled before their sidecars were copied.
    Keep that behavior for non-striped entries.  Manifest-v2 extents are a
    different contract: their logical coverage and physical file bounds must
    be exact before a store is opened.
    """
    import logging

    logger = logging.getLogger(__name__)

    file_sizes: dict[str, int] = {}
    if weights_files is not None:
        paths = [root / str(f) for f in weights_files]
        if weights_path not in paths:
            paths.append(weights_path)
        for p in paths:
            try:
                try:
                    relative = p.relative_to(root).as_posix()
                except ValueError:
                    relative = p.name
                file_sizes[relative] = p.stat().st_size
            except OSError:
                pass
    else:
        w = weights_path
        try:
            file_sizes[str(w)] = w.stat().st_size
        except OSError:
            pass

    for t in tensors:
        if t.stripes:
            if manifest_version < 2:
                raise ValueError(
                    f"tensor {t.name!r} has striped extents in a pre-v2 manifest"
                )

            # Logical extents must form one exact, non-overlapping partition
            # of the tensor.  Sorting makes the check independent of JSON
            # order while keeping the original order available to callers.
            pieces = sorted(
                t.stripes,
                key=lambda stripe: int(stripe["logical_offset"]),
            )
            cursor = 0
            for index, stripe in enumerate(pieces):
                logical = int(stripe["logical_offset"])
                length = int(stripe["length"])
                physical_offset = int(stripe["offset"])
                physical_length = int(stripe["physical_length"])
                shard_file = str(stripe["shard_file"]).replace("\\", "/")

                if length <= 0 or physical_length <= 0:
                    raise ValueError(
                        f"tensor {t.name!r} stripe {index} must have positive lengths"
                    )
                if logical != cursor:
                    if logical < cursor:
                        raise ValueError(
                            f"tensor {t.name!r} has overlapping logical stripe ranges "
                            f"at offset {logical}"
                        )
                    raise ValueError(
                        f"tensor {t.name!r} has a missing logical stripe range "
                        f"[{cursor}, {logical})"
                    )
                if logical + length > t.length:
                    raise ValueError(
                        f"tensor {t.name!r} logical stripe range exceeds entry.length"
                    )
                if physical_length < length:
                    raise ValueError(
                        f"tensor {t.name!r} stripe physical_length is shorter than length"
                    )
                if physical_offset % t.alignment != 0:
                    raise ValueError(
                        f"tensor {t.name!r} stripe offset {physical_offset} is not "
                        f"aligned to {t.alignment} bytes"
                    )
                if physical_length % t.alignment != 0:
                    raise ValueError(
                        f"tensor {t.name!r} stripe physical_length {physical_length} "
                        f"is not aligned to {t.alignment} bytes"
                    )
                if weights_files is None:
                    raise ValueError(
                        f"tensor {t.name!r} stripes require declared shard files"
                    )
                shard_size = file_sizes.get(shard_file)
                if shard_size is None:
                    raise FileNotFoundError(
                        f"stripe shard file not found: {root / shard_file}"
                    )
                if physical_offset + physical_length > shard_size:
                    raise ValueError(
                        f"tensor {t.name!r} stripe [{physical_offset}, "
                        f"{physical_offset + physical_length}) exceeds shard "
                        f"{shard_file} size {shard_size}"
                    )
                cursor = logical + length

            if cursor != t.length:
                raise ValueError(
                    f"tensor {t.name!r} stripes cover {cursor} logical bytes, "
                    f"expected exactly {t.length}"
                )
            continue
        fpath = t.shard_file or "weights.bin"
        normalized = fpath.replace("\\", "/")
        if weights_files is not None and t.shard_file:
            fsize = file_sizes.get(normalized)
        else:
            full = str(root / fpath) if t.shard_file else str(weights_path)
            fsize = file_sizes.get(full)
            if not t.shard_file and logical_size is not None:
                fsize = logical_size
        if fsize is not None and t.offset + t.length > fsize:
            logger.warning(
                "tensor %r: offset %d + length %d exceeds %s size %d",
                t.name,
                t.offset,
                t.length,
                fpath,
                fsize,
            )
