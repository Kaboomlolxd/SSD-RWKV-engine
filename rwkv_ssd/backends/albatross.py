"""CUDA Albatross adapter for the shared packed-SSD execution pipeline.

Albatross is deliberately an external dependency.  This module owns the
boundary between that checkout and :class:`ManifestWeightProvider`; it does
not copy a checkpoint into a resident VRAM model.  A supported Albatross
variant must expose a layer-wise adapter with the following small protocol::

    zero_state() -> object
    embed(token_id, global_tensors) -> Tensor
    forward_layer(layer_id, x, state, layer_tensors, global_tensors) -> (x, state)
    logits(x, global_tensors) -> Tensor

The adapter may expose ``encode``/``decode`` as well.  A module-level factory
named ``create_ssd_adapter`` (or one of the aliases below) is preferred.  A
monolithic ``forward``/``generate`` implementation is rejected because it
would force all weights into VRAM and bypass the SSD tier machinery.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, TYPE_CHECKING

from rwkv_ssd.backends.pack_backend import PackBackend
from rwkv_ssd.runtime.errors import BackendNotAvailableError
from rwkv_ssd.runtime.generation_control import make_generation_control
from rwkv_ssd.runtime.sampling import sample_torch
from rwkv_ssd.runtime.state_cache import RecurrentState
from rwkv_ssd.runtime.power import throttle_after_work

if TYPE_CHECKING:
    import torch

    from rwkv_ssd.runtime.config import EngineConfig
    from rwkv_ssd.runtime.manifest import Manifest, TensorEntry
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.weight_provider import WeightProvider


_VARIANT_PRIORITY = (
    "faster3a_2606",
    "faster3a_2605",
    "faster3a",
)
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
_GLOBAL_NAMES = frozenset(
    {
        "emb.weight",
        "ln_out.weight",
        "ln_out.bias",
        "head.weight",
    }
)
_REQUIRED_LAYER_SUFFIXES = (
    "ln0.weight",
    "ln0.bias",
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
    "att.x_r",
    "att.x_w",
    "att.x_k",
    "att.x_v",
    "att.x_a",
    "att.x_g",
    "att.w0",
    "att.w1",
    "att.w2",
    "att.a0",
    "att.a1",
    "att.a2",
    "att.v0",
    "att.v1",
    "att.v2",
    "att.g1",
    "att.g2",
    "att.k_k",
    "att.k_a",
    "att.r_k",
    "att.receptance.weight",
    "att.key.weight",
    "att.value.weight",
    "att.output.weight",
    "att.ln_x.weight",
    "att.ln_x.bias",
    "ffn.x_k",
    "ffn.key.weight",
    "ffn.value.weight",
)


def find_albatross_root() -> Path | None:
    """Find an external Albatross checkout without importing it."""
    env = os.environ.get("ALBATROSS_ROOT")
    if env:
        candidate = Path(env).expanduser()
        if (candidate / "README.md").is_file():
            return candidate.resolve()
    here = Path(__file__).resolve().parents[2]
    for candidate in (here / "backends" / "albatross_ref", here.parent / "Albatross"):
        if (candidate / "README.md").is_file():
            return candidate.resolve()
    return None


def _variant_source(root: Path, variant: str) -> Path | None:
    """Return the most useful source file for a named checkout variant."""
    direct = root / f"{variant}.py"
    package = root / variant / "__init__.py"
    nested = root / variant / "model.py"
    nested_named = root / variant / f"{variant}.py"
    nested_main = root / variant / "main.py"
    for candidate in (package, direct, nested, nested_named, nested_main):
        if candidate.is_file():
            return candidate
    return None


def resolve_albatross_variant(root: Path, requested: str | None = None) -> str:
    """Resolve an explicit or checkout-provided Albatross variant name."""
    requested = (requested or os.environ.get("ALBATROSS_VARIANT", "")).strip()
    if requested:
        if _variant_source(root, requested) is not None:
            return requested
        # A dotted module can be importable even when it is not a direct file.
        if (root / requested.replace(".", os.sep)).is_dir():
            return requested
        raise BackendNotAvailableError(
            f"Albatross variant {requested!r} was not found under {root}. "
            "Set ALBATROSS_VARIANT to a layer-wise compatible variant."
        )

    for candidate in _VARIANT_PRIORITY:
        if _variant_source(root, candidate) is not None:
            return candidate
    names: list[str] = []
    try:
        for child in root.iterdir():
            name = child.stem if child.is_file() else child.name
            if name.startswith("faster") and (
                child.is_file() or (child / "__init__.py").is_file()
            ):
                names.append(name)
    except OSError:
        names = []
    if names:
        return sorted(names)[-1]
    raise BackendNotAvailableError(
        f"No Albatross variant was found under {root}. "
        "Set ALBATROSS_VARIANT to a layer-wise compatible variant."
    )


def _import_variant(root: Path, variant: str) -> ModuleType:
    """Import a variant while preserving its sibling-module import behavior."""
    source = _variant_source(root, variant)
    search_paths = [str(root), str(root / variant)]
    for path in reversed(search_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    if source is None:
        try:
            return importlib.import_module(variant)
        except Exception as exc:  # pragma: no cover - depends on external checkout
            raise BackendNotAvailableError(
                f"could not import Albatross variant {variant!r}: {exc}"
            ) from exc

    # Normal package/module imports give relative imports inside a checkout
    # the best chance of working.  Fall back to a file import for variants
    # shipped as a standalone script.
    if source.name == "__init__.py" or source.parent == root:
        try:
            return importlib.import_module(variant)
        except Exception:
            pass
    module_name = f"_rwkv_ssd_albatross_{re.sub(r'[^A-Za-z0-9_]', '_', variant)}"
    kwargs: dict[str, Any] = {}
    if source.name == "__init__.py":
        kwargs["submodule_search_locations"] = [str(source.parent)]
    spec = importlib.util.spec_from_file_location(module_name, source, **kwargs)
    if spec is None or spec.loader is None:
        raise BackendNotAvailableError(
            f"could not load Albatross variant {variant!r} from {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on external checkout
        raise BackendNotAvailableError(
            f"could not import Albatross variant {variant!r}: {exc}"
        ) from exc
    return module


def _clone_object(value: Any) -> Any:
    """Clone common CUDA state containers without knowing their concrete type."""
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "clone"):
        try:
            return value.clone()
        except Exception:
            pass
    if isinstance(value, list):
        return [_clone_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_object(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_object(item) for key, item in value.items()}
    copier = getattr(value, "copy", None)
    if callable(copier):
        try:
            return copier()
        except Exception:
            pass
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise TypeError(
            f"Albatross recurrent state of type {type(value).__name__} cannot be cloned; "
            "the external adapter must provide clone_state()"
        ) from exc


@dataclass
class _AlbatrossState:
    adapter_state: Any
    pending_logits: Any | None = None

    def copy(self) -> "_AlbatrossState":
        return _AlbatrossState(
            _clone_object(self.adapter_state),
            _clone_object(self.pending_logits),
        )


def _call_candidates(
    fn: Callable[..., Any],
    *,
    values: dict[str, Any],
    positional: Iterable[tuple[Any, ...]],
) -> Any:
    """Call an external hook using named aliases, then known positional forms."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - extension callables
        signature = None

    if signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        aliases = {
            "meta": "metadata",
            "model_config": "metadata",
            "config": "metadata",
            "manifest": "metadata",
            "manifest_meta": "metadata",
            "torch_device": "device",
            "idx": "token_id",
            "token": "token_id",
            "hidden": "x",
            "hidden_state": "x",
            "input": "x",
            "recurrent_state": "state",
            "rnn_state": "state",
            "layer_weights": "weights",
            "tensors": "weights",
            "params": "weights",
            "global_tensors": "globals",
            "model_tensors": "globals",
            "i": "layer_id",
        }
        kwargs: dict[str, Any] = {}
        for name, parameter in parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            key = name if name in values else aliases.get(name)
            if key is not None and key in values:
                kwargs[name] = values[key]
        if accepts_kwargs:
            for key, value in values.items():
                kwargs.setdefault(key, value)
        try:
            signature.bind(**kwargs)
        except TypeError:
            pass
        else:
            return fn(**kwargs)

        for args in positional:
            try:
                signature.bind(*args)
            except TypeError:
                continue
            return fn(*args)

    # Opaque extension callables: the first protocol spelling is the least
    # surprising fallback.  A TypeError from the extension is intentionally
    # allowed to surface as an adapter compatibility error to the caller.
    first = next(iter(positional), ())
    return fn(*first)


def _find_callable(target: Any, names: Iterable[str]) -> Callable[..., Any] | None:
    for name in names:
        fn = getattr(target, name, None)
        if callable(fn):
            return fn
    return None


class _AlbatrossAdapter:
    """Normalize one external variant's layer-wise adapter surface."""

    def __init__(self, target: Any, *, metadata: dict[str, Any], device: Any) -> None:
        self.target = target
        self.metadata = metadata
        self.device = device
        self._zero = _find_callable(target, ("zero_state", "generate_zero_state", "init_state"))
        self._embed = _find_callable(target, ("embed_token", "embed", "embedding"))
        self._forward_layer = _find_callable(
            target, ("forward_layer", "layer_forward", "block_forward", "forward_block")
        )
        self._logits = _find_callable(target, ("logits", "head", "output_logits", "project_logits"))
        self._clone_state = _find_callable(target, ("clone_state", "copy_state", "state_clone"))
        self._encode = _find_callable(target, ("encode", "tokenize"))
        self._decode = _find_callable(target, ("decode", "detokenize"))
        if self._zero is None or self._forward_layer is None:
            raise BackendNotAvailableError(
                "the selected Albatross variant does not expose the required "
                "zero_state()/forward_layer() layer-wise adapter; a monolithic "
                "forward()/generate() variant cannot use the SSD pipeline"
            )
        if self._embed is None or self._logits is None:
            # Embedding and the output head have simple dense fallbacks below,
            # so only the recurrent block hook is mandatory.
            pass

    def _invoke(self, fn: Callable[..., Any], values: dict[str, Any], positional: Iterable[tuple[Any, ...]]) -> Any:
        try:
            return _call_candidates(fn, values=values, positional=positional)
        except TypeError as exc:
            raise BackendNotAvailableError(
                f"Albatross layer adapter call {getattr(fn, '__name__', fn)!r} "
                f"does not match the SSD compatibility protocol: {exc}"
            ) from exc

    def zero_state(self) -> Any:
        return self._invoke(
            self._zero,
            {"metadata": self.metadata, "device": self.device, "variant": self.metadata.get("albatross_variant")},
            ((self.metadata, self.device), (self.device,), ()),
        )

    def clone_state(self, state: Any) -> Any:
        if self._clone_state is not None:
            return self._invoke(
                self._clone_state,
                {"state": state},
                ((state,),),
            )
        return _clone_object(state)

    def embed(self, token_id: int, globals_: dict[str, Any]) -> Any:
        if self._embed is None:
            return globals_["emb.weight"][int(token_id)]
        return self._invoke(
            self._embed,
            {"token_id": int(token_id), "globals": globals_, "weights": globals_},
            ((int(token_id), globals_), (globals_, int(token_id)), (int(token_id), globals_["emb.weight"]), (int(token_id),)),
        )

    def forward_layer(
        self,
        layer_id: int,
        x: Any,
        state: Any,
        weights: dict[str, Any],
        globals_: dict[str, Any],
    ) -> tuple[Any, Any]:
        result = self._invoke(
            self._forward_layer,
            {
                "layer_id": int(layer_id),
                "x": x,
                "state": state,
                "weights": weights,
                "globals": globals_,
            },
            (
                (int(layer_id), x, state, weights, globals_),
                (x, state, weights, int(layer_id), globals_),
                (x, state, weights, int(layer_id)),
                (x, state, weights),
                (int(layer_id), x, state, weights),
            ),
        )
        if isinstance(result, dict):
            next_x = result.get("x", result.get("hidden", result.get("output")))
            next_state = result.get("state", result.get("next_state", state))
            if next_x is None:
                raise BackendNotAvailableError(
                    "Albatross forward_layer() returned a mapping without x/hidden/output"
                )
            return next_x, next_state
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result[0], result[1]
        if result is not None:
            # A stateful adapter may mutate its state and return only x.
            return result, state
        raise BackendNotAvailableError(
            "Albatross forward_layer() must return (x, state), a mapping, or x"
        )

    def logits(self, x: Any, globals_: dict[str, Any]) -> Any:
        if self._logits is None:
            return x @ globals_["head.weight"]
        return self._invoke(
            self._logits,
            {"x": x, "globals": globals_, "weights": globals_},
            ((x, globals_), (x, globals_["head.weight"]), (x,)),
        )

    def encode(self, text: str) -> list[int] | None:
        if self._encode is None:
            return None
        result = self._invoke(self._encode, {"text": text, "prompt": text}, ((text,),))
        if hasattr(result, "ids"):
            result = result.ids
        return [int(value) for value in result]

    def decode(self, token_ids: list[int]) -> str | None:
        if self._decode is None:
            return None
        return str(self._invoke(self._decode, {"token_ids": token_ids, "tokens": token_ids}, ((token_ids,),)))

    def configure(self, globals_: dict[str, Any]) -> None:
        fn = _find_callable(self.target, ("configure", "set_global_tensors", "bind_globals"))
        if fn is None:
            return
        self._invoke(
            fn,
            {"globals": globals_, "weights": globals_, "metadata": self.metadata},
            ((globals_,), (globals_, self.metadata), ()),
        )

    def close(self) -> None:
        fn = _find_callable(self.target, ("close", "shutdown", "release"))
        if fn is not None:
            fn()


def _make_external_adapter(
    module: ModuleType,
    *,
    metadata: dict[str, Any],
    device: Any,
) -> _AlbatrossAdapter:
    names = (
        "create_ssd_adapter",
        "create_pack_adapter",
        "create_albatross_adapter",
        "build_ssd_adapter",
    )
    target: Any = None
    for name in names:
        factory = getattr(module, name, None)
        if callable(factory):
            target = _call_candidates(
                factory,
                values={"metadata": metadata, "device": device, "variant": metadata.get("albatross_variant")},
                positional=((metadata, device), (device,), (metadata,), ()),
            )
            break
    if target is None:
        for name in ("SSDAdapter", "PackAdapter", "AlbatrossSSDAdapter", "AlbatrossAdapter"):
            cls = getattr(module, name, None)
            if callable(cls):
                target = _call_candidates(
                    cls,
                    values={"metadata": metadata, "device": device, "variant": metadata.get("albatross_variant")},
                    positional=((metadata, device), (device,), (metadata,), ()),
                )
                break
    if target is None:
        # A variant may expose the protocol directly at module scope.
        target = module
    return _AlbatrossAdapter(target, metadata=metadata, device=device)


class _RWKVTokenizer:
    """Small dependency-free reader for the bundled RWKV world vocabulary."""

    def __init__(self, path: Path) -> None:
        self.idx2token: dict[int, bytes] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                index_text, rest = line.split(" ", 1)
                payload_text, _length = rest.rsplit(" ", 1)
                payload = ast.literal_eval(payload_text)
            except (ValueError, SyntaxError) as exc:
                raise BackendNotAvailableError(
                    f"could not parse bundled RWKV vocabulary {path}: {exc}"
                ) from exc
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            if not isinstance(payload, bytes):
                raise BackendNotAvailableError(f"invalid token payload in {path}")
            self.idx2token[int(index_text)] = payload
        if not self.idx2token:
            raise BackendNotAvailableError(f"RWKV vocabulary {path} is empty")

    def encode(self, text: str) -> list[int]:
        source = text.encode("utf-8")
        output: list[int] = []
        cursor = 0
        while cursor < len(source):
            best_id: int | None = None
            best_length = 0
            for token_id, payload in self.idx2token.items():
                if len(payload) > best_length and source.startswith(payload, cursor):
                    best_id = token_id
                    best_length = len(payload)
            if best_id is None:
                raise ValueError(f"RWKV vocabulary cannot encode byte at offset {cursor}")
            output.append(best_id)
            cursor += best_length
        return output

    def decode(self, token_ids: list[int]) -> str:
        payload = b"".join(self.idx2token[int(token_id)] for token_id in token_ids)
        return payload.decode("utf-8", errors="replace")


def _find_vocab(root: Path) -> Path | None:
    here = Path(__file__).resolve().parents[2]
    candidates = (
        root / "rwkv_vocab_v20230424.txt",
        root / "tokenizer" / "rwkv_vocab_v20230424.txt",
        root / "rwkv_pip_package" / "src" / "rwkv" / "rwkv_vocab_v20230424.txt",
        here / "test_model" / "ChatRWKV" / "rwkv_pip_package" / "src" / "rwkv" / "rwkv_vocab_v20230424.txt",
        here / "test_model" / "ChatRWKV" / "tokenizer" / "rwkv_vocab_v20230424.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class AlbatrossBackend(PackBackend):
    """RWKV-7 CUDA backend that streams packed layers through the SSD provider."""

    supports_streaming = True

    def __init__(self, variant: str | None = None) -> None:
        self._requested_variant = variant
        self._variant: str | None = variant
        self._root: Path | None = None
        self._manifest: Manifest | None = None
        self._device: Any = None
        self._torch: Any = None
        self._adapter: _AlbatrossAdapter | None = None
        self._tokenizer: _RWKVTokenizer | None = None
        self._globals: dict[str, Any] = {}
        self._layer_entries: dict[int, list[TensorEntry]] = {}
        self._n_layer = 0
        self._n_embd = 0
        self._vocab_size = 0
        self._last_state: RecurrentState | None = None

    @classmethod
    def availability_error(cls, variant: str | None = None) -> str | None:
        """Return a user-facing reason when the backend cannot be selected."""
        try:
            import torch
        except ImportError:
            return "Albatross requires a PyTorch installation with CUDA support"
        if not torch.cuda.is_available():
            return "Albatross requires an available CUDA device"
        root = find_albatross_root()
        if root is None:
            return (
                "Albatross was not found; clone the external checkout and set "
                "ALBATROSS_ROOT"
            )
        try:
            resolve_albatross_variant(root, variant)
        except BackendNotAvailableError as exc:
            return str(exc)
        return None

    @property
    def num_layers(self) -> int:
        return self._n_layer

    @property
    def variant(self) -> str | None:
        return self._variant

    def load(self, model_path: str, strategy: str, device: str) -> None:
        del model_path, strategy, device
        raise BackendNotAvailableError(
            "Albatross is pack-only; use InferenceEngine with a RWKV-7 runtime pack"
        )

    def _validate_manifest(self, manifest: Manifest) -> None:
        family = str(manifest.model_family or manifest.meta.get("model_family", "")).lower()
        if not family.startswith("rwkv7") or bool(manifest.meta.get("deepembed")):
            raise BackendNotAvailableError(
                "Albatross supports ordinary RWKV-7 packs only; DeepEmbed and other "
                f"model families are not compatible (got {family or 'unknown'!r})"
            )
        try:
            rwkv_version = int(manifest.meta.get("rwkv_version", 7))
        except (TypeError, ValueError):
            rwkv_version = 0
        if rwkv_version != 7:
            raise BackendNotAvailableError(
                f"Albatross requires rwkv_version=7, got {rwkv_version}"
            )
        names = {entry.name for entry in manifest.tensors}
        missing_globals = sorted(_GLOBAL_NAMES - names)
        if missing_globals:
            raise BackendNotAvailableError(
                "RWKV-7 pack is missing Albatross global tensors: "
                + ", ".join(missing_globals)
            )
        layer_ids = sorted(
            {
                int(match.group(1))
                for entry in manifest.tensors
                if (match := _BLOCK_RE.match(entry.name)) is not None
            }
        )
        if not layer_ids or layer_ids != list(range(len(layer_ids))):
            raise BackendNotAvailableError(
                "Albatross requires contiguous blocks.0..blocks.N-1 tensors in the pack"
            )
        try:
            declared_layers = int(manifest.meta.get("n_layer", len(layer_ids)))
        except (TypeError, ValueError):
            declared_layers = 0
        if declared_layers > 0 and declared_layers != len(layer_ids):
            raise BackendNotAvailableError(
                "RWKV-7 pack metadata n_layer does not match its block tensors: "
                f"metadata={declared_layers}, blocks={len(layer_ids)}"
            )
        by_name = {entry.name: entry for entry in manifest.tensors}
        try:
            declared_embd = int(manifest.meta.get("n_embd", 0) or 0)
            declared_vocab = int(manifest.meta.get("vocab_size", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise BackendNotAvailableError(
                "RWKV-7 Albatross metadata n_embd/vocab_size must be integers"
            ) from exc
        if declared_embd <= 0 or declared_vocab <= 0:
            raise BackendNotAvailableError(
                "RWKV-7 Albatross packs must declare positive n_embd and vocab_size"
            )
        for name, expected in (
            ("emb.weight", [declared_vocab, declared_embd]),
            ("head.weight", [declared_embd, declared_vocab]),
            ("ln_out.weight", [declared_embd]),
            ("ln_out.bias", [declared_embd]),
        ):
            actual = [int(value) for value in by_name[name].shape]
            if actual != expected:
                raise BackendNotAvailableError(
                    f"Albatross tensor {name!r} has shape {actual}, expected {expected}"
                )
        missing_by_layer: list[str] = []
        for layer_id in layer_ids:
            present = {
                entry.name[len(f"blocks.{layer_id}.") :]
                for entry in manifest.tensors
                if entry.name.startswith(f"blocks.{layer_id}.")
            }
            missing_by_layer.extend(
                f"blocks.{layer_id}.{suffix}"
                for suffix in _REQUIRED_LAYER_SUFFIXES
                if suffix not in present
            )
        if missing_by_layer:
            preview = ", ".join(missing_by_layer[:8])
            suffix = " ..." if len(missing_by_layer) > 8 else ""
            raise BackendNotAvailableError(
                "RWKV-7 pack is missing required Albatross layer tensors: "
                f"{preview}{suffix}"
            )

    def load_pack(self, manifest: Manifest, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Albatross requires PyTorch with CUDA support"
            ) from exc
        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise BackendNotAvailableError(
                "Albatross requires an available CUDA device; it has no CPU fallback"
            )
        self._validate_manifest(manifest)
        root = find_albatross_root()
        if root is None:
            raise BackendNotAvailableError(
                "Albatross was not found; clone it and set ALBATROSS_ROOT"
            )
        variant = resolve_albatross_variant(root, self._requested_variant)
        metadata = dict(manifest.meta)
        metadata["model_family"] = manifest.model_family
        metadata["albatross_variant"] = variant
        module = _import_variant(root, variant)
        try:
            adapter = _make_external_adapter(
                module, metadata=metadata, device=torch_device
            )
        except BackendNotAvailableError:
            raise
        except Exception as exc:  # pragma: no cover - depends on external checkout
            raise BackendNotAvailableError(
                f"Albatross variant {variant!r} is not compatible with the "
                f"layer-wise SSD adapter protocol: {exc}"
            ) from exc
        tokenizer = None
        vocab = _find_vocab(root)
        if adapter._encode is None or adapter._decode is None:
            if vocab is None:
                raise BackendNotAvailableError(
                    "Albatross needs a tokenizer: provide encode/decode in the "
                    "variant or a bundled rwkv_vocab_v20230424.txt"
                )
            tokenizer = _RWKVTokenizer(vocab)
        self._manifest = manifest
        self._root = root
        self._variant = variant
        self._device = torch_device
        self._torch = torch
        self._adapter = adapter
        self._tokenizer = tokenizer
        self._n_layer = len(
            {
                int(match.group(1))
                for entry in manifest.tensors
                if (match := _BLOCK_RE.match(entry.name)) is not None
            }
        )
        self._n_embd = int(manifest.meta.get("n_embd", 0) or 0)
        self._vocab_size = int(manifest.meta.get("vocab_size", 0) or 0)
        self._reindex_manifest(manifest)

    def refresh_manifest(self, manifest: Manifest) -> None:
        self._validate_manifest(manifest)
        self._manifest = manifest
        self._reindex_manifest(manifest)

    def _reindex_manifest(self, manifest: Manifest) -> None:
        self._layer_entries = {}
        for entry in manifest.tensors:
            match = _BLOCK_RE.match(entry.name)
            if match is not None:
                self._layer_entries.setdefault(int(match.group(1)), []).append(entry)

    def _load_globals(self, provider: WeightProvider) -> None:
        if len(self._globals) == len(_GLOBAL_NAMES):
            return
        if self._manifest is None:
            raise RuntimeError("Albatross backend is not loaded")
        for layer_id in sorted({entry.layer_id for entry in self._manifest.tensors}):
            entries = [
                entry
                for entry in self._manifest.tensors
                if entry.layer_id == layer_id and entry.name in _GLOBAL_NAMES
            ]
            if entries:
                self._globals.update(provider.load_layer_tensors_dense(entries))
        missing = sorted(_GLOBAL_NAMES - self._globals.keys())
        if missing:
            raise BackendNotAvailableError(
                "provider could not materialize Albatross global tensors: "
                + ", ".join(missing)
            )
        if self._adapter is not None:
            self._adapter.configure(self._globals)

    def prepare_provider(self, provider: WeightProvider, *, config: EngineConfig) -> None:
        """Load globals and honor F5 through the provider-owned dense cache."""
        self._load_globals(provider)
        tier = str(getattr(config, "_ram_budget_tier_applied", "")).upper()
        full = tier == "F5" or bool(getattr(config, "warm_z", False)) or (
            str(getattr(config, "cache_format", "")).lower() == "dense"
        )
        if full and self._layer_entries:
            for layer_id in sorted(self._layer_entries):
                # The returned mapping is intentionally not copied into the
                # backend.  F5 retention belongs to ManifestWeightProvider,
                # which reports and bounds the memory consistently with the
                # other pack backends.
                provider.load_layer_tensors_dense(self._layer_entries[layer_id])

    def _encode(self, prompt: str) -> list[int]:
        if self._adapter is not None:
            ids = self._adapter.encode(prompt)
            if ids is not None:
                return ids
        if self._tokenizer is None:
            raise BackendNotAvailableError("Albatross tokenizer is not loaded")
        return self._tokenizer.encode(prompt)

    def decode_text(self, token_ids: list[int]) -> str:
        if self._adapter is not None:
            text = self._adapter.decode(token_ids)
            if text is not None:
                return text
        if self._tokenizer is None:
            raise BackendNotAvailableError("Albatross tokenizer is not loaded")
        return self._tokenizer.decode(token_ids)

    def _clone_runtime_state(self, state: _AlbatrossState) -> _AlbatrossState:
        adapter_state = (
            self._adapter.clone_state(state.adapter_state)
            if self._adapter is not None
            else _clone_object(state.adapter_state)
        )
        return _AlbatrossState(adapter_state, _clone_object(state.pending_logits))

    def _layer_step(
        self,
        token_id: int,
        runtime_state: _AlbatrossState,
        provider: WeightProvider,
        metrics: MetricsCollector,
        *,
        control: Any | None = None,
    ) -> tuple[Any, _AlbatrossState]:
        if self._adapter is None:
            raise RuntimeError("Albatross adapter is not loaded")
        x = self._adapter.embed(int(token_id), self._globals)
        state = runtime_state.adapter_state
        for layer_id in sorted(self._layer_entries):
            if control is not None:
                control.check()
            entries = self._layer_entries[layer_id]
            if layer_id + 1 in self._layer_entries:
                provider.prefetch_layer(self._layer_entries[layer_id + 1])
            layer_tensors = provider.load_layer_tensors_dense(entries)
            started = time.perf_counter()
            x, state = self._adapter.forward_layer(
                layer_id, x, state, layer_tensors, self._globals
            )
            row = metrics.start_layer(layer_id)
            row.compute_ms += (time.perf_counter() - started) * 1000.0
        logits = self._adapter.logits(x, self._globals)
        return logits, _AlbatrossState(state)

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
        self._load_globals(provider)
        if self._adapter is None:
            raise RuntimeError("Albatross adapter is not loaded")
        if initial_state is not None:
            if not isinstance(initial_state.external_state, _AlbatrossState):
                raise ValueError("Albatross initial state has an incompatible representation")
            runtime = self._clone_runtime_state(initial_state.external_state)
            last = int(initial_state.last_token_id)
        else:
            runtime = _AlbatrossState(self._adapter.zero_state())
            last = 0
        token_ids = self._encode(text)
        if not token_ids:
            token_ids = [0]
        control = make_generation_control(
            cancel_event=cancel_event,
            deadline=deadline,
        )
        logits: Any | None = runtime.pending_logits
        for token_id in token_ids:
            if control is not None:
                control.check()
            logits, runtime = self._layer_step(
                int(token_id), runtime, provider, metrics, control=control
            )
            last = int(token_id)
        runtime.pending_logits = logits
        state = RecurrentState(last_token_id=last, external_state=runtime)
        self._last_state = RecurrentState(
            last_token_id=state.last_token_id,
            external_state=self._clone_runtime_state(runtime),
        )
        return state

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
        if not isinstance(state.external_state, _AlbatrossState):
            raise ValueError("Albatross state has an incompatible representation")
        self._load_globals(provider)
        runtime = self._clone_runtime_state(state.external_state)
        logits = runtime.pending_logits
        if logits is None:
            logits, runtime = self._layer_step(
                state.last_token_id, runtime, provider, metrics
            )
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        output: list[int] = []
        for _ in range(max(0, int(max_tokens))):
            if control is not None:
                control.check()
            started = time.perf_counter()
            token_id = sample_torch(
                logits,
                temperature=float(temperature),
                greedy=float(temperature) <= 0.0,
            )
            output.append(int(token_id))
            # Advance the recurrent state after every emitted token, including
            # the final one, so follow-up generation starts at the right point.
            logits, runtime = self._layer_step(
                int(token_id), runtime, provider, metrics, control=control
            )
            runtime.pending_logits = logits
            if control is not None:
                control.emit(int(token_id))
            throttle_after_work(started, int(getattr(metrics, "power_percent", 100)))
            token_ms = (time.perf_counter() - started) * 1000.0
            metrics.token_latencies_ms.append(token_ms)
            if len(metrics.token_latencies_ms) == 1 and metrics.ttft_s <= 0.0:
                metrics.ttft_s = token_ms / 1000.0
        metrics.tokens_generated = len(output)
        self._last_state = RecurrentState(
            last_token_id=(output[-1] if output else state.last_token_id),
            external_state=self._clone_runtime_state(runtime),
        )
        return output

    def generate_greedy(
        self,
        prompt: str,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[int]:
        state = self.prefill_text(prompt, provider, metrics)
        return self.decode_greedy(state, provider, max_tokens, metrics)

    def get_recurrent_state(self) -> RecurrentState | None:
        if self._last_state is None:
            return None
        state = self._last_state
        if not isinstance(state.external_state, _AlbatrossState):
            return None
        return RecurrentState(
            last_token_id=state.last_token_id,
            external_state=self._clone_runtime_state(state.external_state),
        )

    def set_recurrent_state(self, state: RecurrentState) -> None:
        if not isinstance(state.external_state, _AlbatrossState):
            raise ValueError("Albatross state has an incompatible representation")
        self._last_state = RecurrentState(
            last_token_id=int(state.last_token_id),
            external_state=self._clone_runtime_state(state.external_state),
        )

    def probe_logits(self, state: RecurrentState | None = None) -> Any | None:
        current = state or self._last_state
        if current is None or not isinstance(current.external_state, _AlbatrossState):
            return None
        logits = current.external_state.pending_logits
        if logits is None:
            return None
        return logits.detach().clone() if hasattr(logits, "detach") else _clone_object(logits)

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
        self._adapter = None
        self._globals.clear()
        self._last_state = None


__all__ = [
    "AlbatrossBackend",
    "find_albatross_root",
    "resolve_albatross_variant",
]
