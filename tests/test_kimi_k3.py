"""Kimi-K3 compatibility and opt-in real-checkpoint smoke coverage.

Kimi-K3 is deliberately not routed through the generic packed Transformer
executor yet.  These tests make that boundary explicit and provide a useful
local validation command when the small Hugging Face checkpoint is available.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rwkv_ssd.backends.kimi_k3 import _resolve_execution_dtype
from rwkv_ssd.backends.sequence import HFTransformerPackBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.errors import CapabilityNotSupportedError
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from rwkv_ssd.runtime.manifest import Manifest
from rwkv_ssd.tools.pack_runtime import detect_model_family_from_state, pack


def _kimi_root() -> Path | None:
    raw = os.environ.get("RWKV_KIMI_K3_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def test_kimi_k3_family_detection_is_explicit() -> None:
    state = {
        "language_model.model.embed_tokens.weight": torch.zeros(8, 4),
        "language_model.model.layers.0.linear_attn.in_proj.weight": torch.zeros(4, 4),
        "language_model.model.layers.1.mlp.gate.weight": torch.zeros(4, 4),
        "lm_head.weight": torch.zeros(8, 4),
    }
    assert detect_model_family_from_state(state) == "kimi_k3"


def test_kimi_k3_strategy_selects_reduced_resident_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RWKV_KIMI_DTYPE", raising=False)
    monkeypatch.delenv("RWKV_KIMI_DYNAMIC_INT8", raising=False)
    dtype, quantized = _resolve_execution_dtype("cpu fp16")
    assert dtype is torch.float16
    assert quantized is False
    dtype, quantized = _resolve_execution_dtype("cpu bf16")
    assert dtype is torch.bfloat16
    assert quantized is False
    dtype, quantized = _resolve_execution_dtype("cpu int8")
    assert dtype is torch.float32
    assert quantized is True

    monkeypatch.setenv("RWKV_KIMI_DTYPE", "fp32")
    dtype, quantized = _resolve_execution_dtype("cpu bf16")
    assert dtype is torch.float32
    assert quantized is False


def test_kimi_k3_pack_fails_with_capability_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "kimi"
    model_dir.mkdir()
    save_file(
        {
            "language_model.model.embed_tokens.weight": torch.zeros(8, 4),
            "language_model.model.layers.0.linear_attn.in_proj.weight": torch.zeros(4, 4),
            "lm_head.weight": torch.zeros(8, 4),
        },
        str(model_dir / "model.safetensors"),
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "architectures": ["KimiK3ForConditionalGeneration"],
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "vocab_size": 8,
            }
        ),
        encoding="utf-8",
    )

    pack_dir = tmp_path / "pack"
    pack(
        model_dir,
        pack_dir,
        model_family="kimi_k3",
        hash_weights=False,
        copy_hf_metadata=True,
        quiet=True,
    )
    backend = HFTransformerPackBackend()
    with pytest.raises(CapabilityNotSupportedError, match="model_architecture"):
        backend.load_pack(Manifest.load(pack_dir), "cpu")


def test_kimi_k3_raw_directory_rejects_generic_transformer(tmp_path: Path) -> None:
    model_dir = tmp_path / "kimi"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "kimi_k3"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        InferenceEngine(
            EngineConfig(
                pack_dir=model_dir,
                backend="transformer",
                mode="resident",
                device="cpu",
            )
        )


@pytest.mark.integration
def test_kimi_k3_real_checkpoint_metadata() -> None:
    """Inspect the downloaded asset without importing its custom kernels."""

    root = _kimi_root()
    if root is None or not (root / "config.json").is_file() or not (
        root / "model.safetensors"
    ).is_file():
        pytest.skip("set RWKV_KIMI_K3_DIR to run the real Kimi checkpoint checks")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "kimi_k3"
    assert config["text_config"]["model_type"] == "kimi_linear"
    with safe_open(str(root / "model.safetensors"), framework="pt") as handle:
        keys = list(handle.keys())
    assert any(key.startswith("language_model.model.layers.0.") for key in keys)
    assert any(key.startswith("language_model.model.layers.3.") for key in keys)
    assert any(key.endswith("embed_tokens.weight") for key in keys)
    assert "language_model.lm_head.weight" in keys


@pytest.mark.integration
def test_kimi_k3_real_cpu_reference_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the actual tiny Kimi checkpoint when explicitly supplied by CI/dev.

    The 720 MB checkpoint and ``transformers`` are intentionally not normal
    CI dependencies.  Set ``RWKV_KIMI_K3_DIR`` to a local download to run this
    test; the custom modeling files are loaded only with explicit trust.
    """

    root = _kimi_root()
    if root is None or not (root / "config.json").is_file() or not (
        root / "model.safetensors"
    ).is_file():
        pytest.skip(
            "set RWKV_KIMI_K3_DIR to a local Kimi-K3-0.18B checkout to run the "
            "real CPU smoke test"
        )
    # Transformers' trust_remote_code loader copies the repository's custom
    # Python modules into its module cache. Keep that cache inside the
    # writable test temp root instead of assuming a user-level HF cache is
    # writable in a sandboxed CI runner.
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_MODULES_CACHE", str(hf_home / "modules"))
    transformers = pytest.importorskip("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=True
    )
    try:
        import fla.modules  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(
            "Kimi-K3 custom KDA execution requires fla-core with a working "
            f"Triton runtime on this host: {exc}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    encoded = tokenizer("According to all known laws of aviation", return_tensors="pt")
    with torch.no_grad():
        result = model(**encoded, use_cache=True, return_dict=True)
        generated = model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
        )
    assert result.logits.ndim == 3
    assert result.logits.shape[0] == 1
    assert torch.isfinite(result.logits).all()
    assert generated.shape[0] == 1
    assert generated.shape[1] >= encoded["input_ids"].shape[1]
    assert int(generated.max().item()) < int(getattr(model.config, "vocab_size"))


@pytest.mark.integration
def test_kimi_k3_engine_cpu_compat_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real resident CPU path without requiring Triton/FLA."""

    root = _kimi_root()
    if root is None or not (root / "config.json").is_file() or not (
        root / "model.safetensors"
    ).is_file():
        pytest.skip("set RWKV_KIMI_K3_DIR to run the real Kimi engine contract")
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_MODULES_CACHE", str(hf_home / "modules"))
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=root,
            backend="kimi_k3",
            mode="resident",
            device="cpu",
            strategy="cpu fp32",
            max_tokens=3,
            greedy=True,
        )
    )
    try:
        engine.load()
        assert engine.supports_capability("generation")
        assert engine.supports_capability("state_transfer")
        assert engine.supports_capability("followup_generation")
        assert engine.supports_capability("streaming")
        assert not engine.supports_capability("layer_streaming")

        delivered: list[int] = []
        tokens = engine.generate_tokens(
            "According to all known laws of aviation",
            token_callback=delivered.append,
        )
        assert tokens == delivered
        assert tokens
        assert all(0 <= token < engine.backend.vocab_size for token in tokens)
        assert torch.isfinite(engine.backend.probe_logits()).all()

        state = engine.backend.get_recurrent_state()
        assert state is not None and state.external_state is not None
        follow_metrics = engine.metrics.__class__()
        follow = engine.backend.generate_followup_native(
            " and",
            2,
            metrics=follow_metrics,
            temperature=0.0,
            greedy=True,
        )
        engine.backend.set_recurrent_state(state)
        repeat_follow = engine.backend.generate_followup_native(
            " and",
            2,
            metrics=engine.metrics.__class__(),
            temperature=0.0,
            greedy=True,
        )
        assert follow == repeat_follow
        restored = engine.generate_tokens("", greedy=True)
        assert restored

        # The same seed must produce the same valid IDs even though sampling
        # is deliberately not required to match another backend bit-for-bit.
        first = engine.generate_tokens("hello", temperature=0.8, greedy=False, seed=17)
        second = engine.generate_tokens("hello", temperature=0.8, greedy=False, seed=17)
        assert first == second

        cancelled = threading.Event()
        cancelled.set()
        with pytest.raises(GenerationCancelled):
            engine.generate_tokens("hello", cancel_event=cancelled)
    finally:
        engine.close()


@pytest.mark.integration
def test_kimi_k3_real_reduced_dtype_state_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the half-width resident path, including parked BF16 state."""

    root = _kimi_root()
    if root is None or not (root / "config.json").is_file() or not (
        root / "model.safetensors"
    ).is_file():
        pytest.skip("set RWKV_KIMI_K3_DIR to run the reduced-dtype Kimi contract")
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_MODULES_CACHE", str(hf_home / "modules"))
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=root,
            backend="kimi_k3",
            mode="resident",
            device="cpu",
            strategy="cpu bf16",
            max_tokens=3,
            greedy=True,
        )
    )
    try:
        engine.load()
        assert engine.backend.weight_dtype is torch.bfloat16
        tokens = engine.generate_tokens("According to all known laws of aviation")
        state = engine.backend.get_recurrent_state()
        assert tokens and state is not None and state.external_state is not None
        engine.backend.set_recurrent_state(state)
        restored = engine.backend.decode_greedy(
            state,
            None,
            2,
            engine.metrics,
            temperature=0.0,
            greedy=True,
        )
        assert restored
        assert all(0 <= token < engine.backend.vocab_size for token in restored)
    finally:
        engine.close()
