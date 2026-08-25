"""Integration coverage for the packed Mamba/Transformer paths."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import threading

import pytest
import torch
from safetensors.torch import save_file

from research.real_sequence_models import ReferenceLlama, ReferenceMamba2
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine
from rwkv_ssd.runtime.errors import CapabilityNotSupportedError
from rwkv_ssd.runtime.generation_control import GenerationCancelled
from rwkv_ssd.runtime.state_cache import PrefixStateCache
from rwkv_ssd.tools.pack_runtime import pack


ROOT = Path(__file__).resolve().parents[1]
MAMBA_SOURCE = ROOT / "test_model" / "Small mamba"
TRANSFORMER_SOURCE = ROOT / "test_model" / "Small_transformer"


class _FixedTokenizer:
    """Small local tokenizer shim for public engine contract tests.

    The model-math tests use fixed IDs, but the common engine methods accept
    text.  Avoid making the suite depend on the optional Transformers runtime
    while still exercising the exact text entry points.
    """

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(char) % 1024) for char in str(text)] or [0]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(token)) for token in token_ids)

    def bos_id(self) -> int:
        return 0


@pytest.fixture(scope="module")
def sequence_packs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("sequence-packs")
    mamba = root / "mamba"
    transformer = root / "transformer"
    pack(
        MAMBA_SOURCE,
        mamba,
        model_family="mamba2",
        pack_codec="none",
        copy_hf_metadata=True,
        quiet=True,
    )
    pack(
        TRANSFORMER_SOURCE,
        transformer,
        model_family="llama",
        pack_codec="none",
        copy_hf_metadata=True,
        quiet=True,
    )
    return {"mamba": mamba, "transformer": transformer}


def _engine(
    pack_dir: Path,
    backend: str,
    mode: str,
    *,
    stream_layer_cache: bool = False,
    max_provider_cache_layers: int = 0,
) -> InferenceEngine:
    engine = InferenceEngine(
        EngineConfig(
            pack_dir=pack_dir,
            backend=backend,
            mode=mode,
            device="cpu",
            max_tokens=8,
            verify_hash=False,
            stream_layer_cache=stream_layer_cache,
            max_provider_cache_layers=max_provider_cache_layers,
        )
    )
    engine.load()
    return engine


def _public_engine(
    pack_dir: Path,
    backend: str,
    mode: str,
    *,
    max_tokens: int = 4,
    state_cache: bool = False,
    system_prefix: str | None = None,
) -> InferenceEngine:
    engine = _engine(
        pack_dir,
        backend,
        mode,
        stream_layer_cache=state_cache,
    )
    engine.config.max_tokens = max_tokens
    engine.config.state_cache = state_cache
    engine.config.system_prefix = system_prefix
    engine.backend._tokenizer = _FixedTokenizer()  # type: ignore[attr-defined]
    if state_cache and engine.prefix_cache is None:
        # The fixture is intentionally created after load so this test helper
        # does not alter the production load path.
        from rwkv_ssd.runtime.state_cache import PrefixStateCache

        engine._prefix_cache = PrefixStateCache(
            max_entries=4,
            disk_dir=pack_dir / ".public-prefix-test",
        )
    return engine


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pack_name", "backend", "reference", "source"),
    [
        ("mamba", "mamba2", ReferenceMamba2, MAMBA_SOURCE),
        ("transformer", "transformer", ReferenceLlama, TRANSFORMER_SOURCE),
    ],
)
def test_sequence_pack_matches_reference_and_modes(
    sequence_packs: dict[str, Path],
    pack_name: str,
    backend: str,
    reference: type,
    source: Path,
) -> None:
    ids = [1, 7, 3, 11, 4]
    reference_model = reference.from_pretrained_local(source)
    expected_logits, reference_state = reference_model.forward(torch.tensor(ids))
    expected: list[int] = []
    logits = expected_logits
    for _ in range(8):
        token = int(torch.argmax(logits[0, -1]).item())
        expected.append(token)
        logits, reference_state = reference_model.forward(
            torch.tensor([[token]]), reference_state
        )

    outputs: dict[str, list[int]] = {}
    for mode in ("resident", "partial", "streaming"):
        engine = _engine(sequence_packs[pack_name], backend, mode)
        try:
            provider = engine._get_or_create_pack_provider()
            state = engine.backend.prefill_ids(ids, provider, engine.metrics)
            assert state.sequence_state is not None
            assert state.sequence_state.next_logits is not None
            assert torch.allclose(
                state.sequence_state.next_logits.float()[0],
                expected_logits[0, -1].float(),
                atol=2e-5,
                rtol=2e-5,
            )
            outputs[mode] = engine.backend.decode_ids(
                state, provider, 8, engine.metrics
            )
        finally:
            engine.close()

    assert outputs["resident"] == expected
    assert outputs["partial"] == expected
    assert outputs["streaming"] == expected


@pytest.mark.integration
def test_mamba_batch_matches_independent_sessions(sequence_packs: dict[str, Path]) -> None:
    engine = _engine(sequence_packs["mamba"], "mamba2", "streaming")
    try:
        provider = engine._get_or_create_pack_provider()
        batches = [[1, 7, 3], [4, 2, 9, 6]]
        batched = engine.backend.generate_greedy_ids_batch(
            batches, provider, 5, engine.metrics
        )
        independent = [
            engine.backend.generate_greedy_ids(ids, provider, 5, engine.metrics)
            for ids in batches
        ]
        assert batched == independent
    finally:
        engine.close()


@pytest.mark.integration
def test_mamba_grouped_ssm_retains_updated_state(
    sequence_packs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grouped Mamba update must remain recurrent across generated tokens."""
    monkeypatch.delenv("RWKV_MAMBA_GROUPED_SSM", raising=False)
    reference = _engine(sequence_packs["mamba"], "mamba2", "resident")
    try:
        provider = reference._get_or_create_pack_provider()
        expected = reference.backend.generate_greedy_ids(
            [1, 7, 3], provider, 4, reference.metrics
        )
    finally:
        reference.close()

    monkeypatch.setenv("RWKV_MAMBA_GROUPED_SSM", "1")
    grouped = _engine(sequence_packs["mamba"], "mamba2", "resident")
    try:
        provider = grouped._get_or_create_pack_provider()
        actual = grouped.backend.generate_greedy_ids([1, 7, 3], provider, 4, grouped.metrics)
        assert grouped.backend._sequence_kernel.startswith("torch_grouped_ssm")
    finally:
        grouped.close()
    assert actual == expected


@pytest.mark.integration
def test_sequence_provider_cache_obeys_layer_cap(
    sequence_packs: dict[str, Path],
) -> None:
    engine = _engine(
        sequence_packs["mamba"],
        "mamba2",
        "streaming",
        stream_layer_cache=True,
        max_provider_cache_layers=1,
    )
    try:
        provider = engine._get_or_create_pack_provider()
        engine.backend.generate_greedy_ids([1, 7, 3], provider, 4, engine.metrics)
        assert len(provider.cached_layer_ids()) <= 1
        assert provider.cache_stats()["provider_cache_evictions"] > 0
    finally:
        engine.close()


@pytest.mark.integration
def test_sequence_snapshot_and_prefix_cache_round_trip(
    sequence_packs: dict[str, Path], tmp_path: Path
) -> None:
    engine = _engine(sequence_packs["mamba"], "mamba2", "streaming")
    snapshot = tmp_path / "mamba.snapshot"
    try:
        provider = engine._get_or_create_pack_provider()
        engine.backend.generate_greedy_ids([1, 7, 3], provider, 3, engine.metrics)
        state = engine.backend.get_recurrent_state()
        assert state is not None and state.sequence_state is not None
        engine.save_snapshot(snapshot, prompt="fixed ids")

        cache = PrefixStateCache(max_entries=2, disk_dir=tmp_path / "prefix")
        cache.put("fixed", state)
        restarted = PrefixStateCache(max_entries=2, disk_dir=tmp_path / "prefix")
        restored = restarted.get("fixed")
        assert restored is not None and restored.sequence_state is not None
        assert restored.sequence_state.position == state.sequence_state.position
        assert restarted.stats.hits == 1
    finally:
        engine.close()

    restored_engine = _engine(sequence_packs["mamba"], "mamba2", "streaming")
    try:
        restored_engine.load_snapshot(snapshot)
        restored_state = restored_engine.backend.get_recurrent_state()
        assert restored_state is not None and restored_state.sequence_state is not None
        provider = restored_engine._get_or_create_pack_provider()
        continuation = restored_engine.backend.decode_ids(
            restored_state, provider, 3, restored_engine.metrics
        )
        assert len(continuation) == 3
    finally:
        restored_engine.close()


@pytest.mark.integration
def test_transformer_batch_is_explicitly_unsupported(
    sequence_packs: dict[str, Path]
) -> None:
    engine = _engine(sequence_packs["transformer"], "transformer", "streaming")
    try:
        with pytest.raises(Exception, match="does not support weight-stationary batching"):
            engine.generate_batch(["a", "b"], max_tokens=1)
    finally:
        engine.close()


@pytest.mark.integration
def test_transformer_qualified_batch_matches_independent_sessions(
    sequence_packs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A qualified equal-length Transformer pack uses one batched KV sweep."""
    monkeypatch.setenv("RWKV_TRANSFORMER_BATCH", "1")
    batch_engine = _public_engine(
        sequence_packs["transformer"], "transformer", "streaming", max_tokens=2
    )
    try:
        batched = batch_engine.generate_batch(["a", "b"], max_tokens=2)
        assert batch_engine.supports_capability("batch") is True
        assert batch_engine.metrics.batch_size == 2
        assert batch_engine.metrics.batch_prefill_wall_s > 0.0
        assert batch_engine.metrics.batch_decode_wall_s > 0.0
        assert batch_engine.metrics.tokens_generated == 4
        assert len(batch_engine.backend._last_batch_states) == 2  # type: ignore[attr-defined]
        assert all(
            state.sequence_state is not None
            and state.sequence_state.batch_size == 1
            for state in batch_engine.backend._last_batch_states  # type: ignore[attr-defined]
        )

        independent: list[str] = []
        for prompt in ("a", "b"):
            engine = _public_engine(
                sequence_packs["transformer"],
                "transformer",
                "streaming",
                max_tokens=2,
            )
            try:
                independent.append(engine.generate(prompt))
            finally:
                engine.close()
        assert batched == independent

        with pytest.raises(CapabilityNotSupportedError, match="batch_shape"):
            batch_engine.generate_batch(["a", "long"], max_tokens=1)
    finally:
        batch_engine.close()


@pytest.mark.integration
def test_transformer_fused_qkv_gqa_fixture(tmp_path: Path) -> None:
    """Exercise the common Qwen/Llama fused-QKV + GQA structural variant."""
    source = tmp_path / "source"
    pack_dir = tmp_path / "pack"
    source.mkdir()
    generator = torch.Generator().manual_seed(19)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    hidden, vocab, q_heads, kv_heads, head_dim, intermediate = 8, 24, 4, 2, 2, 16
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": random(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": random(vocab, hidden),
    }
    prefix = "model.layers.0."
    state.update(
        {
            prefix + "input_layernorm.weight": torch.ones(hidden),
            prefix + "post_attention_layernorm.weight": torch.ones(hidden),
            prefix + "self_attn.qkv_proj.weight": random(
                q_heads * head_dim + 2 * kv_heads * head_dim, hidden
            ),
            prefix + "self_attn.o_proj.weight": random(hidden, hidden),
            prefix + "self_attn.q_norm.weight": torch.ones(head_dim),
            prefix + "self_attn.k_norm.weight": torch.ones(head_dim),
            prefix + "mlp.gate_up_proj.weight": random(2 * intermediate, hidden),
            prefix + "mlp.down_proj.weight": random(hidden, intermediate),
        }
    )
    save_file(state, str(source / "model.safetensors"))
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 1,
                "hidden_size": hidden,
                "num_attention_heads": q_heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "intermediate_size": intermediate,
                "vocab_size": vocab,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    pack(
        source,
        pack_dir,
        model_family="qwen2",
        pack_codec="none",
        copy_hf_metadata=True,
        quiet=True,
    )

    outputs: dict[str, list[int]] = {}
    for mode in ("resident", "streaming"):
        engine = _engine(pack_dir, "transformer", mode)
        try:
            provider = engine._get_or_create_pack_provider()
            outputs[mode] = engine.backend.generate_greedy_ids(
                [1, 2, 3], provider, 5, engine.metrics
            )
            state_after = engine.backend.get_recurrent_state()
            assert state_after is not None and state_after.sequence_state is not None
            assert state_after.sequence_state.context_length == 8
        finally:
            engine.close()
    assert outputs["resident"] == outputs["streaming"]


@pytest.mark.integration
def test_transformer_sdpa_prefix_continuation_matches_manual(
    sequence_packs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDPA must preserve the absolute offset of an existing KV prefix."""
    ids = [1, 7, 3, 11, 4]

    monkeypatch.delenv("RWKV_SEQUENCE_SDPA", raising=False)
    manual = _engine(sequence_packs["transformer"], "transformer", "resident")
    try:
        provider = manual._get_or_create_pack_provider()
        prefix = manual.backend.prefill_ids(ids[:3], provider, manual.metrics)
        continued = manual.backend.prefill_ids(
            ids[3:], provider, manual.metrics, initial_state=prefix
        )
        manual_logits = continued.sequence_state.next_logits
        assert manual_logits is not None
    finally:
        manual.close()

    monkeypatch.setenv("RWKV_SEQUENCE_SDPA", "1")
    accelerated = _engine(sequence_packs["transformer"], "transformer", "resident")
    try:
        provider = accelerated._get_or_create_pack_provider()
        prefix = accelerated.backend.prefill_ids(ids[:3], provider, accelerated.metrics)
        continued = accelerated.backend.prefill_ids(
            ids[3:], provider, accelerated.metrics, initial_state=prefix
        )
        sdpa_logits = continued.sequence_state.next_logits
        assert sdpa_logits is not None
        torch.testing.assert_close(sdpa_logits, manual_logits, rtol=1e-6, atol=1e-5)
        assert accelerated.backend._sequence_kernel == "sdpa_prefill_manual_decode"
    finally:
        accelerated.close()


@pytest.mark.integration
def test_transformer_sliding_window_bounds_prefill_cache(
    sequence_packs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compacted window must stay correct across prefill and decode."""
    sliding_pack = tmp_path / "sliding-transformer"
    shutil.copytree(sequence_packs["transformer"], sliding_pack)
    config_path = sliding_pack / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sliding_window"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    ids = [1, 7, 3, 11, 4]

    monkeypatch.delenv("RWKV_SEQUENCE_SDPA", raising=False)
    manual = _engine(sliding_pack, "transformer", "resident")
    try:
        provider = manual._get_or_create_pack_provider()
        state = manual.backend.prefill_ids(ids, provider, manual.metrics)
        assert state.sequence_state is not None
        assert state.sequence_state.context_length == 3
        manual_tokens = manual.backend.decode_ids(state, provider, 4, manual.metrics)
    finally:
        manual.close()

    monkeypatch.setenv("RWKV_SEQUENCE_SDPA", "1")
    accelerated = _engine(sliding_pack, "transformer", "resident")
    try:
        provider = accelerated._get_or_create_pack_provider()
        state = accelerated.backend.prefill_ids(ids, provider, accelerated.metrics)
        assert state.sequence_state is not None
        assert state.sequence_state.context_length == 3
        sdpa_tokens = accelerated.backend.decode_ids(state, provider, 4, accelerated.metrics)
    finally:
        accelerated.close()
    assert sdpa_tokens == manual_tokens


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pack_name", "backend"),
    [("mamba", "mamba2"), ("transformer", "transformer")],
)
def test_public_sequence_engine_contract(
    sequence_packs: dict[str, Path],
    pack_name: str,
    backend: str,
    tmp_path: Path,
) -> None:
    """Exercise the user-facing engine API, not only direct backend methods."""

    outputs: dict[str, list[int]] = {}
    followups: dict[str, list[int]] = {}
    for mode in ("resident", "streaming"):
        engine = _public_engine(
            sequence_packs[pack_name], backend, mode, max_tokens=4
        )
        try:
            outputs[mode] = engine.generate_tokens("hello")
            assert len(outputs[mode]) == 4
            assert engine.metrics.tokens_generated == 4
            assert engine.metrics.prefill_wall_s > 0.0
            assert engine.metrics.decode_wall_s > 0.0

            state = engine.backend.get_recurrent_state()
            assert state is not None
            snapshot = tmp_path / f"{backend}-{mode}.snapshot"
            engine.save_snapshot(snapshot, prompt="hello")

            callback_ids: list[int] = []
            engine.generate_followup(
                "!",
                max_tokens=3,
                token_callback=callback_ids.append,
            )
            followups[mode] = callback_ids
            assert len(callback_ids) == 3

            # Incremental text streaming is the same token contract exposed
            # through decoded chunks.
            stream_engine = _public_engine(
                sequence_packs[pack_name], backend, mode, max_tokens=3
            )
            try:
                chunks = list(stream_engine.generate_stream("stream"))
                assert len(chunks) == 3
                assert stream_engine.metrics.tokens_generated == 3
            finally:
                stream_engine.close()

            restored = _public_engine(
                sequence_packs[pack_name], backend, mode, max_tokens=3
            )
            try:
                restored.load_snapshot(snapshot)
                restored_ids: list[int] = []
                restored.generate_followup(
                    "!",
                    max_tokens=3,
                    token_callback=restored_ids.append,
                )
                assert restored_ids == callback_ids
            finally:
                restored.close()

            cancelled = threading.Event()
            cancelled.set()
            with pytest.raises(GenerationCancelled):
                engine.generate_tokens("cancel", cancel_event=cancelled)
            with pytest.raises(GenerationCancelled):
                engine.generate_tokens("deadline", deadline=0.0)
            continued_ids: list[int] = []
            engine.continue_generate(
                max_tokens=1,
                token_callback=continued_ids.append,
                temperature=0.0,
                greedy=True,
            )
            assert len(continued_ids) == 1
            cancelled_continuation = threading.Event()
            cancelled_continuation.set()
            with pytest.raises(GenerationCancelled):
                engine.continue_generate(
                    max_tokens=2,
                    cancel_event=cancelled_continuation,
                )
        finally:
            engine.close()

    assert outputs["resident"] == outputs["streaming"]
    assert followups["resident"] == followups["streaming"]


@pytest.mark.integration
def test_public_sequence_batch_capabilities(
    sequence_packs: dict[str, Path],
) -> None:
    mamba = _public_engine(sequence_packs["mamba"], "mamba2", "streaming")
    try:
        rows = mamba.generate_batch(["a", "b"], max_tokens=2)
        assert len(rows) == 2
        assert mamba.metrics.batch_size == 2
        assert mamba.metrics.tokens_generated == 4
        assert mamba.metrics.batch_prefill_wall_s > 0.0
        assert mamba.metrics.batch_decode_wall_s > 0.0
        assert mamba.supports_capability("batch") is True
    finally:
        mamba.close()

    transformer = _public_engine(
        sequence_packs["transformer"], "transformer", "streaming"
    )
    try:
        assert transformer.supports_capability("batch") is False
        with pytest.raises(Exception, match="weight-stationary batching"):
            transformer.generate_batch(["a", "b"], max_tokens=1)
    finally:
        transformer.close()


@pytest.mark.integration
def test_public_sequence_prefix_cache_reuse(
    sequence_packs: dict[str, Path],
) -> None:
    engine = _public_engine(
        sequence_packs["mamba"],
        "mamba2",
        "streaming",
        max_tokens=2,
        state_cache=True,
        system_prefix="system:",
    )
    try:
        engine.generate_tokens("first")
        engine.generate_tokens("second")
        assert engine.prefix_cache is not None
        assert engine.prefix_cache.stats.hits >= 1
    finally:
        engine.close()
