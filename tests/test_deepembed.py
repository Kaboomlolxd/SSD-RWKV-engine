from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
from rwkv_ssd.runtime.deepembed import (
    DeepEmbedReferenceModel,
    DeepEmbedSidecar,
    detect_deepembed_variant,
    infer_deepembed_meta,
    is_deepembed_checkpoint,
    is_rwkv7a_deepembed_checkpoint,
    write_deepembed_sidecar,
)
from rwkv_ssd.runtime.metrics import MetricsCollector
from rwkv_ssd.tools.pack_runtime import detect_model_family_from_state, pack


def _rwkv7a_v1_state() -> dict[str, torch.Tensor]:
    return {
        "emb.weight": torch.zeros(8, 4),
        "blocks.0.ffn.s_emb.weight": torch.zeros(8, 1024),
        "blocks.0.ffn.s_emb_x.weight": torch.zeros(1024, 4),
        "blocks.0.att.x_r": torch.zeros(4),
    }


def _deepembed_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    vocab, emb, projected = 9, 4, 3
    return {
        "emb.weight": torch.randn(vocab, emb),
        "blocks.0.ln0.weight": torch.ones(emb),
        "blocks.0.ln0.bias": torch.zeros(emb),
        "blocks.0.ffn.s_emb.weight": torch.randn(vocab, projected),
        "blocks.0.ffn.s_emb_x.weight": torch.randn(projected, emb),
        "blocks.0.qkv.k_emb.weight": torch.randn(vocab, projected),
        "blocks.0.qkv.k_emb_x.weight": torch.randn(projected, emb),
        "blocks.0.qkv.v_emb.weight": torch.randn(vocab, projected),
        "blocks.0.qkv.v_emb_x.weight": torch.randn(projected, emb),
        "blocks.0.qkv.qq.weight": torch.randn(emb, projected),
        "blocks.0.qkv.k1": torch.randn(emb, projected),
        "blocks.0.qkv.v1": torch.randn(emb, projected),
    }


def _complete_deepembed_state() -> dict[str, torch.Tensor]:
    """Small complete qkv/DEA state shared by resident and batch tests."""
    torch.manual_seed(11)
    vocab, width, hidden = 16, 8, 4
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


def test_deepembed_is_detected_as_a_distinct_family() -> None:
    state = _deepembed_state()
    assert is_deepembed_checkpoint(state)
    meta = infer_deepembed_meta(state)
    assert meta["deepembed"] is True
    assert meta["deepembed_variant"] == "qkv_dea"
    assert meta["deepembed_sidecar_required"] is True
    assert meta["deepembed_streaming_supported"] is True
    assert detect_model_family_from_state(state) == "rwkv7_deepembed"


def test_rwkv7a_deepembed_v1_is_distinct_from_qkv_dea() -> None:
    state = _rwkv7a_v1_state()
    assert is_rwkv7a_deepembed_checkpoint(state)
    assert detect_deepembed_variant(state) == "rwkv7a_v1"
    meta = infer_deepembed_meta(state)
    assert meta["deepembed_variant"] == "rwkv7a_v1"
    assert meta["deepembed_streaming_supported"] is True
    assert meta["deepembed_sidecar_required"] is False
    assert detect_model_family_from_state(state) == "rwkv7_deepembed"


def test_deepembed_sidecar_round_trip(tmp_path) -> None:
    state = _deepembed_state()
    path = tmp_path / "DeepEmbed.bin"
    info = write_deepembed_sidecar(state, path)
    assert info["tensor_count"] == 3
    with DeepEmbedSidecar(path) as sidecar:
        rows = sidecar.lookup("k_emb", 0, [1, 7])
        norm_emb = torch.nn.functional.layer_norm(
            state["emb.weight"],
            (4,),
            weight=state["blocks.0.ln0.weight"],
            bias=state["blocks.0.ln0.bias"],
        )
        expected = state["blocks.0.qkv.k_emb.weight"] + norm_emb @ state[
            "blocks.0.qkv.k_emb_x.weight"
        ].t()
        assert torch.allclose(rows, expected[[1, 7]])
        repeated = sidecar.lookup("k_emb", 0, [7, 1, 7])
        assert torch.allclose(repeated, expected[[7, 1, 7]])
        with pytest.raises(IndexError):
            sidecar.lookup("k_emb", 0, [9])


def test_pack_runtime_records_deepembed_sidecar(tmp_path) -> None:
    checkpoint = tmp_path / "deepembed.pth"
    state = _deepembed_state()
    torch.save(state, checkpoint)
    output = tmp_path / "pack"
    pack(checkpoint, output, quiet=True)
    manifest = __import__("json").loads((output / "manifest.json").read_text())
    assert manifest["model_family"] == "rwkv7_deepembed"
    assert manifest["meta"]["deepembed"] is True
    assert manifest["meta"]["deepembed_sidecar"]["path"] == "DeepEmbed.bin"
    assert (output / "DeepEmbed.bin").is_file()


def test_deepembed_reference_model_smoke(tmp_path) -> None:
    state = _complete_deepembed_state()
    vocab = int(state["emb.weight"].shape[0])
    model = DeepEmbedReferenceModel(state)
    logits, next_state = model.forward([1, 2, 3], model.generate_zero_state(), full_output=True)
    assert list(logits.shape) == [3, vocab]
    assert len(next_state) == 7
    assert torch.isfinite(logits).all()

    sidecar_path = tmp_path / "DeepEmbed.bin"
    write_deepembed_sidecar(state, sidecar_path)
    with DeepEmbedSidecar(sidecar_path) as sidecar:
        sidecar_model = DeepEmbedReferenceModel(state, sidecar)
        sidecar_logits, _ = sidecar_model.forward(
            [1, 2, 3], sidecar_model.generate_zero_state(), full_output=True
        )
    assert torch.allclose(logits, sidecar_logits)


def test_deepembed_reference_layer_streaming_matches_resident(tmp_path) -> None:
    state = _deepembed_state()
    state.update(
        {
            "blocks.0.ln0.weight": torch.ones(4),
            "blocks.0.ln0.bias": torch.zeros(4),
            "ln_out.weight": torch.ones(4),
            "ln_out.bias": torch.zeros(4),
            "head.weight": torch.randn(4, 9),
            "blocks.0.ffn.s_emb.weight": torch.randn(9, 1024),
            "blocks.0.ffn.s_emb_x.weight": torch.randn(1024, 4),
            "blocks.0.qkv.k_emb.weight": torch.randn(9, 4),
            "blocks.0.qkv.k_emb_x.weight": torch.randn(4, 4),
            "blocks.0.qkv.v_emb.weight": torch.randn(9, 4),
            "blocks.0.qkv.v_emb_x.weight": torch.randn(4, 4),
            "blocks.0.qkv.qq.weight": torch.randn(4, 4),
            "blocks.0.qkv.k1": torch.randn(4, 3),
            "blocks.0.qkv.v1": torch.randn(4, 3),
            "blocks.0.qkv.k2": torch.randn(3, 4),
            "blocks.0.qkv.v2": torch.randn(3, 4),
            "blocks.0.qkv.x_q": torch.randn(4),
            "blocks.0.qkv.x_k": torch.randn(4),
            "blocks.0.qkv.x_v": torch.randn(4),
            "blocks.0.qkv.lnq.weight": torch.ones(4),
            "blocks.0.qkv.lnq.bias": torch.zeros(4),
            "blocks.0.qkv.lnk.weight": torch.ones(4),
            "blocks.0.qkv.lnk.bias": torch.zeros(4),
            "blocks.0.qkv.lnv.weight": torch.ones(4),
            "blocks.0.qkv.lnv.bias": torch.zeros(4),
        }
    )
    # Add the small but complete block tensors used by the reference equations.
    width = 4
    hidden = 2
    state.update(
        {
            "blocks.0.ln1.weight": torch.ones(width),
            "blocks.0.ln1.bias": torch.zeros(width),
            "blocks.0.ln2.weight": torch.ones(width),
            "blocks.0.ln2.bias": torch.zeros(width),
            "blocks.0.att.r_k": torch.randn(1, width),
            "blocks.0.att.x_r": torch.randn(width),
            "blocks.0.att.x_w": torch.randn(width),
            "blocks.0.att.x_k": torch.randn(width),
            "blocks.0.att.x_v": torch.randn(width),
            "blocks.0.att.x_a": torch.randn(width),
            "blocks.0.att.x_g": torch.randn(width),
            "blocks.0.att.w0": torch.randn(width),
            "blocks.0.att.w1": torch.randn(width, hidden),
            "blocks.0.att.w2": torch.randn(hidden, width),
            "blocks.0.att.k_k": torch.randn(width),
            "blocks.0.att.k_a": torch.randn(width),
            "blocks.0.att.a0": torch.randn(width),
            "blocks.0.att.a1": torch.randn(width, hidden),
            "blocks.0.att.a2": torch.randn(hidden, width),
            "blocks.0.att.v0": torch.randn(width),
            "blocks.0.att.v1": torch.randn(width, hidden),
            "blocks.0.att.v2": torch.randn(hidden, width),
            "blocks.0.att.g1": torch.randn(width, width),
            "blocks.0.att.g2": torch.randn(width, width),
            "blocks.0.att.receptance.weight": torch.randn(width, width),
            "blocks.0.att.key.weight": torch.randn(width, width),
            "blocks.0.att.value.weight": torch.randn(width, width),
            "blocks.0.att.output.weight": torch.randn(width, width),
            "blocks.0.att.ln_x.weight": torch.ones(width),
            "blocks.0.att.ln_x.bias": torch.zeros(width),
            "blocks.0.ffn.x_k": torch.randn(width),
            "blocks.0.ffn.s1": torch.randn(width, 32),
            "blocks.0.ffn.s2": torch.randn(32, width),
            "blocks.0.ffn.s0": torch.randn(width),
            "blocks.0.ffn.key.weight": torch.randn(width, width),
            "blocks.0.ffn.value.weight": torch.randn(width, width),
        }
    )
    # The qkv projection dimensions in _deepembed_state are already compatible
    # with the four-wide reference block; keep this test focused on I/O parity.
    sidecar_path = tmp_path / "DeepEmbed.bin"
    write_deepembed_sidecar(state, sidecar_path)

    class Provider:
        def load_layer_tensors(self, entries):
            return {entry.name: state[entry.name] for entry in entries}

    entries = {
        0: [
            SimpleNamespace(name=name)
            for name in state
            if name.startswith("blocks.0.")
        ]
    }
    ids = [1, 2, 3]
    with DeepEmbedSidecar(sidecar_path) as resident_sidecar, DeepEmbedSidecar(
        sidecar_path
    ) as streaming_sidecar:
        resident = DeepEmbedReferenceModel(state, resident_sidecar)
        streaming = DeepEmbedReferenceModel(
            state,
            streaming_sidecar,
            resident_layers=set(),
        )
        resident_logits, _ = resident.forward(
            ids, resident.generate_zero_state(), full_output=True
        )
        streamed_logits, _ = streaming.forward_streaming(
            ids,
            streaming.generate_zero_state(),
            Provider(),
            entries,
            full_output=True,
        )
    torch.testing.assert_close(streamed_logits, resident_logits, rtol=0, atol=0)


def test_deepembed_reference_batch_streaming_matches_independent(tmp_path) -> None:
    state = _complete_deepembed_state()
    sidecar_path = tmp_path / "DeepEmbed.bin"
    write_deepembed_sidecar(state, sidecar_path)

    class Provider:
        def __init__(self) -> None:
            self.loads = 0

        def load_layer_tensors(self, entries):
            self.loads += 1
            return {entry.name: state[entry.name] for entry in entries}

    entries = {
        0: [
            SimpleNamespace(name=name)
            for name in state
            if name.startswith("blocks.0.")
        ]
    }
    prompts = ([1], [3])
    decode_tokens = ([4, 5, 6], [7, 8, 9])
    with DeepEmbedSidecar(sidecar_path) as sidecar:
        independent = DeepEmbedReferenceModel(
            state, sidecar, resident_layers=set()
        )
        batched = DeepEmbedReferenceModel(
            state, sidecar, resident_layers=set()
        )
        independent_provider = Provider()
        batch_provider = Provider()
        independent_states = [
            independent.generate_zero_state() for _ in prompts
        ]
        batch_states = [batched.generate_zero_state() for _ in prompts]
        for prompt, state_i in zip(prompts, independent_states, strict=True):
            independent.forward_streaming(
                prompt, state_i, independent_provider, entries
            )
        for prompt, state_i in zip(prompts, batch_states, strict=True):
            batched.forward_streaming(prompt, state_i, batch_provider, entries)

        metrics = MetricsCollector()
        for position in range(len(decode_tokens[0])):
            step_logits = []
            for index, state_i in enumerate(independent_states):
                logits, _ = independent.forward_streaming(
                    [decode_tokens[index][position]],
                    state_i,
                    independent_provider,
                    entries,
                )
                step_logits.append(logits)
            batch_logits, batch_states = batched.forward_batch_streaming(
                [tokens[position] for tokens in decode_tokens],
                batch_states,
                batch_provider,
                entries,
                metrics=metrics,
            )
            for expected, actual in zip(
                step_logits, batch_logits, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Both approaches pay one independent layer load per prompt during
        # prefill.  Only the decode phase is shared by the batch path.
        assert batch_provider.loads == len(prompts) + len(decode_tokens[0])
        assert independent_provider.loads == len(prompts) * (1 + len(decode_tokens[0]))
        assert metrics.batch_size == len(prompts)
        assert metrics.weight_sweeps == len(decode_tokens[0])
        assert metrics.weight_layer_loads == len(decode_tokens[0])
        for independent_state, batch_state in zip(
            independent_states, batch_states, strict=True
        ):
            for expected, actual in zip(
                independent_state, batch_state, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepembed_reference_batch_prefill_streaming_matches_independent(
    tmp_path,
) -> None:
    """Variable-length qkv/DEA prompts share each layer without padding."""
    state = _complete_deepembed_state()
    sidecar_path = tmp_path / "DeepEmbed.bin"
    write_deepembed_sidecar(state, sidecar_path)

    class Provider:
        def __init__(self) -> None:
            self.loads = 0

        def load_layer_tensors(self, entries):
            self.loads += 1
            return {entry.name: state[entry.name] for entry in entries}

    entries = {
        0: [
            SimpleNamespace(name=name)
            for name in state
            if name.startswith("blocks.0.")
        ]
    }
    prompts = ([1, 2], [3, 4, 5])
    with DeepEmbedSidecar(sidecar_path) as sidecar:
        independent = DeepEmbedReferenceModel(
            state, sidecar, resident_layers=set()
        )
        batched = DeepEmbedReferenceModel(
            state, sidecar, resident_layers=set()
        )
        independent_provider = Provider()
        batch_provider = Provider()
        independent_states = [
            independent.generate_zero_state() for _ in prompts
        ]
        batch_states = [batched.generate_zero_state() for _ in prompts]

        expected_logits = []
        for prompt, state_i in zip(prompts, independent_states, strict=True):
            logits, _ = independent.forward_streaming(
                prompt, state_i, independent_provider, entries
            )
            expected_logits.append(logits)

        actual_logits, batch_states = batched.forward_batch_prefill_streaming(
            prompts,
            batch_states,
            batch_provider,
            entries,
        )
        for expected, actual in zip(expected_logits, actual_logits, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for independent_state, batch_state in zip(
            independent_states, batch_states, strict=True
        ):
            for expected, actual in zip(
                independent_state, batch_state, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        assert independent_provider.loads == len(prompts)
        assert batch_provider.loads == 1


def test_chatrwkv_deepembed_batch_generation_uses_prefill_logits(tmp_path) -> None:
    """The backend batch wrapper must not replay each prompt's final token."""
    state = _complete_deepembed_state()
    sidecar_path = tmp_path / "DeepEmbed.bin"
    write_deepembed_sidecar(state, sidecar_path)

    class Provider:
        def __init__(self) -> None:
            self.loads = 0

        def load_layer_tensors(self, entries):
            self.loads += 1
            return {entry.name: state[entry.name] for entry in entries}

    entries = {
        0: [
            SimpleNamespace(name=name)
            for name in state
            if name.startswith("blocks.0.")
        ]
    }
    prompt_ids = {"first": [1, 2], "second": [3]}

    class Pipeline:
        def encode(self, prompt):
            return prompt_ids[prompt]

    expected_model = DeepEmbedReferenceModel(state)
    expected = []
    for prompt in ("first", "second"):
        current_state = expected_model.generate_zero_state()
        logits, current_state = expected_model.forward(
            prompt_ids[prompt], current_state
        )
        generated = []
        for _ in range(3):
            token = int(logits.argmax().item())
            generated.append(token)
            logits, current_state = expected_model.forward(
                [token], current_state
            )
        expected.append(generated)

    with DeepEmbedSidecar(sidecar_path) as sidecar:
        backend = ChatRWKVBackend()
        backend._model = DeepEmbedReferenceModel(
            state, sidecar, resident_layers=set()
        )
        backend._pipeline = Pipeline()
        backend._rwkv7 = True
        backend._deepembed_streaming_reference = True
        metrics = MetricsCollector()
        provider = Provider()
        actual = backend.generate_greedy_batch_streaming(
            ["first", "second"],
            3,
            provider,
            entries,
            [0],
            metrics,
        )

        assert actual == expected
        # One shared prefill sweep plus two shared advances produces three
        # layer transactions for the three generated distributions.
        assert provider.loads == 3
        assert metrics.weight_sweeps == 3
        assert metrics.weight_layer_loads == 3
        assert metrics.tokens_generated == 6
