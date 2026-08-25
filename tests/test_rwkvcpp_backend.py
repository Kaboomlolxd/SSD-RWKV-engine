"""Optional rwkv.cpp backend wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pytest

from rwkv_ssd.backends.capabilities import backend_capabilities, supports_true_streaming
from rwkv_ssd.backends.factory import ensure_v0_backend, supports_streaming_mode
from rwkv_ssd.backends.rwkvcpp import (
    RWKVCppBackend,
    _bridge_slot_config,
    _create_weight_bridge,
    _resolve_thread_count,
    _sync_provider_every_token,
    find_rwkvcpp_root,
)
from rwkv_ssd.runtime.errors import BackendNotAvailableError
from rwkv_ssd.runtime.state_cache import RecurrentState


def test_rwkvcpp_streaming_mode() -> None:
    assert supports_streaming_mode("rwkvcpp") is True


def test_rwkvcpp_capabilities() -> None:
    caps = backend_capabilities(RWKVCppBackend())
    assert caps.supports_resident is True
    assert caps.supports_pack_streaming is True
    assert caps.supports_skeleton is True
    assert caps.requires_ggml_bin is True
    assert supports_true_streaming(RWKVCppBackend()) is True


def test_rwkvcpp_thread_policy_is_conservative_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_CPU_THREADS", raising=False)
    assert _resolve_thread_count() == 1
    monkeypatch.setenv("RWKV_CPU_THREADS", "auto")
    assert _resolve_thread_count() == 1
    monkeypatch.setenv("RWKV_CPU_THREADS", "8")
    assert _resolve_thread_count() == 8


def test_rwkvcpp_thread_policy_uses_model_width_for_auto(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_CPU_THREADS", raising=False)
    monkeypatch.setattr("rwkv_ssd.backends.rwkvcpp.os.cpu_count", lambda: 16)
    assert _resolve_thread_count(768) == 1
    assert _resolve_thread_count(1536) == 4
    assert _resolve_thread_count(2560) == 4

    monkeypatch.setenv("RWKV_CPU_THREADS", "auto")
    assert _resolve_thread_count(2560) == 4

    monkeypatch.setattr("rwkv_ssd.backends.rwkvcpp.os.cpu_count", lambda: 8)
    assert _resolve_thread_count(2560) == 2


def test_rwkvcpp_explicit_thread_count_overrides_model_width(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_CPU_THREADS", "3")
    assert _resolve_thread_count(2560) == 3


def test_rwkvcpp_backend_accepts_manifest_width_hint() -> None:
    backend = RWKVCppBackend()
    backend.set_model_width_hint(2560)
    assert backend._n_embd_hint == 2560
    backend.set_model_width_hint(None)
    assert backend._n_embd_hint is None


def test_rwkvcpp_slot_config_is_explicit_and_paired(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_GGML_SLOT_COUNT", raising=False)
    monkeypatch.delenv("RWKV_GGML_SLOT_BYTES", raising=False)
    assert _bridge_slot_config() == (0, 0)
    monkeypatch.setenv("RWKV_GGML_SLOT_COUNT", "2")
    assert _bridge_slot_config() == (0, 0)
    monkeypatch.setenv("RWKV_GGML_SLOT_BYTES", "4096")
    assert _bridge_slot_config() == (2, 4096)


def test_rwkvcpp_provider_sync_is_weight_stationary_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RWKVCPP_SYNC_EVERY_TOKEN", raising=False)
    assert _sync_provider_every_token() is False
    monkeypatch.setenv("RWKVCPP_SYNC_EVERY_TOKEN", "1")
    assert _sync_provider_every_token() is True


def _mock_streaming_backend() -> RWKVCppBackend:
    backend = RWKVCppBackend()
    backend._model = MagicMock()
    backend._model.eval_sequence_in_chunks.return_value = (
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    backend._model.eval.return_value = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    backend._encode_fn = lambda text: [1] if text else []
    return backend


def test_rwkvcpp_streaming_prefix_miss_does_not_resync(monkeypatch) -> None:
    monkeypatch.delenv("RWKVCPP_SYNC_EVERY_TOKEN", raising=False)
    backend = _mock_streaming_backend()
    backend._sync_provider_layers = MagicMock()
    from rwkv_ssd.runtime.state_cache import PrefixStateCache

    backend.generate_greedy_pack_streaming(
        "user",
        1,
        MagicMock(),
        {},
        [0],
        system_prefix="system:",
        prefix_cache=PrefixStateCache(),
    )

    assert backend._sync_provider_layers.call_count == 1


def test_rwkvcpp_state_decode_syncs_once_by_default_and_honors_strict_mode(monkeypatch) -> None:
    backend = _mock_streaming_backend()
    backend._sync_provider_layers = MagicMock()
    state = RecurrentState(last_token_id=7, external_state=np.zeros(4, dtype=np.float32))

    monkeypatch.delenv("RWKVCPP_SYNC_EVERY_TOKEN", raising=False)
    backend.generate_greedy_from_state(state, 3, MagicMock(), {}, [0])
    assert backend._sync_provider_layers.call_count == 1

    backend._sync_provider_layers.reset_mock()
    monkeypatch.setenv("RWKVCPP_SYNC_EVERY_TOKEN", "1")
    backend.generate_greedy_from_state(state, 3, MagicMock(), {}, [0])
    assert backend._sync_provider_layers.call_count == 4

    backend._sync_provider_layers.reset_mock()
    monkeypatch.delenv("RWKVCPP_SYNC_EVERY_TOKEN", raising=False)
    backend.generate_greedy_from_state(
        state,
        3,
        MagicMock(),
        {},
        [0],
        layers_already_synced=True,
    )
    assert backend._sync_provider_layers.call_count == 0


def test_rwkvcpp_bridge_creation_forwards_slot_config(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_GGML_SLOT_COUNT", "3")
    monkeypatch.setenv("RWKV_GGML_SLOT_BYTES", "8192")
    model = object()
    with patch("rwkv_ssd.runtime.ggml_weight_bridge.GgmlWeightBridge") as bridge:
        _create_weight_bridge(model)
    bridge.assert_called_once_with(model, slot_count=3, slot_bytes=8192)


@pytest.mark.integration
def test_real_rwkvcpp_native_tensor_slot_abi() -> None:
    root = find_rwkvcpp_root()
    if root is None:
        pytest.skip("rwkv.cpp source unavailable")
    from rwkv_ssd.backends.rwkvcpp import find_rwkvcpp_dll

    dll = find_rwkvcpp_dll(root)
    model_path = Path("test_model/rwkv7-g1d-0.01b-bench-FP16.bin")
    if dll is None or not model_path.is_file():
        pytest.skip("built rwkv.cpp DLL or 0.01B GGML model unavailable")
    import sys
    sys.path.insert(0, str(root / "python"))
    from rwkv_cpp import rwkv_cpp_model, rwkv_cpp_shared_library

    library = rwkv_cpp_shared_library.RWKVSharedLibrary(str(dll))
    model = rwkv_cpp_model.RWKVModel(library, str(model_path), thread_count=1)
    try:
        assert model.supports_tensor_slot_upload
        assert model.supports_tensor_type
        size = model.tensor_nbytes("blocks.0.ln1.weight")
        assert model.tensor_type("emb.weight") == 1  # GGML_TYPE_F16
        slot = bytearray(size)
        model.register_tensor_slot(0, slot)
        model.set_tensor_from_slot("blocks.0.ln1.weight", 0, 0, size, 1)
    finally:
        model.free()


def test_ensure_v0_backend_rwkvcpp_without_dll() -> None:
    with patch("rwkv_ssd.backends.factory.is_rwkvcpp_available", return_value=False):
        with pytest.raises(BackendNotAvailableError, match="rwkv.cpp"):
            ensure_v0_backend("rwkvcpp")


def test_ensure_v0_backend_rwkvcpp_with_dll() -> None:
    with patch("rwkv_ssd.backends.factory.is_rwkvcpp_available", return_value=True):
        ensure_v0_backend("rwkvcpp")


def test_rwkvcpp_generate_simple_mocked() -> None:
    backend = RWKVCppBackend()
    backend._model = MagicMock()
    backend._encode_fn = lambda s: [1, 2]
    backend._decode_fn = lambda ids: "".join(chr(i) for i in ids)
    backend._model.eval_sequence_in_chunks.return_value = (
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    backend._model.eval.return_value = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    text = backend.generate_simple("hi", 2)
    assert text == "\x01\x02"
    assert backend._last_token_id == 2


def test_rwkvcpp_decode_rejects_invalid_token_with_actionable_error() -> None:
    backend = RWKVCppBackend()
    backend._decode_fn = lambda ids: {1: b"ok"}[ids[0]].decode("utf-8")
    with pytest.raises(RuntimeError, match="token_id=0"):
        backend.decode_text([0])


def test_rwkvcpp_state_snapshot() -> None:
    backend = RWKVCppBackend()
    arr = np.zeros(8, dtype=np.float32)
    backend._state = arr
    backend._last_token_id = 42
    snap = backend.get_recurrent_state()
    assert snap is not None
    assert snap.last_token_id == 42
    assert snap.external_state is not None

    backend2 = RWKVCppBackend()
    backend2.set_recurrent_state(snap)
    assert backend2._last_token_id == 42
    assert backend2._state is not None


@pytest.mark.skipif(not RWKVCppBackend.is_available(), reason="rwkv.cpp DLL not built")
def test_rwkvcpp_smoke_generate() -> None:
    from pathlib import Path

    bin_path = Path("test_model/rwkv7-g1d-0.01b-bench-FP16.bin")
    if not bin_path.is_file():
        pytest.skip("converted ggml bin missing")
    backend = RWKVCppBackend()
    backend.load(str(bin_path), "cpu", "cpu")
    ids = backend.generate_greedy_native("Hello", 4)
    assert len(ids) == 4
    assert all(isinstance(i, int) for i in ids)
    backend.close()


@pytest.mark.skipif(find_rwkvcpp_root() is None, reason="rwkv.cpp repo not cloned")
def test_rwkvcpp_root_detected() -> None:
    root = find_rwkvcpp_root()
    assert root is not None
    assert (root / "CMakeLists.txt").is_file()
