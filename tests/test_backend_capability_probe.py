from rwkv_ssd.backends.capabilities import probe_backend_capabilities
from rwkv_ssd.backends.capabilities import capability_supported
from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend
from rwkv_ssd.backends.synthetic import SyntheticBackend
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


def test_runtime_capability_probe_distinguishes_declared_and_observed() -> None:
    backend = SyntheticBackend()
    report = probe_backend_capabilities(backend)
    assert report["declared"]["pack_streaming"] is True
    assert report["runtime"]["loaded"] is False
    assert report["runtime"]["state_get"] is True
    assert report["runtime"]["batch_generate"] is True
    assert report["runtime"]["tensor_slot_upload"] is False


def test_backend_and_engine_expose_common_capability_queries(tmp_path) -> None:
    backend = SyntheticBackend()
    assert backend.supports_capability("batching") is True
    engine = InferenceEngine(EngineConfig(pack_dir=tmp_path, backend="synthetic"))
    report = engine.capabilities()
    assert report["config"]["backend"] == "synthetic"
    assert engine.supports_capability("token_generation") is True


def test_native_layer_capability_is_runtime_queryable() -> None:
    backend = RWKVCppBackend()
    assert capability_supported(backend, "native_layer_streaming") is False
    backend._model = type("Model", (), {"supports_layer_streaming": True})()
    assert capability_supported(backend, "native_layer_streaming") is True


def test_native_cached_layer_capability_requires_complete_runtime_abi() -> None:
    backend = RWKVCppBackend()
    backend._model = type(
        "Model",
        (),
        {
            "supports_layer_cached_step": True,
            "layer_step_cached": lambda self, *args: None,
            "layer_cache_ready": lambda self: True,
        },
    )()

    assert capability_supported(backend, "native_cached_layer_step") is True
    assert capability_supported(backend, "cached_token_step") is True

    # A boolean flag without both callable entry points is not enough to
    # advertise the optimization to the scheduler.
    backend._model = type("IncompleteModel", (), {"supports_layer_cached_step": True})()
    assert capability_supported(backend, "native_cached_layer_step") is False
