from rwkv_ssd.runtime.manifest import Manifest, TensorEntry

__all__ = [
    "Manifest",
    "TensorEntry",
    "ParityStep",
    "ParityThresholds",
    "ParityTrace",
]


def __getattr__(name: str):
    if name in ("InferenceEngine", "RuntimeConfig"):
        from rwkv_ssd.runtime.engine import InferenceEngine, RuntimeConfig

        return InferenceEngine if name == "InferenceEngine" else RuntimeConfig
    if name in ("ParityStep", "ParityThresholds", "ParityTrace"):
        from rwkv_ssd.runtime.parity import ParityStep, ParityThresholds, ParityTrace

        return {
            "ParityStep": ParityStep,
            "ParityThresholds": ParityThresholds,
            "ParityTrace": ParityTrace,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
