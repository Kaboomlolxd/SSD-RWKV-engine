"""Structured errors for the inference engine."""


class EngineError(Exception):
    """Base engine error."""


class PackError(EngineError):
    """Invalid or incompatible runtime pack."""


class ManifestVersionError(PackError):
    """manifest.json version is not supported."""


class BackendNotAvailableError(EngineError):
    """Requested backend or dependency is missing."""


class StreamingNotSupportedError(EngineError):
    """Backend cannot run true weight streaming yet."""


class CapabilityNotSupportedError(StreamingNotSupportedError):
    """The selected backend does not implement a requested engine operation."""

    def __init__(self, capability: str, backend: str, detail: str | None = None) -> None:
        self.capability = str(capability)
        self.backend = str(backend)
        message = (
            f"backend {self.backend!r} does not support capability "
            f"{self.capability!r}"
        )
        if detail:
            message += f": {detail}"
        super().__init__(message)
