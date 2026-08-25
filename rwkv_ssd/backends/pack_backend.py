"""Pack-driven backend: weights come from runtime pack, not a separate checkpoint."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.runtime.generation_control import make_generation_control

if TYPE_CHECKING:
    from rwkv_ssd.runtime.manifest import Manifest
    from rwkv_ssd.runtime.metrics import MetricsCollector
    from rwkv_ssd.runtime.state_cache import RecurrentState
    from rwkv_ssd.runtime.weight_provider import WeightProvider


class PackBackend(RecurrentBackend):


    """Backend that decodes using tensors loaded from weights.bin."""

    supports_streaming: bool = True

    @abstractmethod
    def load_pack(
        self,
        manifest: Manifest,
        device: str,
    ) -> None:
        ...

    @abstractmethod
    def generate_greedy(
        self,
        prompt: str,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[int]:
        ...

    def prepare_provider(
        self,
        provider: WeightProvider,
        *,
        config: Any,
    ) -> None:
        """Prepare backend-owned global state from the shared pack provider.

        Most pack backends load their global tensors on first use.  Accelerator
        adapters may need to materialize them before the first token, and F5
        adapters may use this hook to warm every layer through the provider.
        The default is intentionally a no-op so existing pack backends keep
        their lazy behavior.
        """
        del provider, config

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
        raise NotImplementedError

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
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
        greedy: bool = True,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        """Common generation contract with sampling and cancellation support."""
        control = make_generation_control(
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if control is not None:
            control.check()
        state = self.prefill_text(
            prompt,
            provider,
            metrics,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if control is not None:
            control.check()
        return self.decode_greedy(
            state,
            provider,
            max_tokens,
            metrics,
            temperature=0.0 if greedy else float(temperature),
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_tokens(
        self,
        prompt: str,
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
        *,
        temperature: float = 0.0,
        greedy: bool = True,
        token_callback: Any | None = None,
        cancel_event: Any | None = None,
        deadline: float | None = None,
    ) -> list[int]:
        """Common token-ID spelling used by conformance and serving code."""
        return self.generate(
            prompt,
            provider,
            max_tokens,
            metrics,
            temperature=temperature,
            greedy=greedy,
            token_callback=token_callback,
            cancel_event=cancel_event,
            deadline=deadline,
        )

    def generate_greedy_batch(
        self,
        prompts: list[str],
        provider: WeightProvider,
        max_tokens: int,
        metrics: MetricsCollector,
    ) -> list[list[int]]:
        """Optional weight-stationary multi-session generation contract."""
        self.capability_error(
            "batching",
            f"{type(self).__name__} does not implement weight-stationary batching",
        )

    def generate_simple(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError("use InferenceEngine.generate with a WeightProvider")
