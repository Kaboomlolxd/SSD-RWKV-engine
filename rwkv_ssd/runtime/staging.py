"""Pinned host staging + device transfer ring.

The original implementation called this a ping-pong buffer, but it always
waited for the copy stream immediately after submitting a transfer.  The
implementation below keeps the old API while adding explicit event ownership
to a three-slot ring.  CUDA callers can submit work and rotate slots without
reusing a device allocation before its copy has completed; CPU callers retain
the synchronous, allocation-free behaviour.
"""

from __future__ import annotations

import torch


class _TransferSlot:
    __slots__ = ("host", "device", "event", "busy")

    def __init__(self, host: torch.Tensor, device: torch.Tensor, event: object | None) -> None:
        self.host = host
        self.device = device
        self.event = event
        self.busy = False


class PingPongStaging:
    """Pinned host buffers and an event-owned device transfer ring.

    ``slots`` defaults to three.  Three slots are enough to cover a read,
    decode, and in-flight H2D copy; callers may set two for compatibility with
    the historical ping-pong footprint.  ``active_host``/``active_dev``,
    ``h2d_async``, ``sync_copy_stream`` and ``swap`` remain source compatible.
    New code can use :meth:`acquire` and :meth:`release` to make ownership
    explicit.
    """

    def __init__(
        self,
        byte_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        slots: int = 3,
    ) -> None:
        if slots < 2:
            raise ValueError("staging ring requires at least two slots")
        self.byte_size = byte_size
        self.device = device
        self.dtype = dtype
        numel = byte_size // torch.tensor([], dtype=dtype).element_size()
        pin = device.type == "cuda"

        self._host = [torch.empty(numel, dtype=dtype, pin_memory=pin) for _ in range(slots)]
        self._dev = [torch.empty(numel, dtype=dtype, device=device) for _ in range(slots)]
        # Keep the old public names for callers that inspect the buffers.
        self.host_a, self.host_b = self._host[:2]
        self.dev_a, self.dev_b = self._dev[:2]
        self._idx = 0

        self.copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._slots = [
            _TransferSlot(
                host,
                dev,
                torch.cuda.Event(blocking=False) if self.copy_stream is not None else None,
            )
            for host, dev in zip(self._host, self._dev)
        ]

    @property
    def slots(self) -> int:
        return len(self._slots)

    def _wait_slot(self, idx: int) -> None:
        slot = self._slots[idx]
        if slot.busy and slot.event is not None:
            slot.event.synchronize()
            slot.busy = False

    def acquire(self, idx: int | None = None) -> int:
        """Acquire a reusable slot, waiting only if its prior copy is live."""
        if idx is None:
            idx = self._idx
        idx %= len(self._slots)
        self._wait_slot(idx)
        self._idx = idx
        return idx

    def release(self, idx: int | None = None) -> None:
        """Mark a slot reusable after its event has completed."""
        self._wait_slot(self._idx if idx is None else idx % len(self._slots))

    @property
    def active_host(self) -> torch.Tensor:
        return self._host[self._idx]

    @property
    def active_dev(self) -> torch.Tensor:
        return self._dev[self._idx]

    def swap(self) -> None:
        self._idx = (self._idx + 1) % len(self._slots)
        # Do not let a producer overwrite a slot whose copy is still in flight.
        self._wait_slot(self._idx)

    def h2d_async(self, host_tensor: torch.Tensor, dev_tensor: torch.Tensor) -> None:
        slot = self._slots[self._idx]
        self._wait_slot(self._idx)
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                dev_tensor.copy_(host_tensor, non_blocking=True)
                if slot.event is not None:
                    slot.event.record(self.copy_stream)
                    slot.busy = True
        else:
            dev_tensor.copy_(host_tensor)

    def sync_copy_stream(self) -> None:
        if self.copy_stream is not None:
            # Synchronizing the active event is narrower than synchronizing
            # unrelated work submitted to the copy stream.
            self._wait_slot(self._idx)

    def close(self) -> None:
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        for slot in self._slots:
            slot.busy = False
