"""Bounded reusable byte slots for native layer-weight uploads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlotLease:
    slot_id: int
    layer_id: int
    generation: int
    buffer: bytearray
    used_bytes: int = 0

    def reset(self) -> None:
        self.used_bytes = 0

    def write(self, payload: bytes) -> tuple[int, int]:
        start = self.used_bytes
        end = start + len(payload)
        if end > len(self.buffer):
            raise MemoryError(
                f"layer {self.layer_id} needs {end} slot bytes, capacity is {len(self.buffer)}"
            )
        self.buffer[start:end] = payload
        self.used_bytes = end
        return start, len(payload)


class LayerSlotArena:
    """LRU mapping from logical layers to a fixed number of upload slots."""

    def __init__(self, slot_count: int, slot_bytes: int) -> None:
        if slot_count <= 0 or slot_bytes <= 0:
            raise ValueError("slot_count and slot_bytes must be positive")
        self.slot_count = int(slot_count)
        self.slot_bytes = int(slot_bytes)
        self._leases = [
            SlotLease(index, -1, 0, bytearray(self.slot_bytes))
            for index in range(self.slot_count)
        ]
        self._layer_to_slot: dict[int, int] = {}
        self._lru: list[int] = []
        self.evictions = 0

    @property
    def allocated_bytes(self) -> int:
        return self.slot_count * self.slot_bytes

    @property
    def leases(self) -> tuple[SlotLease, ...]:
        return tuple(self._leases)

    def acquire(self, layer_id: int) -> tuple[SlotLease, int | None]:
        layer_id = int(layer_id)
        existing = self._layer_to_slot.get(layer_id)
        if existing is not None:
            if existing in self._lru:
                self._lru.remove(existing)
            self._lru.append(existing)
            lease = self._leases[existing]
            lease.reset()
            return lease, None
        free = next((lease for lease in self._leases if lease.layer_id < 0), None)
        evicted_layer: int | None = None
        if free is None:
            slot_id = self._lru.pop(0)
            free = self._leases[slot_id]
            evicted_layer = free.layer_id
            self._layer_to_slot.pop(evicted_layer, None)
            self.evictions += 1
        free.layer_id = layer_id
        free.generation += 1
        free.reset()
        self._layer_to_slot[layer_id] = free.slot_id
        self._lru.append(free.slot_id)
        return free, evicted_layer

    def invalidate(self, layer_id: int | None = None) -> None:
        if layer_id is None:
            self._layer_to_slot.clear()
            self._lru.clear()
            for lease in self._leases:
                lease.layer_id = -1
                lease.reset()
            return
        slot_id = self._layer_to_slot.pop(int(layer_id), None)
        if slot_id is None:
            return
        if slot_id in self._lru:
            self._lru.remove(slot_id)
        self._leases[slot_id].layer_id = -1
        self._leases[slot_id].reset()


__all__ = ["LayerSlotArena", "SlotLease"]
