"""Portable, mockable coverage for process RSS telemetry."""

from __future__ import annotations

from types import SimpleNamespace

from rwkv_ssd.runtime import memory_usage


def test_process_rss_dispatches_to_windows_reader(monkeypatch) -> None:
    monkeypatch.setattr(memory_usage, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        memory_usage, "_process_rss_windows_bytes", lambda: 123456789
    )
    assert memory_usage.process_rss_bytes() == 123456789


def test_process_rss_dispatches_to_posix_reader(monkeypatch) -> None:
    monkeypatch.setattr(memory_usage, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(memory_usage, "_process_rss_posix_bytes", lambda: 987654321)
    assert memory_usage.process_rss_bytes() == 987654321


def test_windows_counter_uses_pointer_width_for_working_set() -> None:
    fields = dict(memory_usage._ProcessMemoryCounters._fields_)
    assert fields["working_set"] is memory_usage.ctypes.c_size_t
    assert fields["peak_working_set"] is memory_usage.ctypes.c_size_t
