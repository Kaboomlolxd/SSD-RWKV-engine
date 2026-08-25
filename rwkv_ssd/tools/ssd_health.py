#!/usr/bin/env python3
"""Best-effort SSD / NVMe health and temperature snapshot.

Usage:
  python -m rwkv_ssd.tools.ssd_health
  python -m rwkv_ssd.tools.ssd_health --json
  python -m rwkv_ssd.tools.ssd_health --watch 5
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DriveHealth:
    name: str
    model: str = ""
    media_type: str = ""
    health_status: str = ""
    temperature_c: float | None = None
    percentage_used: float | None = None
    power_on_hours: int | None = None
    unsafe_shutdowns: int | None = None
    media_errors: int | None = None
    read_errors: int | None = None
    write_errors: int | None = None
    available_spare_pct: float | None = None
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], *, timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _parse_smartctl_nvme(text: str, drive: DriveHealth) -> None:
    for line in text.splitlines():
        low = line.lower()
        if "temperature" in low and ":" in line and drive.temperature_c is None:
            m = re.search(r"(\d+)\s*(?:celsius|c\b)", line, re.I)
            if m:
                drive.temperature_c = float(m.group(1))
        if "percentage used" in low:
            m = re.search(r":\s*(\d+)", line)
            if m:
                drive.percentage_used = float(m.group(1))
        if "power on hours" in low:
            m = re.search(r":\s*(\d+)", line)
            if m:
                drive.power_on_hours = int(m.group(1))
        if "unsafe shutdowns" in low:
            m = re.search(r":\s*(\d+)", line)
            if m:
                drive.unsafe_shutdowns = int(m.group(1))
        if "media errors" in low:
            m = re.search(r":\s*(\d+)", line)
            if m:
                drive.media_errors = int(m.group(1))
        if "available spare" in low and "%" in line:
            m = re.search(r":\s*(\d+)%", line)
            if m:
                drive.available_spare_pct = float(m.group(1))


def _collect_linux_sysfs() -> list[DriveHealth]:
    from pathlib import Path

    drives: list[DriveHealth] = []
    nvme_root = Path("/sys/class/nvme")
    if not nvme_root.is_dir():
        return drives
    for ctrl in sorted(nvme_root.glob("nvme*")):
        if not ctrl.is_dir() or ctrl.name.count("nvme") > 1:
            continue
        name = ctrl.name
        model = ""
        model_path = ctrl / "model"
        if model_path.is_file():
            model = model_path.read_text(encoding="utf-8", errors="replace").strip()
        temp_c: float | None = None
        for hwmon in ctrl.glob("device/hwmon/hwmon*"):
            for temp_file in sorted(hwmon.glob("temp*_input")):
                try:
                    milli = int(temp_file.read_text(encoding="utf-8").strip())
                    temp_c = milli / 1000.0
                    break
                except (OSError, ValueError):
                    continue
            if temp_c is not None:
                break
        drives.append(
            DriveHealth(
                name=name,
                model=model,
                media_type="NVMe",
                health_status="unknown",
                temperature_c=temp_c,
            )
        )
    return drives


def _collect_linux_smartctl() -> list[DriveHealth]:
    if not shutil.which("smartctl"):
        return []
    code, out, err = _run(["smartctl", "--scan", "-j"])
    if code != 0:
        return []
    try:
        scan = json.loads(out)
    except json.JSONDecodeError:
        return []
    drives: list[DriveHealth] = []
    for dev in scan.get("devices", []):
        path = dev.get("name") or dev.get("device", {}).get("name")
        if not path:
            continue
        code2, smart_out, _ = _run(["smartctl", "-a", "-j", str(path)])
        if code2 != 0:
            continue
        try:
            data = json.loads(smart_out)
        except json.JSONDecodeError:
            continue
        model = str(data.get("model_name") or "")
        name = str(path)
        dh = DriveHealth(name=name, model=model, media_type="NVMe")
        nvme = data.get("nvme_smart_health_information_log") or {}
        if nvme.get("temperature") is not None:
            dh.temperature_c = float(nvme["temperature"])
        if nvme.get("percentage_used") is not None:
            dh.percentage_used = float(nvme["percentage_used"])
        if nvme.get("power_on_hours") is not None:
            dh.power_on_hours = int(nvme["power_on_hours"])
        if nvme.get("unsafe_shutdowns") is not None:
            dh.unsafe_shutdowns = int(nvme["unsafe_shutdowns"])
        if nvme.get("media_errors") is not None:
            dh.media_errors = int(nvme["media_errors"])
        if nvme.get("available_spare") is not None:
            dh.available_spare_pct = float(nvme["available_spare"])
        drives.append(dh)
    if not drives and err:
        pass
    return drives


def _collect_windows() -> list[DriveHealth]:
    ps = r"""
$disks = Get-PhysicalDisk | Select-Object FriendlyName, Model, MediaType, HealthStatus, OperationalStatus, DeviceId
$rel = @()
try { $rel = Get-PhysicalDisk | Get-StorageReliabilityCounter -ErrorAction Stop } catch { }
$out = foreach ($d in $disks) {
  $r = $rel | Where-Object { $_.DeviceId -eq $d.DeviceId } | Select-Object -First 1
  [PSCustomObject]@{
    name = $d.FriendlyName
    model = $d.Model
    media_type = $d.MediaType
    health_status = $d.HealthStatus
    operational_status = $d.OperationalStatus
    temperature_c = if ($r) { $r.Temperature } else { $null }
    wear = if ($r) { $r.Wear } else { $null }
    read_errors = if ($r) { $r.ReadErrorsTotal } else { $null }
    write_errors = if ($r) { $r.WriteErrorsTotal } else { $null }
  }
}
$out | ConvertTo-Json -Compress
"""
    code, out, err = _run(
        ["powershell", "-NoProfile", "-Command", ps],
        timeout=30.0,
    )
    if code != 0 or not out.strip():
        return [
            DriveHealth(
                name="unknown",
                notes=[f"PowerShell query failed: {err.strip() or 'no output'}"],
            )
        ]
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return [DriveHealth(name="unknown", notes=["Failed to parse PowerShell JSON"])]
    if isinstance(raw, dict):
        raw = [raw]
    drives: list[DriveHealth] = []
    for item in raw:
        temp = item.get("temperature_c")
        drives.append(
            DriveHealth(
                name=str(item.get("name") or ""),
                model=str(item.get("model") or ""),
                media_type=str(item.get("media_type") or ""),
                health_status=str(item.get("health_status") or ""),
                temperature_c=float(temp) if temp not in (None, "", 0) else None,
                read_errors=int(item["read_errors"])
                if item.get("read_errors") is not None
                else None,
                write_errors=int(item["write_errors"])
                if item.get("write_errors") is not None
                else None,
                notes=(
                    ["Wear indicator: " + str(item["wear"])]
                    if item.get("wear") is not None
                    else []
                ),
            )
        )
    if drives and all(d.temperature_c is None for d in drives):
        drives[0].notes.append(
            "Temperature unavailable — run as Administrator or install vendor NVMe tool"
        )
    return drives


def collect_drive_health() -> list[DriveHealth]:
    system = platform.system().lower()
    if system == "windows":
        return _collect_windows()
    drives = _collect_linux_smartctl()
    if drives:
        return drives
    sysfs = _collect_linux_sysfs()
    if sysfs:
        return sysfs
    if shutil.which("smartctl"):
        code, out, _ = _run(["smartctl", "--scan"])
        drives = []
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            dev = parts[0]
            code2, smart_text, _ = _run(["smartctl", "-a", dev])
            if code2 != 0:
                continue
            dh = DriveHealth(name=dev, media_type="block")
            _parse_smartctl_nvme(smart_text, dh)
            drives.append(dh)
        return drives
    return [DriveHealth(name="unknown", notes=["No SSD telemetry source found on this OS"])]


def _thermal_assessment(temp_c: float | None) -> str:
    if temp_c is None:
        return "temperature unknown"
    if temp_c < 50:
        return "cool"
    if temp_c < 70:
        return "normal"
    if temp_c < 80:
        return "warm — monitor during long inference runs"
    if temp_c < 90:
        return "hot — NVMe likely throttling; add heatsink or lower RWKV_SSD_IO_CAP_MBPS"
    return "critical — stop sustained load and improve cooling"


def format_report(drives: list[DriveHealth]) -> str:
    lines: list[str] = []
    lines.append(f"SSD health snapshot ({platform.system()})")
    lines.append("")
    for i, d in enumerate(drives, 1):
        lines.append(f"[{i}] {d.name or 'drive'}")
        if d.model:
            lines.append(f"    model:         {d.model}")
        if d.media_type:
            lines.append(f"    media:         {d.media_type}")
        if d.health_status:
            lines.append(f"    health:        {d.health_status}")
        if d.temperature_c is not None:
            lines.append(
                f"    temperature:   {d.temperature_c:.1f} C ({_thermal_assessment(d.temperature_c)})"
            )
        else:
            lines.append(f"    temperature:   ({_thermal_assessment(None)})")
        if d.percentage_used is not None:
            lines.append(f"    used (life):   {d.percentage_used:.0f}%")
        if d.available_spare_pct is not None:
            lines.append(f"    spare:         {d.available_spare_pct:.0f}%")
        if d.power_on_hours is not None:
            lines.append(f"    power-on h:    {d.power_on_hours}")
        if d.media_errors is not None:
            lines.append(f"    media errors:  {d.media_errors}")
        if d.read_errors is not None:
            lines.append(f"    read errors:   {d.read_errors}")
        if d.write_errors is not None:
            lines.append(f"    write errors:  {d.write_errors}")
        for note in d.notes:
            lines.append(f"    note:          {note}")
        lines.append("")
    lines.append("Engine tips: docs/SSD_HEALTH.md")
    lines.append("  RWKV_SSD_HEALTH=1  — health-conscious streaming preset")
    lines.append("  RWKV_SSD_IO_CAP_MBPS=N  — cap read bandwidth for cooling")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="SSD / NVMe health and temperature snapshot")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="Poll repeatedly (interval in seconds)",
    )
    args = p.parse_args()

    def emit_once() -> list[DriveHealth]:
        drives = collect_drive_health()
        if args.json:
            payload: dict[str, Any] = {
                "platform": platform.system(),
                "drives": [asdict(d) for d in drives],
            }
            print(json.dumps(payload, indent=2))
        else:
            print(format_report(drives))
        return drives

    if args.watch and args.watch > 0:
        try:
            while True:
                if not args.json:
                    print("\033[2J\033[H", end="")
                emit_once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        emit_once()


if __name__ == "__main__":
    main()
