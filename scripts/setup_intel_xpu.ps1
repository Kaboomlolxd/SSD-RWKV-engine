# Intel iGPU (XPU) Torch environment for RWKV-7 + Trinity decode.
# This script installs Python packages only; it never installs or updates drivers.
# Requires: Windows 11, a working Intel GPU driver, Python 3.11 or 3.12 (not 3.14).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venv = Join-Path $Root ".venv-xpu"

# Windows DLL search paths for the bundled Intel oneAPI and XPU Torch builds.
# Keep these in the setup shell so probes and subsequent benchmark commands
# use the same runtime that is installed in the venv.
$xpuDllDirs = @(
    (Join-Path $Venv "Library\bin"),
    (Join-Path $Venv "Scripts"),
    (Join-Path $Venv "Lib\site-packages\torch\lib")
) | Where-Object { Test-Path $_ }
if ($xpuDllDirs.Count -gt 0) {
    $env:Path = (($xpuDllDirs -join ";") + ";" + $env:Path)
}

# Broken PYTHONHOME in project dir breaks non-3.14 interpreters.
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$PyExe = $null
$uvPy = Join-Path $env:USERPROFILE ".local\share\uv\python"
if (-not (Test-Path $uvPy)) {
    $uvPy = Join-Path $env:APPDATA "..\Local\uv\python" -ErrorAction SilentlyContinue
}
$uvCandidates = @()
if (Test-Path "$env:APPDATA\..\Roaming\uv\python") {
    $uvRoot = (Resolve-Path "$env:APPDATA\..\Roaming\uv\python").Path
    $uvCandidates = Get-ChildItem -Path $uvRoot -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
}
$candidates = @($uvCandidates) + @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
foreach ($cand in $candidates) {
    if (-not $cand -or -not (Test-Path $cand)) { continue }
    try {
        $ver = & $cand -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($ver -match "^3\.(11|12)$") { $PyExe = $cand; break }
    } catch {}
}
if (-not $PyExe) {
    Write-Error "Need Python 3.11 or 3.12. Install from python.org, then re-run."
}

Write-Host "Using: $PyExe"
if (Test-Path $Venv) {
    Write-Host "venv exists: $Venv"
} else {
    & $PyExe -m venv $Venv
}

$pip = Join-Path $Venv "Scripts\pip.exe"
$python = Join-Path $Venv "Scripts\python.exe"

& $pip install -U pip
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
& $pip install -e $Root

Write-Host ""
Write-Host "Probe:"
& $python -c "from rwkv_ssd.runtime.trinity_accel import probe_intel_xpu; import json; print(json.dumps(probe_intel_xpu(), indent=2))"

Write-Host ""
Write-Host "Activate:  $($Venv)\Scripts\Activate.ps1"
Write-Host "Bench:     `$env:RWKV_TRINITY_DECODE_DEVICE='xpu'"
Write-Host "           python archive/bench/bench_trinity_decode.py --probe-xpu --device xpu --pack test_model/trinity_eval/trinity_layer"
