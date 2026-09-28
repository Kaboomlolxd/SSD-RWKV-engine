param(
    [string]$Config = (Join-Path $PSScriptRoot "rwkv-ssd.yaml"),
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    $example = Join-Path $PSScriptRoot "rwkv-ssd.example.yaml"
    throw "Config not found: $Config`nCopy '$example' to 'rwkv-ssd.yaml' and set pack_dir, checkpoint_path, and backend."
}

Push-Location $PSScriptRoot
try {
    $arguments = @("-m", "app.serve", "--config", $Config)
    if (-not $NoBrowser) { $arguments += "--open-browser" }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
