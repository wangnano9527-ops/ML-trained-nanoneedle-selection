param(
    [string]$Config = "configs\needle_select_project.toml"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Here
try {
    $Tool = Get-Command needle-select -ErrorAction SilentlyContinue
    if ($Tool) {
        & needle-select doctor --config $Config
    } else {
        & python -m needle_select.cli doctor --config $Config
    }
} finally {
    Pop-Location
}
