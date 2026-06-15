param(
    [string]$Config = "configs\needle_select_project.toml",
    [string]$Steps = ""
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Here
try {
    $Tool = Get-Command needle-select -ErrorAction SilentlyContinue
    $ArgsList = @("plan", "--config", $Config)
    if ($Steps) { $ArgsList += @("--steps", $Steps) }
    if ($Tool) {
        & needle-select @ArgsList
    } else {
        & python -m needle_select.cli @ArgsList
    }
} finally {
    Pop-Location
}
