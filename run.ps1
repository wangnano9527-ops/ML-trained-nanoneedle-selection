param(
    [string]$Config = "configs\example.project.toml",
    [string]$Steps = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$Here\src;$env:PYTHONPATH"
$Python = if (Test-Path "$Here\.venv\Scripts\python.exe") { "$Here\.venv\Scripts\python.exe" } elseif (Test-Path "$Here\.venv\python.exe") { "$Here\.venv\python.exe" } else { "python" }
Push-Location $Here
try {
    $ArgsList = @("-m", "needle_select.cli", "run", "--config", $Config)
    if ($Steps) { $ArgsList += @("--steps", $Steps) }
    if ($DryRun) { $ArgsList += "--dry-run" }
    & $Python @ArgsList
} finally {
    Pop-Location
}
