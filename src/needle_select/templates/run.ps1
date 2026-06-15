param(
    [string]$Config = "configs\needle_select_project.toml",
    [string]$Steps = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Here
try {
    $Tool = Get-Command needle-select -ErrorAction SilentlyContinue
    if ($Tool) {
        $Exe = "needle-select"
        $Prefix = @()
    } else {
        $Exe = "python"
        $Prefix = @("-m", "needle_select.cli")
    }
    $ArgsList = @("run", "--config", $Config)
    if ($Steps) { $ArgsList += @("--steps", $Steps) }
    if ($DryRun) { $ArgsList += "--dry-run" }
    & $Exe @Prefix @ArgsList
} finally {
    Pop-Location
}
