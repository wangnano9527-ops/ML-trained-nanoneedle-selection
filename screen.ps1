param(
    [string]$Config = "configs\example.project.toml",
    [int]$SampleLimit = 3
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$Here\src;$env:PYTHONPATH"
$Python = if (Test-Path "$Here\.venv\Scripts\python.exe") { "$Here\.venv\Scripts\python.exe" } elseif (Test-Path "$Here\.venv\python.exe") { "$Here\.venv\python.exe" } else { "python" }
Push-Location $Here
try {
    & $Python -m needle_select.cli screen --config $Config --sample-limit $SampleLimit
} finally {
    Pop-Location
}
exit $LASTEXITCODE
