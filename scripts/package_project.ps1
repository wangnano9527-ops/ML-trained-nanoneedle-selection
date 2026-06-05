param(
    [string]$Out = "Needle-select-transfer.zip",
    [switch]$IncludeRuns,
    [switch]$IncludePredictions
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PackageName = "needle-select-package-" + [guid]::NewGuid().ToString("N")
$StageRoot = Join-Path ([System.IO.Path]::GetTempPath()) $PackageName
$StageProject = Join-Path $StageRoot "Needle-select"

New-Item -ItemType Directory -Force -Path $StageProject | Out-Null

$Items = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements-ml.txt",
    "configs",
    "docs",
    "needle_select",
    "scripts",
    "raw data",
    "data"
)

if ($IncludeRuns) {
    $Items += "runs"
}
if ($IncludePredictions) {
    $Items += "predictions"
}

foreach ($Item in $Items) {
    $Source = Join-Path $ProjectRoot $Item
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $StageProject -Recurse -Force
    }
}

$OutPath = if ([System.IO.Path]::IsPathRooted($Out)) {
    $Out
} else {
    Join-Path $ProjectRoot $Out
}

if (Test-Path -LiteralPath $OutPath) {
    Remove-Item -LiteralPath $OutPath -Force
}

Compress-Archive -Path (Join-Path $StageRoot "Needle-select") -DestinationPath $OutPath -Force

if ($StageRoot.StartsWith([System.IO.Path]::GetTempPath(), [System.StringComparison]::OrdinalIgnoreCase)) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}

Write-Host "Wrote $OutPath"

