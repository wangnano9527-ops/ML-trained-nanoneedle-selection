param(
    [string]$Config = "configs\needle_select_project.toml",
    [int]$SampleLimit = 3
)

$ErrorActionPreference = "Stop"
if (Get-Command needle-select -ErrorAction SilentlyContinue) {
    & needle-select screen --config $Config --sample-limit $SampleLimit
} else {
    & python -m needle_select.cli screen --config $Config --sample-limit $SampleLimit
}
exit $LASTEXITCODE
