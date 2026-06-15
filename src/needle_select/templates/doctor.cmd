@echo off
setlocal
cd /d "%~dp0"
where needle-select >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  needle-select doctor --config configs\needle_select_project.toml %*
) else (
  python -m needle_select.cli doctor --config configs\needle_select_project.toml %*
)
