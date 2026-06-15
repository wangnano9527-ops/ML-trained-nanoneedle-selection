@echo off
setlocal
set ROOT=%~dp0
set PYTHONPATH=%ROOT%src;%PYTHONPATH%
if exist "%ROOT%.venv\Scripts\python.exe" (
  set PY=%ROOT%.venv\Scripts\python.exe
) else if exist "%ROOT%.venv\python.exe" (
  set PY=%ROOT%.venv\python.exe
) else (
  set PY=python
)
cd /d "%ROOT%"
"%PY%" -m needle_select.cli run --config configs\example.project.toml %*
