#!/usr/bin/env sh
set -eu
if command -v needle-select >/dev/null 2>&1; then
  exec needle-select screen --config configs/needle_select_project.toml "$@"
fi
exec python -m needle_select.cli screen --config configs/needle_select_project.toml "$@"
