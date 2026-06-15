#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if command -v needle-select >/dev/null 2>&1; then
  exec needle-select plan --config configs/needle_select_project.toml "$@"
fi
exec python -m needle_select.cli plan --config configs/needle_select_project.toml "$@"
