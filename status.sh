#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
python -m needle_select.cli plan --config configs/example.project.toml "$@"
