from __future__ import annotations

from pathlib import Path
import tomllib


def load_toml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def section(config: dict, name: str) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return value

