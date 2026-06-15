from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from needle_select.preprocess_parameters import describe_preprocess_parameters


def main() -> None:
    for spec in describe_preprocess_parameters():
        print(f"[{spec.group}] {spec.name}")
        print(f"  default: {spec.default}")
        print(f"  level: {spec.user_level}")
        print(f"  {spec.description}")
        if spec.tune_lower:
            print(f"  lower: {spec.tune_lower}")
        if spec.tune_higher:
            print(f"  higher: {spec.tune_higher}")
        print()


if __name__ == "__main__":
    main()
