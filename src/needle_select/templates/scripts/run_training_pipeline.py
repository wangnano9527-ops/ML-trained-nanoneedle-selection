from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from check_training_env import build_report, print_human_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check environment, prepare splits, and train the model.")
    parser.add_argument("--config", default=Path("configs/train.toml"), type=Path)
    parser.add_argument("--preprocess-config", default=Path("configs/preprocess.toml"), type=Path)
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before split/training.")
    parser.add_argument("--force-splits", action="store_true", help="Regenerate data/splits.csv before training.")
    parser.add_argument("--strict-gpu", action="store_true", help="Do not train unless CUDA is available.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preprocess:
        run([sys.executable, "scripts/preprocess_raw_data.py", "--config", str(args.preprocess_config)])

    if args.force_splits or not Path("data/splits.csv").exists():
        run([sys.executable, "scripts/make_splits.py", "--config", str(args.config)])

    report = build_report(args.config, strict_gpu=args.strict_gpu)
    print_human_report(report)
    if args.check_only:
        raise SystemExit(0 if report["ready_for_training"] else 1)
    if not report["ready_for_training"]:
        raise SystemExit("Training environment is not ready. Fix failures above and rerun.")

    run([sys.executable, "scripts/train_unet.py", "--config", str(args.config)])


def run(command: list[str]) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
