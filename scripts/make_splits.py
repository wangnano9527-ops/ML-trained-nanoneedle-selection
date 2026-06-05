from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from needle_select.ml.config import load_toml, section
from needle_select.ml.data import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits by whole image.")
    parser.add_argument("--config", default=Path("configs/train.toml"), type=Path)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--seed", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    data_config = section(config, "data")
    split_config = section(config, "split")

    manifest = args.manifest or Path(data_config.get("manifest", "data/manifest.csv"))
    out_path = args.out or Path(data_config.get("splits", "data/splits.csv"))
    seed = args.seed if args.seed is not None else int(split_config.get("seed", 42))
    val_fraction = float(split_config.get("val_fraction", 0.15))
    test_fraction = float(split_config.get("test_fraction", 0.15))

    records = read_manifest(
        manifest,
        image_column=data_config.get("image_column", "image_path"),
        mask_column=data_config.get("mask_column", "mask_path"),
    )
    sample_ids = [record.sample_id for record in records]
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_test = max(1, round(n_total * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, round(n_total * val_fraction)) if val_fraction > 0 else 0
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions leave no training samples.")

    test_ids = set(sample_ids[:n_test])
    val_ids = set(sample_ids[n_test : n_test + n_val])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split"])
        writer.writeheader()
        for sample_id in sorted(sample_ids):
            split = "test" if sample_id in test_ids else "val" if sample_id in val_ids else "train"
            writer.writerow({"sample_id": sample_id, "split": split})

    print(f"Wrote {out_path}")
    print(f"train={n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()

