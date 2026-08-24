from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from needle_select.ml.config import section
from needle_select.ml.data import load_image, read_manifest
from needle_select.ml.model import build_model
from needle_select.ml.predict import choose_device, predict_full_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict full-size nano-needle masks.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("data/manifest.csv"), type=Path)
    parser.add_argument("--out-dir", default=Path("predictions"), type=Path)
    parser.add_argument("--threshold", default=None, type=float)
    parser.add_argument("--patch-size", default=None, type=int)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(section(config, "model"))
    model.load_state_dict(checkpoint["model_state"])
    device = choose_device(args.device)
    model.to(device)
    model.eval()

    training_config = section(config, "training")
    patch_config = section(config, "patches")
    threshold = args.threshold if args.threshold is not None else float(training_config.get("threshold", 0.5))
    patch_size = args.patch_size if args.patch_size is not None else int(patch_config.get("patch_size", 512))

    records = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc="predict"):
        image = load_image(record.image_path)
        prob = predict_full_image(model, image, patch_size=patch_size, overlap=args.overlap, device=device)
        binary = (prob >= threshold).astype(np.uint8) * 255
        Image.fromarray(binary).save(args.out_dir / f"{record.sample_id}_mask_pred.png")
        Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)).save(
            args.out_dir / f"{record.sample_id}_prob.png"
        )
    print(f"Wrote predictions to {args.out_dir}")

if __name__ == "__main__":
    main()
