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
import torch.nn.functional as F
from tqdm import tqdm

from needle_select.ml.config import section
from needle_select.ml.data import load_image, read_manifest
from needle_select.ml.model import build_model


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


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    patch_size: int,
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    height, width = image.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))
    y_starts = starts_for_axis(height, patch_size, stride)
    x_starts = starts_for_axis(width, patch_size, stride)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for y0 in y_starts:
        for x0 in x_starts:
            patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            tensor = torch.from_numpy(patch[None, None, ...].astype(np.float32)).to(device)
            if pad_h or pad_w:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            probs = probs[: patch.shape[0], : patch.shape[1]]
            prob_sum[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] += probs
            count[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] += 1.0

    return prob_sum / np.maximum(count, 1.0)


def starts_for_axis(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


if __name__ == "__main__":
    main()

