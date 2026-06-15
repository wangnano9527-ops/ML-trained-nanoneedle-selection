from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch

from needle_select.ml.config import section
from needle_select.ml.data import robust_normalize
from needle_select.ml.inference_profile import (
    load_inference_profile,
    profile_get,
    resize_float_image,
    resolve_image_scale,
    restore_probability_to_shape,
)
from needle_select.ml.model import build_model
from scripts.predict_masks import choose_device, predict_full_image


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained Needle Select model on images or a folder.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory of images.")
    parser.add_argument("--out-dir", default=Path("predictions"), type=Path)
    parser.add_argument("--profile", default=None, type=Path, help="Inference profile JSON with defaults and scale hints.")
    parser.add_argument("--threshold", default=None, type=float)
    parser.add_argument("--patch-size", default=None, type=int)
    parser.add_argument("--overlap", default=None, type=float)
    parser.add_argument("--channel", default=None, type=int, help="Zero-based TIFF frame/band to use.")
    parser.add_argument("--scale", default=None, type=float, help="Direct image scale applied before inference.")
    parser.add_argument("--input-magnification", default=None, type=float, help="Input microscope magnification, e.g. 60.")
    parser.add_argument("--trained-magnification", default=None, type=float, help="Model training magnification, usually 40.")
    parser.add_argument("--auto-scale", action="store_true", help="Estimate image scale from lattice pitch/dot diameter.")
    parser.add_argument("--target-pitch-px", default=None, type=float, help="Expected lattice pitch at model scale.")
    parser.add_argument("--target-dot-diameter-px", default=None, type=float, help="Expected dot diameter at model scale.")
    parser.add_argument("--scale-min", default=None, type=float)
    parser.add_argument("--scale-max", default=None, type=float)
    parser.add_argument("--no-settings-json", action="store_true", help="Do not write inference_settings.json.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config = load_model(args.checkpoint, device=choose_device(args.device))
    profile = load_inference_profile(args.profile, checkpoint_path=args.checkpoint)
    training_config = section(config, "training")
    patch_config = section(config, "patches")
    threshold = args.threshold if args.threshold is not None else float(
        profile_get(profile, "inference", "threshold", training_config.get("threshold", 0.5))
    )
    patch_size = args.patch_size if args.patch_size is not None else int(
        profile_get(profile, "inference", "patch_size", patch_config.get("patch_size", 512))
    )
    overlap = args.overlap if args.overlap is not None else float(profile_get(profile, "inference", "overlap", 0.5))
    channel = args.channel if args.channel is not None else int(profile_get(profile, "inference", "channel", 0))

    inputs = collect_inputs(args.input, recursive=args.recursive)
    if not inputs:
        raise SystemExit(f"No image files found in {args.input}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    settings_rows: list[dict] = []
    for image_path in inputs:
        image = load_inference_image(image_path, channel_index=channel)
        image_scale, geometry, scale_source = resolve_image_scale(
            image,
            profile,
            explicit_scale=args.scale,
            input_magnification=args.input_magnification,
            trained_magnification=args.trained_magnification,
            auto_scale=args.auto_scale,
            target_pitch_px=args.target_pitch_px,
            target_dot_diameter_px=args.target_dot_diameter_px,
            min_scale=args.scale_min,
            max_scale=args.scale_max,
        )
        model_image = resize_float_image(image, image_scale)
        prob_scaled = predict_full_image(model, model_image, patch_size=patch_size, overlap=overlap, device=device)
        prob = restore_probability_to_shape(prob_scaled, image.shape)
        mask = prob >= threshold
        stem = safe_stem(image_path)
        save_outputs(args.out_dir, stem, image, prob, mask)
        settings_rows.append(
            {
                "input": str(image_path),
                "output_mask": str(args.out_dir / (stem + "_mask_pred.png")),
                "threshold": threshold,
                "patch_size": patch_size,
                "overlap": overlap,
                "channel": channel,
                "image_scale": image_scale,
                "scale_source": scale_source,
                "geometry_estimate": geometry.to_dict() if geometry is not None else None,
                "original_shape": list(image.shape),
                "model_shape": list(model_image.shape),
            }
        )
        print(f"{image_path} -> {args.out_dir / (stem + '_mask_pred.png')}")

    if not args.no_settings_json:
        (args.out_dir / "inference_settings.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "profile": profile,
                    "images": settings_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_model(checkpoint_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(section(config, "model"))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def collect_inputs(path: Path, *, recursive: bool) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def load_inference_image(path: Path, *, channel_index: int) -> np.ndarray:
    with Image.open(path) as image:
        n_frames = getattr(image, "n_frames", 1)
        if n_frames > 1:
            if channel_index >= n_frames:
                raise ValueError(f"{path} has only {n_frames} frames; channel={channel_index} is invalid.")
            image.seek(channel_index)
            channel = image.copy()
        elif len(image.getbands()) > 1:
            channel = image.getchannel(channel_index)
        else:
            if channel_index != 0:
                raise ValueError(f"{path} has one channel only.")
            channel = image.copy()
    array = np.asarray(channel, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{path} did not load as a 2D image; got shape {array.shape}.")
    return robust_normalize(array)


def save_outputs(out_dir: Path, stem: str, image: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> None:
    binary = mask.astype(np.uint8) * 255
    probability = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
    overlay = make_overlay(image, mask)
    Image.fromarray(binary).save(out_dir / f"{stem}_mask_pred.png")
    Image.fromarray(probability).save(out_dir / f"{stem}_prob.png")
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    outline = mask & ~erode_binary(mask)
    fill = mask & ~outline
    rgb[fill, 0] = np.maximum(rgb[fill, 0], 170)
    rgb[fill, 1] = (rgb[fill, 1] * 0.55).astype(np.uint8)
    rgb[fill, 2] = (rgb[fill, 2] * 0.55).astype(np.uint8)
    rgb[outline] = np.array([255, 32, 32], dtype=np.uint8)
    return rgb


def erode_binary(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return np.zeros_like(mask, dtype=bool)
    center = mask[1:-1, 1:-1]
    eroded_inner = (
        center
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
        & mask[:-2, :-2]
        & mask[:-2, 2:]
        & mask[2:, :-2]
        & mask[2:, 2:]
    )
    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1] = eroded_inner
    return eroded


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


if __name__ == "__main__":
    main()
