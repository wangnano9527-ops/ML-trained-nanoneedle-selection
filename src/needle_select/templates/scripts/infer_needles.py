from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from needle_select.inference import run_needle_inference


DEFAULT_CHECKPOINT = Path("model_registry/unified_v2/needle_unet_unified_v2.pt")
DEFAULT_PROFILE = Path("configs/inference_profile_unified_v2.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified-v2 Needle Select inference with channel projection, magnification mapping, and circular ROIs."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory of images.")
    parser.add_argument("--out-dir", default=Path("predictions"), type=Path)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, type=Path)
    parser.add_argument("--threshold", default=None, type=float)
    parser.add_argument("--patch-size", default=None, type=int)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--channel-mode", choices=("single", "max", "sum"), default="max")
    parser.add_argument("--channel", default=0, type=int, help="Zero-based channel used when --channel-mode=single.")
    parser.add_argument("--scale", default=None, type=float, help="Advanced override; magnification is preferred.")
    parser.add_argument("--input-magnification", choices=(20.0, 40.0, 60.0), default=None, type=float)
    parser.add_argument("--trained-magnification", default=40.0, type=float)
    parser.add_argument("--no-auto-magnification", action="store_true")
    parser.add_argument("--model-diameter", default=None, type=float)
    parser.add_argument("--expected-diameter", default=None, type=float)
    parser.add_argument("--diameter-tolerance", default=0.35, type=float)
    parser.add_argument("--no-clean-components", action="store_true")
    parser.add_argument("--min-diameter-ratio", default=0.35, type=float)
    parser.add_argument("--max-diameter-ratio", default=2.5, type=float)
    parser.add_argument("--no-circle-rois", action="store_true")
    parser.add_argument("--circle-radius-padding", default=0.0, type=float)
    parser.add_argument("--circle-min-radius", default=None, type=float)
    parser.add_argument("--circle-max-radius", default=None, type=float)
    parser.add_argument("--circle-min-area", default=4, type=int)
    parser.add_argument("--circle-center-mode", choices=("component", "lattice"), default="lattice")
    parser.add_argument("--circle-uniform-radius", action="store_true")
    parser.add_argument("--circle-radius-mode", choices=("image-max", "global-quantile"), default="global-quantile")
    parser.add_argument("--circle-component-coverage-quantile", default=0.99, type=float)
    parser.add_argument("--circle-global-radius-quantile", default=0.99, type=float)
    parser.add_argument("--circle-radius-anomaly-ratio", default=1.5, type=float)
    parser.add_argument("--circle-radius-anomaly-mad", default=6.0, type=float)
    parser.add_argument("--lattice-min-points", default=6, type=int)
    parser.add_argument("--lattice-max-snap-distance", default=4.0, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_needle_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.out_dir,
        profile_path=args.profile,
        channel=args.channel,
        channel_mode=args.channel_mode,
        threshold=args.threshold,
        patch_size=args.patch_size,
        overlap=args.overlap,
        recursive=args.recursive,
        device=args.device,
        model_scale=args.scale,
        trained_magnification=args.trained_magnification,
        input_magnification=args.input_magnification,
        auto_magnification=not args.no_auto_magnification,
        model_diameter_px=args.model_diameter,
        expected_diameter_px=args.expected_diameter,
        diameter_tolerance=args.diameter_tolerance,
        clean_components=not args.no_clean_components,
        min_diameter_ratio=args.min_diameter_ratio,
        max_diameter_ratio=args.max_diameter_ratio,
        save_circle_rois=not args.no_circle_rois,
        circle_radius_padding=args.circle_radius_padding,
        circle_min_radius=args.circle_min_radius,
        circle_max_radius=args.circle_max_radius,
        circle_min_area=args.circle_min_area,
        circle_center_mode=args.circle_center_mode,
        circle_uniform_radius=args.circle_uniform_radius,
        circle_radius_mode=args.circle_radius_mode,
        circle_component_coverage_quantile=args.circle_component_coverage_quantile,
        circle_global_radius_quantile=args.circle_global_radius_quantile,
        circle_radius_anomaly_ratio=args.circle_radius_anomaly_ratio,
        circle_radius_anomaly_mad=args.circle_radius_anomaly_mad,
        lattice_min_points=args.lattice_min_points,
        lattice_max_snap_distance=args.lattice_max_snap_distance,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
