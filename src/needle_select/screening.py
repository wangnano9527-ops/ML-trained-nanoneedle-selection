from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .config import load_project_config, resolve_project_path
from .image_io import IMAGE_SUFFIXES, load_image_projection
from .ml.inference_profile import load_inference_profile, resolve_magnification_scale


def screen_project(config_path: str | Path, *, sample_limit: int = 3) -> dict[str, Any]:
    config = load_project_config(config_path)
    root = config.paths.project_root
    checkpoint = resolve_project_path(config, config.model.checkpoint)
    input_path = resolve_project_path(config, config.inference.input)
    output_path = resolve_project_path(config, config.paths.predictions_dir)
    profile_path = resolve_project_path(config, config.configs.inference_profile)

    packages = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "numpy": "numpy",
            "Pillow": "PIL",
            "scipy": "scipy",
            "tifffile": "tifffile",
            "torch": "torch",
        }.items()
    }
    errors: list[str] = []
    warnings: list[str] = []
    questions: list[str] = []

    if checkpoint is None:
        errors.append("No model checkpoint is configured.")
        questions.append("Set [model].checkpoint to the unified v2 checkpoint path.")
    elif not checkpoint.is_file():
        errors.append(f"Model checkpoint not found: {checkpoint}")
        questions.append("Download Git LFS files or correct [model].checkpoint.")
    elif is_git_lfs_pointer(checkpoint):
        errors.append(f"Model checkpoint is still a Git LFS pointer: {checkpoint}")
        questions.append("Run git lfs pull, then repeat the screen.")

    if input_path is None:
        errors.append("No inference input is configured.")
        questions.append("Set [inference].input to an image file or directory.")
        inputs: list[Path] = []
    else:
        inputs = collect_inputs(input_path, recursive=config.inference.recursive)
        if not inputs:
            errors.append(f"No supported image files found: {input_path}")
            questions.append("Provide TIFF, PNG, or JPEG inputs and check the recursive setting.")

    missing_packages = [name for name, available in packages.items() if not available]
    if missing_packages:
        errors.append("Missing runtime packages: " + ", ".join(missing_packages))
        questions.append('Install the project with ML dependencies: python -m pip install -e ".[ml]"')

    mode = config.inference.channel_mode.lower()
    channel = config.inference.channel if config.inference.channel is not None else 0
    if mode not in {"single", "max", "sum"}:
        errors.append(f"Invalid channel_mode={mode!r}; choose single, max, or sum.")
    if channel < 0:
        errors.append("[inference].channel must be a zero-based non-negative integer.")
    if mode == "single":
        questions.append(f"Confirm that zero-based channel {channel} contains the needle signal.")
    elif mode == "max":
        questions.append("MAX is safest when the signal channel is unknown; confirm that bright noise is not dominant.")
    else:
        questions.append("SUM is useful when signal is distributed across channels; confirm saturation is acceptable.")

    supplied_magnification = config.inference.input_magnification
    if supplied_magnification is not None and not any(
        abs(supplied_magnification - supported) <= 0.5 for supported in (20.0, 40.0, 60.0)
    ):
        errors.append("[inference].input_magnification must be 20, 40, or 60 when supplied.")

    profile = load_inference_profile(profile_path, checkpoint_path=checkpoint)
    samples: list[dict[str, Any]] = []
    selected_magnifications: list[float] = []
    for path in inputs[: max(1, int(sample_limit))]:
        try:
            projection, projection_info = load_image_projection(path, channel_mode=mode, channel=channel)
            scale = resolve_magnification_scale(
                projection,
                profile,
                explicit_scale=config.inference.scale,
                input_magnification=config.inference.input_magnification,
                trained_magnification=config.inference.trained_magnification,
                auto_scale=config.inference.auto_magnification,
            )
            selected_magnifications.append(scale.selected_input_magnification)
            if scale.warning:
                warnings.append(f"{path.name}: {scale.warning}")
            samples.append(
                {
                    "input": str(path),
                    "projection": projection_info.to_dict(),
                    "projected_shape": list(projection.shape),
                    "magnification": scale.to_dict(),
                }
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    unique_magnifications = sorted(set(selected_magnifications))
    if len(unique_magnifications) > 1:
        warnings.append(
            "Sampled inputs mapped to mixed magnifications: "
            + ", ".join(f"{value:g}x" for value in unique_magnifications)
        )
    if config.inference.input_magnification is None:
        questions.append("No magnification was supplied; review the estimated 20x/40x/60x mapping below.")
    questions.append(f"Confirm that the output directory is correct and writable: {output_path}")

    return {
        "screen": "needle-select-inference",
        "ready": not errors,
        "config": str(Path(config_path).resolve()),
        "project_root": str(root),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "input": str(input_path) if input_path else None,
        "output": str(output_path) if output_path else None,
        "input_count": len(inputs),
        "channel_mode": mode,
        "channel": channel,
        "supported_magnifications": [20, 40, 60],
        "packages": packages,
        "samples": samples,
        "warnings": deduplicate(warnings),
        "errors": deduplicate(errors),
        "operator_checklist": deduplicate(questions),
        "outputs": [
            "*_mask_pred.png",
            "*_prob.png",
            "*_overlay.png",
            "*_circle_rois.csv",
            "*_circle_mask.png",
            "*_circle_overlay.png",
            "needle_mask_metrics.csv",
            "needle_circle_rois.csv",
            "needle_circle_radius_summary.csv",
            "inference_settings.json",
        ],
    }


def collect_inputs(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        candidate
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(160)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")
