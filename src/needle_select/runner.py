from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable

from .config import ProjectConfig, resolve_project_path


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    command: list[str]
    required: list[Path]


DEFAULT_STEPS = ["doctor", "preprocess", "make-splits", "train", "predict"]


def public_steps() -> list[dict[str, str]]:
    return [
        {"name": "doctor", "description": "Check Python packages, GPU, manifest, splits, and disk space."},
        {"name": "preprocess", "description": "Convert raw tif/mask pairs into cleaned training images and masks."},
        {"name": "make-splits", "description": "Create train/val/test splits from a manifest."},
        {"name": "train", "description": "Train or fine-tune the U-Net segmentation model."},
        {"name": "predict", "description": "Run model prediction over a manifest."},
        {"name": "infer", "description": "Run profile-aware inference over a file or directory."},
    ]


def build_plan(config: ProjectConfig, steps: Iterable[str] | None = None) -> list[Step]:
    selected = list(steps or DEFAULT_STEPS)
    return [build_step(config, step) for step in selected]


def build_step(config: ProjectConfig, name: str) -> Step:
    python = config.commands.python
    root = config.paths.project_root
    scripts_dir = config.paths.scripts_dir
    train_config = resolve_project_path(config, config.configs.train_config)
    preprocess_config = resolve_project_path(config, config.configs.preprocess_config)
    manifest = resolve_project_path(config, config.paths.manifest)
    predictions_dir = resolve_project_path(config, config.paths.predictions_dir)
    checkpoint = resolve_project_path(config, config.model.checkpoint)
    inference_input = resolve_project_path(config, config.inference.input)
    profile = resolve_project_path(config, config.configs.inference_profile)

    if name == "doctor":
        command = [python, str(root / scripts_dir / "check_training_env.py"), "--config", str(train_config)]
        return Step(name, "Check environment readiness.", command, [root / scripts_dir / "check_training_env.py", train_config])
    if name == "preprocess":
        command = [python, str(root / scripts_dir / "preprocess_raw_data.py"), "--config", str(preprocess_config)]
        return Step(name, "Preprocess raw tif/mask pairs.", command, [root / scripts_dir / "preprocess_raw_data.py", preprocess_config])
    if name in {"make-splits", "make_splits"}:
        command = [python, str(root / scripts_dir / "make_splits.py"), "--config", str(train_config)]
        return Step("make-splits", "Create train/val/test splits.", command, [root / scripts_dir / "make_splits.py", train_config])
    if name == "train":
        command = [python, str(root / scripts_dir / "train_unet.py"), "--config", str(train_config)]
        return Step(name, "Train U-Net.", command, [root / scripts_dir / "train_unet.py", train_config])
    if name == "predict":
        command = [
            python,
            str(root / scripts_dir / "predict_masks.py"),
            "--checkpoint",
            str(checkpoint or Path("MISSING_CHECKPOINT")),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(predictions_dir),
        ]
        return Step(name, "Predict masks for manifest rows.", command, [root / scripts_dir / "predict_masks.py"])
    if name == "infer":
        command = [
            python,
            str(root / scripts_dir / "infer_needles.py"),
            "--checkpoint",
            str(checkpoint or Path("MISSING_CHECKPOINT")),
            "--input",
            str(inference_input or Path("MISSING_INPUT")),
            "--out-dir",
            str(predictions_dir),
        ]
        if profile is not None:
            command.extend(["--profile", str(profile)])
        add_optional(command, "--threshold", config.inference.threshold)
        add_optional(command, "--patch-size", config.inference.patch_size)
        add_optional(command, "--overlap", config.inference.overlap)
        add_optional(command, "--channel", config.inference.channel)
        add_optional(command, "--scale", config.inference.scale)
        add_optional(command, "--input-magnification", config.inference.input_magnification)
        add_optional(command, "--trained-magnification", config.inference.trained_magnification)
        if config.inference.recursive:
            command.append("--recursive")
        return Step(name, "Run direct image/folder inference.", command, [root / scripts_dir / "infer_needles.py"])
    raise ValueError(f"Unknown step: {name}")


def add_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run_plan(plan: list[Step], *, cwd: Path, dry_run: bool) -> dict:
    rows = [step_to_dict(step) for step in plan]
    if dry_run:
        return {"dry_run": True, "steps": rows, "return_codes": []}
    return_codes = []
    for step in plan:
        result = subprocess.run(step.command, cwd=cwd, check=False)
        return_codes.append({"step": step.name, "returncode": result.returncode})
        if result.returncode != 0:
            break
    return {"dry_run": False, "steps": rows, "return_codes": return_codes}


def step_to_dict(step: Step) -> dict:
    return {
        "name": step.name,
        "description": step.description,
        "command": step.command,
        "required": [str(path) for path in step.required],
    }
