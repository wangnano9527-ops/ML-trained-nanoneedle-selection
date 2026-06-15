from __future__ import annotations

import importlib.metadata
import importlib.resources as resources
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

from .config import ProjectConfig, load_project_config
from .runner import build_plan, public_steps, run_plan


def describe_project() -> dict:
    return {
        "name": "Needle Select",
        "package": "needle_select",
        "version": package_version(),
        "summary": "Reusable preprocessing, training, fine-tuning, and inference tooling for nano-needle array masks.",
        "capabilities": list_capabilities(),
        "public_steps": list_public_steps(),
        "data_flow": [
            {
                "step": "preprocess",
                "input": "raw tif images plus paired manual/normalized masks",
                "output": "data/images, data/masks, data/manifest.csv, data/preprocess_summary.json",
            },
            {
                "step": "make-splits",
                "input": "manifest.csv",
                "output": "splits.csv",
            },
            {
                "step": "train",
                "input": "manifest.csv, splits.csv, training config, optional init checkpoint",
                "output": "run directory with checkpoints, config.json, history.json",
            },
            {
                "step": "infer",
                "input": "checkpoint, image or folder, inference profile",
                "output": "mask_pred.png, prob.png, overlay.png, inference_settings.json",
            },
        ],
        "safe_to_git": ["src/", "needle_select/", "scripts/", "configs/", "docs/", "tests/", "pyproject.toml"],
        "keep_out_of_git": ["data/", "runs/", "predictions/", "Training/", "New-raw/", "dist/", "raw data/"],
    }


def describe_pipeline() -> dict:
    return describe_project()


def list_capabilities() -> list[dict[str, str]]:
    return [
        {"name": "mask_preprocessing", "description": "Clean manual/normalized masks using area and lattice filters."},
        {"name": "split_generation", "description": "Create train/val/test splits by whole image."},
        {"name": "unet_training", "description": "Train or fine-tune a binary U-Net segmentation model."},
        {"name": "sliding_window_prediction", "description": "Predict full images from patch-trained models."},
        {"name": "profile_aware_inference", "description": "Run inference with threshold, scale, magnification, and auto pitch estimation."},
        {"name": "project_bootstrap", "description": "Initialize a clean run directory for another project."},
    ]


def list_public_steps() -> list[dict[str, str]]:
    return public_steps()


def init_project(target_dir: str | Path, *, overwrite: bool = False) -> dict:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for folder in ["data", "output", "work", "logs", "configs"]:
        (target / folder).mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    template_root = resources.files("needle_select.templates")
    for relative, resource in walk_resource_files(template_root):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            continue
        destination.write_bytes(resource.read_bytes())
        copied.append(str(relative))
    return {
        "target_dir": str(target.resolve()),
        "created_dirs": ["data", "output", "work", "logs", "configs"],
        "copied_files": copied,
        "next_commands": [
            str(target / "doctor.ps1"),
            str(target / "run.ps1"),
        ],
    }


def walk_resource_files(root, prefix: Path = Path("")):
    for child in root.iterdir():
        if child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo")):
            continue
        if not prefix.parts and child.name == "__init__.py":
            continue
        relative = prefix / child.name
        if child.is_file():
            yield relative, child
        elif child.is_dir():
            yield from walk_resource_files(child, relative)


def run_project(
    config_path: str | Path,
    *,
    steps: Iterable[str] | None = None,
    dry_run: bool = False,
) -> dict:
    config = load_project_config(config_path)
    plan = build_plan(config, steps)
    return run_plan(plan, cwd=config.paths.project_root, dry_run=dry_run)


def check_environment(config_path: str | Path | None = None) -> dict:
    config = load_project_config(config_path) if config_path else ProjectConfig()
    packages = {name: importlib.util.find_spec(module) is not None for name, module in {
        "numpy": "numpy",
        "Pillow": "PIL",
        "scipy": "scipy",
        "torch": "torch",
        "tqdm": "tqdm",
    }.items()}
    paths = {
        "project_root": config.paths.project_root.exists(),
        "data_dir": (config.paths.project_root / config.paths.data_dir).exists(),
        "scripts_dir": (config.paths.project_root / config.paths.scripts_dir).exists(),
    }
    gpu = {"torch_installed": packages["torch"], "cuda_available": False}
    if packages["torch"]:
        import torch

        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda_version"] = getattr(torch.version, "cuda", None)
        gpu["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": packages,
        "paths": paths,
        "gpu": gpu,
        "config": str(config_path) if config_path else None,
        "ready": all(packages[name] for name in ["numpy", "Pillow", "scipy"]) and paths["project_root"],
    }


def package_version() -> str:
    try:
        return importlib.metadata.version("needle-select")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2)
