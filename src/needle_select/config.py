from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys
import tomllib


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    work_dir: Path = Path("work")
    raw_dir: Path = Path("raw data")
    manifest: Path = Path("data/manifest.csv")
    splits: Path = Path("data/splits.csv")
    predictions_dir: Path = Path("output/predictions")
    scripts_dir: Path = Path("scripts")


@dataclass(frozen=True)
class ConfigPaths:
    preprocess_config: Path = Path("configs/preprocess.toml")
    train_config: Path = Path("configs/train.toml")
    inference_profile: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: Path | None = None


@dataclass(frozen=True)
class InferenceConfig:
    input: Path | None = None
    threshold: float | None = None
    patch_size: int | None = None
    overlap: float | None = None
    channel_mode: str = "max"
    channel: int | None = None
    scale: float | None = None
    input_magnification: float | None = None
    trained_magnification: float | None = None
    auto_magnification: bool = True
    model_diameter_px: float | None = None
    expected_diameter_px: float | None = None
    clean_components: bool = True
    min_diameter_ratio: float = 0.35
    max_diameter_ratio: float = 2.5
    save_circle_rois: bool = True
    circle_radius_padding: float = 0.0
    circle_min_radius: float | None = None
    circle_max_radius: float | None = None
    circle_min_area: int = 4
    circle_center_mode: str = "lattice"
    circle_radius_mode: str = "global-quantile"
    circle_component_coverage_quantile: float = 0.99
    circle_global_radius_quantile: float = 0.99
    circle_radius_anomaly_ratio: float = 1.5
    circle_radius_anomaly_mad: float = 6.0
    lattice_min_points: int = 6
    lattice_max_snap_distance: float | None = 4.0
    recursive: bool = False


@dataclass(frozen=True)
class CommandConfig:
    python: str = sys.executable


@dataclass(frozen=True)
class ProjectConfig:
    path: Path | None = None
    name: str = "needle-select-run"
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    configs: ConfigPaths = field(default_factory=ConfigPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    base_dir = config_path.parent.resolve()
    paths_payload = payload.get("paths", {})
    paths = ProjectPaths(
        project_root=resolve_path(paths_payload.get("project_root", "."), base_dir),
        data_dir=Path(paths_payload.get("data_dir", "data")),
        output_dir=Path(paths_payload.get("output_dir", "output")),
        work_dir=Path(paths_payload.get("work_dir", "work")),
        raw_dir=Path(paths_payload.get("raw_dir", "raw data")),
        manifest=Path(paths_payload.get("manifest", "data/manifest.csv")),
        splits=Path(paths_payload.get("splits", "data/splits.csv")),
        predictions_dir=Path(paths_payload.get("predictions_dir", "output/predictions")),
        scripts_dir=Path(paths_payload.get("scripts_dir", "scripts")),
    )
    configs_payload = payload.get("configs", {})
    inference_profile = configs_payload.get("inference_profile")
    configs = ConfigPaths(
        preprocess_config=Path(configs_payload.get("preprocess_config", "configs/preprocess.toml")),
        train_config=Path(configs_payload.get("train_config", "configs/train.toml")),
        inference_profile=Path(inference_profile) if inference_profile else None,
    )
    model_payload = payload.get("model", {})
    checkpoint = model_payload.get("checkpoint")
    inference_payload = payload.get("inference", {})
    commands_payload = payload.get("commands", {})
    return ProjectConfig(
        path=config_path.resolve(),
        name=str(payload.get("project", {}).get("name", "needle-select-run")),
        paths=paths,
        configs=configs,
        model=ModelConfig(checkpoint=Path(checkpoint) if checkpoint else None),
        inference=InferenceConfig(
            input=Path(inference_payload["input"]) if inference_payload.get("input") else None,
            threshold=maybe_float(inference_payload.get("threshold")),
            patch_size=maybe_int(inference_payload.get("patch_size")),
            overlap=maybe_float(inference_payload.get("overlap")),
            channel_mode=str(inference_payload.get("channel_mode", "max")),
            channel=maybe_int(inference_payload.get("channel")),
            scale=maybe_float(inference_payload.get("scale")),
            input_magnification=maybe_float(inference_payload.get("input_magnification")),
            trained_magnification=maybe_float(inference_payload.get("trained_magnification")),
            auto_magnification=bool(inference_payload.get("auto_magnification", True)),
            model_diameter_px=maybe_float(inference_payload.get("model_diameter_px")),
            expected_diameter_px=maybe_float(inference_payload.get("expected_diameter_px")),
            clean_components=bool(inference_payload.get("clean_components", True)),
            min_diameter_ratio=float(inference_payload.get("min_diameter_ratio", 0.35)),
            max_diameter_ratio=float(inference_payload.get("max_diameter_ratio", 2.5)),
            save_circle_rois=bool(inference_payload.get("save_circle_rois", True)),
            circle_radius_padding=float(inference_payload.get("circle_radius_padding", 0.0)),
            circle_min_radius=maybe_float(inference_payload.get("circle_min_radius")),
            circle_max_radius=maybe_float(inference_payload.get("circle_max_radius")),
            circle_min_area=int(inference_payload.get("circle_min_area", 4)),
            circle_center_mode=str(inference_payload.get("circle_center_mode", "lattice")),
            circle_radius_mode=str(inference_payload.get("circle_radius_mode", "global-quantile")),
            circle_component_coverage_quantile=float(
                inference_payload.get("circle_component_coverage_quantile", 0.99)
            ),
            circle_global_radius_quantile=float(
                inference_payload.get("circle_global_radius_quantile", 0.99)
            ),
            circle_radius_anomaly_ratio=float(inference_payload.get("circle_radius_anomaly_ratio", 1.5)),
            circle_radius_anomaly_mad=float(inference_payload.get("circle_radius_anomaly_mad", 6.0)),
            lattice_min_points=int(inference_payload.get("lattice_min_points", 6)),
            lattice_max_snap_distance=maybe_float(inference_payload.get("lattice_max_snap_distance", 4.0)),
            recursive=bool(inference_payload.get("recursive", False)),
        ),
        commands=CommandConfig(python=str(commands_payload.get("python", sys.executable))),
    )


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_project_path(config: ProjectConfig, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return config.paths.project_root / path


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)
