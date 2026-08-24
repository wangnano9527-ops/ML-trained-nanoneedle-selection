from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional at import time, required for auto-scale.
    ndi = None
    cKDTree = None


DEFAULT_PROFILE: dict[str, Any] = {
    "version": 1,
    "model": {
        "trained_magnification": 40.0,
        "target_pitch_px": 31.0,
        "target_dot_diameter_px": 9.4,
    },
    "inference": {
        "threshold": 0.5,
        "patch_size": 256,
        "overlap": 0.5,
        "channel": 0,
        "channel_mode": "max",
    },
    "scale": {
        "mode": "auto",
        "min_scale": 0.5,
        "max_scale": 2.25,
        "supported_magnifications": [20.0, 40.0, 60.0],
        "auto_method": "pitch_then_diameter",
        "candidate_threshold_percentiles": [96.0, 95.0, 94.0, 93.0, 92.0, 90.0],
        "min_components": 12,
        "candidate_min_area_px": 4.0,
        "candidate_max_area_px": 2500.0,
    },
    "postprocess": {
        "clean_components": True,
        "min_diameter_ratio": 0.35,
        "max_diameter_ratio": 2.5,
        "save_circle_rois": True,
        "circle_center_mode": "lattice",
        "circle_radius_mode": "global-quantile",
        "circle_component_coverage_quantile": 0.99,
        "circle_global_radius_quantile": 0.99,
        "circle_radius_anomaly_ratio": 1.5,
        "circle_radius_anomaly_mad": 6.0,
        "lattice_min_points": 6,
        "lattice_max_snap_distance": 4.0,
    },
    "notes": {
        "image_scale": "Scale applied to the input image before model inference. 60x images for a 40x-trained model often use 40/60 = 0.6667.",
        "target_pitch_px": "Expected nano-needle lattice pitch at the model's training scale.",
        "target_dot_diameter_px": "Expected dot diameter at the model's training scale, used as auto-scale fallback.",
    },
}


@dataclass(frozen=True)
class GeometryEstimate:
    pitch_px: float | None
    dot_diameter_px: float | None
    component_count: int
    threshold_percentile: float | None
    scale_from_pitch: float | None
    scale_from_dot_diameter: float | None
    selected_scale: float | None
    selected_method: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MagnificationResolution:
    image_scale: float
    source: str
    trained_magnification: float
    selected_input_magnification: float
    estimated_input_magnification: float | None
    supported_magnifications: tuple[float, ...]
    geometry: GeometryEstimate | None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry"] = self.geometry.to_dict() if self.geometry is not None else None
        return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_inference_profile(path: Path | None = None, *, checkpoint_path: Path | None = None) -> dict[str, Any]:
    profile = DEFAULT_PROFILE
    profile_path = path or find_inference_profile(checkpoint_path)
    if profile_path is not None and profile_path.exists():
        loaded = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = deep_merge(profile, loaded)
    return profile


def find_inference_profile(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    candidates = [
        checkpoint_path.with_name("inference_profile.json"),
        checkpoint_path.parent.parent / "inference_profile.json",
        checkpoint_path.parent / "model_metadata.json",
        checkpoint_path.parent.parent / "model_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.name == "model_metadata.json":
                continue
            return candidate
    return None


def profile_get(profile: dict[str, Any], section: str, name: str, default: Any) -> Any:
    value = profile.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(name, default)


def resize_float_image(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return image.astype(np.float32, copy=False)
    height, width = image.shape
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(image.astype(np.float32, copy=False), mode="F")
    resized = pil_image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def restore_probability_to_shape(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if prob.shape == shape:
        return prob.astype(np.float32, copy=False)
    height, width = shape
    pil_prob = Image.fromarray(prob.astype(np.float32, copy=False), mode="F")
    resized = pil_prob.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def resolve_image_scale(
    image: np.ndarray,
    profile: dict[str, Any],
    *,
    explicit_scale: float | None = None,
    input_magnification: float | None = None,
    trained_magnification: float | None = None,
    auto_scale: bool = False,
    target_pitch_px: float | None = None,
    target_dot_diameter_px: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> tuple[float, GeometryEstimate | None, str]:
    resolution = resolve_magnification_scale(
        image,
        profile,
        explicit_scale=explicit_scale,
        input_magnification=input_magnification,
        trained_magnification=trained_magnification,
        auto_scale=auto_scale,
        target_pitch_px=target_pitch_px,
        target_dot_diameter_px=target_dot_diameter_px,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    return resolution.image_scale, resolution.geometry, resolution.source


def resolve_magnification_scale(
    image: np.ndarray,
    profile: dict[str, Any],
    *,
    explicit_scale: float | None = None,
    input_magnification: float | None = None,
    trained_magnification: float | None = None,
    auto_scale: bool = False,
    target_pitch_px: float | None = None,
    target_dot_diameter_px: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> MagnificationResolution:
    min_scale = float(min_scale if min_scale is not None else profile_get(profile, "scale", "min_scale", 0.45))
    max_scale = float(max_scale if max_scale is not None else profile_get(profile, "scale", "max_scale", 1.5))
    trained = float(
        trained_magnification
        if trained_magnification is not None
        else profile_get(profile, "model", "trained_magnification", 40.0)
    )
    supported = tuple(
        float(value)
        for value in profile_get(profile, "scale", "supported_magnifications", [20.0, 40.0, 60.0])
    )
    if not supported:
        raise ValueError("At least one supported magnification is required.")

    if explicit_scale is not None:
        scale = clamp_scale(float(explicit_scale), min_scale, max_scale)
        estimated = trained / scale
        selected = nearest_supported_magnification(estimated, supported)
        return MagnificationResolution(
            scale,
            "explicit_scale",
            trained,
            selected,
            estimated,
            supported,
            None,
            "Explicit scale bypassed exact 20x/40x/60x scaling.",
        )

    if input_magnification is not None:
        if input_magnification <= 0 or trained <= 0:
            raise ValueError("Magnification values must be positive.")
        selected = require_supported_magnification(float(input_magnification), supported)
        return MagnificationResolution(
            clamp_scale(trained / selected, min_scale, max_scale),
            "explicit_magnification",
            trained,
            selected,
            selected,
            supported,
            None,
        )

    mode = str(profile_get(profile, "scale", "mode", "none"))
    if auto_scale or mode == "auto":
        estimate = estimate_geometry_scale(
            image,
            profile,
            target_pitch_px=target_pitch_px,
            target_dot_diameter_px=target_dot_diameter_px,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        if estimate.selected_scale is not None:
            estimated = trained / float(estimate.selected_scale)
            selected = nearest_supported_magnification(estimated, supported)
            warning = None
            if abs(estimated - selected) > 5.0:
                warning = f"Raw estimate {estimated:.1f}x was mapped to supported {selected:g}x."
            return MagnificationResolution(
                clamp_scale(trained / selected, min_scale, max_scale),
                f"{estimate.selected_method}_mapped_{selected:g}x",
                trained,
                selected,
                estimated,
                supported,
                estimate,
                warning,
            )
        fallback = nearest_supported_magnification(trained, supported)
        return MagnificationResolution(
            clamp_scale(trained / fallback, min_scale, max_scale),
            f"auto_failed_fallback_{fallback:g}x",
            trained,
            fallback,
            None,
            supported,
            estimate,
            "Magnification could not be estimated; confirm the fallback before production use.",
        )

    selected = nearest_supported_magnification(trained, supported)
    return MagnificationResolution(
        clamp_scale(trained / selected, min_scale, max_scale),
        f"default_{selected:g}x",
        trained,
        selected,
        None,
        supported,
        None,
        "No input magnification was supplied and auto estimation is disabled.",
    )


def nearest_supported_magnification(value: float, supported: tuple[float, ...]) -> float:
    return min(supported, key=lambda candidate: (abs(candidate - value), candidate))


def require_supported_magnification(value: float, supported: tuple[float, ...]) -> float:
    selected = nearest_supported_magnification(value, supported)
    if abs(selected - value) > 0.5:
        choices = ", ".join(f"{candidate:g}x" for candidate in supported)
        raise ValueError(f"Unsupported input magnification {value:g}x; choose one of {choices}.")
    return selected


def clamp_scale(value: float, min_scale: float, max_scale: float) -> float:
    return max(float(min_scale), min(float(value), float(max_scale)))


def estimate_geometry_scale(
    image: np.ndarray,
    profile: dict[str, Any],
    *,
    target_pitch_px: float | None = None,
    target_dot_diameter_px: float | None = None,
    min_scale: float,
    max_scale: float,
) -> GeometryEstimate:
    if ndi is None or cKDTree is None:
        return GeometryEstimate(None, None, 0, None, None, None, None, "none", ["scipy_not_available"])

    target_pitch = float(target_pitch_px if target_pitch_px is not None else profile_get(profile, "model", "target_pitch_px", 31.0))
    target_diameter = float(
        target_dot_diameter_px
        if target_dot_diameter_px is not None
        else profile_get(profile, "model", "target_dot_diameter_px", 9.4)
    )
    percentiles = list(profile_get(profile, "scale", "candidate_threshold_percentiles", [96, 95, 94, 92, 90]))
    min_components = int(profile_get(profile, "scale", "min_components", 12))
    min_area = float(profile_get(profile, "scale", "candidate_min_area_px", 4.0))
    max_area = float(profile_get(profile, "scale", "candidate_max_area_px", 2500.0))

    best: tuple[float | None, float | None, int, float] | None = None
    for percentile in percentiles:
        threshold = float(np.percentile(image[np.isfinite(image)], float(percentile)))
        binary = image >= threshold
        labels, count = ndi.label(binary, structure=np.ones((3, 3), dtype=bool))
        if count <= 0:
            continue
        areas, centroids = component_measurements(labels, count)
        keep = (areas[1:] >= min_area) & (areas[1:] <= max_area)
        kept_areas = areas[1:][keep]
        kept_centroids = centroids[1:][keep]
        if len(kept_centroids) < min_components:
            continue
        pitch = estimate_pitch_from_centroids(kept_centroids)
        diameter = estimate_dot_diameter_from_areas(kept_areas)
        best = (pitch, diameter, len(kept_centroids), float(percentile))
        if pitch is not None:
            break

    if best is None:
        return GeometryEstimate(None, None, 0, None, None, None, None, "none", ["not_enough_candidate_components"])

    pitch, diameter, component_count, percentile = best
    scale_from_pitch = target_pitch / pitch if pitch and pitch > 0 else None
    scale_from_diameter = target_diameter / diameter if diameter and diameter > 0 else None
    notes: list[str] = []

    selected_scale = None
    selected_method = "none"
    if scale_from_pitch is not None and np.isfinite(scale_from_pitch):
        selected_scale = clamp_scale(float(scale_from_pitch), min_scale, max_scale)
        selected_method = "auto_pitch"
    elif scale_from_diameter is not None and np.isfinite(scale_from_diameter):
        selected_scale = clamp_scale(float(scale_from_diameter), min_scale, max_scale)
        selected_method = "auto_dot_diameter"
    else:
        notes.append("no_valid_pitch_or_dot_diameter")

    if selected_scale is not None and (selected_scale == min_scale or selected_scale == max_scale):
        notes.append("scale_clamped")

    return GeometryEstimate(
        pitch_px=float(pitch) if pitch is not None else None,
        dot_diameter_px=float(diameter) if diameter is not None else None,
        component_count=int(component_count),
        threshold_percentile=float(percentile),
        scale_from_pitch=float(scale_from_pitch) if scale_from_pitch is not None else None,
        scale_from_dot_diameter=float(scale_from_diameter) if scale_from_diameter is not None else None,
        selected_scale=float(selected_scale) if selected_scale is not None else None,
        selected_method=selected_method,
        notes=notes,
    )


def component_measurements(labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    _, width = labels.shape
    flat_labels = labels.ravel()
    foreground = flat_labels > 0
    foreground_labels = flat_labels[foreground]
    flat_indices = np.flatnonzero(foreground)
    ys, xs = np.divmod(flat_indices, width)
    areas = np.bincount(foreground_labels, minlength=count + 1).astype(np.float64)
    sum_y = np.bincount(foreground_labels, weights=ys, minlength=count + 1)
    sum_x = np.bincount(foreground_labels, weights=xs, minlength=count + 1)
    centroids = np.zeros((count + 1, 2), dtype=np.float64)
    nonzero = areas > 0
    centroids[nonzero, 0] = sum_y[nonzero] / areas[nonzero]
    centroids[nonzero, 1] = sum_x[nonzero] / areas[nonzero]
    return areas, centroids


def estimate_dot_diameter_from_areas(areas: np.ndarray) -> float | None:
    valid = np.asarray(areas, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        return None
    area = estimate_peak(valid)
    if area <= 0:
        return None
    return float(np.sqrt(4.0 * area / np.pi))


def estimate_pitch_from_centroids(centroids_yx: np.ndarray) -> float | None:
    points = np.asarray(centroids_yx, dtype=np.float64)
    if len(points) < 8:
        return None
    tree = cKDTree(points)
    k = min(5, len(points))
    distances, _ = tree.query(points, k=k)
    nearest = distances[:, 1:].ravel()
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size < 8:
        return None
    pitch = estimate_peak(nearest)
    return float(pitch) if np.isfinite(pitch) and pitch > 0 else None


def estimate_peak(values: np.ndarray, *, bins: int = 80) -> float:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        return 0.0
    if valid.size < 8:
        return float(np.median(valid))
    lower, upper = np.quantile(valid, [0.02, 0.98])
    central = valid[(valid >= lower) & (valid <= upper)]
    if central.size >= 8:
        valid = central
    if np.min(valid) == np.max(valid):
        return float(valid[0])
    hist, edges = np.histogram(valid, bins=bins, range=(float(np.min(valid)), float(np.max(valid))))
    if hist.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        hist = np.convolve(hist, kernel / kernel.sum(), mode="same")
    peak_bin = int(np.argmax(hist))
    return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)
