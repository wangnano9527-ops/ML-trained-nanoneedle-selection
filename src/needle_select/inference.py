from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .image_io import IMAGE_SUFFIXES, load_image_projection, safe_stem as image_safe_stem
from .ml.inference_profile import (
    load_inference_profile,
    profile_get,
    resolve_magnification_scale,
)


@dataclass(frozen=True)
class LoadedNeedleModel:
    model: Any
    config: dict[str, Any]
    predict_full_image: Any
    torch_device: Any


@dataclass(frozen=True)
class ComponentStats:
    count: int
    mask_fraction: float
    diameter_mean: float | None
    diameter_median: float | None
    diameter_p10: float | None
    diameter_p90: float | None


@dataclass(frozen=True)
class CircleROI:
    roi_id: int
    component_label: int
    center_x: float
    center_y: float
    radius_px: float
    required_radius_px: float
    component_area_px: int
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int
    touches_edge: bool
    center_mode: str = "component"
    seed_center_x: float | None = None
    seed_center_y: float | None = None
    lattice_node_i: int | None = None
    lattice_node_j: int | None = None
    lattice_snap_distance_px: float | None = None
    radius_padding_px: float = 0.0
    min_radius_px: float | None = None
    _component_x: np.ndarray | None = field(default=None, repr=False, compare=False)
    _component_y: np.ndarray | None = field(default=None, repr=False, compare=False)
    _image_width: int | None = field(default=None, repr=False, compare=False)
    _image_height: int | None = field(default=None, repr=False, compare=False)


CIRCLE_ROI_FIELDNAMES = [
    "image",
    "stem",
    "roi_id",
    "component_label",
    "center_x",
    "center_y",
    "radius_px",
    "required_radius_px",
    "diameter_px",
    "component_area_px",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
    "touches_edge",
    "center_mode",
    "seed_center_x",
    "seed_center_y",
    "lattice_node_i",
    "lattice_node_j",
    "lattice_snap_distance_px",
]


@dataclass
class CircleImageRecord:
    image_path: Path
    stem: str
    rois: list[CircleROI]
    metrics: dict[str, Any]
    image_radius_px: float | None = None
    final_radius_px: float | None = None
    anomaly: bool = False
    anomaly_reason: str = ""


def run_needle_inference(
    *,
    project_dir: Path | None = None,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    profile_path: Path | None = None,
    channel: int = 0,
    channel_mode: str = "max",
    threshold: float | None = None,
    patch_size: int | None = None,
    overlap: float = 0.5,
    recursive: bool = False,
    device: str = "auto",
    model_scale: float | None = None,
    trained_magnification: float | None = None,
    input_magnification: float | None = None,
    auto_magnification: bool = True,
    model_diameter_px: float | None = None,
    expected_diameter_px: float | None = None,
    diameter_tolerance: float = 0.35,
    clean_components: bool = True,
    min_diameter_ratio: float = 0.35,
    max_diameter_ratio: float = 2.5,
    save_circle_rois: bool = True,
    circle_radius_padding: float = 0.0,
    circle_min_radius: float | None = None,
    circle_max_radius: float | None = None,
    circle_min_area: int = 1,
    circle_center_mode: str = "component",
    circle_uniform_radius: bool = False,
    circle_radius_mode: str = "image-max",
    circle_component_coverage_quantile: float = 1.0,
    circle_global_radius_quantile: float = 0.99,
    circle_radius_anomaly_ratio: float = 1.5,
    circle_radius_anomaly_mad: float = 6.0,
    lattice_min_points: int = 6,
    lattice_max_snap_distance: float | None = None,
    overwrite: bool = False,
) -> int:
    """Run unified-v2 inference with channel projection and exact magnification mapping.

    The model consumes one 2D plane. Inputs can select one channel or combine all
    channels with MAX/SUM. Known 20x/40x/60x inputs use their exact model scale;
    unknown magnification is estimated and mapped to one of those three values.
    """
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_needle_model(checkpoint, device_name=device)
    profile = load_inference_profile(profile_path, checkpoint_path=checkpoint)
    training_config = config_section(loaded.config, "training")
    patch_config = config_section(loaded.config, "patches")
    threshold_value = threshold if threshold is not None else float(
        profile_get(profile, "inference", "threshold", training_config.get("threshold", 0.5))
    )
    patch_size_value = patch_size if patch_size is not None else int(
        profile_get(profile, "inference", "patch_size", patch_config.get("patch_size", 256))
    )
    if model_diameter_px is None:
        model_diameter_px = float(profile_get(profile, "model", "target_dot_diameter_px", 9.4))

    inputs = collect_inputs(input_path, recursive=recursive)
    if not inputs:
        raise FileNotFoundError(f"No image files found in {input_path}")

    if circle_radius_mode not in {"image-max", "global-quantile"}:
        raise ValueError("circle_radius_mode must be 'image-max' or 'global-quantile'.")
    component_quantile = validate_quantile(
        circle_component_coverage_quantile,
        name="circle_component_coverage_quantile",
    )
    global_quantile = validate_quantile(
        circle_global_radius_quantile,
        name="circle_global_radius_quantile",
    )

    rows: list[dict[str, Any]] = []
    all_circle_rows: list[dict[str, Any]] = []
    circle_records: list[CircleImageRecord] = []
    for image_path in inputs:
        stem = safe_stem(image_path)
        mask_path = output_dir / f"{stem}_mask_pred.png"
        if mask_path.exists() and not overwrite:
            print(f"needle-mask skip existing: {image_path.name}")
            continue

        projection, projection_info = load_image_projection(
            image_path,
            channel_mode=channel_mode,
            channel=channel,
        )
        scale_resolution = resolve_magnification_scale(
            projection,
            profile,
            explicit_scale=model_scale,
            input_magnification=input_magnification,
            trained_magnification=trained_magnification,
            auto_scale=auto_magnification,
        )
        scale = scale_resolution.image_scale
        selected_magnification = scale_resolution.selected_input_magnification
        training_magnification_value = scale_resolution.trained_magnification
        expected_diameter = resolve_expected_diameter(
            expected_diameter_px=expected_diameter_px,
            model_diameter_px=model_diameter_px,
            training_magnification=training_magnification_value,
            imaging_magnification=selected_magnification,
        )
        original_shape = projection.shape
        model_input = resize_float_image(projection, scale) if not math.isclose(scale, 1.0) else projection

        prob_model = loaded.predict_full_image(
            loaded.model,
            model_input,
            patch_size=patch_size_value,
            overlap=overlap,
            device=loaded.torch_device,
        )
        prob = (
            resize_float_image(prob_model, 1.0 / scale, output_shape=original_shape)
            if not math.isclose(scale, 1.0)
            else prob_model
        )
        prob = np.clip(prob, 0.0, 1.0)
        raw_mask = prob >= threshold_value
        raw_stats = component_stats(raw_mask)

        mask = raw_mask
        if clean_components and expected_diameter is not None:
            mask = filter_components_by_diameter(
                raw_mask,
                min_diameter=expected_diameter * min_diameter_ratio,
                max_diameter=expected_diameter * max_diameter_ratio,
            )
        clean_stats = component_stats(mask)

        circle_rois: list[CircleROI] = []
        circle_mask_fraction: float | None = None
        if save_circle_rois:
            circle_rois = extract_circle_rois(
                mask,
                weight_image=prob,
                radius_padding=circle_radius_padding,
                min_radius=resolve_circle_min_radius(circle_min_radius, expected_diameter),
                max_radius=circle_max_radius,
                min_area=circle_min_area,
            )
            circle_rois = regularize_circle_rois(
                circle_rois,
                center_mode=circle_center_mode,
                uniform_radius=circle_uniform_radius and circle_radius_mode == "image-max",
                lattice_min_points=lattice_min_points,
                lattice_max_snap_distance=lattice_max_snap_distance,
            )

        save_outputs(output_dir, stem, projection, prob, mask, raw_mask=raw_mask)

        warning = diameter_warning(
            clean_stats,
            expected_diameter=expected_diameter,
            tolerance=diameter_tolerance,
        )
        row = {
            "image": str(image_path),
            "stem": stem,
            "height": original_shape[0],
            "width": original_shape[1],
            "channel_mode": channel_mode,
            "source_channels": projection_info.channel_count,
            "source_axes": projection_info.source_axes,
            "source_shape": "x".join(str(value) for value in projection_info.source_shape),
            "selected_channel": projection_info.selected_channel,
            "model_scale": scale,
            "scale_source": scale_resolution.source,
            "trained_magnification": training_magnification_value,
            "input_magnification": selected_magnification,
            "estimated_input_magnification": scale_resolution.estimated_input_magnification,
            "magnification_warning": scale_resolution.warning or "",
            "model_diameter_px": model_diameter_px,
            "expected_diameter_px": expected_diameter,
            "threshold": threshold_value,
            "patch_size": patch_size_value,
            "overlap": overlap,
            "raw_components": raw_stats.count,
            "raw_mask_fraction": raw_stats.mask_fraction,
            "raw_diameter_mean": raw_stats.diameter_mean,
            "raw_diameter_median": raw_stats.diameter_median,
            "raw_diameter_p10": raw_stats.diameter_p10,
            "raw_diameter_p90": raw_stats.diameter_p90,
            "clean_components": clean_stats.count,
            "clean_mask_fraction": clean_stats.mask_fraction,
            "clean_diameter_mean": clean_stats.diameter_mean,
            "clean_diameter_median": clean_stats.diameter_median,
            "clean_diameter_p10": clean_stats.diameter_p10,
            "clean_diameter_p90": clean_stats.diameter_p90,
            "circle_rois": len(circle_rois),
            "circle_radius_mean": mean_or_none([roi.radius_px for roi in circle_rois]),
            "circle_radius_median": median_or_none([roi.radius_px for roi in circle_rois]),
            "circle_mask_fraction": circle_mask_fraction,
            "circle_center_mode": circle_center_mode,
            "circle_uniform_radius": circle_uniform_radius,
            "circle_radius_mode": circle_radius_mode,
            "circle_component_coverage_quantile": component_quantile,
            "circle_global_radius_quantile": global_quantile,
            "circle_image_radius_px": None,
            "circle_final_radius_px": None,
            "circle_radius_anomaly": False,
            "circle_radius_anomaly_reason": "",
            "circle_component_coverage_fraction": None,
            "circle_missed_mask_pixels": None,
            "circle_missed_mask_fraction": None,
            "warning": "; ".join(
                value for value in [warning, scale_resolution.warning] if value
            ),
        }
        rows.append(row)
        if save_circle_rois:
            circle_records.append(
                CircleImageRecord(
                    image_path=image_path,
                    stem=stem,
                    rois=strip_circle_runtime_fields(circle_rois),
                    metrics=row,
                )
            )
        print(format_row_summary(row))

    if save_circle_rois:
        final_circle_rows = finalize_circle_records(
            circle_records,
            output_dir=output_dir,
            radius_mode=circle_radius_mode,
            uniform_radius=circle_uniform_radius,
            component_quantile=component_quantile,
            global_quantile=global_quantile,
            anomaly_ratio=circle_radius_anomaly_ratio,
            anomaly_mad_threshold=circle_radius_anomaly_mad,
        )
        all_circle_rows.extend(final_circle_rows)
        write_circle_radius_summary(output_dir / "needle_circle_radius_summary.csv", circle_records)

    write_metrics_csv(output_dir / "needle_mask_metrics.csv", rows)
    if save_circle_rois:
        write_circle_rois_csv(output_dir / "needle_circle_rois.csv", all_circle_rows)
    (output_dir / "inference_settings.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "profile": profile,
                "channel_mode": channel_mode,
                "channel": channel,
                "images": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def load_needle_model(checkpoint: Path, *, device_name: str) -> LoadedNeedleModel:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Needle-select checkpoint not found: {checkpoint}")

    import torch
    from .ml.model import build_model
    from .ml.predict import predict_full_image

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]
    model = build_model(config_section(config, "model"))
    model.load_state_dict(checkpoint_data["model_state"])
    torch_device = choose_torch_device(torch, device_name)
    model.to(torch_device)
    model.eval()
    return LoadedNeedleModel(
        model=model,
        config=config,
        predict_full_image=predict_full_image,
        torch_device=torch_device,
    )


def choose_torch_device(torch_module: Any, value: str) -> Any:
    if value != "auto":
        return torch_module.device(value)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def collect_inputs(path: Path, *, recursive: bool) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def load_model_input_projection(
    path: Path,
    *,
    channel_mode: str,
    channel: int,
) -> tuple[np.ndarray, int]:
    projection, info = load_image_projection(path, channel_mode=channel_mode, channel=channel)
    return projection, info.channel_count


def extract_channel_yx(data: np.ndarray, axes: str, *, channel: int) -> np.ndarray:
    if "C" not in axes:
        if channel != 0:
            raise ValueError("Cannot select channel>0 from an image with no C axis.")
        return collapse_to_yx(data, axes)
    c_axis = axes.index("C")
    c_count = data.shape[c_axis]
    if channel < 0 or channel >= c_count:
        raise ValueError(f"channel={channel} is out of range for {c_count} channels in TIFF.")
    channel_data = np.take(data, channel, axis=c_axis)
    channel_axes = axes[:c_axis] + axes[c_axis + 1 :]
    return collapse_to_yx(channel_data, channel_axes)


def collapse_to_yx(data: np.ndarray, axes: str) -> np.ndarray:
    axes_list = list(axes)
    if "Y" not in axes_list or "X" not in axes_list:
        if data.ndim == 2:
            return data.astype(np.float32, copy=False)
        raise ValueError(f"Cannot collapse axes {axes!r} to YX.")

    reduced = data
    for axis_index in reversed([i for i, axis in enumerate(axes_list) if axis not in {"Y", "X"}]):
        reduced = np.max(reduced, axis=axis_index)
        axes_list.pop(axis_index)

    if axes_list != ["Y", "X"]:
        reduced = np.transpose(reduced, [axes_list.index("Y"), axes_list.index("X")])
    if reduced.ndim != 2:
        raise ValueError(f"Expected a 2D YX image after collapse; got shape {reduced.shape}.")
    return reduced.astype(np.float32, copy=False)


def robust_normalize(image: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def resolve_model_scale(
    *,
    model_scale: float | None,
    training_magnification: float | None,
    imaging_magnification: float | None,
) -> float:
    if model_scale is not None:
        if model_scale <= 0:
            raise ValueError("model_scale must be > 0.")
        return float(model_scale)
    if training_magnification is not None and imaging_magnification is not None:
        if training_magnification <= 0 or imaging_magnification <= 0:
            raise ValueError("magnification values must be > 0.")
        return float(training_magnification) / float(imaging_magnification)
    return 1.0


def resolve_expected_diameter(
    *,
    expected_diameter_px: float | None,
    model_diameter_px: float | None,
    training_magnification: float | None,
    imaging_magnification: float | None,
) -> float | None:
    if expected_diameter_px is not None:
        return float(expected_diameter_px)
    if (
        model_diameter_px is not None
        and training_magnification is not None
        and imaging_magnification is not None
    ):
        return float(model_diameter_px) * float(imaging_magnification) / float(training_magnification)
    return None


def resize_float_image(
    image: np.ndarray,
    scale: float,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if output_shape is None:
        if scale <= 0:
            raise ValueError("scale must be > 0.")
        output_shape = (
            max(1, int(round(image.shape[0] * scale))),
            max(1, int(round(image.shape[1] * scale))),
        )
    if output_shape == image.shape:
        return image.astype(np.float32, copy=False)
    pil = Image.fromarray(image.astype(np.float32, copy=False), mode="F")
    resized = pil.resize((output_shape[1], output_shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def filter_components_by_diameter(
    mask: np.ndarray,
    *,
    min_diameter: float,
    max_diameter: float,
) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    ids = np.arange(1, count + 1)
    areas = ndimage.sum(mask, labels, ids)
    diameters = equivalent_diameters(np.asarray(areas, dtype=np.float64))
    keep_ids = ids[(diameters >= min_diameter) & (diameters <= max_diameter)]
    if keep_ids.size == 0:
        return mask
    return np.isin(labels, keep_ids)


def resolve_circle_min_radius(
    circle_min_radius: float | None,
    expected_diameter: float | None,
) -> float | None:
    if circle_min_radius is not None:
        return None if circle_min_radius <= 0 else float(circle_min_radius)
    if expected_diameter is None:
        return None
    return float(expected_diameter) / 2.0


def extract_circle_rois(
    mask: np.ndarray,
    *,
    weight_image: np.ndarray | None = None,
    radius_padding: float = 0.0,
    min_radius: float | None = None,
    max_radius: float | None = None,
    min_area: int = 1,
) -> list[CircleROI]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    if weight_image is not None and np.asarray(weight_image).shape != mask.shape:
        raise ValueError("weight_image shape must match mask shape.")
    if min_radius is not None and min_radius <= 0:
        min_radius = None
    if max_radius is not None and max_radius <= 0:
        max_radius = None
    if min_radius is not None and max_radius is not None and min_radius > max_radius:
        raise ValueError("circle_min_radius cannot be larger than circle_max_radius.")

    labels, count = ndimage.label(mask)
    if count == 0:
        return []

    height, width = mask.shape
    objects = ndimage.find_objects(labels)
    rois: list[CircleROI] = []
    roi_id = 1
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        component = labels[slices] == label_id
        area = int(component.sum())
        if area < int(min_area):
            continue

        rows, cols = np.nonzero(component)
        global_rows = rows + int(slices[0].start)
        global_cols = cols + int(slices[1].start)
        center_y, center_x = weighted_component_center(
            global_rows,
            global_cols,
            weight_image=weight_image,
        )
        distances = np.sqrt(np.square(global_cols - center_x) + np.square(global_rows - center_y))
        fit_radius = (float(distances.max()) if distances.size else 0.0) + 0.5
        radius = fit_radius + max(0.0, float(radius_padding))
        if min_radius is not None:
            radius = max(radius, float(min_radius))
        if max_radius is not None and radius > float(max_radius):
            continue

        touches_edge = (
            center_x - radius < 0
            or center_y - radius < 0
            or center_x + radius >= width
            or center_y + radius >= height
        )
        rois.append(
            CircleROI(
                roi_id=roi_id,
                component_label=label_id,
                center_x=float(center_x),
                center_y=float(center_y),
                radius_px=float(radius),
                required_radius_px=float(radius),
                component_area_px=area,
                bbox_min_x=int(global_cols.min()),
                bbox_min_y=int(global_rows.min()),
                bbox_max_x=int(global_cols.max()),
                bbox_max_y=int(global_rows.max()),
                touches_edge=bool(touches_edge),
                seed_center_x=float(center_x),
                seed_center_y=float(center_y),
                radius_padding_px=max(0.0, float(radius_padding)),
                min_radius_px=min_radius,
                _component_x=global_cols.astype(np.float32, copy=True),
                _component_y=global_rows.astype(np.float32, copy=True),
                _image_width=width,
                _image_height=height,
            )
        )
        roi_id += 1
    return rois


def regularize_circle_rois(
    rois: list[CircleROI],
    *,
    center_mode: str,
    uniform_radius: bool,
    lattice_min_points: int,
    lattice_max_snap_distance: float | None,
) -> list[CircleROI]:
    if center_mode == "component":
        regularized = rois
    elif center_mode == "lattice":
        regularized = snap_circle_rois_to_lattice(
            rois,
            min_points=lattice_min_points,
            max_snap_distance=lattice_max_snap_distance,
        )
    else:
        raise ValueError(f"Unsupported circle_center_mode={center_mode!r}.")
    if uniform_radius:
        regularized = set_uniform_circle_radius(regularized)
    return reindex_circle_rois(regularized)


def snap_circle_rois_to_lattice(
    rois: list[CircleROI],
    *,
    min_points: int = 6,
    max_snap_distance: float | None = None,
) -> list[CircleROI]:
    if len(rois) < min_points:
        return rois
    from .lattice import fit_lattice, snap_points_to_lattice_network

    points = np.asarray([[roi.center_x, roi.center_y] for roi in rois], dtype=np.float64)
    try:
        model = fit_lattice(points, min_points=min_points)
        snaps = snap_points_to_lattice_network(points, model, max_distance=max_snap_distance)
    except ValueError:
        return rois

    regularized: list[CircleROI] = []
    for roi, snap in zip(rois, snaps):
        if not snap.accepted:
            regularized.append(roi)
            continue
        radius = required_circle_radius(roi, center_x=snap.center_x, center_y=snap.center_y)
        snapped_roi = replace(
            roi,
            center_x=snap.center_x,
            center_y=snap.center_y,
            radius_px=radius,
            required_radius_px=radius,
            touches_edge=circle_touches_edge(roi, snap.center_x, snap.center_y, radius),
            center_mode="lattice",
            seed_center_x=roi.center_x,
            seed_center_y=roi.center_y,
            lattice_node_i=snap.node_i,
            lattice_node_j=snap.node_j,
            lattice_snap_distance_px=snap.distance_px,
        )
        regularized.append(snapped_roi)
    return regularized


def set_uniform_circle_radius(rois: list[CircleROI]) -> list[CircleROI]:
    if not rois:
        return rois
    radius = max(roi.radius_px for roi in rois)
    return [
        replace(
            roi,
            radius_px=radius,
            touches_edge=circle_touches_edge(roi, roi.center_x, roi.center_y, radius),
        )
        for roi in rois
    ]


def reindex_circle_rois(rois: list[CircleROI]) -> list[CircleROI]:
    return [replace(roi, roi_id=index) for index, roi in enumerate(rois, start=1)]


def strip_circle_runtime_fields(rois: list[CircleROI]) -> list[CircleROI]:
    return [
        replace(
            roi,
            _component_x=None,
            _component_y=None,
        )
        for roi in rois
    ]


def finalize_circle_records(
    records: list[CircleImageRecord],
    *,
    output_dir: Path,
    radius_mode: str,
    uniform_radius: bool,
    component_quantile: float,
    global_quantile: float,
    anomaly_ratio: float,
    anomaly_mad_threshold: float,
) -> list[dict[str, Any]]:
    if not records:
        return []
    for record in records:
        record.image_radius_px = circle_required_radius_quantile(
            record.rois,
            component_quantile,
        )

    if radius_mode == "global-quantile":
        global_radius, anomaly_flags, anomaly_reasons = choose_global_circle_radius(
            [record.image_radius_px for record in records],
            global_quantile=global_quantile,
            anomaly_ratio=anomaly_ratio,
            anomaly_mad_threshold=anomaly_mad_threshold,
        )
    else:
        global_radius = None
        anomaly_flags = [False] * len(records)
        anomaly_reasons = [""] * len(records)

    all_rows: list[dict[str, Any]] = []
    for record, anomaly, reason in zip(records, anomaly_flags, anomaly_reasons):
        record.anomaly = anomaly
        record.anomaly_reason = reason
        if radius_mode == "global-quantile":
            final_radius = global_radius
        elif uniform_radius:
            final_radius = record.image_radius_px
        else:
            final_radius = None
        record.final_radius_px = final_radius
        final_rois = (
            apply_final_circle_radius(record.rois, final_radius)
            if final_radius is not None
            else record.rois
        )

        mask = load_saved_mask(output_dir / f"{record.stem}_mask_pred.png")
        base = load_saved_float_image(output_dir / f"{record.stem}_prob.png")
        circle_mask = make_circle_mask(mask.shape, final_rois)
        missed = mask & ~circle_mask
        coverage = component_coverage_fraction(final_rois, final_radius)

        record.metrics.update(
            {
                "circle_rois": len(final_rois),
                "circle_radius_mean": mean_or_none([roi.radius_px for roi in final_rois]),
                "circle_radius_median": median_or_none([roi.radius_px for roi in final_rois]),
                "circle_mask_fraction": float(np.mean(circle_mask)) if circle_mask.size else 0.0,
                "circle_image_radius_px": record.image_radius_px,
                "circle_final_radius_px": final_radius,
                "circle_radius_anomaly": anomaly,
                "circle_radius_anomaly_reason": reason,
                "circle_component_coverage_fraction": coverage,
                "circle_missed_mask_pixels": int(missed.sum()),
                "circle_missed_mask_fraction": float(missed.sum() / max(int(mask.sum()), 1)),
            }
        )

        rows = circle_rois_to_rows(final_rois, image_path=record.image_path, stem=record.stem)
        write_circle_rois_csv(output_dir / f"{record.stem}_circle_rois.csv", rows)
        save_circle_outputs(output_dir, record.stem, base, mask, final_rois, circle_mask)
        record.rois = final_rois
        all_rows.extend(rows)
    return all_rows


def apply_final_circle_radius(rois: list[CircleROI], radius: float | None) -> list[CircleROI]:
    if radius is None:
        return rois
    return [
        replace(
            roi,
            radius_px=float(radius),
            touches_edge=circle_touches_edge(roi, roi.center_x, roi.center_y, float(radius)),
        )
        for roi in rois
    ]


def circle_required_radius_quantile(rois: list[CircleROI], quantile: float) -> float | None:
    radii = [roi.required_radius_px for roi in rois if np.isfinite(roi.required_radius_px)]
    if not radii:
        return None
    return float(np.quantile(np.asarray(radii, dtype=np.float64), quantile))


def choose_global_circle_radius(
    radii: list[float | None],
    *,
    global_quantile: float,
    anomaly_ratio: float,
    anomaly_mad_threshold: float,
) -> tuple[float | None, list[bool], list[str]]:
    values = np.asarray([np.nan if value is None else float(value) for value in radii], dtype=np.float64)
    valid = np.isfinite(values)
    flags = [False] * len(values)
    reasons = [""] * len(values)
    if not np.any(valid):
        return None, flags, reasons

    valid_values = values[valid]
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    robust_sigma = 1.4826 * mad
    mad_threshold = (
        median + float(anomaly_mad_threshold) * robust_sigma
        if robust_sigma > 0 and anomaly_mad_threshold > 0
        else math.inf
    )
    ratio_threshold = median * float(anomaly_ratio) if anomaly_ratio > 0 else math.inf
    threshold = min(mad_threshold, ratio_threshold)
    if not np.isfinite(threshold):
        threshold = math.inf

    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if value > threshold:
            flags[index] = True
            reasons[index] = (
                f"image radius {value:.3f}px exceeds calibration threshold {threshold:.3f}px"
            )

    calibration = values[valid & ~np.asarray(flags, dtype=bool)]
    if calibration.size == 0:
        calibration = valid_values
        flags = [False] * len(values)
        reasons = [""] * len(values)
    return float(np.quantile(calibration, global_quantile)), flags, reasons


def component_coverage_fraction(rois: list[CircleROI], radius: float | None) -> float | None:
    if radius is None or not rois:
        return None
    covered = sum(1 for roi in rois if roi.required_radius_px <= radius)
    return float(covered / len(rois))


def load_saved_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_saved_float_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    return np.clip(arr / 255.0, 0.0, 1.0)


def write_circle_radius_summary(path: Path, records: list[CircleImageRecord]) -> None:
    rows = [
        {
            "image": str(record.image_path),
            "stem": record.stem,
            "roi_count": len(record.rois),
            "image_radius_px": record.image_radius_px,
            "final_radius_px": record.final_radius_px,
            "component_coverage_fraction": record.metrics.get("circle_component_coverage_fraction"),
            "missed_mask_pixels": record.metrics.get("circle_missed_mask_pixels"),
            "missed_mask_fraction": record.metrics.get("circle_missed_mask_fraction"),
            "anomaly": record.anomaly,
            "anomaly_reason": record.anomaly_reason,
        }
        for record in records
    ]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def required_circle_radius(roi: CircleROI, *, center_x: float, center_y: float) -> float:
    if roi._component_x is None or roi._component_y is None or len(roi._component_x) == 0:
        return roi.radius_px
    distances = np.sqrt(
        np.square(roi._component_x.astype(np.float64, copy=False) - float(center_x))
        + np.square(roi._component_y.astype(np.float64, copy=False) - float(center_y))
    )
    radius = float(distances.max()) + 0.5 + max(0.0, float(roi.radius_padding_px))
    if roi.min_radius_px is not None:
        radius = max(radius, float(roi.min_radius_px))
    return radius


def circle_touches_edge(roi: CircleROI, center_x: float, center_y: float, radius: float) -> bool:
    if roi._image_width is None or roi._image_height is None:
        return roi.touches_edge
    return (
        center_x - radius < 0
        or center_y - radius < 0
        or center_x + radius >= roi._image_width
        or center_y + radius >= roi._image_height
    )


def weighted_component_center(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    weight_image: np.ndarray | None,
) -> tuple[float, float]:
    if weight_image is not None:
        weights = np.asarray(weight_image, dtype=np.float64)[rows, cols]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        total = float(weights.sum())
        if total > 0:
            return float(np.dot(rows, weights) / total), float(np.dot(cols, weights) / total)
    return float(np.mean(rows)), float(np.mean(cols))


def circle_rois_to_rows(
    rois: list[CircleROI],
    *,
    image_path: Path,
    stem: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for roi in rois:
        rows.append(
            {
                "image": str(image_path),
                "stem": stem,
                "roi_id": roi.roi_id,
                "component_label": roi.component_label,
                "center_x": roi.center_x,
                "center_y": roi.center_y,
                "radius_px": roi.radius_px,
                "required_radius_px": roi.required_radius_px,
                "diameter_px": roi.radius_px * 2.0,
                "component_area_px": roi.component_area_px,
                "bbox_min_x": roi.bbox_min_x,
                "bbox_min_y": roi.bbox_min_y,
                "bbox_max_x": roi.bbox_max_x,
                "bbox_max_y": roi.bbox_max_y,
                "touches_edge": roi.touches_edge,
                "center_mode": roi.center_mode,
                "seed_center_x": roi.seed_center_x,
                "seed_center_y": roi.seed_center_y,
                "lattice_node_i": roi.lattice_node_i,
                "lattice_node_j": roi.lattice_node_j,
                "lattice_snap_distance_px": roi.lattice_snap_distance_px,
            }
        )
    return rows


def write_circle_rois_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CIRCLE_ROI_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_circle_mask(shape: tuple[int, int], rois: list[CircleROI]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for roi in rois:
        add_circle_to_mask(mask, roi.center_x, roi.center_y, roi.radius_px)
    return mask


def add_circle_to_mask(mask: np.ndarray, center_x: float, center_y: float, radius: float) -> None:
    height, width = mask.shape
    min_x = max(0, int(math.floor(center_x - radius)))
    max_x = min(width - 1, int(math.ceil(center_x + radius)))
    min_y = max(0, int(math.floor(center_y - radius)))
    max_y = min(height - 1, int(math.ceil(center_y + radius)))
    if min_x > max_x or min_y > max_y:
        return
    yy, xx = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
    patch = np.square(xx - center_x) + np.square(yy - center_y) <= radius * radius
    mask[min_y : max_y + 1, min_x : max_x + 1] |= patch


def save_circle_outputs(
    out_dir: Path,
    stem: str,
    image: np.ndarray,
    mask: np.ndarray,
    rois: list[CircleROI],
    circle_mask: np.ndarray,
) -> None:
    Image.fromarray((circle_mask.astype(np.uint8) * 255)).save(
        out_dir / f"{stem}_circle_mask.png"
    )
    Image.fromarray(make_circle_overlay(image, mask, rois)).save(
        out_dir / f"{stem}_circle_overlay.png"
    )


def make_circle_overlay(image: np.ndarray, mask: np.ndarray, rois: list[CircleROI]) -> np.ndarray:
    rgb = make_overlay(image, mask)
    circle_edges = np.zeros(mask.shape, dtype=bool)
    centers = np.zeros(mask.shape, dtype=bool)
    for roi in rois:
        add_circle_outline_to_mask(
            circle_edges,
            roi.center_x,
            roi.center_y,
            roi.radius_px,
            thickness=1.5,
        )
        add_circle_to_mask(centers, roi.center_x, roi.center_y, radius=1.5)
    rgb[circle_edges] = np.array([0, 220, 255], dtype=np.uint8)
    rgb[centers] = np.array([255, 230, 0], dtype=np.uint8)
    return rgb


def add_circle_outline_to_mask(
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    thickness: float,
) -> None:
    height, width = mask.shape
    outer = radius + thickness / 2.0
    inner = max(0.0, radius - thickness / 2.0)
    min_x = max(0, int(math.floor(center_x - outer)))
    max_x = min(width - 1, int(math.ceil(center_x + outer)))
    min_y = max(0, int(math.floor(center_y - outer)))
    max_y = min(height - 1, int(math.ceil(center_y + outer)))
    if min_x > max_x or min_y > max_y:
        return
    yy, xx = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
    dist2 = np.square(xx - center_x) + np.square(yy - center_y)
    patch = (dist2 <= outer * outer) & (dist2 >= inner * inner)
    mask[min_y : max_y + 1, min_x : max_x + 1] |= patch


def component_stats(mask: np.ndarray) -> ComponentStats:
    labels, count = ndimage.label(mask)
    mask_fraction = float(np.mean(mask)) if mask.size else 0.0
    if count == 0:
        return ComponentStats(
            count=0,
            mask_fraction=mask_fraction,
            diameter_mean=None,
            diameter_median=None,
            diameter_p10=None,
            diameter_p90=None,
        )
    ids = np.arange(1, count + 1)
    areas = ndimage.sum(mask, labels, ids)
    diameters = equivalent_diameters(np.asarray(areas, dtype=np.float64))
    return ComponentStats(
        count=int(count),
        mask_fraction=mask_fraction,
        diameter_mean=float(np.mean(diameters)),
        diameter_median=float(np.median(diameters)),
        diameter_p10=float(np.percentile(diameters, 10)),
        diameter_p90=float(np.percentile(diameters, 90)),
    )


def equivalent_diameters(areas: np.ndarray) -> np.ndarray:
    return np.sqrt(4.0 * areas / math.pi)


def diameter_warning(
    stats: ComponentStats,
    *,
    expected_diameter: float | None,
    tolerance: float,
) -> str | None:
    if expected_diameter is None or stats.diameter_median is None:
        return None
    lower = expected_diameter * (1.0 - tolerance)
    upper = expected_diameter * (1.0 + tolerance)
    if lower <= stats.diameter_median <= upper:
        return None
    return (
        f"median diameter {stats.diameter_median:.2f}px is outside expected "
        f"{expected_diameter:.2f}px +/- {tolerance:.0%}; check magnification/model_diameter"
    )


def save_outputs(
    out_dir: Path,
    stem: str,
    image: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    *,
    raw_mask: np.ndarray,
) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / f"{stem}_mask_pred.png")
    if not np.array_equal(mask, raw_mask):
        Image.fromarray((raw_mask.astype(np.uint8) * 255)).save(out_dir / f"{stem}_mask_raw.png")
    Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)).save(
        out_dir / f"{stem}_prob.png"
    )
    Image.fromarray(make_overlay(image, mask)).save(out_dir / f"{stem}_overlay.png")


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


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_quantile(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the range (0, 1].")
    return value


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def format_row_summary(row: dict[str, Any]) -> str:
    warning = f" WARNING: {row['warning']}" if row["warning"] else ""
    median = row["clean_diameter_median"]
    median_text = "NA" if median is None else f"{median:.2f}px"
    circle_radius = row["circle_radius_median"]
    circle_text = "NA" if circle_radius is None else f"{circle_radius:.2f}px"
    return (
        f"needle-mask: {Path(row['image']).name} -> "
        f"components={row['clean_components']}, median_diameter={median_text}, "
        f"circles={row['circle_rois']}, circle_radius={circle_text}, "
        f"scale={row['model_scale']:.4f}, channels={row['source_channels']}{warning}"
    )


def safe_stem(path: Path) -> str:
    return image_safe_stem(path)
