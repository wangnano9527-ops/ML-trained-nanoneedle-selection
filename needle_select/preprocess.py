from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    ndi = None
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class MaskCleanConfig:
    """Parameters for reusable mask cleaning."""

    binarize_threshold: int = 0
    connectivity: int = 2
    area_hist_bins: int = 48
    min_area_factor: float = 0.45
    remove_large_components: bool = False
    max_area_factor: float = 4.0
    min_components_for_network: int = 20
    network_radius_factor: float = 2.4
    network_min_neighbors: int = 2
    min_network_cluster_size: int = 20
    keep_cluster_min_fraction_of_largest: float = 0.10
    use_lattice_filter: bool = True
    lattice_neighbor_k: int = 10
    lattice_angle_bins: int = 120
    lattice_phase_bins: int = 96
    lattice_vector_distance_tolerance: float = 0.45
    lattice_axis_angle_tolerance_deg: float = 18.0
    lattice_phase_tolerance: float = 0.36
    lattice_min_phase_tolerance_px: float = 4.0
    lattice_axial_distance_tolerance: float = 0.32
    lattice_axial_lateral_tolerance: float = 0.28
    lattice_min_axial_neighbors: int = 2
    lattice_min_final_fraction: float = 0.35


@dataclass(frozen=True)
class LatticeModel:
    angle_deg: float
    pitch_px: float
    phase_u_px: float
    phase_v_px: float
    phase_tolerance_px: float
    basis_u_xy: tuple[float, float]
    basis_v_xy: tuple[float, float]


@dataclass(frozen=True)
class ArrayFilterResult:
    component_ids: np.ndarray
    typical_spacing_px: float | None
    network_radius_px: float | None
    lattice_model: LatticeModel | None
    lattice_candidate_components: int | None
    removed_lattice_components: int | None
    removed_cluster_components: int | None


@dataclass(frozen=True)
class MaskCleanStats:
    source_mask: str
    total_components: int
    area_peak_px: float
    min_area_px: float
    max_area_px: float | None
    after_area_components: int
    typical_spacing_px: float | None
    network_radius_px: float | None
    final_components: int
    removed_small_components: int
    removed_large_components: int
    removed_network_components: int
    lattice_angle_deg: float | None
    lattice_pitch_px: float | None
    lattice_phase_tolerance_px: float | None
    lattice_candidate_components: int | None
    removed_lattice_components: int | None
    removed_cluster_components: int | None
    foreground_px_before: int
    foreground_px_after: int


@dataclass(frozen=True)
class PreprocessResult:
    sample_id: str
    source_tif: str
    source_mask: str
    image_path: str
    mask_path: str
    stats: MaskCleanStats


def require_scipy() -> None:
    if ndi is None or cKDTree is None:
        raise RuntimeError(
            "Mask cleaning requires scipy. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from _SCIPY_IMPORT_ERROR


def extract_tif_channel(
    tif_path: Path,
    output_path: Path,
    *,
    channel_index: int = 0,
    overwrite: bool = True,
) -> Path:
    """Extract one channel/page from a TIFF and save it as a TIFF.

    The microscope files in this project are multi-page TIFFs, where channel1 is
    the first page. For RGB-like TIFFs, the selected band is extracted instead.
    """

    tif_path = Path(tif_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    with Image.open(tif_path) as image:
        n_frames = getattr(image, "n_frames", 1)
        if n_frames > 1:
            if channel_index >= n_frames:
                raise ValueError(
                    f"{tif_path} has {n_frames} frames; channel_index={channel_index} is invalid."
                )
            image.seek(channel_index)
            channel = image.copy()
        elif len(image.getbands()) > 1:
            channel = image.getchannel(channel_index)
        else:
            if channel_index != 0:
                raise ValueError(f"{tif_path} has one channel only.")
            channel = image.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel.save(output_path)
    return output_path


def load_binary_mask(mask_path: Path, *, threshold: int = 0) -> np.ndarray:
    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L")) > threshold


def save_binary_mask(mask: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)
    return output_path


def clean_mask(
    mask_path: Path,
    output_path: Path | None = None,
    *,
    config: MaskCleanConfig | None = None,
) -> tuple[np.ndarray, MaskCleanStats]:
    """Clean one hand-drawn mask and optionally save the result."""

    require_scipy()
    config = config or MaskCleanConfig()
    mask = load_binary_mask(mask_path, threshold=config.binarize_threshold)
    labels, count = label_components(mask, connectivity=config.connectivity)
    areas, centroids = component_measurements(labels, count)

    if count == 0:
        cleaned = np.zeros_like(mask, dtype=bool)
        stats = MaskCleanStats(
            source_mask=str(mask_path),
            total_components=0,
            area_peak_px=0.0,
            min_area_px=0.0,
            max_area_px=None,
            after_area_components=0,
            typical_spacing_px=None,
            network_radius_px=None,
            final_components=0,
            removed_small_components=0,
            removed_large_components=0,
            removed_network_components=0,
            lattice_angle_deg=None,
            lattice_pitch_px=None,
            lattice_phase_tolerance_px=None,
            lattice_candidate_components=None,
            removed_lattice_components=None,
            removed_cluster_components=None,
            foreground_px_before=int(mask.sum()),
            foreground_px_after=0,
        )
        if output_path is not None:
            save_binary_mask(cleaned, output_path)
        return cleaned, stats

    area_peak = estimate_component_area_peak(areas[1:], bins=config.area_hist_bins)
    min_area = area_peak * config.min_area_factor
    max_area = area_peak * config.max_area_factor if config.remove_large_components else None

    component_ids = np.arange(1, count + 1)
    area_keep = areas[1:] >= min_area
    if max_area is not None:
        area_keep &= areas[1:] <= max_area

    area_kept_ids = component_ids[area_keep]
    array_filter = keep_array_network_components(
        area_kept_ids,
        centroids[area_kept_ids],
        config=config,
    )
    network_kept_ids = array_filter.component_ids

    keep_table = np.zeros(count + 1, dtype=bool)
    keep_table[network_kept_ids] = True
    cleaned = keep_table[labels]

    small_removed = int(np.sum(areas[1:] < min_area))
    large_removed = int(np.sum(areas[1:] > max_area)) if max_area is not None else 0
    network_removed = int(len(area_kept_ids) - len(network_kept_ids))
    stats = MaskCleanStats(
        source_mask=str(mask_path),
        total_components=int(count),
        area_peak_px=float(area_peak),
        min_area_px=float(min_area),
        max_area_px=float(max_area) if max_area is not None else None,
        after_area_components=int(len(area_kept_ids)),
        typical_spacing_px=(
            float(array_filter.typical_spacing_px)
            if array_filter.typical_spacing_px is not None
            else None
        ),
        network_radius_px=(
            float(array_filter.network_radius_px)
            if array_filter.network_radius_px is not None
            else None
        ),
        final_components=int(len(network_kept_ids)),
        removed_small_components=small_removed,
        removed_large_components=large_removed,
        removed_network_components=network_removed,
        lattice_angle_deg=(
            array_filter.lattice_model.angle_deg if array_filter.lattice_model is not None else None
        ),
        lattice_pitch_px=(
            array_filter.lattice_model.pitch_px if array_filter.lattice_model is not None else None
        ),
        lattice_phase_tolerance_px=(
            array_filter.lattice_model.phase_tolerance_px
            if array_filter.lattice_model is not None
            else None
        ),
        lattice_candidate_components=array_filter.lattice_candidate_components,
        removed_lattice_components=array_filter.removed_lattice_components,
        removed_cluster_components=array_filter.removed_cluster_components,
        foreground_px_before=int(mask.sum()),
        foreground_px_after=int(cleaned.sum()),
    )

    if output_path is not None:
        save_binary_mask(cleaned, output_path)
    return cleaned, stats


def label_components(mask: np.ndarray, *, connectivity: int = 2) -> tuple[np.ndarray, int]:
    require_scipy()
    if connectivity == 1:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    elif connectivity == 2:
        structure = np.ones((3, 3), dtype=bool)
    else:
        raise ValueError("connectivity must be 1 or 2 for 2D masks.")
    labels, count = ndi.label(mask, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


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


def estimate_component_area_peak(areas: np.ndarray, *, bins: int = 48) -> float:
    """Estimate the dominant needle-dot area using an area-weighted log histogram."""

    valid = np.asarray(areas, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        return 0.0
    if valid.size < 8:
        return float(np.median(valid))

    log_area = np.log(valid)
    lower, upper = np.quantile(log_area, [0.02, 0.98])
    central = (log_area >= lower) & (log_area <= upper)
    if np.sum(central) >= 8:
        log_area = log_area[central]
        valid = valid[central]

    hist, edges = np.histogram(log_area, bins=bins, weights=valid)
    if hist.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        hist = np.convolve(hist, kernel / kernel.sum(), mode="same")
    peak_bin = int(np.argmax(hist))
    coarse_peak = float(np.exp((edges[peak_bin] + edges[peak_bin + 1]) / 2))

    near_peak = valid[(valid >= coarse_peak * 0.5) & (valid <= coarse_peak * 2.0)]
    if near_peak.size:
        return float(np.median(near_peak))
    return coarse_peak


def keep_array_network_components(
    component_ids: np.ndarray,
    centroids_yx: np.ndarray,
    *,
    config: MaskCleanConfig,
) -> ArrayFilterResult:
    """Keep components belonging to the main centroid-neighbor array network."""

    require_scipy()
    component_ids = np.asarray(component_ids, dtype=np.int32)
    centroids_yx = np.asarray(centroids_yx, dtype=np.float64)
    n_components = len(component_ids)
    if n_components < config.min_components_for_network:
        return ArrayFilterResult(
            component_ids=component_ids,
            typical_spacing_px=None,
            network_radius_px=None,
            lattice_model=None,
            lattice_candidate_components=None,
            removed_lattice_components=None,
            removed_cluster_components=None,
        )

    if config.use_lattice_filter:
        lattice_result = keep_lattice_components(component_ids, centroids_yx, config=config)
        if lattice_result is not None:
            return lattice_result

    return keep_density_network_components(component_ids, centroids_yx, config=config)


def keep_density_network_components(
    component_ids: np.ndarray,
    centroids_yx: np.ndarray,
    *,
    config: MaskCleanConfig,
) -> ArrayFilterResult:
    """Legacy broad-density filter used as a safe fallback."""

    tree = cKDTree(centroids_yx)
    n_components = len(component_ids)
    k = min(7, n_components)
    distances, _ = tree.query(centroids_yx, k=k)
    nearest = distances[:, 1:] if distances.ndim == 2 else distances
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        return ArrayFilterResult(
            component_ids=component_ids,
            typical_spacing_px=None,
            network_radius_px=None,
            lattice_model=None,
            lattice_candidate_components=None,
            removed_lattice_components=None,
            removed_cluster_components=None,
        )

    typical_spacing = float(np.median(nearest))
    radius = typical_spacing * config.network_radius_factor
    neighbor_lists = tree.query_ball_point(centroids_yx, r=radius)
    neighbor_counts = np.array([len(neighbors) - 1 for neighbors in neighbor_lists])
    active = neighbor_counts >= config.network_min_neighbors

    if np.sum(active) < config.min_components_for_network:
        kept_ids = component_ids[active]
        return ArrayFilterResult(
            component_ids=kept_ids,
            typical_spacing_px=typical_spacing,
            network_radius_px=radius,
            lattice_model=None,
            lattice_candidate_components=None,
            removed_lattice_components=None,
            removed_cluster_components=int(n_components - len(kept_ids)),
        )

    clusters = connected_clusters(neighbor_lists, active)
    if not clusters:
        kept_ids = component_ids[active]
        return ArrayFilterResult(
            component_ids=kept_ids,
            typical_spacing_px=typical_spacing,
            network_radius_px=radius,
            lattice_model=None,
            lattice_candidate_components=None,
            removed_lattice_components=None,
            removed_cluster_components=int(n_components - len(kept_ids)),
        )

    largest = max(len(cluster) for cluster in clusters)
    min_size = max(
        config.min_network_cluster_size,
        ceil(largest * config.keep_cluster_min_fraction_of_largest),
    )
    keep_indices = np.array(
        sorted(index for cluster in clusters if len(cluster) >= min_size for index in cluster),
        dtype=np.int32,
    )
    kept_ids = component_ids[keep_indices]
    return ArrayFilterResult(
        component_ids=kept_ids,
        typical_spacing_px=typical_spacing,
        network_radius_px=radius,
        lattice_model=None,
        lattice_candidate_components=None,
        removed_lattice_components=None,
        removed_cluster_components=int(n_components - len(kept_ids)),
    )


def keep_lattice_components(
    component_ids: np.ndarray,
    centroids_yx: np.ndarray,
    *,
    config: MaskCleanConfig,
) -> ArrayFilterResult | None:
    model = estimate_lattice_model(centroids_yx, config=config)
    if model is None:
        return None

    points_xy = centroids_yx[:, ::-1]
    phase_keep = lattice_phase_keep(points_xy, model)
    axial_support = count_axial_neighbors(points_xy, model, config=config)
    lattice_candidates = phase_keep | (axial_support >= config.lattice_min_axial_neighbors)
    if np.sum(lattice_candidates) < config.min_components_for_network:
        return None

    radius = model.pitch_px * config.network_radius_factor
    tree = cKDTree(points_xy)
    neighbor_lists = tree.query_ball_point(points_xy, r=radius)
    clusters = connected_clusters(neighbor_lists, lattice_candidates)
    if not clusters:
        return None

    largest = max(len(cluster) for cluster in clusters)
    min_size = max(
        config.min_network_cluster_size,
        ceil(largest * config.keep_cluster_min_fraction_of_largest),
    )
    keep_indices = np.array(
        sorted(index for cluster in clusters if len(cluster) >= min_size for index in cluster),
        dtype=np.int32,
    )
    min_final = max(config.min_components_for_network, int(len(component_ids) * config.lattice_min_final_fraction))
    if len(keep_indices) < min_final:
        return None

    kept_ids = component_ids[keep_indices]
    return ArrayFilterResult(
        component_ids=kept_ids,
        typical_spacing_px=model.pitch_px,
        network_radius_px=radius,
        lattice_model=model,
        lattice_candidate_components=int(np.sum(lattice_candidates)),
        removed_lattice_components=int(len(component_ids) - np.sum(lattice_candidates)),
        removed_cluster_components=int(np.sum(lattice_candidates) - len(keep_indices)),
    )


def estimate_lattice_model(
    centroids_yx: np.ndarray,
    *,
    config: MaskCleanConfig,
) -> LatticeModel | None:
    """Estimate a rotated square-lattice basis from local nearest-neighbor vectors."""

    points_xy = np.asarray(centroids_yx, dtype=np.float64)[:, ::-1]
    n_points = len(points_xy)
    if n_points < config.min_components_for_network:
        return None

    tree = cKDTree(points_xy)
    k = min(max(3, config.lattice_neighbor_k), n_points)
    distances, indices = tree.query(points_xy, k=k)
    if distances.ndim != 2 or distances.shape[1] < 2:
        return None

    pitch_seed = estimate_distance_peak(distances[:, 1 : min(k, 5)].ravel())
    if not np.isfinite(pitch_seed) or pitch_seed <= 0:
        return None

    vectors: list[np.ndarray] = []
    vector_distances: list[float] = []
    lower = pitch_seed * (1.0 - config.lattice_vector_distance_tolerance)
    upper = pitch_seed * (1.0 + config.lattice_vector_distance_tolerance)
    for source_index in range(n_points):
        for neighbor_slot in range(1, k):
            target_index = int(indices[source_index, neighbor_slot])
            distance = float(distances[source_index, neighbor_slot])
            if target_index <= source_index or not lower <= distance <= upper:
                continue
            vectors.append(points_xy[target_index] - points_xy[source_index])
            vector_distances.append(distance)

    if len(vectors) < config.min_components_for_network:
        return None

    vectors_xy = np.asarray(vectors, dtype=np.float64)
    vector_distances_arr = np.asarray(vector_distances, dtype=np.float64)
    angles = np.arctan2(vectors_xy[:, 1], vectors_xy[:, 0])
    distance_weights = np.exp(
        -0.5 * ((vector_distances_arr - pitch_seed) / max(1e-6, 0.25 * pitch_seed)) ** 2
    )
    theta = circular_histogram_mode(
        angles,
        period=np.pi / 2,
        bins=config.lattice_angle_bins,
        weights=distance_weights,
    )

    angle_delta = angle_distance_to_lattice_axes(angles, theta)
    aligned = (
        (angle_delta <= np.deg2rad(config.lattice_axis_angle_tolerance_deg))
        & (vector_distances_arr >= pitch_seed * 0.65)
        & (vector_distances_arr <= pitch_seed * 1.35)
    )
    pitch_values = vector_distances_arr[aligned] if np.any(aligned) else vector_distances_arr
    pitch = estimate_distance_peak(pitch_values)
    if not np.isfinite(pitch) or pitch <= 0:
        return None

    basis_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    basis_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    coord_u = points_xy @ basis_u
    coord_v = points_xy @ basis_v
    phase_u = circular_histogram_mode(coord_u, period=pitch, bins=config.lattice_phase_bins)
    phase_v = circular_histogram_mode(coord_v, period=pitch, bins=config.lattice_phase_bins)
    phase_tolerance = max(
        config.lattice_min_phase_tolerance_px,
        pitch * config.lattice_phase_tolerance,
    )
    return LatticeModel(
        angle_deg=float(np.rad2deg(theta) % 90.0),
        pitch_px=float(pitch),
        phase_u_px=float(phase_u),
        phase_v_px=float(phase_v),
        phase_tolerance_px=float(phase_tolerance),
        basis_u_xy=(float(basis_u[0]), float(basis_u[1])),
        basis_v_xy=(float(basis_v[0]), float(basis_v[1])),
    )


def estimate_distance_peak(values: np.ndarray, *, bins: int = 80) -> float:
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


def circular_histogram_mode(
    values: np.ndarray,
    *,
    period: float,
    bins: int,
    weights: np.ndarray | None = None,
) -> float:
    wrapped = np.mod(values, period)
    hist, edges = np.histogram(wrapped, bins=bins, range=(0.0, period), weights=weights)
    if hist.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        kernel = kernel / kernel.sum()
        hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], kernel, mode="same")[2:-2]
    peak_bin = int(np.argmax(hist))
    return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)


def angle_distance_to_lattice_axes(angles: np.ndarray, axis_angle: float) -> np.ndarray:
    period = np.pi / 2
    return np.abs(np.mod(angles - axis_angle + period / 2, period) - period / 2)


def circular_distance(values: np.ndarray, phase: float, period: float) -> np.ndarray:
    return np.abs(np.mod(values - phase + period / 2, period) - period / 2)


def lattice_phase_keep(points_xy: np.ndarray, model: LatticeModel) -> np.ndarray:
    basis_u = np.asarray(model.basis_u_xy, dtype=np.float64)
    basis_v = np.asarray(model.basis_v_xy, dtype=np.float64)
    residual_u = circular_distance(points_xy @ basis_u, model.phase_u_px, model.pitch_px)
    residual_v = circular_distance(points_xy @ basis_v, model.phase_v_px, model.pitch_px)
    return np.maximum(residual_u, residual_v) <= model.phase_tolerance_px


def count_axial_neighbors(
    points_xy: np.ndarray,
    model: LatticeModel,
    *,
    config: MaskCleanConfig,
) -> np.ndarray:
    basis_u = np.asarray(model.basis_u_xy, dtype=np.float64)
    basis_v = np.asarray(model.basis_v_xy, dtype=np.float64)
    pitch = model.pitch_px
    tree = cKDTree(points_xy)
    neighbor_lists = tree.query_ball_point(
        points_xy,
        r=pitch * (1.0 + config.lattice_axial_distance_tolerance),
    )
    support = np.zeros(len(points_xy), dtype=np.int32)
    max_parallel_delta = pitch * config.lattice_axial_distance_tolerance
    max_lateral_delta = pitch * config.lattice_axial_lateral_tolerance

    for point_index, neighbors in enumerate(neighbor_lists):
        directions = [False, False, False, False]
        for neighbor_index in neighbors:
            if neighbor_index == point_index:
                continue
            vector = points_xy[neighbor_index] - points_xy[point_index]
            along_u = float(vector @ basis_u)
            along_v = float(vector @ basis_v)
            if (
                abs(along_v) <= max_lateral_delta
                and abs(abs(along_u) - pitch) <= max_parallel_delta
            ):
                directions[0 if along_u > 0 else 1] = True
            if (
                abs(along_u) <= max_lateral_delta
                and abs(abs(along_v) - pitch) <= max_parallel_delta
            ):
                directions[2 if along_v > 0 else 3] = True
        support[point_index] = sum(directions)
    return support


def connected_clusters(neighbor_lists: Iterable[list[int]], active: np.ndarray) -> list[list[int]]:
    neighbor_lists = list(neighbor_lists)
    visited = np.zeros(len(neighbor_lists), dtype=bool)
    clusters: list[list[int]] = []

    for start, is_active in enumerate(active):
        if visited[start] or not is_active:
            continue
        stack = [start]
        visited[start] = True
        cluster: list[int] = []
        while stack:
            node = stack.pop()
            cluster.append(node)
            for neighbor in neighbor_lists[node]:
                if not active[neighbor] or visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        clusters.append(cluster)
    return clusters


def find_tif_mask_pairs(raw_dir: Path, *, mask_suffix: str = "_normalized_mask.png") -> list[tuple[Path, Path]]:
    raw_dir = Path(raw_dir)
    pairs: list[tuple[Path, Path]] = []
    for tif_path in sorted(raw_dir.glob("*.tif")):
        mask_path = tif_path.with_name(tif_path.name + mask_suffix)
        if mask_path.exists():
            pairs.append((tif_path, mask_path))
    return pairs


def preprocess_dataset(
    raw_dir: Path,
    out_dir: Path,
    *,
    config: MaskCleanConfig | None = None,
    channel_index: int = 0,
    overwrite: bool = True,
) -> list[PreprocessResult]:
    config = config or MaskCleanConfig()
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    pairs = find_tif_mask_pairs(raw_dir)
    if not pairs:
        raise FileNotFoundError(f"No tif/mask pairs found in {raw_dir}.")

    results: list[PreprocessResult] = []
    for tif_path, mask_path in pairs:
        sample_id = tif_path.stem
        image_out = image_dir / f"{sample_id}_channel{channel_index + 1}.tif"
        mask_out = mask_dir / f"{sample_id}_mask_clean.png"

        extract_tif_channel(
            tif_path,
            image_out,
            channel_index=channel_index,
            overwrite=overwrite,
        )
        _, stats = clean_mask(mask_path, mask_out, config=config)
        results.append(
            PreprocessResult(
                sample_id=sample_id,
                source_tif=str(tif_path),
                source_mask=str(mask_path),
                image_path=str(image_out),
                mask_path=str(mask_out),
                stats=stats,
            )
        )

    write_manifest(results, out_dir / "manifest.csv")
    write_summary(results, out_dir / "preprocess_summary.json", config=config)
    return results


def write_manifest(results: list[PreprocessResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_tif",
        "source_mask",
        "image_path",
        "mask_path",
        "total_components",
        "after_area_components",
        "final_components",
        "removed_small_components",
        "removed_network_components",
        "removed_lattice_components",
        "removed_cluster_components",
        "lattice_candidate_components",
        "lattice_angle_deg",
        "lattice_pitch_px",
        "lattice_phase_tolerance_px",
        "foreground_px_before",
        "foreground_px_after",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            stats = result.stats
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "source_tif": result.source_tif,
                    "source_mask": result.source_mask,
                    "image_path": result.image_path,
                    "mask_path": result.mask_path,
                    "total_components": stats.total_components,
                    "after_area_components": stats.after_area_components,
                    "final_components": stats.final_components,
                    "removed_small_components": stats.removed_small_components,
                    "removed_network_components": stats.removed_network_components,
                    "removed_lattice_components": stats.removed_lattice_components,
                    "removed_cluster_components": stats.removed_cluster_components,
                    "lattice_candidate_components": stats.lattice_candidate_components,
                    "lattice_angle_deg": stats.lattice_angle_deg,
                    "lattice_pitch_px": stats.lattice_pitch_px,
                    "lattice_phase_tolerance_px": stats.lattice_phase_tolerance_px,
                    "foreground_px_before": stats.foreground_px_before,
                    "foreground_px_after": stats.foreground_px_after,
                }
            )


def write_summary(
    results: list[PreprocessResult],
    output_path: Path,
    *,
    config: MaskCleanConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "n_samples": len(results),
        "samples": [
            {
                "sample_id": result.sample_id,
                "source_tif": result.source_tif,
                "source_mask": result.source_mask,
                "image_path": result.image_path,
                "mask_path": result.mask_path,
                "stats": asdict(result.stats),
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
