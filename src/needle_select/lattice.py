from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LatticeModel:
    origin: tuple[float, float]
    vector_i: tuple[float, float]
    vector_j: tuple[float, float]
    rms_error: float


@dataclass(frozen=True)
class LatticeSnap:
    source_index: int
    node_i: int
    node_j: int
    seed_x: float
    seed_y: float
    center_x: float
    center_y: float
    distance_px: float
    accepted: bool


def fit_lattice(
    points_xy: np.ndarray,
    *,
    min_points: int = 6,
    neighbor_count: int = 4,
    iterations: int = 8,
) -> LatticeModel:
    """Fit an affine 2D lattice to detected point centers.

    The fitted grid is only used to regularize centers for already detected points.
    Missing lattice intersections are not generated here.
    """
    points = normalize_points(points_xy)
    if len(points) < min_points:
        raise ValueError(f"Need at least {min_points} points to fit a lattice; got {len(points)}.")

    vectors = nearest_neighbor_vectors(points, neighbor_count=neighbor_count)
    vector_i, vector_j = estimate_lattice_basis(vectors)
    origin, nodes, snapped = refine_lattice_origin(points, vector_i, vector_j, iterations=iterations)
    distances = np.linalg.norm(points - snapped, axis=1)

    min_node = nodes.min(axis=0)
    origin = origin + min_node[0] * vector_i + min_node[1] * vector_j
    snapped = snapped
    distances = np.linalg.norm(points - snapped, axis=1)
    return LatticeModel(
        origin=(float(origin[0]), float(origin[1])),
        vector_i=(float(vector_i[0]), float(vector_i[1])),
        vector_j=(float(vector_j[0]), float(vector_j[1])),
        rms_error=float(np.sqrt(np.mean(np.square(distances)))),
    )


def snap_points_to_lattice(
    points_xy: np.ndarray,
    model: LatticeModel,
    *,
    max_distance: float | None = None,
) -> list[LatticeSnap]:
    points = normalize_points(points_xy)
    vector_i = np.asarray(model.vector_i, dtype=np.float64)
    vector_j = np.asarray(model.vector_j, dtype=np.float64)
    origin = np.asarray(model.origin, dtype=np.float64)
    nodes, centers = assign_lattice_nodes(points, origin, vector_i, vector_j)
    distances = np.linalg.norm(points - centers, axis=1)
    if max_distance is None:
        max_distance = default_snap_distance(model)

    return [
        LatticeSnap(
            source_index=index,
            node_i=int(nodes[index, 0]),
            node_j=int(nodes[index, 1]),
            seed_x=float(points[index, 0]),
            seed_y=float(points[index, 1]),
            center_x=float(centers[index, 0]),
            center_y=float(centers[index, 1]),
            distance_px=float(distances[index]),
            accepted=bool(distances[index] <= max_distance),
        )
        for index in range(len(points))
    ]


def snap_points_to_lattice_network(
    points_xy: np.ndarray,
    model: LatticeModel,
    *,
    max_distance: float | None = None,
    min_line_points: int = 3,
) -> list[LatticeSnap]:
    """Snap points to intersections of fitted row/column network lines.

    The affine lattice provides node identities. Then each row and column is
    refit as its own line through observed points, which handles mild field
    curvature better than one global affine grid.
    """
    points = normalize_points(points_xy)
    vector_i = np.asarray(model.vector_i, dtype=np.float64)
    vector_j = np.asarray(model.vector_j, dtype=np.float64)
    origin = np.asarray(model.origin, dtype=np.float64)
    nodes, affine_centers = assign_lattice_nodes(points, origin, vector_i, vector_j)

    row_lines = fit_index_lines(
        points,
        nodes[:, 1],
        fallback_direction=vector_i,
        min_line_points=min_line_points,
    )
    column_lines = fit_index_lines(
        points,
        nodes[:, 0],
        fallback_direction=vector_j,
        min_line_points=min_line_points,
    )

    centers = []
    for index, (node_i, node_j) in enumerate(nodes):
        row = row_lines.get(int(node_j))
        column = column_lines.get(int(node_i))
        if row is None or column is None:
            centers.append(affine_centers[index])
            continue
        centers.append(intersect_lines(row[0], row[1], column[0], column[1], affine_centers[index]))
    centers = np.asarray(centers, dtype=np.float64)
    distances = np.linalg.norm(points - centers, axis=1)
    if max_distance is None:
        max_distance = default_snap_distance(model)

    return [
        LatticeSnap(
            source_index=index,
            node_i=int(nodes[index, 0]),
            node_j=int(nodes[index, 1]),
            seed_x=float(points[index, 0]),
            seed_y=float(points[index, 1]),
            center_x=float(centers[index, 0]),
            center_y=float(centers[index, 1]),
            distance_px=float(distances[index]),
            accepted=bool(distances[index] <= max_distance),
        )
        for index in range(len(points))
    ]


def default_snap_distance(model: LatticeModel, fraction: float = 0.35) -> float:
    spacing_i = np.linalg.norm(np.asarray(model.vector_i, dtype=np.float64))
    spacing_j = np.linalg.norm(np.asarray(model.vector_j, dtype=np.float64))
    return float(min(spacing_i, spacing_j) * fraction)


def normalize_points(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 point array, got shape {points.shape}.")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("No finite points were provided.")
    return points


def nearest_neighbor_vectors(points: np.ndarray, *, neighbor_count: int) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("Need at least two points to estimate neighbor vectors.")
    k = min(len(points), max(2, int(neighbor_count) + 1))
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(points).query(points, k=k)
        if indices.ndim == 1:
            indices = indices[:, None]
    except Exception:
        distances = np.sqrt(np.square(points[:, None, :] - points[None, :, :]).sum(axis=2))
        indices = np.argsort(distances, axis=1)[:, :k]

    vectors: list[np.ndarray] = []
    for source_index, row in enumerate(indices):
        for target_index in row:
            if int(target_index) == source_index:
                continue
            vector = points[int(target_index)] - points[source_index]
            if np.linalg.norm(vector) > 0:
                vectors.append(vector)
    if not vectors:
        raise ValueError("Could not estimate nonzero neighbor vectors.")
    return np.asarray(vectors, dtype=np.float64)


def estimate_lattice_basis(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 0
    vectors = vectors[keep]
    lengths = lengths[keep]
    if len(vectors) < 2:
        raise ValueError("Need at least two neighbor vectors to estimate a lattice basis.")

    base_spacing = float(np.percentile(lengths, 25))
    step_vectors = vectors[lengths <= base_spacing * 1.55]
    if len(step_vectors) < 2:
        step_vectors = vectors[lengths <= np.percentile(lengths, 50)]
    if len(step_vectors) < 2:
        step_vectors = vectors

    angles = canonical_angles(step_vectors)
    first = dominant_angle(angles)
    separation = np.asarray([angle_separation(angle, first) for angle in angles])
    second_candidates = separation > math.radians(35.0)
    if np.any(second_candidates):
        second = dominant_angle(angles[second_candidates])
    else:
        second = (first + math.pi / 2.0) % math.pi

    first = refine_angle(angles, first)
    second = refine_angle(angles, second)
    if angle_separation(first, second) < math.radians(25.0):
        second = (first + math.pi / 2.0) % math.pi

    vector_i = step_vector_for_angle(step_vectors, first)
    vector_j = step_vector_for_angle(step_vectors, second)
    if abs(cross2(vector_i, vector_j)) < 1e-6:
        raise ValueError("Estimated lattice vectors are degenerate.")
    if cross2(vector_i, vector_j) < 0:
        vector_j = -vector_j
    return vector_i, vector_j


def canonical_angles(vectors: np.ndarray) -> np.ndarray:
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    return np.mod(angles, math.pi)


def dominant_angle(angles: np.ndarray, *, bins: int = 90) -> float:
    hist, edges = np.histogram(angles, bins=bins, range=(0.0, math.pi))
    index = int(np.argmax(hist))
    return float((edges[index] + edges[index + 1]) / 2.0)


def refine_angle(angles: np.ndarray, target: float, *, window_degrees: float = 18.0) -> float:
    window = math.radians(window_degrees)
    keep = np.asarray([angle_separation(angle, target) <= window for angle in angles])
    if not np.any(keep):
        return target
    doubled = angles[keep] * 2.0
    mean = math.atan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean())) / 2.0
    return float(mean % math.pi)


def step_vector_for_angle(vectors: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    projections = vectors @ axis
    keep = np.abs(projections) > np.percentile(np.abs(projections), 60) * 0.5
    if not np.any(keep):
        keep = np.ones(len(vectors), dtype=bool)
    spacing = float(np.median(np.abs(projections[keep])))
    return axis * spacing


def angle_separation(a: float, b: float) -> float:
    diff = abs((a - b) % math.pi)
    return min(diff, math.pi - diff)


def refine_lattice_origin(
    points: np.ndarray,
    vector_i: np.ndarray,
    vector_j: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = points[np.argmin(points[:, 0] + points[:, 1])].copy()
    nodes = np.zeros((len(points), 2), dtype=np.int64)
    snapped = points.copy()
    for _ in range(max(1, int(iterations))):
        nodes, snapped = assign_lattice_nodes(points, origin, vector_i, vector_j)
        origin = np.mean(
            points - nodes[:, 0, None] * vector_i - nodes[:, 1, None] * vector_j,
            axis=0,
        )
    nodes, snapped = assign_lattice_nodes(points, origin, vector_i, vector_j)
    return origin, nodes, snapped


def assign_lattice_nodes(
    points: np.ndarray,
    origin: np.ndarray,
    vector_i: np.ndarray,
    vector_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = np.column_stack([vector_i, vector_j])
    coords = np.linalg.solve(basis, (points - origin).T).T
    nodes = np.rint(coords).astype(np.int64)
    centers = origin + nodes[:, 0, None] * vector_i + nodes[:, 1, None] * vector_j
    return nodes, centers


def fit_index_lines(
    points: np.ndarray,
    indices: np.ndarray,
    *,
    fallback_direction: np.ndarray,
    min_line_points: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    direction = unit_vector(fallback_direction)
    for index in sorted(set(int(value) for value in indices)):
        group = points[indices == index]
        if len(group) == 0:
            continue
        if len(group) < max(2, min_line_points):
            lines[index] = (group.mean(axis=0), direction)
            continue
        center = group.mean(axis=0)
        centered = group - center
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            line_direction = unit_vector(vh[0])
        except np.linalg.LinAlgError:
            line_direction = direction
        if np.dot(line_direction, direction) < 0:
            line_direction = -line_direction
        lines[index] = (center, line_direction)
    return lines


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def intersect_lines(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    denominator = cross2(direction_a, direction_b)
    if abs(denominator) < 1e-6:
        return fallback
    t = cross2(point_b - point_a, direction_b) / denominator
    return point_a + t * direction_a


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
