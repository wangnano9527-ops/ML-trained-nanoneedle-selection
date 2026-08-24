import numpy as np

from needle_select.lattice import fit_lattice, snap_points_to_lattice, snap_points_to_lattice_network


def test_fit_lattice_snaps_existing_points_without_filling_missing_nodes() -> None:
    rng = np.random.default_rng(42)
    origin = np.asarray([21.0, 17.0])
    vector_i = np.asarray([18.0, 3.0])
    vector_j = np.asarray([-4.0, 22.0])
    missing_node = (3, 2)

    true_points = []
    for i in range(7):
        for j in range(5):
            if (i, j) == missing_node:
                continue
            true_points.append(origin + i * vector_i + j * vector_j)
    true_points = np.asarray(true_points, dtype=np.float64)
    seeds = true_points + rng.normal(0.0, 0.35, size=true_points.shape)

    model = fit_lattice(seeds)
    snaps = snap_points_to_lattice(seeds, model, max_distance=2.0)

    accepted = [snap for snap in snaps if snap.accepted]
    snapped_points = np.asarray([[snap.center_x, snap.center_y] for snap in accepted])
    distances_to_true = np.sqrt(
        np.square(snapped_points[:, None, :] - true_points[None, :, :]).sum(axis=2)
    )
    missing_point = origin + missing_node[0] * vector_i + missing_node[1] * vector_j
    distances_to_missing = np.linalg.norm(snapped_points - missing_point, axis=1)

    assert len(accepted) == len(seeds)
    assert distances_to_true.min(axis=1).max() < 1.0
    assert distances_to_missing.min() > 4.0
    assert model.rms_error < 0.75


def test_lattice_network_lines_reduce_curved_grid_snap_error() -> None:
    points = []
    origin = np.asarray([30.0, 25.0])
    vector_i = np.asarray([18.0, 2.0])
    vector_j = np.asarray([-2.0, 20.0])
    normal_i = np.asarray([-vector_i[1], vector_i[0]]) / np.linalg.norm(vector_i)
    normal_j = np.asarray([-vector_j[1], vector_j[0]]) / np.linalg.norm(vector_j)
    for i in range(9):
        for j in range(7):
            row_shift = (j - 3) * 0.9 * normal_i
            column_shift = math_like_wave(i) * 1.1 * normal_j
            bend = row_shift + column_shift
            points.append(origin + i * vector_i + j * vector_j + bend)
    points = np.asarray(points, dtype=np.float64)

    model = fit_lattice(points)
    affine = snap_points_to_lattice(points, model)
    network = snap_points_to_lattice_network(points, model)

    affine_distances = np.asarray([snap.distance_px for snap in affine])
    network_distances = np.asarray([snap.distance_px for snap in network])

    assert np.percentile(network_distances, 95) < np.percentile(affine_distances, 95)
    assert network_distances.max() < affine_distances.max()


def math_like_wave(index: int) -> float:
    return [0.0, 1.0, -0.5, 1.5, -1.0, 0.75, -1.25, 0.5, 0.0][index]
