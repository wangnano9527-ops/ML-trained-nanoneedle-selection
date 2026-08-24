from pathlib import Path

import numpy as np
from PIL import Image

from needle_select.inference import (
    circle_rois_to_rows,
    choose_global_circle_radius,
    extract_circle_rois,
    make_circle_mask,
    regularize_circle_rois,
    save_circle_outputs,
    write_circle_rois_csv,
)


def test_extract_circle_rois_uses_weighted_center_and_min_radius() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:2, 0:2] = True
    mask[9:12, 14:17] = True
    weights = np.ones_like(mask, dtype=np.float32)
    weights[10, 15] = 20.0

    rois = extract_circle_rois(
        mask,
        weight_image=weights,
        radius_padding=2.0,
        min_radius=5.0,
        min_area=4,
    )

    assert len(rois) == 2
    edge_roi = next(roi for roi in rois if roi.touches_edge)
    center_roi = next(roi for roi in rois if not roi.touches_edge)
    assert edge_roi.component_area_px == 4
    assert center_roi.component_area_px == 9
    assert center_roi.center_x == 15.0
    assert center_roi.center_y == 10.0
    assert center_roi.radius_px == 5.0


def test_make_circle_mask_covers_source_components() -> None:
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:9, 7:10] = True

    rois = extract_circle_rois(mask, radius_padding=1.0, min_radius=4.0)
    circle_mask = make_circle_mask(mask.shape, rois)

    assert np.all(circle_mask[mask])
    assert circle_mask.sum() > mask.sum()


def test_regularize_circle_rois_can_snap_to_lattice_and_use_uniform_radius() -> None:
    mask = np.zeros((96, 96), dtype=bool)
    for row in range(4):
        for col in range(5):
            if (row, col) == (2, 3):
                continue
            y = 12 + row * 18
            x = 11 + col * 16
            mask[y - 1 : y + 2, x - 1 : x + 2] = True

    rois = extract_circle_rois(mask, min_radius=3.0)
    source_radii = [roi.radius_px for roi in rois]
    snapped = regularize_circle_rois(
        rois,
        center_mode="lattice",
        uniform_radius=False,
        lattice_min_points=6,
        lattice_max_snap_distance=2.0,
    )
    regularized = regularize_circle_rois(
        rois,
        center_mode="lattice",
        uniform_radius=True,
        lattice_min_points=6,
        lattice_max_snap_distance=2.0,
    )
    circle_mask = make_circle_mask(mask.shape, regularized)

    assert len(snapped) == len(rois)
    assert source_radii
    assert np.all(make_circle_mask(mask.shape, snapped)[mask])
    assert len(regularized) == len(rois)
    assert np.all(circle_mask[mask])
    assert {roi.center_mode for roi in regularized} == {"lattice"}
    assert len({roi.radius_px for roi in regularized}) == 1
    assert all(roi.lattice_node_i is not None for roi in regularized)
    assert all(roi.lattice_snap_distance_px is not None for roi in regularized)


def test_circle_roi_csv_and_outputs_are_written(tmp_path: Path) -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    image[8:12, 8:12] = 1.0
    mask = image > 0
    rois = extract_circle_rois(mask, min_radius=4.0)
    rows = circle_rois_to_rows(rois, image_path=tmp_path / "image.tif", stem="image")

    write_circle_rois_csv(tmp_path / "image_circle_rois.csv", rows)
    save_circle_outputs(tmp_path, "image", image, mask, rois, make_circle_mask(mask.shape, rois))

    csv_text = (tmp_path / "image_circle_rois.csv").read_text(encoding="utf-8")
    assert "center_x" in csv_text
    assert "radius_px" in csv_text
    assert "required_radius_px" in csv_text
    assert (tmp_path / "image_circle_mask.png").exists()
    assert (tmp_path / "image_circle_overlay.png").exists()
    with Image.open(tmp_path / "image_circle_overlay.png") as overlay:
        assert overlay.size == (20, 20)


def test_global_circle_radius_excludes_anomaly_from_calibration() -> None:
    radius, flags, reasons = choose_global_circle_radius(
        [10.0, 10.5, 11.0, 60.0],
        global_quantile=0.99,
        anomaly_ratio=1.5,
        anomaly_mad_threshold=6.0,
    )

    assert radius is not None
    assert radius < 12.0
    assert flags == [False, False, False, True]
    assert reasons[-1]
