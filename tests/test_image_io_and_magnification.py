from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from needle_select.image_io import load_image_projection
from needle_select.ml.inference_profile import (
    DEFAULT_PROFILE,
    resolve_magnification_scale,
)


def write_three_channel_tiff(path: Path) -> None:
    yy, xx = np.mgrid[:64, :64]
    data = np.stack(
        [
            xx.astype(np.float32),
            yy.astype(np.float32),
            ((xx - 32) ** 2 + (yy - 32) ** 2 < 100).astype(np.float32) * 100,
        ],
        axis=0,
    )
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX"}, photometric="minisblack")


def test_channel_projection_supports_single_max_and_sum(tmp_path: Path) -> None:
    path = tmp_path / "channels.tif"
    write_three_channel_tiff(path)

    single, single_info = load_image_projection(path, channel_mode="single", channel=1)
    maximum, max_info = load_image_projection(path, channel_mode="max")
    summed, sum_info = load_image_projection(path, channel_mode="sum")

    assert single.shape == maximum.shape == summed.shape == (64, 64)
    assert single_info.channel_count == max_info.channel_count == sum_info.channel_count == 3
    assert single_info.selected_channel == 1
    assert max_info.selected_channel is None
    assert not np.allclose(single, maximum)
    assert not np.allclose(maximum, summed)


def test_plain_multipage_tiff_is_treated_as_channels(tmp_path: Path) -> None:
    image = np.zeros((3, 24, 24), dtype=np.uint16)
    image[0, 2:5, 2:5] = 100
    image[1, 9:12, 9:12] = 200
    image[2, 16:19, 16:19] = 300
    path = tmp_path / "plain-pages.tif"
    tifffile.imwrite(path, image, photometric="minisblack")

    projection, info = load_image_projection(path, channel_mode="single", channel=1)

    assert info.source_axes == "CYX"
    assert info.channel_count == 3
    assert info.selected_channel == 1
    assert projection[10, 10] == 1.0
    assert projection[3, 3] == 0.0


def test_single_channel_rejects_out_of_range_index(tmp_path: Path) -> None:
    path = tmp_path / "channels.tif"
    write_three_channel_tiff(path)
    with pytest.raises(ValueError, match="3 channels"):
        load_image_projection(path, channel_mode="single", channel=3)


@pytest.mark.parametrize(
    ("pitch", "expected_magnification", "expected_scale"),
    [(16, 20.0, 2.0), (31, 40.0, 1.0), (47, 60.0, 2.0 / 3.0)],
)
def test_auto_magnification_maps_to_supported_values(
    pitch: int,
    expected_magnification: float,
    expected_scale: float,
) -> None:
    image = np.zeros((512, 512), dtype=np.float32)
    for y in range(pitch, 512 - pitch, pitch):
        for x in range(pitch, 512 - pitch, pitch):
            image[y - 6 : y + 7, x - 6 : x + 7] = 1.0

    resolution = resolve_magnification_scale(image, DEFAULT_PROFILE, auto_scale=True)

    assert resolution.selected_input_magnification == expected_magnification
    assert resolution.image_scale == pytest.approx(expected_scale)
    assert resolution.source.startswith("auto_pitch")


def test_explicit_magnification_must_be_20_40_or_60() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="20x, 40x, 60x"):
        resolve_magnification_scale(image, DEFAULT_PROFILE, input_magnification=10.0)

    resolution = resolve_magnification_scale(image, DEFAULT_PROFILE, input_magnification=20.0)
    assert resolution.image_scale == 2.0
    assert resolution.source == "explicit_magnification"
