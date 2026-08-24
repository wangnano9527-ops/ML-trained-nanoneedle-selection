from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:  # Let doctor/screen report the missing package cleanly.
    tifffile = None

from .ml.data import robust_normalize


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CHANNEL_MODES = {"single", "max", "sum"}


@dataclass(frozen=True)
class ImageSeries:
    path: Path
    data: np.ndarray
    axes: str


@dataclass(frozen=True)
class ProjectionInfo:
    path: str
    source_shape: tuple[int, ...]
    source_axes: str
    channel_count: int
    channel_mode: str
    selected_channel: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source_shape": list(self.source_shape),
            "source_axes": self.source_axes,
            "channel_count": self.channel_count,
            "channel_mode": self.channel_mode,
            "selected_channel": self.selected_channel,
        }


def read_image_series(path: str | Path) -> ImageSeries:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        if tifffile is None:
            raise RuntimeError("TIFF input requires tifffile; install the Needle Select runtime dependencies.")
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            data = np.asarray(series.asarray())
            axes = normalize_axes(getattr(series, "axes", None), data)
        return ImageSeries(path, data, axes)

    with Image.open(path) as image:
        data = np.asarray(image)
    if data.ndim == 2:
        axes = "YX"
    elif data.ndim == 3 and data.shape[-1] <= 8:
        axes = "YXC"
    else:
        axes = infer_axes(data.shape)
    return ImageSeries(path, data, axes)


def load_image_projection(
    path: str | Path,
    *,
    channel_mode: str = "max",
    channel: int = 0,
) -> tuple[np.ndarray, ProjectionInfo]:
    mode = normalize_channel_mode(channel_mode)
    series = read_image_series(path)
    data = series.data.astype(np.float32, copy=False)
    axes = series.axes.upper()
    channel_axis = find_channel_axis(axes)

    if channel_axis is None:
        if mode == "single" and channel != 0:
            raise ValueError(f"{series.path} has one channel; channel={channel} is invalid.")
        projection = collapse_to_yx(data, axes)
        count = 1
        selected = 0 if mode == "single" else None
    else:
        axis_index = axes.index(channel_axis)
        count = int(data.shape[axis_index])
        channel_axes = axes[:axis_index] + axes[axis_index + 1 :]
        if mode == "single":
            if channel < 0 or channel >= count:
                raise ValueError(f"{series.path} has {count} channels; channel={channel} is invalid.")
            projection = collapse_to_yx(np.take(data, channel, axis=axis_index), channel_axes)
            selected = int(channel)
        else:
            planes = [
                collapse_to_yx(np.take(data, index, axis=axis_index), channel_axes)
                for index in range(count)
            ]
            stacked = np.stack(planes, axis=0)
            projection = np.max(stacked, axis=0) if mode == "max" else np.sum(stacked, axis=0)
            selected = None

    info = ProjectionInfo(
        path=str(series.path),
        source_shape=tuple(int(value) for value in data.shape),
        source_axes=axes,
        channel_count=count,
        channel_mode=mode,
        selected_channel=selected,
    )
    return robust_normalize(projection), info


def normalize_channel_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "channel":
        normalized = "single"
    if normalized not in CHANNEL_MODES:
        choices = ", ".join(sorted(CHANNEL_MODES))
        raise ValueError(f"Unsupported channel_mode={value!r}; expected one of: {choices}.")
    return normalized


def find_channel_axis(axes: str) -> str | None:
    if "C" in axes:
        return "C"
    if "S" in axes:
        return "S"
    return None


def collapse_to_yx(data: np.ndarray, axes: str) -> np.ndarray:
    axes_list = list(axes.upper())
    if "Y" not in axes_list or "X" not in axes_list:
        if data.ndim == 2:
            return data.astype(np.float32, copy=False)
        raise ValueError(f"Cannot collapse axes {axes!r} to YX for shape {data.shape}.")

    reduced = np.asarray(data, dtype=np.float32)
    extra_axes = [index for index, axis in enumerate(axes_list) if axis not in {"Y", "X"}]
    for axis_index in reversed(extra_axes):
        reduced = np.max(reduced, axis=axis_index)
        axes_list.pop(axis_index)
    if axes_list != ["Y", "X"]:
        reduced = np.transpose(reduced, [axes_list.index("Y"), axes_list.index("X")])
    if reduced.ndim != 2:
        raise ValueError(f"Expected a 2D YX projection, got shape {reduced.shape}.")
    return reduced.astype(np.float32, copy=False)


def normalize_axes(axes: str | None, data: np.ndarray) -> str:
    if axes:
        normalized = str(axes).upper()
        if len(normalized) == data.ndim:
            # Plain multi-page TIFF files are commonly reported as QYX/IYX
            # instead of CYX. A small unknown page axis is a channel axis;
            # explicit Z/T axes remain projection dimensions.
            if "C" not in normalized and "S" not in normalized:
                unknown_axes = [
                    index
                    for index, axis in enumerate(normalized)
                    if axis in {"I", "Q"} and int(data.shape[index]) <= 8
                ]
                if len(unknown_axes) == 1:
                    index = unknown_axes[0]
                    normalized = normalized[:index] + "C" + normalized[index + 1 :]
            return normalized
    return infer_axes(data.shape)


def infer_axes(shape: tuple[int, ...]) -> str:
    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        if shape[-1] <= 8:
            return "YXC"
        return "CYX" if shape[0] <= 8 else "ZYX"
    if len(shape) == 4:
        if shape[0] <= 8:
            return "CZYX"
        if shape[1] <= 8:
            return "ZCYX"
        return "TZYX"
    if len(shape) == 5:
        return "TZCYX"
    raise ValueError(f"Cannot infer image axes for shape {shape}.")


def safe_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return name[: -len(suffix)].replace(" ", "_")
    return Path(name).stem.replace(" ", "_")
