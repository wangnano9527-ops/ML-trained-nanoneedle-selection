from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - depends on optional ML environment.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    mask_path: Path | None = None
    split: str | None = None


def read_manifest(
    manifest_path: Path,
    *,
    image_column: str = "image_path",
    mask_column: str = "mask_path",
) -> list[SampleRecord]:
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "data" else manifest_path.parent
    records: list[SampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = resolve_path(row[image_column], base_dir)
            mask_value = row.get(mask_column, "")
            mask_path = resolve_path(mask_value, base_dir) if mask_value else None
            records.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    image_path=image_path,
                    mask_path=mask_path,
                    split=row.get("split") or None,
                )
            )
    return records


def attach_splits(records: list[SampleRecord], splits_path: Path) -> list[SampleRecord]:
    split_by_id: dict[str, str] = {}
    with Path(splits_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_by_id[row["sample_id"]] = row["split"]
    return [
        SampleRecord(
            sample_id=record.sample_id,
            image_path=record.image_path,
            mask_path=record.mask_path,
            split=split_by_id.get(record.sample_id),
        )
        for record in records
    ]


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def robust_normalize(image: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image[np.isfinite(image)], [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32, copy=False)


def load_image(path: Path, *, cache: bool = False) -> np.ndarray:
    if cache:
        return _load_image_cached(str(Path(path).resolve()))
    return _load_image_uncached(path)


def _load_image_uncached(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return robust_normalize(np.asarray(image, dtype=np.float32))


@lru_cache(maxsize=128)
def _load_image_cached(path: str) -> np.ndarray:
    return _load_image_uncached(Path(path))


def load_mask(path: Path, *, cache: bool = False) -> np.ndarray:
    if cache:
        return _load_mask_cached(str(Path(path).resolve()))
    return _load_mask_uncached(path)


def _load_mask_uncached(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L")) > 0).astype(np.float32)


@lru_cache(maxsize=128)
def _load_mask_cached(path: str) -> np.ndarray:
    return _load_mask_uncached(Path(path))


class NeedlePatchDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        *,
        patch_size: int,
        patches_per_image: int,
        positive_patch_fraction: float,
        augment: bool,
        intensity_jitter: float = 0.0,
        cache_images: bool = False,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("NeedlePatchDataset requires at least one sample.")
        for record in records:
            if record.mask_path is None:
                raise ValueError(f"Training sample has no mask: {record.sample_id}")
        self.records = records
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.positive_patch_fraction = float(positive_patch_fraction)
        self.augment = bool(augment)
        self.intensity_jitter = float(intensity_jitter)
        self.cache_images = bool(cache_images)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if torch is None:
            raise RuntimeError(
                "NeedlePatchDataset requires PyTorch. Install training dependencies "
                "with `python -m pip install -r requirements-ml.txt`."
            )
        record = self.records[index % len(self.records)]
        rng = random.Random(self.seed + index * 1009)
        image = load_image(record.image_path, cache=self.cache_images)
        mask = load_mask(record.mask_path, cache=self.cache_images)
        image_patch, mask_patch = crop_patch(
            image,
            mask,
            self.patch_size,
            positive=rng.random() < self.positive_patch_fraction,
            rng=rng,
        )
        if self.augment:
            image_patch, mask_patch = augment_patch(
                image_patch,
                mask_patch,
                rng=rng,
                intensity_jitter=self.intensity_jitter,
            )
        return {
            "image": torch.from_numpy(image_patch[None, ...].copy()),
            "mask": torch.from_numpy(mask_patch[None, ...].copy()),
            "sample_id": record.sample_id,
        }


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    *,
    positive: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    patch_size = min(patch_size, height, width)

    if positive and np.any(mask > 0):
        ys, xs = np.nonzero(mask > 0)
        point_index = rng.randrange(len(ys))
        center_y = int(ys[point_index])
        center_x = int(xs[point_index])
    else:
        center_y = rng.randrange(height)
        center_x = rng.randrange(width)

    y0 = clamp(center_y - patch_size // 2, 0, height - patch_size)
    x0 = clamp(center_x - patch_size // 2, 0, width - patch_size)
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def augment_patch(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    rng: random.Random,
    intensity_jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    rotations = rng.randrange(4)
    if rotations:
        image = np.rot90(image, rotations)
        mask = np.rot90(mask, rotations)

    if intensity_jitter > 0:
        scale = 1.0 + rng.uniform(-intensity_jitter, intensity_jitter)
        offset = rng.uniform(-intensity_jitter, intensity_jitter)
        image = np.clip(image * scale + offset, 0.0, 1.0)

    return image.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)
