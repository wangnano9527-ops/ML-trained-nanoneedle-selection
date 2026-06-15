# Needle Select

Machine-learning project for extracting nano-needle array masks from fluorescence images.

This repository now also exposes an installable package and CLI for reuse in other projects:

```powershell
python -m pip install -e .
needle-select describe
needle-select init-project E:\work\needle-run
needle-select run --config configs\example.project.toml --dry-run
```

Start with:

- `docs/OVERVIEW.md` for current structure and data flow.
- `docs/QUICKSTART.md` for install, CLI, and Python API usage.
- `docs/CONFIG_REFERENCE.md` for portable config fields.
- `docs/DELIVERY.md` for migration and handoff rules.

## Preprocessing

The raw folder contains paired files such as:

- `sample_01.tif`
- `sample_01.tif_normalized_mask.png`

Run preprocessing from the project root:

```powershell
python scripts/preprocess_raw_data.py --config configs/preprocess.toml
```

Useful tuning flags:

```powershell
python scripts/preprocess_raw_data.py --lattice-phase-tolerance 0.36 --lattice-min-axial-neighbors 2
```

Outputs:

- `data/images/<sample>_channel1.tif`
- `data/masks/<sample>_mask_clean.png`
- `data/manifest.csv`
- `data/preprocess_summary.json`

Parameter guide:

- Editable config: `configs/preprocess.toml`
- Detailed parameter notes: `docs/preprocess_parameters.md`
- Program-readable parameter specs: `needle_select.preprocess_parameters`
- CLI summary: `python scripts/show_preprocess_parameters.py`
- Automatically estimated values are recorded in `data/manifest.csv` and `data/preprocess_summary.json`

The preprocessing code is importable:

```python
from pathlib import Path
from needle_select.preprocess import MaskCleanConfig, preprocess_dataset

config = MaskCleanConfig(min_area_factor=0.45, network_radius_factor=2.4)
preprocess_dataset(Path("raw data"), Path("data"), config=config)
```

## Algorithm

Mask cleaning uses two reusable steps:

1. Estimate the main nano-needle point size from the area-weighted connected-component area peak, then remove components that are much smaller than that peak.
2. Estimate the rotated square-lattice axes and pitch from local nearest-neighbor vectors.
3. Remove points whose centroids fall off the lattice phase, which catches points in the gaps between nano-needles.
4. Keep edge and missing-neighbor cases when they are phase-aligned or have axial neighbors along the two perpendicular lattice axes, then remove isolated clusters far from the main array.

## Training

The first model is a binary U-Net segmentation baseline.

```powershell
python scripts/make_splits.py --config configs/train.toml
python scripts/train_unet.py --config configs/train.toml
python scripts/predict_masks.py --checkpoint runs/unet_baseline/best.pt --manifest data/manifest.csv --out-dir predictions
```

Training config:

- `configs/train.toml`
- Details: `docs/training_setup.md`

On a new training machine, first run:

```powershell
python scripts/check_training_env.py --config configs/train.toml
```

To let the project check, prepare splits, and start training:

```powershell
python scripts/run_training_pipeline.py --config configs/train.toml
```

Environment and migration notes:

- Minimal preprocessing dependencies: `requirements.txt`
- Training dependencies: `requirements-ml.txt`
- Migration guide: `docs/environment.md`
