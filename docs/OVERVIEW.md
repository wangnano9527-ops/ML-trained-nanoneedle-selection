# Needle Select Overview

Needle Select is a nano-needle mask segmentation project. The repository now supports two use modes:

- Legacy scripts in `scripts/` for the original preprocessing, training, prediction, and monitoring workflow.
- Installable Python package in `src/needle_select/` with stable public APIs, CLI commands, project templates, and portable configuration.

## Current Entry Points

- `scripts/preprocess_raw_data.py`: converts paired raw `.tif` images and manual/normalized masks into cleaned images, masks, manifest, and summary files.
- `scripts/make_splits.py`: creates train/validation/test splits from `data/manifest.csv`.
- `scripts/train_unet.py`: trains or fine-tunes the U-Net model from a training TOML file.
- `scripts/predict_masks.py`: runs checkpoint prediction over manifest rows.
- `scripts/infer_needles.py`: runs direct image/folder inference with threshold, patch, scale, magnification, and inference profile settings.
- `scripts/check_training_env.py`: checks Python, CUDA/PyTorch, configured paths, and training readiness.
- `scripts/run_training_pipeline.py`: runs the legacy check/split/train flow.
- `needle-select`: installable CLI entry point from `src/needle_select/cli.py`.

## Core Modules

- `needle_select.preprocess`: reusable mask cleaning and preprocessing pipeline.
- `needle_select.preprocess_parameters`: program-readable parameter metadata for preprocessing.
- `needle_select.ml.config`: training configuration loader.
- `needle_select.ml.data`: image/mask dataset and patch sampling utilities.
- `needle_select.ml.model`: U-Net model implementation.
- `needle_select.ml.losses`: segmentation losses and metrics.
- `needle_select.ml.inference_profile`: profile-aware inference settings and auto-scale helpers.
- `needle_select.image_io`: TIFF axes handling and single/MAX/SUM channel projection.
- `needle_select.inference`: unified-v2 inference, diameter filtering, and circular ROI outputs.
- `needle_select.lattice`: affine/network lattice fitting and center correction.
- `needle_select.screening`: operator pre-run validation and sample setting resolution.
- `needle_select.config`: portable project-level TOML config parser.
- `needle_select.runner`: builds and executes workflow plans.
- `needle_select.project_api`: stable public API for other projects.
- `needle_select.cli`: command-line interface.

## Data Flow

| Step | Input | Output |
| --- | --- | --- |
| `preprocess` | raw `.tif` files and paired manual/normalized masks | `data/images/`, `data/masks/`, `data/manifest.csv`, `data/preprocess_summary.json` |
| `make-splits` | `data/manifest.csv` | `data/splits.csv` |
| `train` | manifest, splits, training config, optional checkpoint | run directory with checkpoints, `config.json`, `history.json` |
| `predict` | checkpoint and manifest | prediction masks/probability images under configured output directory |
| `screen` | project config, checkpoint, sample inputs | readiness, channel interpretation, 20x/40x/60x mapping, operator checklist |
| `infer` | unified-v2 checkpoint, image or folder, inference profile | masks/probability/overlays, ROI CSV, circle masks/overlays, metrics and radius summary |

## Dependencies

Minimal preprocessing dependencies are in `requirements.txt`:

- `numpy`
- `Pillow`
- `scipy`
- `tifffile`

Training and inference dependencies are in `requirements-ml.txt`:

- `torch`
- `tqdm`

The installable package declares the minimal dependencies and exposes an optional `ml` extra for the training stack.

## Runtime Directories

These directories are generated or machine-local and should normally not be committed:

- `raw data/`
- `data/`
- `Training/`
- `runs/`
- `predictions/`
- `output/`
- `work/`
- `logs/`
- `models/`
- `dist/`
- `New-raw/`

## Public API Candidates

The stable API surface is intentionally small:

- `describe_project()`
- `describe_pipeline()`
- `list_capabilities()`
- `list_public_steps()`
- `init_project(target_dir, overwrite=False)`
- `run_project(config_path, steps=None, dry_run=False)`
- `check_environment(config_path=None)`
- `screen_project(config_path, sample_limit=3)`

Old scripts remain available for compatibility, but new integrations should call the package API or CLI.
