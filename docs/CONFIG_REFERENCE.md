# Config Reference

Needle Select uses a project-level TOML config to connect portable package commands with the legacy preprocessing, training, and inference configs.

Example files:

- `configs/example.project.toml`
- `configs/smoke.project.toml`
- generated `configs/needle_select_project.toml` inside `init-project` output

## `[project]`

- `name`: human-readable run name.

## `[paths]`

- `project_root`: root directory for this run. In configs stored under `configs/`, use `..`.
- `data_dir`: runtime data directory.
- `output_dir`: runtime output directory.
- `work_dir`: scratch/work directory.
- `raw_dir`: source raw image directory for preprocessing.
- `manifest`: manifest CSV path, usually `data/manifest.csv`.
- `splits`: split CSV path, usually `data/splits.csv`.
- `predictions_dir`: prediction/inference output path.
- `scripts_dir`: directory containing compatibility runner scripts.

Relative paths are resolved under `project_root`.

## `[configs]`

- `preprocess_config`: TOML consumed by `scripts/preprocess_raw_data.py`.
- `train_config`: TOML consumed by `scripts/make_splits.py`, `scripts/train_unet.py`, and environment checks.
- `inference_profile`: JSON profile consumed by profile-aware inference.

## `[model]`

- `checkpoint`: local model checkpoint for prediction/inference. Keep checkpoint files outside Git, usually under `models/`.

## `[inference]`

- `input`: file or directory for direct inference.
- `threshold`: probability threshold. Lower keeps more faint needles; higher reduces false positives.
- `patch_size`: sliding-window patch size. Existing models were trained around 256 or 512 pixel crops depending on the run config.
- `overlap`: sliding-window overlap fraction.
- `channel`: zero-based TIFF page/band index.
- `scale`: optional manual resize factor before inference.
- `input_magnification`: microscope magnification of the new input, such as `60.0`.
- `trained_magnification`: microscope magnification used to train the model, currently `40.0` for the main unified model.
- `recursive`: whether to search input directories recursively.

For 60x inputs using a 40x-trained model, a useful manual scale is approximately `40 / 60 = 0.6667`. The inference profile can also estimate scale automatically from lattice pitch or dot diameter.

## Inference Profile JSON

`configs/inference_profile_unified_v2.json` and generated `configs/inference_profile.json` describe:

- model-trained magnification
- expected lattice pitch in pixels
- expected dot diameter in pixels
- default threshold, patch size, overlap, and channel
- auto-scale search range and component filters
- recommended 40x, 60x, and auto presets
- tunable parameter names and CLI flags

This file is intended to travel with an exported model so other projects can read the model's recommended inference settings without reading source code.

## `[commands]`

- `python`: Python executable used by workflow plans. Use `python`, a venv path, or a conda environment Python path.
