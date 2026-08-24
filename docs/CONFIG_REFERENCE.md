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

- `checkpoint`: model checkpoint for prediction/inference. The maintained default is `model_registry/unified_v2/needle_unet_unified_v2.pt`, tracked with Git LFS.

## `[inference]`

- `input`: file or directory for direct inference.
- `threshold`: probability threshold. Lower keeps more faint needles; higher reduces false positives.
- `patch_size`: sliding-window patch size. Existing models were trained around 256 or 512 pixel crops depending on the run config.
- `overlap`: sliding-window overlap fraction.
- `channel_mode`: `single`, `max`, or `sum`. `max` is the safe default when the signal channel is unknown.
- `channel`: zero-based TIFF page/band index, used only with `channel_mode = "single"`.
- `scale`: optional manual resize factor before inference.
- `input_magnification`: optional microscope magnification; accepted values are exactly `20.0`, `40.0`, or `60.0`.
- `trained_magnification`: microscope magnification used to train the model, currently `40.0` for the main unified model.
- `auto_magnification`: estimate image geometry when magnification is omitted, then map it to 20x, 40x, or 60x.
- `clean_components`, `min_diameter_ratio`, `max_diameter_ratio`: expected-diameter component filtering.
- `save_circle_rois`: write circular ROI CSV/mask/overlay outputs.
- `circle_center_mode`: `lattice` corrects detected centers against the fitted grid; `component` keeps measured centers.
- `circle_radius_mode`: `global-quantile` calibrates a shared dataset radius while excluding anomalous images; `image-max` stays image-local.
- `circle_component_coverage_quantile`, `circle_global_radius_quantile`: component and dataset radius quantiles.
- `lattice_min_points`, `lattice_max_snap_distance`: center-correction acceptance settings.
- `recursive`: whether to search input directories recursively.

The exact model-space scales are 2.0 for 20x, 1.0 for 40x, and approximately 0.6667 for 60x. Manual `scale` remains an advanced compatibility override; normal operation should use or estimate magnification.

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
