# Needle Select Quickstart

## Install

From this repository:

```powershell
python -m pip install -e .
```

For training or GPU inference:

```powershell
python -m pip install -e ".[ml]"
```

On this migrated machine, you can also run the root helper scripts without installing first:

```powershell
.\doctor.ps1
.\status.ps1
.\run.ps1 -DryRun
```

## Inspect The Package

```powershell
needle-select describe
needle-select describe --json
needle-select steps --json
needle-select capabilities --json
```

## Initialize A New Run Directory

```powershell
needle-select init-project E:\somewhere\needle-run
cd E:\somewhere\needle-run
.\doctor.ps1
.\status.ps1
```

The initialized directory includes small compatibility scripts and sample configs, but not data, model checkpoints, or old results.

## Run A Dry Plan

```powershell
needle-select plan --config configs\example.project.toml
needle-select run --config configs\example.project.toml --dry-run
```

To run only direct inference:

```powershell
needle-select run --config configs\example.project.toml --steps infer
```

## Call From Python

```python
from needle_select import describe_project, init_project, run_project

print(describe_project()["summary"])
init_project(r"E:\work\needle-run")
plan = run_project(r"E:\work\needle-run\configs\needle_select_project.toml", steps=["infer"], dry_run=True)
print(plan["steps"][0]["command"])
```

## Existing Legacy Flow

The original commands still work from the repository root:

```powershell
python scripts\preprocess_raw_data.py --config configs\preprocess.toml
python scripts\make_splits.py --config configs\train.toml
python scripts\train_unet.py --config configs\train.toml
python scripts\predict_masks.py --checkpoint runs\unet_baseline\best.pt --manifest data\manifest.csv --out-dir predictions
```

Use `scripts\infer_needles.py` for profile-aware direct inference:

```powershell
python scripts\infer_needles.py --checkpoint models\needle_unet_unified_v2.pt --input data\input --out-dir predictions --profile configs\inference_profile_unified_v2.json --recursive
```
