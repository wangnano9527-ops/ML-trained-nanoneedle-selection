# Environment And Migration

The repository tracks code and configuration. Large/generated data are ignored by git:

- `raw data/`
- `data/`
- `.venv/`
- `runs/`
- `predictions/`

When moving to another machine, copy or recreate `raw data/`, then rerun preprocessing.

## Minimal Preprocessing Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\preprocess_raw_data.py --config configs/preprocess.toml
```

## Training Environment

PyTorch installation depends on CPU/GPU/CUDA. Use the official selector when possible:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

CPU-only fallback:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

For NVIDIA GPU, install the PyTorch wheel that matches the machine's CUDA driver, then install the remaining packages from `requirements-ml.txt`.

## Reproducible Project State

Recommended files to keep in git:

- `needle_select/`
- `scripts/`
- `configs/`
- `docs/`
- `README.md`
- `requirements*.txt`

Recommended files to keep outside git or in external storage:

- raw microscope TIFFs
- generated `data/`
- model checkpoints in `runs/`
- prediction masks

## What To Copy To Another Computer

Best practical option: copy the whole `Needle-select` folder, but do not rely on the copied `.venv`.

You can also create a transfer zip from the current machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_project.ps1 -Out Needle-select-transfer.zip
```

Add trained models and previous predictions only when needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_project.ps1 -IncludeRuns -IncludePredictions
```

Must have:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `requirements-ml.txt`
- `configs/`
- `docs/`
- `needle_select/`
- `scripts/`
- `raw data/` if you want to regenerate preprocessing
- `data/` if you want to train immediately from the already cleaned masks

Optional:

- `runs/` if you already trained models and want the checkpoints
- `predictions/` if you want previous prediction outputs

Can skip:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- any temporary zip/cache files

If copying by git only, remember that `raw data/` and `data/` are ignored. You must copy them separately or rerun preprocessing after adding raw data.

## First Commands On The Training Computer

```powershell
cd path\to\Needle-select
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe scripts\check_training_env.py --config configs\train.toml
```

If the check passes:

```powershell
.\.venv\Scripts\python.exe scripts\run_training_pipeline.py --config configs\train.toml
```

If you require GPU training, use:

```powershell
.\.venv\Scripts\python.exe scripts\check_training_env.py --config configs\train.toml --strict-gpu
```

## Environment Doctor

`scripts/check_training_env.py` checks:

- Python version
- required packages
- PyTorch installation
- CUDA availability and GPU memory when PyTorch can see CUDA
- `nvidia-smi` availability
- `data/manifest.csv`
- image/mask file existence
- `data/splits.csv`
- free disk space near `runs/`

It exits with code 0 when training can start, and non-zero when a required item is missing.

Machine-readable output:

```powershell
.\.venv\Scripts\python.exe scripts\check_training_env.py --json
```
