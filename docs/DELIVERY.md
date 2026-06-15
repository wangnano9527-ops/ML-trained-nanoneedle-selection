# Delivery And Migration

## What To Ship

For a reusable code delivery, ship:

- `pyproject.toml`
- `src/needle_select/`
- `scripts/`
- `configs/example.project.toml`
- `configs/smoke.project.toml`
- `configs/inference_profile_unified_v2.json`
- `docs/`
- `tests/`
- `requirements.txt`
- `requirements-ml.txt`
- root helper scripts: `run.*`, `doctor.*`, `status.*`

For a model delivery, also ship the checkpoint and its inference profile together:

- `models/<checkpoint>.pt`
- `configs/inference_profile_unified_v2.json`

Do not commit or package large local data and results unless explicitly creating an external artifact.

## What Not To Put In Git

Keep these out of Git:

- raw `.tif` and `.tiff` microscope files
- `raw data/`
- `data/`
- `New-raw/`
- `Training/`
- `runs/`
- `predictions/`
- `output/`
- `work/`
- `logs/`
- `models/`
- `dist/`
- model files: `.pt`, `.pth`, `.ckpt`
- generated `.zip` archives

The repository `.gitignore` includes these patterns.

## Moving To Another Computer

1. Clone or copy the repository.
2. Create a Python environment.
3. Install the package:

   ```powershell
   python -m pip install -e .
   ```

4. Install ML dependencies if training or GPU inference is needed:

   ```powershell
   python -m pip install -e ".[ml]"
   ```

5. Put local data and checkpoints into machine-local directories such as `data/` and `models/`.
6. Update `configs/example.project.toml` or a generated `configs/needle_select_project.toml`.
7. Run:

   ```powershell
   needle-select doctor --config configs\example.project.toml
   needle-select run --config configs\example.project.toml --dry-run
   ```

## Initializing A Clean Consumer Project

```powershell
needle-select init-project E:\target\needle-consumer
cd E:\target\needle-consumer
.\doctor.ps1
.\status.ps1
```

The initialized project includes small compatibility scripts, configs, and runtime folders. Add data and model checkpoints locally, then edit `configs/needle_select_project.toml`.

## External Software And Paths

The package itself requires Python. Training/inference requires PyTorch, and CUDA is optional but recommended for GPU acceleration.

Machine-specific items that must be configured after migration:

- raw data location
- output/work directories
- model checkpoint path
- Python executable or environment
- GPU/CUDA/PyTorch installation
- 40x/60x inference scale behavior if the microscope setup changes
