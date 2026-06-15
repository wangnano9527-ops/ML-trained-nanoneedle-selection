# Needle Select Run Directory

This directory was initialized by `needle-select init-project`.

## Quick Commands

```powershell
.\doctor.ps1
.\status.ps1
.\run.ps1 -DryRun
.\run.ps1 -Steps infer
```

The same commands are available as `.cmd` and `.sh` scripts for other shells.

## Layout

- `configs/needle_select_project.toml` is the portable project-level config.
- `configs/preprocess.toml` controls mask preprocessing.
- `configs/train.toml` controls split generation, training, and fine-tuning.
- `configs/inference_profile.json` stores inference defaults, scale estimation, threshold, pitch, and diameter settings.
- `scripts/` contains small legacy runner scripts copied for compatibility.
- `data/`, `output/`, `work/`, and `logs/` are local runtime directories and should normally stay out of Git.

## Migration Notes

Install the package in the Python environment first:

```powershell
pip install -e <path-to-Needle-select>
```

Then edit `configs/needle_select_project.toml` to point at your local data, model checkpoint, and desired inference profile. Keep large `.tif`, `.pt`, `.pth`, `.ckpt`, `data/`, `output/`, `runs/`, and `work/` artifacts outside Git.
