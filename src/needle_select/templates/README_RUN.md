# Needle Select Run Directory

This directory was initialized by `needle-select init-project`.

## Quick Commands

```powershell
.\screen.ps1
.\doctor.ps1
.\status.ps1
.\run.ps1 -DryRun
.\run.ps1 -Steps infer
```

Run `screen` before the first inference run and whenever the input microscope or channel layout changes. It reports the model path, detected channels, channel projection, estimated or supplied 20x/40x/60x magnification, scale, required operator confirmations, and output files. The same commands are available as `.cmd` and `.sh` scripts for other shells.

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

Then edit `configs/needle_select_project.toml` to point at your local data and unified v2 model checkpoint. Choose `channel_mode = "single"`, `"max"`, or `"sum"`; set `input_magnification` to 20, 40, or 60 when known, otherwise leave it unset for automatic estimation and mapping. Keep large runtime artifacts outside Git unless they are explicitly managed with Git LFS.
