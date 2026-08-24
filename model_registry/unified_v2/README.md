# Needle Select Unified V2 Model

This directory is the runtime model registry for Needle Select. The project uses one
checkpoint by default:

```text
model_registry/unified_v2/needle_unet_unified_v2.pt
```

The checkpoint is tracked with Git LFS. After cloning, run `git lfs pull` before
inference. `model_metadata.json` records the checkpoint SHA-256, architecture,
training lineage, and evaluation metrics.

## Why V2 Is The Default

Unified v2 was fine-tuned from the earlier unified model with 25 unique manually
corrected 256x256 smoke tiles. On those 25 tiles, Dice improved from `0.789529` to
`0.916834`. The v2 manual test split Dice was `0.851392`.

The CSV files in this directory retain full comparison results. Experimental
manual-holdout and manual-fit checkpoints are not distributed here and are not used
at runtime; this keeps model selection unambiguous.

## Runtime Behavior

- Input channel mode: `single`, `max`, or `sum`; default is `max`.
- Supported input magnifications: 20x, 40x, and 60x only.
- Known magnification is used directly.
- Unknown magnification is estimated from lattice pitch, then dot diameter, and
  mapped to the nearest supported value.
- Model-space scales for a 40x-trained model are 2.0, 1.0, and 0.6667.
- Postprocessing includes diameter filtering, circular ROIs, lattice-center
  correction, global radius calibration, CSV summaries, masks, and overlays.

## Recommended Run

Edit `configs/example.project.toml`, then inspect the resolved inputs before running:

```powershell
needle-select screen --config configs\example.project.toml
needle-select run --config configs\example.project.toml --steps infer
```

For direct script use:

```powershell
python scripts\infer_needles.py --input D:\your_images --out-dir D:\needle_predictions --recursive
```

Use `--channel-mode single --channel 1` for a known zero-based channel, or
`--channel-mode sum` when signal is distributed across all channels. Supply
`--input-magnification 20`, `40`, or `60` when known.

See `docs/INFERENCE_SCREEN.md` for the operator checklist and output contract.
