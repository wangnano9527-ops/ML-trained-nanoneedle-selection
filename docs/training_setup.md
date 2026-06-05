# Model Training Setup

The first model should be a binary semantic-segmentation model:

- **Input**: `data/images/*_channel1.tif`
- **Target**: `data/masks/*_mask_clean.png`
- **Output**: one probability mask per image, later thresholded to a binary mask.

## Recommended Pipeline

1. Run preprocessing to create channel1 images and cleaned masks.
2. Create train/val/test splits from `data/manifest.csv`.
3. Train on random patches instead of full 2272 x 2272 images.
4. Validate on held-out full samples.
5. Predict full-size masks with sliding-window inference.

## Why Patches

The images are large and the dots are sparse. Patch training gives more batches from 40 images, fits on smaller GPUs/CPUs, and allows biased sampling toward patches that contain positive mask pixels.

Current defaults:

- patch size: `512`
- patches per image per epoch: `48`
- positive patch fraction: `0.70`
- model: small U-Net
- loss: BCEWithLogits + Dice

## Commands

```powershell
python scripts/check_training_env.py --config configs/train.toml
python scripts/make_splits.py --config configs/train.toml
python scripts/train_unet.py --config configs/train.toml
python scripts/predict_masks.py --checkpoint runs/unet_baseline/best.pt --manifest data/manifest.csv --out-dir predictions
```

Or run the guarded pipeline:

```powershell
python scripts/run_training_pipeline.py --config configs/train.toml
```

## Data Handling Notes

- Normalize each image by robust percentiles inside the dataset loader, not by saving modified TIFFs.
- Keep image/mask pairing through `manifest.csv`; do not rely on sorting two folders independently.
- Split by whole image, not by patch, to avoid leakage between train and validation.
- If later you add more manual labels, rerun preprocessing and regenerate splits or append rows carefully.

## First Experiments

Start with the default `configs/train.toml`. If the model misses dots, increase positive patch fraction or train longer. If it overpredicts background noise, lower the inference threshold only after checking validation precision/recall; otherwise use more hard-negative patches.
