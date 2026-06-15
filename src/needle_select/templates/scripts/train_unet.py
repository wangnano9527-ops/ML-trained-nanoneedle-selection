from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from needle_select.ml.config import load_toml, section
from needle_select.ml.data import NeedlePatchDataset, attach_splits, read_manifest
from needle_select.ml.losses import BCEDiceLoss, segmentation_metrics
from needle_select.ml.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net nano-needle mask model.")
    parser.add_argument("--config", default=Path("configs/train.toml"), type=Path)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        type=Path,
        help="Load model weights from an existing checkpoint before training.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    data_config = section(config, "data")
    patch_config = section(config, "patches")
    aug_config = section(config, "augmentation")
    model_config = section(config, "model")
    training_config = section(config, "training")
    output_config = section(config, "output")

    records = read_manifest(
        Path(data_config.get("manifest", "data/manifest.csv")),
        image_column=data_config.get("image_column", "image_path"),
        mask_column=data_config.get("mask_column", "mask_path"),
    )
    records = attach_splits(records, Path(data_config.get("splits", "data/splits.csv")))
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    if not train_records or not val_records:
        raise ValueError("Need both train and val samples. Run scripts/make_splits.py first.")

    device = choose_device(str(training_config.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training_config.get("cudnn_benchmark", True))

    seed = int(section(config, "split").get("seed", 42))
    cache_images = bool(data_config.get("cache_images", False))
    train_dataset = NeedlePatchDataset(
        train_records,
        patch_size=int(patch_config.get("patch_size", 512)),
        patches_per_image=int(patch_config.get("patches_per_image", 48)),
        positive_patch_fraction=float(patch_config.get("positive_patch_fraction", 0.70)),
        augment=bool(aug_config.get("enable", True)),
        intensity_jitter=float(aug_config.get("intensity_jitter", 0.10)),
        cache_images=cache_images,
        seed=seed,
    )
    val_dataset = NeedlePatchDataset(
        val_records,
        patch_size=int(patch_config.get("patch_size", 512)),
        patches_per_image=max(8, int(patch_config.get("patches_per_image", 48)) // 4),
        positive_patch_fraction=0.50,
        augment=False,
        cache_images=cache_images,
        seed=seed + 10_000,
    )

    num_workers = int(training_config.get("num_workers", 0))
    batch_size = int(training_config.get("batch_size", 4))
    pin_memory = bool(training_config.get("pin_memory", device.type == "cuda"))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training_config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(training_config.get("prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(model_config).to(device)
    checkpoint_config = section(config, "checkpoint")
    init_checkpoint = args.init_checkpoint or checkpoint_config.get("init")
    if init_checkpoint:
        load_model_weights(model, Path(init_checkpoint), device=device)

    criterion = BCEDiceLoss(
        bce_weight=float(training_config.get("bce_weight", 0.5)),
        dice_weight=float(training_config.get("dice_weight", 0.5)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    threshold = float(training_config.get("threshold", 0.5))

    run_dir = Path(output_config.get("run_dir", "runs/unet_baseline"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_dice = -1.0
    history: list[dict] = []
    epochs = int(training_config.get("epochs", 80))
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} train"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * images.shape[0]
        train_loss /= len(train_dataset)

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device=device, threshold=threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)
        print(row)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(run_dir / "best.pt", model, config, epoch, best_dice)
        if epoch % int(output_config.get("save_every_epochs", 10)) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, config, epoch, val_metrics["dice"])

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation dice: {best_dice:.4f}")
    print(f"Wrote checkpoints to {run_dir}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metric_sums = {"precision": 0.0, "recall": 0.0, "dice": 0.0}
    total = 0
    for batch in tqdm(loader, desc="val"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        metrics = segmentation_metrics(logits, masks, threshold=threshold)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size
        total += batch_size
    return total_loss / total, {key: value / total for key, value in metric_sums.items()}


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, epoch: int, dice: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "dice": dice,
        },
        path,
    )


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    print(f"Loaded initial model weights from {checkpoint_path}")


if __name__ == "__main__":
    main()
