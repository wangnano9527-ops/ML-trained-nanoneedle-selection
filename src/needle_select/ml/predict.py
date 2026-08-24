from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    patch_size: int,
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    height, width = image.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))
    y_starts = starts_for_axis(height, patch_size, stride)
    x_starts = starts_for_axis(width, patch_size, stride)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for y0 in y_starts:
        for x0 in x_starts:
            patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            tensor = torch.from_numpy(patch[None, None, ...].astype(np.float32)).to(device)
            if pad_h or pad_w:
                # Reflection padding requires each pad size to be smaller than
                # its source dimension. Small crops can need an entire image-width
                # of padding, so use replication for those valid inference inputs.
                can_reflect = (
                    patch.shape[0] > 1
                    and patch.shape[1] > 1
                    and pad_h < patch.shape[0]
                    and pad_w < patch.shape[1]
                )
                mode = "reflect" if can_reflect else "replicate"
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            probs = probs[: patch.shape[0], : patch.shape[1]]
            prob_sum[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] += probs
            count[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] += 1.0

    return prob_sum / np.maximum(count, 1.0)


def starts_for_axis(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts
