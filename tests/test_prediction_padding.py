from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from needle_select.ml.predict import predict_full_image


class IdentityLogits(torch.nn.Module):
    def forward(self, value):
        return value


@pytest.mark.parametrize("shape", [(1, 1), (64, 96), (128, 128), (200, 300)])
def test_predict_full_image_supports_inputs_smaller_than_patch(shape: tuple[int, int]) -> None:
    image = np.zeros(shape, dtype=np.float32)

    probability = predict_full_image(
        IdentityLogits(),
        image,
        patch_size=256,
        overlap=0.5,
        device=torch.device("cpu"),
    )

    assert probability.shape == shape
    assert np.allclose(probability, 0.5)
