from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from needle_select.screening import screen_project


def test_screen_reports_operator_inputs_and_resolved_sample(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "model_registry" / "unified_v2").mkdir(parents=True)
    checkpoint = tmp_path / "model_registry" / "unified_v2" / "needle_unet_unified_v2.pt"
    checkpoint.write_bytes(b"screen-only-placeholder")

    image = np.zeros((3, 96, 96), dtype=np.float32)
    image[:, 10::20, 10::20] = 1.0
    tifffile.imwrite(
        tmp_path / "input" / "sample.tif",
        image,
        imagej=True,
        metadata={"axes": "CYX"},
        photometric="minisblack",
    )
    config = tmp_path / "configs" / "run.toml"
    config.write_text(
        """
[paths]
project_root = ".."
predictions_dir = "output/predictions"

[configs]
inference_profile = ""

[model]
checkpoint = "model_registry/unified_v2/needle_unet_unified_v2.pt"

[inference]
input = "input"
channel_mode = "max"
channel = 0
trained_magnification = 40.0
auto_magnification = true
recursive = false
""".strip(),
        encoding="utf-8",
    )

    result = screen_project(config, sample_limit=1)

    assert result["input_count"] == 1
    assert result["samples"][0]["projection"]["channel_count"] == 3
    assert result["samples"][0]["magnification"]["selected_input_magnification"] in {20.0, 40.0, 60.0}
    assert any("MAX" in item for item in result["operator_checklist"])
    assert "*_circle_mask.png" in result["outputs"]

    checkpoint.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:241fd18f3d01f921a00ca727ffea2c01b7b66e4e1ea5906c3b4cf3e07d99f1e9\n"
        "size 7745833\n",
        encoding="ascii",
    )
    pointer_result = screen_project(config, sample_limit=1)
    assert any("Git LFS pointer" in error for error in pointer_result["errors"])
