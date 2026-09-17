from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pitch_2p5_candidate_registry_files_match_release_metadata() -> None:
    model_root = ROOT / "model_registry" / "pitch_2p5_v1"
    metadata = json.loads((model_root / "model_metadata.json").read_text(encoding="utf-8"))
    checkpoint = model_root / metadata["checkpoint"]
    profile = model_root / metadata["inference_profile"]

    assert metadata["status"] == "pilot-specialized"
    assert metadata["generalization_claimed"] is False
    assert _sha256(checkpoint) == metadata["checkpoint_sha256"]
    assert _sha256(profile) == metadata["inference_profile_sha256"]
