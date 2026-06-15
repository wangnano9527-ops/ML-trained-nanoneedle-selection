"""Utilities for nano-needle image preprocessing.

The repository keeps this legacy top-level package for backward compatibility.
When running from the repository root, extend the package path to prefer the
new installable implementation under ``src/needle_select``.
"""

from pathlib import Path

_src_package = Path(__file__).resolve().parents[1] / "src" / "needle_select"
if _src_package.exists():
    __path__.insert(0, str(_src_package))

from .preprocess import MaskCleanConfig, PreprocessResult, clean_mask, preprocess_dataset
from .preprocess_parameters import ParameterSpec, describe_preprocess_parameters
from .project_api import (
    check_environment,
    describe_pipeline,
    describe_project,
    init_project,
    list_capabilities,
    list_public_steps,
    run_project,
)

__all__ = [
    "MaskCleanConfig",
    "ParameterSpec",
    "PreprocessResult",
    "check_environment",
    "clean_mask",
    "describe_pipeline",
    "describe_project",
    "describe_preprocess_parameters",
    "init_project",
    "list_capabilities",
    "list_public_steps",
    "preprocess_dataset",
    "run_project",
]
