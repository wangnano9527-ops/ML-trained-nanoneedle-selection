"""Public preprocessing/geometry contract v1; no model or ROI policy replacement."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import inference as _core
from .inference import (
    extract_channel_yx, collapse_to_yx, robust_normalize, resolve_model_scale,
    resolve_expected_diameter, resize_float_image, filter_components_by_diameter,
    resolve_circle_min_radius, weighted_component_center, equivalent_diameters,
    erode_binary, validate_quantile, mean_or_none, median_or_none,
)

API_VERSION = 1
__all__ = ['extract_channel_yx', 'collapse_to_yx', 'robust_normalize', 'resolve_model_scale',
           'resolve_expected_diameter', 'resize_float_image', 'filter_components_by_diameter',
           'resolve_circle_min_radius', 'weighted_component_center', 'equivalent_diameters',
           'erode_binary', 'validate_quantile', 'mean_or_none', 'median_or_none',
           'API_VERSION', 'preprocessing_source']


def preprocessing_source() -> dict:
    path = Path(_core.__file__).resolve()
    return {'api_version':API_VERSION, 'provider':'needle-selection',
            'capability':'shared-needle-preprocessing/v1', 'module':_core.__name__,
            'module_file':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'helpers':__all__[:-2]}
