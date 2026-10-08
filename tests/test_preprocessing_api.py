import importlib.util
from needle_select import inference

NAMES = ('extract_channel_yx', 'collapse_to_yx', 'robust_normalize', 'resolve_model_scale',
         'resolve_expected_diameter', 'resize_float_image', 'filter_components_by_diameter',
         'resolve_circle_min_radius', 'weighted_component_center', 'equivalent_diameters',
         'erode_binary', 'validate_quantile', 'mean_or_none', 'median_or_none')


def test_public_preprocessing_aliases_existing_core_without_copying_math():
    assert importlib.util.find_spec('needle_select.preprocessing_api'), 'Shared preprocessing API is required'
    from needle_select import preprocessing_api
    assert preprocessing_api.API_VERSION == 1
    assert all(getattr(preprocessing_api, name) is getattr(inference, name) for name in NAMES)
    assert set(NAMES) <= set(preprocessing_api.__all__)
