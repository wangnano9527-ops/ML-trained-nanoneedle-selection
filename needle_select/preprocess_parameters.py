from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    group: str
    default: str
    user_level: str
    description: str
    tune_lower: str = ""
    tune_higher: str = ""


AUTO_ESTIMATED_PARAMETERS = [
    ParameterSpec(
        name="area_peak_px",
        group="automatic",
        default="per image",
        user_level="read_only",
        description="Dominant connected-component area peak for real nano-needle dots.",
    ),
    ParameterSpec(
        name="min_area_px",
        group="automatic",
        default="area_peak_px * min_area_factor",
        user_level="read_only",
        description="Actual small-component cutoff used for that image.",
    ),
    ParameterSpec(
        name="lattice_angle_deg",
        group="automatic",
        default="per image",
        user_level="read_only",
        description="Estimated array rotation angle modulo 90 degrees.",
    ),
    ParameterSpec(
        name="lattice_pitch_px",
        group="automatic",
        default="per image",
        user_level="read_only",
        description="Estimated center-to-center nano-needle spacing.",
    ),
    ParameterSpec(
        name="lattice_phase_tolerance_px",
        group="automatic",
        default="max(4 px, lattice_pitch_px * lattice_phase_tolerance)",
        user_level="read_only",
        description="Actual grid residual tolerance after pitch estimation.",
    ),
]


MANUAL_PARAMETERS = [
    ParameterSpec(
        name="channel",
        group="manual",
        default="0",
        user_level="basic",
        description="Zero-based TIFF page/channel index. channel1 is 0.",
    ),
    ParameterSpec(
        name="min_area_factor",
        group="manual",
        default="0.45",
        user_level="basic",
        description="Fraction of the automatic dot-area peak used as the minimum component area.",
        tune_lower="Keep more small real needles.",
        tune_higher="Remove more tiny debris.",
    ),
    ParameterSpec(
        name="lattice_phase_tolerance",
        group="manual",
        default="0.36",
        user_level="basic",
        description="Allowed off-grid residual as a fraction of the automatic pitch.",
        tune_lower="Remove more points in lattice gaps.",
        tune_higher="Keep more warped edge points.",
    ),
    ParameterSpec(
        name="lattice_min_axial_neighbors",
        group="manual",
        default="2",
        user_level="basic",
        description="Minimum up/down/left/right lattice-axis neighbors that can rescue a point.",
        tune_lower="Keep more edge or missing-neighbor points.",
        tune_higher="Remove more unsupported isolated points.",
    ),
    ParameterSpec(
        name="use_lattice_filter",
        group="manual",
        default="true",
        user_level="basic",
        description="Use lattice-aware filtering instead of only broad density-network cleanup.",
    ),
]


ADVANCED_PARAMETERS = [
    ParameterSpec(
        name="lattice_axis_angle_tolerance_deg",
        group="advanced",
        default="18.0",
        user_level="advanced",
        description="Angular window used when refining the two perpendicular lattice axes.",
    ),
    ParameterSpec(
        name="lattice_axial_distance_tolerance",
        group="advanced",
        default="0.32",
        user_level="advanced",
        description="Allowed distance error for one-pitch axial neighbor support.",
    ),
    ParameterSpec(
        name="lattice_axial_lateral_tolerance",
        group="advanced",
        default="0.28",
        user_level="advanced",
        description="Allowed sideways offset for axial neighbor support.",
    ),
    ParameterSpec(
        name="network_radius_factor",
        group="advanced",
        default="2.4",
        user_level="advanced",
        description="Final connected-cluster cleanup radius in pitch units.",
    ),
]


def describe_preprocess_parameters() -> list[ParameterSpec]:
    return AUTO_ESTIMATED_PARAMETERS + MANUAL_PARAMETERS + ADVANCED_PARAMETERS

