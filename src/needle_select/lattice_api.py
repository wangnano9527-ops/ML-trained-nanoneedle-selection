"""Supported lattice contract v1, backed by the unchanged Needle calculation core."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import lattice as _core
from .lattice import (
    LatticeModel, LatticeSnap, fit_lattice, snap_points_to_lattice,
    snap_points_to_lattice_network, default_snap_distance, normalize_points,
    nearest_neighbor_vectors, estimate_lattice_basis, canonical_angles,
    dominant_angle, refine_angle, step_vector_for_angle, angle_separation,
    refine_lattice_origin, assign_lattice_nodes, fit_index_lines, unit_vector,
    intersect_lines, cross2,
)

API_VERSION = 1
__all__ = [
    'LatticeModel', 'LatticeSnap', 'fit_lattice', 'snap_points_to_lattice',
    'snap_points_to_lattice_network', 'default_snap_distance', 'normalize_points',
    'nearest_neighbor_vectors', 'estimate_lattice_basis', 'canonical_angles',
    'dominant_angle', 'refine_angle', 'step_vector_for_angle', 'angle_separation',
    'refine_lattice_origin', 'assign_lattice_nodes', 'fit_index_lines', 'unit_vector',
    'intersect_lines', 'cross2', 'API_VERSION', 'lattice_source',
]


def lattice_source() -> dict:
    """Report the actual loaded calculation file, not an assumed checkout path."""
    path = Path(_core.__file__).resolve()
    return {'provider': 'needle-selection', 'capability': '3-4-a',
            'api_version': API_VERSION, 'module': _core.__name__,
            'module_file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
