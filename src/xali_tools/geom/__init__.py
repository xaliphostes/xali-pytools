"""
Geometry utilities for triangular surfaces.

This module provides functions for computing geometric properties
of triangular meshes, including local coordinate systems.
"""

from .surface_data import SurfaceData
from .local_axes import (
    compute_local_axes,
    compute_triangle_areas,
    compute_triangle_normals,
    set_local_axes,
    triangle_to_vertex_property,
)

__all__ = [
    "SurfaceData",
    "compute_local_axes",
    "compute_triangle_areas",
    "compute_triangle_normals",
    "set_local_axes",
    "triangle_to_vertex_property",
]
