"""
Compute local coordinate systems for triangular surfaces.

The local coordinate system for each triangle is computed as:
- normal: Unit normal to the triangle (V21 × V31)
- strike: eZ × normal (horizontal line in the plane), or eY * n[2] for horizontal planes
- dip: normal × strike (perpendicular to both)

The transformation matrix l2g = [normal, strike, dip] transforms local to global coordinates:
- toLocal(v): l2g.T @ v
- toGlobal(v): l2g @ v
"""

import numpy as np
from typing import Tuple

from .surface_data import SurfaceData


def compute_triangle_areas(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute the area of each triangle.

    Args:
        positions: Vertex positions, shape (n_vertices, 3)
        indices: Triangle indices, shape (n_triangles, 3)

    Returns:
        Triangle areas, shape (n_triangles,)
    """
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


def triangle_to_vertex_property(
    values: np.ndarray,
    indices: np.ndarray,
    n_vertices: int,
    weights: np.ndarray = None
) -> np.ndarray:
    """
    Convert per-triangle property values to per-vertex by weighted averaging.

    For each vertex, computes a weighted average of values from all triangles
    that share it. Vector properties are normalized after averaging.

    Args:
        values: Per-triangle values, shape (n_triangles,) or (n_triangles, k)
        indices: Triangle indices, shape (n_triangles, 3)
        n_vertices: Number of vertices in the mesh
        weights: Per-triangle weights (e.g., areas), shape (n_triangles,).
                 If None, uses uniform weights.

    Returns:
        Per-vertex values, shape (n_vertices,) or (n_vertices, k)
    """
    values = np.asarray(values)
    indices = np.asarray(indices)

    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)

    n_triangles = len(indices)

    if weights is None:
        weights = np.ones(n_triangles, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    is_vector = values.ndim == 2 and values.shape[1] > 1

    if is_vector:
        n_components = values.shape[1]
        vertex_values = np.zeros((n_vertices, n_components), dtype=np.float64)
    else:
        values = values.flatten()
        vertex_values = np.zeros(n_vertices, dtype=np.float64)

    # Sum of weights per vertex
    vertex_weights = np.zeros(n_vertices, dtype=np.float64)

    # Accumulate weighted values from each triangle to its vertices
    for tri_idx in range(n_triangles):
        w = weights[tri_idx]
        for vertex_idx in indices[tri_idx]:
            if is_vector:
                vertex_values[vertex_idx] += w * values[tri_idx]
            else:
                vertex_values[vertex_idx] += w * values[tri_idx]
            vertex_weights[vertex_idx] += w

    # Weighted average
    mask = vertex_weights > 0
    if is_vector:
        vertex_values[mask] /= vertex_weights[mask, np.newaxis]
        # Normalize vectors to maintain unit length
        norms = np.linalg.norm(vertex_values, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vertex_values = vertex_values / norms
    else:
        vertex_values[mask] /= vertex_weights[mask]

    return vertex_values


def compute_triangle_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute unit normal vectors for triangles.

    Args:
        positions: Vertex positions, shape (n_vertices, 3) or flat array
        indices: Triangle indices, shape (n_triangles, 3) or flat array

    Returns:
        Unit normals, shape (n_triangles, 3)
    """
    positions = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(indices)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)

    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    normals = np.cross(e1, e2)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)

    return normals / lengths


def compute_local_axes(
    positions: np.ndarray,
    indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local coordinate axes for each triangle.

    The local coordinate system is computed as:
    - normal: Unit normal (V21 × V31, normalized)
    - strike: eZ × normal (horizontal), or eY * n[2] for horizontal planes
    - dip: normal × strike

    For horizontal planes (normal ≈ vertical), strike = eY * n[2] to preserve
    orientation consistency.

    Args:
        positions: Vertex positions, shape (n_vertices, 3) or flat array
        indices: Triangle indices, shape (n_triangles, 3) or flat array

    Returns:
        Tuple of (dip, strike, normal), each shape (n_triangles, 3)
    """
    positions = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(indices)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)

    n_triangles = indices.shape[0]

    # Compute triangle normals
    normals = compute_triangle_normals(positions, indices)

    # Vertical direction (up)
    eZ = np.array([0.0, 0.0, 1.0])
    eY = np.array([0.0, 1.0, 0.0])

    # Strike = eZ × normal (horizontal line in the plane)
    strike = np.cross(eZ, normals)
    strike_lengths = np.linalg.norm(strike, axis=1, keepdims=True)

    # For horizontal planes (normal nearly vertical), strike is undefined
    # Use convention: strike = eY * n[2] (accounts for normal orientation)
    horizontal_threshold = 1e-6
    is_horizontal = strike_lengths.flatten() < horizontal_threshold

    # Normalize strike where valid
    strike_lengths = np.maximum(strike_lengths, 1e-10)
    strike = strike / strike_lengths

    # Set conventional axes for horizontal planes (multiply by n[2] for sign)
    strike[is_horizontal] = eY * normals[is_horizontal, 2:3]

    # Dip = normal × strike (points down-slope along the plane)
    dip = np.cross(normals, strike)

    # Normalize dip (should already be unit length, but ensure numerical stability)
    dip_lengths = np.linalg.norm(dip, axis=1, keepdims=True)
    dip_lengths = np.maximum(dip_lengths, 1e-10)
    dip = dip / dip_lengths

    return dip, strike, normals


def set_local_axes(
    surface: SurfaceData,
    dip_name: str = "dip_axis",
    strike_name: str = "strike_axis",
    normal_name: str = "normal_axis"
) -> SurfaceData:
    """
    Compute local coordinate axes and store them as properties on the SurfaceData.

    The local coordinate system is computed per-triangle and then averaged
    to vertices for storage. Vector properties are re-normalized after averaging.

    The local coordinate system for each triangle is:
    - normal: Unit normal (X-local in transformation matrix)
    - strike: Horizontal direction eZ × normal (Y-local)
    - dip: normal × strike (Z-local)

    Args:
        surface: SurfaceData to compute axes for.
        dip_name: Property name for dip axis.
        strike_name: Property name for strike axis.
        normal_name: Property name for normal axis.

    Returns:
        The same SurfaceData with added properties (modified in place).
    """
    if surface.indices is None:
        raise ValueError("SurfaceData must have triangle indices to compute local axes")

    positions = surface.get_positions_matrix()
    indices = surface.get_indices_matrix()
    n_vertices = len(positions)

    # Compute per-triangle values
    dip, strike, normal = compute_local_axes(positions, indices)

    # Compute triangle areas for weighted averaging
    areas = compute_triangle_areas(positions, indices)

    # Convert to per-vertex by area-weighted averaging (for TSurf PVRTX format)
    dip_vertex = triangle_to_vertex_property(dip, indices, n_vertices, weights=areas)
    strike_vertex = triangle_to_vertex_property(strike, indices, n_vertices, weights=areas)
    normal_vertex = triangle_to_vertex_property(normal, indices, n_vertices, weights=areas)

    surface.set_property(dip_name, dip_vertex, item_size=3)
    surface.set_property(strike_name, strike_vertex, item_size=3)
    surface.set_property(normal_name, normal_vertex, item_size=3)

    return surface
