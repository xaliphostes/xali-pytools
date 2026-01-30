"""
Traction vector computation on triangular surfaces.

Provides functions to compute:
- Triangle unit normals from mesh geometry
- Traction vectors (stress applied to surface)
- Resolved normal and shear stresses

Sign convention: Normals point outward, positive sigma_n = tension
"""

import numpy as np
from typing import Tuple


def compute_triangle_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute unit normal vectors for triangles.

    Args:
        positions: Vertex positions, shape (n_vertices, 3)
        indices: Triangle indices, shape (n_triangles, 3)

    Returns:
        Unit normals, shape (n_triangles, 3)
    """
    # Get triangle vertices
    v0 = positions[indices[:, 0]]
    v1 = positions[indices[:, 1]]
    v2 = positions[indices[:, 2]]

    # Compute edges
    e1 = v1 - v0
    e2 = v2 - v0

    # Cross product gives normal (right-hand rule)
    normals = np.cross(e1, e2)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)  # Avoid division by zero

    return normals / lengths


def stress_6_to_matrix(stress: np.ndarray) -> np.ndarray:
    """
    Convert 6-component stress to 3x3 symmetric matrix.

    Storage order: [Sxx, Sxy, Sxz, Syy, Syz, Szz]

    Args:
        stress: Stress tensor, shape (6,) or (n, 6)

    Returns:
        Symmetric matrix, shape (3, 3) or (n, 3, 3)
    """
    stress = np.asarray(stress)
    single = stress.ndim == 1
    if single:
        stress = stress.reshape(1, 6)

    n = stress.shape[0]
    matrices = np.zeros((n, 3, 3))

    # Fill symmetric matrix
    # [Sxx, Sxy, Sxz, Syy, Syz, Szz] -> indices [0, 1, 2, 3, 4, 5]
    matrices[:, 0, 0] = stress[:, 0]  # Sxx
    matrices[:, 0, 1] = stress[:, 1]  # Sxy
    matrices[:, 0, 2] = stress[:, 2]  # Sxz
    matrices[:, 1, 0] = stress[:, 1]  # Sxy (symmetric)
    matrices[:, 1, 1] = stress[:, 3]  # Syy
    matrices[:, 1, 2] = stress[:, 4]  # Syz
    matrices[:, 2, 0] = stress[:, 2]  # Sxz (symmetric)
    matrices[:, 2, 1] = stress[:, 4]  # Syz (symmetric)
    matrices[:, 2, 2] = stress[:, 5]  # Szz

    if single:
        return matrices[0]
    return matrices


def compute_traction(stress: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute traction vectors on surfaces with given normals.

    Traction vector T = S @ n where S is the stress tensor and n is the unit normal.

    Args:
        stress: Stress tensor, shape (6,) for uniform stress, or (n_triangles, 6) for per-triangle
        normals: Unit normal vectors, shape (n_triangles, 3)

    Returns:
        Traction vectors, shape (n_triangles, 3)
    """
    stress = np.asarray(stress)
    normals = np.asarray(normals)
    n_triangles = normals.shape[0]

    # Convert stress to matrix form
    if stress.ndim == 1:
        # Single stress for all triangles
        stress_matrix = stress_6_to_matrix(stress)
        # T = S @ n for each triangle
        traction = np.einsum('ij,nj->ni', stress_matrix, normals)
    else:
        # Per-triangle stress
        stress_matrices = stress_6_to_matrix(stress)
        # T_n = S_n @ n_n for each triangle
        traction = np.einsum('nij,nj->ni', stress_matrices, normals)

    return traction


def resolve_stress(stress: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve stress into normal and shear components on surfaces.

    The traction vector T = S @ n is decomposed into:
    - Normal component: sigma_n = T . n (scalar, positive = tension)
    - Shear component: tau = ||T - sigma_n * n|| (magnitude, always positive)

    Sign convention (tension positive):
    - sigma_n > 0: tension (surface being pulled apart)
    - sigma_n < 0: compression (surface being pushed together)

    Args:
        stress: Stress tensor, shape (6,) for uniform stress, or (n_triangles, 6)
        normals: Unit normal vectors, shape (n_triangles, 3)

    Returns:
        sigma_n: Normal stress, shape (n_triangles,) - positive = tension
        tau: Shear stress magnitude, shape (n_triangles,) - always positive
    """
    # Compute traction
    traction = compute_traction(stress, normals)

    # Normal component: sigma_n = T . n
    sigma_n = np.einsum('ni,ni->n', traction, normals)

    # Shear component: tau = ||T - sigma_n * n||
    # T_shear = T - sigma_n * n
    traction_shear = traction - sigma_n[:, np.newaxis] * normals
    tau = np.linalg.norm(traction_shear, axis=1)

    return sigma_n, tau


def compute_shear_direction(stress: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute the direction of maximum shear stress on each surface.

    The shear direction lies in the plane of the surface and points in the
    direction of the shear traction component.

    Args:
        stress: Stress tensor, shape (6,) or (n_triangles, 6)
        normals: Unit normal vectors, shape (n_triangles, 3)

    Returns:
        Unit shear directions, shape (n_triangles, 3)
        Returns zero vector where shear stress is negligible.
    """
    # Compute traction
    traction = compute_traction(stress, normals)

    # Normal component: sigma_n = T . n
    sigma_n = np.einsum('ni,ni->n', traction, normals)

    # Shear component vector: T_shear = T - sigma_n * n
    traction_shear = traction - sigma_n[:, np.newaxis] * normals

    # Normalize to get direction
    tau = np.linalg.norm(traction_shear, axis=1, keepdims=True)
    tau = np.maximum(tau, 1e-10)  # Avoid division by zero

    shear_direction = traction_shear / tau

    # Set to zero where shear is negligible
    negligible = tau.flatten() < 1e-10
    shear_direction[negligible] = 0

    return shear_direction
