"""
Utility functions for stress tensor operations.
"""

import numpy as np
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector or array of vectors."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v / norms


def stress_to_tensor(stress: np.ndarray) -> np.ndarray:
    """
    Convert 6-component stress vector to 3x3 symmetric tensor.

    Args:
        stress: Array of 6 components [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        3x3 symmetric stress tensor matrix.
    """
    s = np.asarray(stress, dtype=np.float64)
    return np.array([
        [s[0], s[1], s[2]],
        [s[1], s[3], s[4]],
        [s[2], s[4], s[5]]
    ])


def principal_directions(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal stress values and directions.

    Args:
        stress: Array of 6 components [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Tuple of (values, directions) where:
        - values: array [σ1, σ2, σ3] with σ1 >= σ2 >= σ3
        - directions: array (3, 3) where row i is the direction for σi
    """
    tensor = stress_to_tensor(stress)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)

    # Sort in descending order (σ1 >= σ2 >= σ3)
    idx = np.argsort(eigenvalues)[::-1]
    values = eigenvalues[idx]
    directions = eigenvectors[:, idx].T  # Rows are directions

    return values, directions


def generate_random_stress(sigma_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Generate a random stress tensor with specified eigenvalue range.

    Args:
        sigma_range: (min, max) range for principal stress values.

    Returns:
        6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
    """
    # Generate random principal values
    s1, s2, s3 = np.sort(np.random.uniform(sigma_range[0], sigma_range[1], 3))[::-1]

    # Generate random orthonormal directions
    v1 = np.random.randn(3)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.random.randn(3)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    v3 = np.cross(v1, v2)

    # Build rotation matrix (columns are eigenvectors)
    R = np.column_stack([v1, v2, v3])

    # Build stress tensor: R @ diag(s1, s2, s3) @ R.T
    D = np.diag([s1, s2, s3])
    tensor = R @ D @ R.T

    # Extract 6 components [σxx, σxy, σxz, σyy, σyz, σzz]
    return np.array([
        tensor[0, 0], tensor[0, 1], tensor[0, 2],
        tensor[1, 1], tensor[1, 2], tensor[2, 2]
    ])


def generate_random_andersonian_stress(sigma_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Generate a random Andersonian stress tensor (one principal axis vertical).

    For Andersonian stress, σxz = σyz = 0 (no shear on vertical planes),
    meaning one principal stress is always vertical.

    The 4 free parameters are: σxx, σxy, σyy, σzz

    Args:
        sigma_range: (min, max) range for stress component values.

    Returns:
        6-component stress [σxx, σxy, 0, σyy, 0, σzz]
    """
    # Generate random horizontal stress components
    sigma_xx = np.random.uniform(sigma_range[0], sigma_range[1])
    sigma_xy = np.random.uniform(sigma_range[0], sigma_range[1])
    sigma_yy = np.random.uniform(sigma_range[0], sigma_range[1])
    sigma_zz = np.random.uniform(sigma_range[0], sigma_range[1])

    # σxz = σyz = 0 for Andersonian stress
    return np.array([sigma_xx, sigma_xy, 0.0, sigma_yy, 0.0, sigma_zz])


def andersonian_4_to_6(params: np.ndarray) -> np.ndarray:
    """
    Convert 4 Andersonian parameters to 6-component stress.

    Args:
        params: Array [σxx, σxy, σyy, σzz]

    Returns:
        6-component stress [σxx, σxy, 0, σyy, 0, σzz]
    """
    return np.array([params[0], params[1], 0.0, params[2], 0.0, params[3]])


def stress_6_to_andersonian_4(stress: np.ndarray) -> np.ndarray:
    """
    Extract 4 Andersonian parameters from 6-component stress.

    Args:
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Array [σxx, σxy, σyy, σzz]
    """
    return np.array([stress[0], stress[1], stress[3], stress[5]])
