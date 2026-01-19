"""
Joint/fracture cost functions for stress inversion.

A fracture joint is characterized by its normal vector. A joint is well-aligned
with a stress tensor if the minimum principal stress direction (S3) is collinear
with the joint normal.

Cost function: c = 1 - |dot(n, S3)|
- c = 0: perfect alignment (n parallel to S3)
- c = 1: worst alignment (n perpendicular to S3)
"""

import numpy as np
from .stress_utils import principal_directions


def cost_single_joint(normal: np.ndarray, stress: np.ndarray) -> float:
    """
    Compute cost for a single joint given a stress tensor.

    Cost = 1 - |dot(n, S3)|

    A joint opens perpendicular to the minimum principal stress (S3).
    Cost is 0 when the joint normal is aligned with S3.

    Args:
        normal: Joint normal vector [nx, ny, nz]
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Cost value between 0 (perfect alignment) and 1 (worst alignment).
    """
    n = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm

    _, directions = principal_directions(stress)
    s3 = directions[2]  # Minimum principal stress direction

    return 1.0 - np.abs(np.dot(n, s3))


def cost_multiple_joints(normals: np.ndarray, stress: np.ndarray,
                         weights: np.ndarray = None) -> float:
    """
    Compute total cost for multiple joints given a stress tensor.

    Args:
        normals: Array of shape (n, 3) with joint normal vectors.
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        weights: Optional weights for each joint. If None, equal weights.

    Returns:
        Weighted mean cost value.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n_joints = normals.shape[0]

    if weights is None:
        weights = np.ones(n_joints)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()  # Normalize

    _, directions = principal_directions(stress)
    s3 = directions[2]  # Minimum principal stress direction

    # Normalize all normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # Compute costs: 1 - |dot(n, S3)|
    dots = np.abs(normals @ s3)
    costs = 1.0 - dots

    return np.sum(weights * costs)
