"""
Stylolite cost functions for stress inversion.

A stylolite is a pressure-solution feature characterized by its normal vector.
A stylolite is well-aligned with a stress tensor if the maximum principal stress
direction (S1) is collinear with the stylolite normal.

Cost function: c = 1 - |dot(n, S1)|
- c = 0: perfect alignment (n parallel to S1)
- c = 1: worst alignment (n perpendicular to S1)
"""

import numpy as np
from .stress_utils import principal_directions


def cost_single_stylolite(normal: np.ndarray, stress: np.ndarray) -> float:
    """
    Compute cost for a single stylolite given a stress tensor.

    Cost = 1 - |dot(n, S1)|

    A stylolite forms perpendicular to the maximum principal stress (S1).
    Cost is 0 when the stylolite normal is aligned with S1.

    Args:
        normal: Stylolite normal vector [nx, ny, nz]
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Cost value between 0 (perfect alignment) and 1 (worst alignment).
    """
    n = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm

    _, directions = principal_directions(stress)
    s1 = directions[0]  # Maximum principal stress direction

    return 1.0 - np.abs(np.dot(n, s1))


def cost_multiple_stylolites(normals: np.ndarray, stress: np.ndarray,
                              weights: np.ndarray = None) -> float:
    """
    Compute total cost for multiple stylolites given a stress tensor.

    Args:
        normals: Array of shape (n, 3) with stylolite normal vectors.
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        weights: Optional weights for each stylolite. If None, equal weights.

    Returns:
        Weighted mean cost value.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n_stylolites = normals.shape[0]

    if weights is None:
        weights = np.ones(n_stylolites)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()  # Normalize

    _, directions = principal_directions(stress)
    s1 = directions[0]  # Maximum principal stress direction

    # Normalize all normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # Compute costs: 1 - |dot(n, S1)|
    dots = np.abs(normals @ s1)
    costs = 1.0 - dots

    return np.sum(weights * costs)
