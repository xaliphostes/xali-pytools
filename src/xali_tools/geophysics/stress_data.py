"""
Cost functions for stress inversion using principal direction and stress ratio data.

Common data types:
- SHmax: Maximum horizontal principal stress direction
- R = (S2-S3)/(S1-S3): Stress ratio (shape factor)
  - R = 0: S2 = S3 (prolate/uniaxial compression)
  - R = 0.5: S2 = (S1+S3)/2 (triaxial)
  - R = 1: S1 = S2 (oblate/uniaxial extension)
"""

import numpy as np
from .stress_utils import principal_directions


def compute_stress_ratio(stress: np.ndarray) -> float:
    """
    Compute the stress ratio R = (S2-S3)/(S1-S3).

    Args:
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Stress ratio R in [0, 1]. Returns 0.5 if S1 = S3.
    """
    values, _ = principal_directions(stress)
    s1, s2, s3 = values  # Already sorted: s1 >= s2 >= s3

    denominator = s1 - s3
    if abs(denominator) < 1e-10:
        return 0.5  # Hydrostatic stress, ratio undefined

    return (s2 - s3) / denominator


def cost_direction(
    observed_direction: np.ndarray,
    stress: np.ndarray,
    principal_index: int = 0
) -> float:
    """
    Compute cost for principal direction alignment.

    Cost = 1 - |dot(observed, computed)|

    Args:
        observed_direction: Observed principal direction [nx, ny, nz]
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        principal_index: Which principal direction to compare:
            0 = S1 (maximum), 1 = S2 (intermediate), 2 = S3 (minimum)

    Returns:
        Cost value between 0 (perfect alignment) and 1 (perpendicular).
    """
    obs = np.asarray(observed_direction, dtype=np.float64)
    norm = np.linalg.norm(obs)
    if norm > 0:
        obs = obs / norm

    _, directions = principal_directions(stress)
    computed = directions[principal_index]

    return 1.0 - np.abs(np.dot(obs, computed))


def cost_stress_ratio(observed_R: float, stress: np.ndarray) -> float:
    """
    Compute cost for stress ratio.

    Cost = |R_observed - R_computed|

    Args:
        observed_R: Observed stress ratio R = (S2-S3)/(S1-S3)
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]

    Returns:
        Cost value between 0 (perfect match) and 1 (maximum mismatch).
    """
    computed_R = compute_stress_ratio(stress)
    return abs(observed_R - computed_R)


def cost_direction_and_ratio(
    observed_direction: np.ndarray,
    observed_R: float,
    stress: np.ndarray,
    principal_index: int = 0,
    weight_direction: float = 0.5,
    weight_ratio: float = 0.5
) -> float:
    """
    Combined cost for principal direction and stress ratio.

    Cost = w_dir * cost_direction + w_ratio * cost_ratio

    Args:
        observed_direction: Observed principal direction [nx, ny, nz]
        observed_R: Observed stress ratio R = (S2-S3)/(S1-S3)
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        principal_index: Which principal direction to compare (0=S1, 1=S2, 2=S3)
        weight_direction: Weight for direction cost
        weight_ratio: Weight for ratio cost

    Returns:
        Combined cost value.
    """
    c_dir = cost_direction(observed_direction, stress, principal_index)
    c_ratio = cost_stress_ratio(observed_R, stress)

    # Normalize weights
    total_weight = weight_direction + weight_ratio
    w_dir = weight_direction / total_weight
    w_ratio = weight_ratio / total_weight

    return w_dir * c_dir + w_ratio * c_ratio


def cost_multiple_observations(
    observed_directions: np.ndarray,
    observed_Rs: np.ndarray,
    stress: np.ndarray,
    principal_index: int = 0,
    weight_direction: float = 0.5,
    weight_ratio: float = 0.5,
    observation_weights: np.ndarray = None
) -> float:
    """
    Compute cost for multiple direction + ratio observations.

    Args:
        observed_directions: Array of shape (n, 3) with observed directions.
        observed_Rs: Array of n stress ratios.
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        principal_index: Which principal direction to compare (0=S1, 1=S2, 2=S3)
        weight_direction: Weight for direction cost
        weight_ratio: Weight for ratio cost
        observation_weights: Optional weights for each observation.

    Returns:
        Weighted mean cost.
    """
    observed_directions = np.asarray(observed_directions, dtype=np.float64).reshape(-1, 3)
    observed_Rs = np.asarray(observed_Rs, dtype=np.float64).flatten()
    n_obs = observed_directions.shape[0]

    if observation_weights is None:
        observation_weights = np.ones(n_obs)
    observation_weights = np.asarray(observation_weights, dtype=np.float64)
    observation_weights = observation_weights / observation_weights.sum()

    # Get computed direction and ratio
    values, directions = principal_directions(stress)
    computed_dir = directions[principal_index]
    s1, s2, s3 = values
    denom = s1 - s3
    computed_R = (s2 - s3) / denom if abs(denom) > 1e-10 else 0.5

    # Normalize observed directions
    norms = np.linalg.norm(observed_directions, axis=1, keepdims=True)
    norms[norms == 0] = 1
    observed_directions = observed_directions / norms

    # Direction costs
    dots = np.abs(observed_directions @ computed_dir)
    dir_costs = 1.0 - dots

    # Ratio costs
    ratio_costs = np.abs(observed_Rs - computed_R)

    # Combine
    total_weight = weight_direction + weight_ratio
    w_dir = weight_direction / total_weight
    w_ratio = weight_ratio / total_weight

    individual_costs = w_dir * dir_costs + w_ratio * ratio_costs

    return np.sum(observation_weights * individual_costs)


def cost_shmax_and_ratio(
    shmax_azimuths: np.ndarray,
    observed_Rs: np.ndarray,
    stress: np.ndarray,
    weight_direction: float = 0.5,
    weight_ratio: float = 0.5,
    observation_weights: np.ndarray = None
) -> float:
    """
    Compute cost using SHmax azimuth (horizontal plane) and stress ratio.

    SHmax is the maximum horizontal principal stress, given as azimuth in degrees
    (0° = North, 90° = East).

    Args:
        shmax_azimuths: Array of SHmax azimuths in degrees.
        observed_Rs: Array of stress ratios.
        stress: 6-component stress [σxx, σxy, σxz, σyy, σyz, σzz]
        weight_direction: Weight for direction cost.
        weight_ratio: Weight for ratio cost.
        observation_weights: Optional weights for each observation.

    Returns:
        Weighted mean cost.
    """
    shmax_azimuths = np.asarray(shmax_azimuths, dtype=np.float64).flatten()
    observed_Rs = np.asarray(observed_Rs, dtype=np.float64).flatten()
    n_obs = len(shmax_azimuths)

    if observation_weights is None:
        observation_weights = np.ones(n_obs)
    observation_weights = np.asarray(observation_weights, dtype=np.float64)
    observation_weights = observation_weights / observation_weights.sum()

    # Convert azimuths to unit vectors in horizontal plane (x=East, y=North, z=Up)
    # Azimuth 0° = North (+y), 90° = East (+x)
    azimuths_rad = np.radians(shmax_azimuths)
    observed_directions = np.zeros((n_obs, 3))
    observed_directions[:, 0] = np.sin(azimuths_rad)  # East component
    observed_directions[:, 1] = np.cos(azimuths_rad)  # North component
    # z = 0 (horizontal)

    # Get principal stress info
    values, directions = principal_directions(stress)
    s1, s2, s3 = values
    denom = s1 - s3
    computed_R = (s2 - s3) / denom if abs(denom) > 1e-10 else 0.5

    # For SHmax, we need to find the most compressive horizontal direction
    # Project S1, S2, S3 onto horizontal plane and find which is most compressive
    # Typically S1 in thrust, S2 in strike-slip, S3 in normal faulting regimes

    # Project all principal directions onto horizontal plane
    horizontal_projections = directions.copy()
    horizontal_projections[:, 2] = 0  # Zero out vertical component
    horizontal_norms = np.linalg.norm(horizontal_projections, axis=1)

    # Find the principal direction with largest horizontal component
    # weighted by stress magnitude for compression
    # SHmax direction is the one with most compressive stress in horizontal
    best_dir_idx = 0
    best_horizontal_stress = -np.inf

    for i in range(3):
        if horizontal_norms[i] > 0.1:  # Has significant horizontal component
            # Project stress value onto horizontal
            horizontal_stress = values[i] * horizontal_norms[i]
            if horizontal_stress > best_horizontal_stress:
                best_horizontal_stress = horizontal_stress
                best_dir_idx = i

    # Get the horizontal direction of SHmax
    shmax_computed = horizontal_projections[best_dir_idx]
    if np.linalg.norm(shmax_computed) > 0:
        shmax_computed = shmax_computed / np.linalg.norm(shmax_computed)
    else:
        shmax_computed = np.array([1.0, 0.0, 0.0])  # Default to East

    # Compute direction costs (only compare horizontal components)
    # Use angular difference: cost = 1 - |cos(angle_diff)|
    # Since both are unit vectors in xy plane: dot = cos(angle)
    dots = np.abs(observed_directions[:, 0] * shmax_computed[0] +
                  observed_directions[:, 1] * shmax_computed[1])
    dir_costs = 1.0 - dots

    # Ratio costs
    ratio_costs = np.abs(observed_Rs - computed_R)

    # Combine
    total_weight = weight_direction + weight_ratio
    w_dir = weight_direction / total_weight
    w_ratio = weight_ratio / total_weight

    individual_costs = w_dir * dir_costs + w_ratio * ratio_costs

    return np.sum(observation_weights * individual_costs)
